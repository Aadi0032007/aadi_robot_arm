"""Main tool: teleoperate the follower, record sessions, and host the AI link.

Arguments follow lerobot-teleoperate / lerobot-record:

    python -m src.teleop --robot.type=so101_follower --robot.port=COM4 --robot.id=aadi_so101_follower \\
                         --teleop.type=so101_leader --teleop.port=COM5 --teleop.id=aadi_so101_leader
    python -m src.teleop --preset so101 --display_data=true          # presets fill in the rest
    python -m src.teleop --preset so101 --resume                      # append to the latest run
    python -m src.teleop --preset so101 --run_dir <run>               # append to a given run
    python -m src.teleop --preset so101_keyboard --teleop_time_s=120

Keys (global, via LeRobot's keyboard listener):

    Space       idle -> start recording a new session_<i>
                recording -> stop and save it (one session = one episode)
    Left or r   recording -> delete the ongoing session and start it again
                idle -> delete the last session saved in this launch and start it again
                (--record.redo_starts_recording=false goes idle instead)
    a           hand control to the AI (inference.py must be running) / take it back
    Esc or q    quit; an ongoing session is saved first

With --teleop.type=vr_ee the second VR controller has Space (X) and Left (Y) too.

Every tick performs the same steps as LeRobot's ``record_loop``: observation
(motors + frames from ``cameras.py``) → observation processor; leader action →
teleop processor → robot action processor → ``send_action``. While a session is
open the processed observation and the action actually applied (human or AI) are
packed with ``build_dataset_frame`` and written by ``recorder.SessionWriter``.

This process is also the robot server for ``inference.py``: it publishes the joint
state on ``ai.state_port`` every tick, the cameras publish frames on the frame bus,
and it applies actions received on ``ai.action_port`` while AI control is on. Moving
the leader more than ``ai.human_takeover_threshold`` hands control straight back.
"""

from __future__ import annotations

if __package__ in (None, ""):  # allow `python src/teleop.py`
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "src"

import logging
import shutil
import time
from pathlib import Path
from typing import Any

from .ai_link import UdpJsonListener, UdpJsonSender
from .cameras import CamerasManager, RobotWithCameras
from .config import ArmConfig
from .devices import add_bool_flag, describe, load_config, make_parser, make_robot, make_teleop
from .recorder import SessionWriter, latest_run_dir, new_run_dir, next_session_dir

logger = logging.getLogger(__name__)

CONTROLS_HELP = "Space=record/stop  Left/r=redo  a=AI on/off  Esc/q=quit"


def build_features(robot) -> dict:
    """Same feature dict ``lerobot-record`` would create for this robot (videos on)."""
    from lerobot.utils.constants import ACTION, OBS_STR
    from lerobot.utils.feature_utils import hw_to_dataset_features

    return {
        **hw_to_dataset_features(robot.action_features, ACTION),
        **hw_to_dataset_features(robot.observation_features, OBS_STR, use_video=True),
    }


def _dispatch(events: dict[str, bool]):
    def on_key(name: str) -> None:
        key = name.lower()
        if key == "space":
            events["toggle_record"] = True
        elif key in ("left", "r"):
            events["redo"] = True
        elif key == "a":
            events["toggle_ai"] = True
        elif key in ("esc", "q"):
            events["stop"] = True

    return on_key


def teleop_and_record(
    cfg: ArmConfig,
    run_dir: Path,
    teleop_time_s: float | None = None,
) -> list[Path]:
    from lerobot.processor import make_default_processors
    from lerobot.utils.constants import ACTION, OBS_STR
    from lerobot.utils.feature_utils import build_dataset_frame
    from lerobot.utils.keyboard_input import create_key_listener
    from lerobot.utils.robot_utils import precise_sleep
    from lerobot.utils.utils import log_say, move_cursor_up
    from lerobot.utils.visualization_utils import init_visualization, log_visualization_data, shutdown_visualization

    rec = cfg.record
    fps = cfg.fps
    task = cfg.task
    play_sounds = rec.play_sounds
    display_data = rec.display_data
    display_mode = rec.display_mode
    compress_images = rec.display_compressed_images or (
        display_data and rec.display_ip is not None and rec.display_port is not None
    )

    cameras = CamerasManager(cfg)
    cameras.start()
    robot = RobotWithCameras(make_robot(cfg), cameras, stale_s=rec.frame_stale_s)
    teleop = make_teleop(cfg)
    teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()
    features = build_features(robot)
    state_names = [k for k, t in robot.observation_features.items() if t is float]
    action_names = list(robot.action_features)

    ai = cfg.ai
    # AI link only when asked for (--ai.enabled=true): no sockets otherwise.
    ai_rx = UdpJsonListener(ai.host, ai.action_port, "ai-action") if ai.enabled else None
    state_tx = UdpJsonSender(ai.host, ai.state_port) if ai.enabled else None
    controls_help = CONTROLS_HELP if ai.enabled else CONTROLS_HELP.replace("  a=AI on/off", "")

    base_meta = {
        "robot_type": robot.name,
        "robot_id": robot.id,
        "teleop_type": cfg.teleop_type,
        "task": task,
        "preset": cfg.name,
        "camera_backends": {n: c.backend for n, c in cameras.captures.items()},
    }

    logging.info("Run directory: %s", run_dir)
    logging.info("Features: %s", {k: (v["dtype"], tuple(v["shape"])) for k, v in features.items()})

    events = {"toggle_record": False, "redo": False, "stop": False, "toggle_ai": False}
    listener = create_key_listener(_dispatch(events), controls_help=controls_help)
    if listener is None:
        logging.warning("No keyboard backend available: teleop runs, but the keys will not work.")

    if display_data:
        init_visualization(display_mode, session_name="revobots_arm_teleop", ip=rec.display_ip, port=rec.display_port)

    session: SessionWriter | None = None
    last_saved: Path | None = None
    saved: list[Path] = []
    ai_active = False
    ai_leader_ref: dict[str, float] | None = None
    ai_waiting_logged = False
    ai_keys_warned = False
    period = 1.0 / fps

    # -- session helpers --------------------------------------------------------

    def start_session() -> None:
        nonlocal session
        session_dir = next_session_dir(run_dir)
        session = SessionWriter(
            session_dir, fps=fps, features=features, meta={**base_meta, "session_name": session_dir.name}, codec=rec.codec
        )
        log_say(f"Recording {session_dir.name}", play_sounds)

    def stop_session() -> None:
        nonlocal session, last_saved
        assert session is not None
        n_frames = session.num_frames
        path = session.close()
        session = None
        if path is None:
            return
        if n_frames == 0:
            shutil.rmtree(path, ignore_errors=True)
            logging.info("%s had no frames, removed.", path.name)
            return
        last_saved = path
        saved.append(path)
        log_say(f"Saved {path.name} ({n_frames} frames)", play_sounds)

    def redo() -> bool:
        nonlocal session, last_saved
        if session is not None:
            name = session.session_dir.name
            session.close(discard=True)
            session = None
            log_say(f"Discarded {name}, redo", play_sounds)
            return True
        if last_saved is not None and last_saved.exists():
            shutil.rmtree(last_saved, ignore_errors=True)
            if last_saved in saved:
                saved.remove(last_saved)
            log_say(f"Deleted {last_saved.name}, redo", play_sounds)
            last_saved = saved[-1] if saved else None
            return True
        logging.info("Nothing to redo.")
        return False

    # -- AI helpers ------------------------------------------------------------

    def ai_action_from_packet(pkt: dict[str, Any] | None) -> dict[str, float] | None:
        nonlocal ai_keys_warned
        if not pkt:
            return None
        raw = pkt.get("action") or {}
        missing = [k for k in action_names if k not in raw]
        if missing:
            if not ai_keys_warned:
                logging.warning("AI action lacks %s; ignoring AI packets until they match", missing)
                ai_keys_warned = True
            return None
        return {k: float(raw[k]) for k in action_names}

    def leader_moved(leader_action: dict[str, float]) -> bool:
        if ai_leader_ref is None:
            return False
        delta = max((abs(leader_action[k] - ai_leader_ref.get(k, leader_action[k])) for k in leader_action), default=0.0)
        return delta > ai.human_takeover_threshold

    def reseed_teleop() -> None:
        """Keyboard-jog style devices integrate their own target; restart it from the arm's pose."""
        if hasattr(teleop, "needs_seed"):
            teleop.needs_seed = True

    # -- run -------------------------------------------------------------------

    try:
        robot.connect()
        teleop.connect()
        if ai_rx is not None:
            ai_rx.start()
        if hasattr(teleop, "help_text"):
            logging.info(teleop.help_text)
        log_say("Teleop ready. " + controls_help, play_sounds)
        t_start = time.perf_counter()

        while not events["stop"]:
            loop_start = time.perf_counter()
            if teleop_time_s is not None and loop_start - t_start >= teleop_time_s:
                logging.info("teleop_time_s=%.0f reached", teleop_time_s)
                break

            # keyboard events
            if events["redo"]:
                events["redo"] = False
                if redo() and rec.redo_starts_recording:
                    events["toggle_record"] = True
            if events["toggle_record"]:
                events["toggle_record"] = False
                if session is None:
                    start_session()
                else:
                    stop_session()
            if events["toggle_ai"] and not ai.enabled:
                events["toggle_ai"] = False
                logging.info("AI link is off. Restart teleop with --ai.enabled=true to use inference.py.")
            if events["toggle_ai"]:
                events["toggle_ai"] = False
                ai_active = not ai_active
                ai_leader_ref = None
                ai_waiting_logged = False
                if not ai_active:
                    reseed_teleop()
                log_say("AI control ON. Move the leader or press a to take over." if ai_active else "AI control OFF",
                        play_sounds)

            # observation (motors + camera frames)
            obs = robot.get_observation()
            if getattr(teleop, "needs_seed", False):
                teleop.seed_from_observation(obs)

            # leader
            leader_action: dict[str, float] | None = None
            if not ai_active or ai.read_leader_while_ai:
                leader_action = teleop_action_processor((teleop.get_action(), obs))
            # buttons on the input device itself (VR controller: X = Space, Y = Left) act like the keys
            if hasattr(teleop, "pop_events"):
                for name in teleop.pop_events():
                    if name in events:
                        events[name] = True

            # arbitration: human first
            source = "human"
            action_values: dict[str, float] | None = leader_action
            if ai_active:
                if leader_action is not None and ai_leader_ref is None:
                    ai_leader_ref = dict(leader_action)
                if leader_action is not None and leader_moved(leader_action):
                    ai_active = False
                    ai_leader_ref = None
                    reseed_teleop()
                    log_say("Human took over, AI control OFF", play_sounds)
                else:
                    ai_action = ai_action_from_packet(ai_rx.latest(max_age_s=ai.action_ttl_s) if ai_rx else None)
                    if ai_action is None:
                        source = "ai-waiting"
                        action_values = None  # hold position
                        if not ai_waiting_logged:
                            logging.info("AI on but no fresh action on udp:%s (is inference.py running?)", ai.action_port)
                            ai_waiting_logged = True
                    else:
                        source = "ai"
                        action_values = ai_action
                        ai_waiting_logged = False

            if action_values is not None:
                robot.send_action(robot_action_processor((action_values, obs)))

            obs_processed = None
            if session is not None or display_data:
                obs_processed = robot_observation_processor(obs)

            # state feed for inference.py (scalars only; frames go over the frame bus)
            if state_tx is not None:
                state_tx.send({
                    "observation": {k: float(obs[k]) for k in state_names if k in obs},
                    "action": action_values,
                    "source": source,
                    "ai": ai_active,
                    "recording": session is not None,
                })

            # dataset frame
            if session is not None and action_values is not None:
                frame = {
                    **build_dataset_frame(features, obs_processed, prefix=OBS_STR),
                    **build_dataset_frame(features, action_values, prefix=ACTION),
                    "task": task,
                }
                session.add_frame(frame, extra={"source": source})

            if display_data:
                log_visualization_data(
                    display_mode, observation=obs_processed, action=action_values, compress_images=compress_images
                )

            # pacing + status line
            dt = time.perf_counter() - loop_start
            precise_sleep(max(period - dt, 0.0))
            loop_s = time.perf_counter() - loop_start
            if session is not None:
                secs = session.num_frames / fps
                state = f"REC {session.session_dir.name} {int(secs) // 60:02d}:{int(secs) % 60:02d} {session.num_frames}f"
            else:
                state = f"idle  (saved {len(saved)})"
            ctrl = {"human": "HUMAN", "ai": "AI", "ai-waiting": "AI(wait)"}[source]
            print(f"{state:<34} {ctrl:<9} loop {loop_s * 1e3:6.1f}ms ({1 / loop_s if loop_s else 0:3.0f} Hz)   {controls_help}")
            move_cursor_up(1)
            if hasattr(teleop, "show_status"):  # e.g. the VR headset's status panel
                teleop.show_status(f"{state}  {ctrl}")
    except KeyboardInterrupt:
        pass
    finally:
        print()
        if session is not None:
            logging.info("Saving the ongoing session before exit.")
            stop_session()
        for label, fn in (
            ("robot", lambda: robot.disconnect() if robot.is_connected else None),
            ("teleop", lambda: teleop.disconnect() if teleop.is_connected else None),
            ("ai_rx", ai_rx.stop if ai_rx is not None else (lambda: None)),
            ("state_tx", state_tx.close if state_tx is not None else (lambda: None)),
            ("cameras", cameras.stop),
            ("listener", listener.stop if listener is not None else (lambda: None)),
        ):
            try:
                fn()
            except Exception as exc:  # noqa: BLE001
                logging.warning("%s stop error: %s", label, exc)
        if display_data:
            shutdown_visualization(display_mode)
        log_say("Teleop stopped", play_sounds, blocking=True)

    return saved


def resolve_run_dir(cfg: ArmConfig, run_dir: str | None, resume: bool) -> Path:
    """``--run_dir`` wins; ``--resume`` appends to the latest run; otherwise a new run folder."""
    if run_dir:
        return Path(run_dir).expanduser()
    if resume:
        latest = latest_run_dir(cfg.sessions_root)
        if latest is not None:
            logging.info("Resuming run %s", latest)
            return latest
        logging.warning("--resume given but no previous run under %s; starting a new one", cfg.sessions_root)
    return new_run_dir(cfg.sessions_root)


def main() -> None:
    parser = make_parser(__doc__)
    parser.add_argument("--run_dir", default=None, help="Append sessions to this run folder")
    add_bool_flag(parser, "--resume", "Append sessions to the most recent run under sessions_root")
    parser.add_argument("--teleop_time_s", type=float, default=None, help="Stop after this many seconds (default: until Esc)")
    args, overrides = parser.parse_known_args()

    cfg = load_config(args.preset, overrides)

    from lerobot.utils.utils import init_logging

    init_logging()
    logging.info("\n%s", describe(cfg))

    run_dir = resolve_run_dir(cfg, args.run_dir, args.resume)
    saved = teleop_and_record(cfg, run_dir=run_dir, teleop_time_s=args.teleop_time_s)

    print(f"\n{len(saved)} session(s) saved under {run_dir}")
    if saved:
        print(f'Next: python -m src.convert --sessions_dir "{run_dir}" --dataset.repo_id={cfg.dataset.repo_id}')


if __name__ == "__main__":
    main()

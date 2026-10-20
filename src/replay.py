"""Replay recorded actions on the follower arm.

Same arguments as lerobot-replay, plus ``--session_dir`` for a raw session:

    python -m src.replay --robot.type=so101_follower --robot.port=COM4 --robot.id=aadi_so101_follower \\
                         --session_dir <run>/session_0
    python -m src.replay --robot.type=so101_follower --robot.port=COM4 --robot.id=aadi_so101_follower \\
                         --dataset.repo_id=Aadi/revobots_so101_v1 --dataset.episode=0
    python -m src.replay --preset so101 --dataset.episode=3 --dataset.fps=15 --replay.ramp_s=0

Each action goes through the default robot action processor and is sent at the
recording fps (``--dataset.fps`` overrides). Before the first action the arm glides from
its current pose to the first recorded pose over ``replay.ramp_s`` seconds (``0`` disables)
so a replay never starts with a violent jump. Cameras are not opened.
"""

from __future__ import annotations

if __package__ in (None, ""):  # allow `python src/replay.py`
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "src"

import logging
import time
from pathlib import Path

from .devices import describe, load_config, make_parser, make_robot, overridden
from .recorder import load_meta, load_rows

logger = logging.getLogger(__name__)


def actions_from_session(session_dir: Path) -> tuple[list[dict[str, float]], int]:
    meta = load_meta(session_dir)
    rows = load_rows(session_dir)
    actions = [{k: float(v) for k, v in row["action"].items()} for row in rows]
    return actions, int(meta["fps"])


def actions_from_dataset(repo_id: str, episode: int, root: str | None) -> tuple[list[dict[str, float]], int]:
    from lerobot.datasets import LeRobotDataset
    from lerobot.utils.constants import ACTION

    dataset = LeRobotDataset(repo_id, root=root, episodes=[episode])
    names = dataset.features[ACTION]["names"]
    column = dataset.select_columns(ACTION)
    actions = []
    for idx in range(dataset.num_frames):
        vec = column[idx][ACTION]
        actions.append({name: float(vec[i]) for i, name in enumerate(names)})
    return actions, int(dataset.fps)


def ramp_to(robot, robot_action_processor, target: dict[str, float], seconds: float, fps: int) -> None:
    from lerobot.utils.robot_utils import precise_sleep

    steps = max(1, int(seconds * fps))
    obs = robot.get_observation()
    start = {k: float(obs[k]) for k in target if k in obs}
    if not start:
        return
    logger.info("Ramping to the first recorded pose over %.1fs", seconds)
    for i in range(1, steps + 1):
        alpha = i / steps
        action = {k: start[k] + (target[k] - start[k]) * alpha for k in start}
        t0 = time.perf_counter()
        obs = robot.get_observation()
        robot.send_action(robot_action_processor((action, obs)))
        precise_sleep(max(1 / fps - (time.perf_counter() - t0), 0.0))


def replay(robot, actions: list[dict[str, float]], fps: int, ramp_s: float = 0.0, play_sounds: bool = True) -> None:
    from lerobot.processor import make_default_robot_action_processor
    from lerobot.utils.robot_utils import precise_sleep
    from lerobot.utils.utils import log_say

    robot_action_processor = make_default_robot_action_processor()

    missing = [k for k in robot.action_features if k not in actions[0]]
    if missing:
        raise ValueError(f"Recorded actions lack joints the robot expects: {missing}")

    if ramp_s > 0:
        ramp_to(robot, robot_action_processor, actions[0], ramp_s, fps)

    log_say("Replaying episode", play_sounds, blocking=True)
    overruns = 0
    for action in actions:
        t0 = time.perf_counter()
        obs = robot.get_observation()
        robot.send_action(robot_action_processor((action, obs)))
        dt = time.perf_counter() - t0
        if dt > 1 / fps:
            overruns += 1
        precise_sleep(max(1 / fps - dt, 0.0))
    if overruns:
        logger.warning("%d/%d replay steps overran the %d fps budget", overruns, len(actions), fps)


def main() -> None:
    parser = make_parser(__doc__)
    parser.add_argument("--session_dir", default=None, help="Raw session folder (contains data.jsonl). Omit to replay --dataset.repo_id / --dataset.episode")
    args, overrides = parser.parse_known_args()

    cfg = load_config(args.preset, overrides)

    from lerobot.utils.utils import init_logging

    init_logging()
    logging.info("\n%s", describe(cfg))

    if args.session_dir:
        actions, fps = actions_from_session(Path(args.session_dir).expanduser())
        logging.info("Loaded %d actions from %s", len(actions), args.session_dir)
    else:
        actions, fps = actions_from_dataset(cfg.dataset.repo_id, cfg.dataset.episode, cfg.dataset.root)
        logging.info("Loaded %d actions from %s episode %d", len(actions), cfg.dataset.repo_id, cfg.dataset.episode)
    if not actions:
        raise SystemExit("Nothing to replay.")

    if overridden(overrides, "fps", "dataset.fps"):
        fps = cfg.fps

    robot = make_robot(cfg)
    robot.connect()
    try:
        replay(robot, actions, fps=fps, ramp_s=cfg.replay.ramp_s, play_sounds=cfg.record.play_sounds)
    except KeyboardInterrupt:
        logging.info("Interrupted.")
    finally:
        if robot.is_connected:
            robot.disconnect()


if __name__ == "__main__":
    main()

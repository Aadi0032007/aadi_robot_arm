"""Run a trained policy on the arm through a running ``teleop.py``.

Same arguments as lerobot_inference (``--robot.*`` is taken from the preset or given
explicitly; only its type matters here, since teleop owns the hardware):

    python -m src.inference --robot.type=so101_follower \\
                            --policy.path=outputs/train/act_so101/checkpoints/last/pretrained_model \\
                            --dataset_repo_id=Aadi/revobots_so101_v1 --single_task="Pick the cube" \\
                            --fps=30 --control_time_s=120 --policy.device=cuda --display_data=true
    python -m src.inference --preset so101 --policy.path=<ckpt>

Teleop keeps the cameras and the arm. This process attaches to the frame bus for
images, receives joint state on ``ai.state_port`` and sends actions to
``ai.action_port`` through ``remote_robot.ArmProxyRobot``. The arm only follows
those actions after you press ``a`` in the teleop window; moving the leader or
pressing ``a`` again takes control back.

Policy loading and the control loop come from
``lerobot.scripts.lerobot_inference_api`` (``build_policy_pipeline`` and
``run_inference``), unchanged.
"""

from __future__ import annotations

if __package__ in (None, ""):  # allow `python src/inference.py`
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "src"

import logging

from .devices import describe, load_config, make_parser

logger = logging.getLogger(__name__)


def main() -> None:
    parser = make_parser(__doc__)
    args, overrides = parser.parse_known_args()

    cfg = load_config(args.preset, overrides)
    pol = cfg.policy

    policy_path = pol.path
    dataset_repo_id = pol.dataset_repo_id or cfg.dataset.repo_id
    if not policy_path or not dataset_repo_id:
        raise SystemExit("Need --policy.path and --dataset_repo_id (or policy.path / dataset.repo_id in config.py).")
    dataset_root = pol.dataset_root or cfg.dataset.root
    device = pol.device
    duration_s = pol.duration_s
    fps = int(pol.fps or cfg.fps)
    task = cfg.task or None  # empty task → no prompt (ACT ignores it; VLA policies need --single_task)
    display_data = cfg.record.display_data

    from lerobot.scripts.lerobot_inference_api import build_policy_pipeline, run_inference
    from lerobot.utils.utils import init_logging, log_say

    from .remote_robot import ArmProxyRobot

    init_logging()
    logging.info("\n%s", describe(cfg))
    logging.info("policy=%s dataset=%s device=%s fps=%s control_time_s=%s", policy_path, dataset_repo_id, device, fps, duration_s)

    pipeline = build_policy_pipeline(
        policy_path=policy_path,
        dataset_repo_id=dataset_repo_id,
        device=device,
        dataset_root=dataset_root,
    )
    features = pipeline[5]

    robot = ArmProxyRobot(features=features, robot_type=cfg.robot_type, ai=cfg.ai)
    trained_cams = sorted(k for k, v in robot.observation_features.items() if isinstance(v, tuple))
    if set(trained_cams) != set(cfg.camera_names):
        logging.warning(
            "policy was trained with cameras %s, config has %s. Frames come from the frame bus by name, "
            "so teleop must publish exactly the trained names.", trained_cams, cfg.camera_names,
        )

    print()
    print("=" * 66)
    print("  ARM INFERENCE  (teleop.py must be running)")
    print("=" * 66)
    print(f"  policy   : {policy_path}")
    print(f"  dataset  : {dataset_repo_id}")
    print(f"  cameras  : {trained_cams}  ← frame bus lab_<name>")
    print(f"  state    : udp://{cfg.ai.host}:{cfg.ai.state_port}  ← teleop")
    print(f"  actions  : udp://{cfg.ai.host}:{cfg.ai.action_port}  → teleop")
    print("  teleop must run with --ai.enabled=true (and the same --cameras.* as training).")
    print("  The arm follows the policy only after `a` is pressed in teleop.")
    print("  Moving the leader, or pressing `a` again, hands control back.")
    print("=" * 66)
    print()

    robot.connect()  # raises with a clear hint when teleop is not running
    try:
        log_say("Policy running. Press a in teleop to hand over control.", cfg.record.play_sounds, blocking=False)
        run_inference(
            robot=robot,
            pipeline=pipeline,
            duration_s=duration_s,
            fps=fps,
            single_task=task,
            display_data=display_data,
        )
    except KeyboardInterrupt:
        logging.info("Interrupted.")
    finally:
        # Stop sending; teleop's action_ttl then makes the arm hold its pose.
        robot.disconnect()


if __name__ == "__main__":
    main()

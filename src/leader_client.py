"""Run this on the PC the leader arm is plugged into.

It reads the leader at ``fps`` and streams every pose over UDP to the robot PC, where
``teleop.py`` picks it up. Which client runs follows the preset's ``teleop["type"]``:

    koch_leader_remote   → lerobot.teleoperators.koch_leader_remote.koch_leader_remote_client
    so101_leader_remote  → lerobot.teleoperators.so_leader_remote.so_leader_remote_client

    python -m src.leader_client                              # nero preset: Koch leader, Nero mapping on
    python -m src.leader_client --preset koch --ip 192.168.1.42
    python -m src.leader_client --preset so101_remote --com /dev/ttyACM1

The leader owns the joint mapping. With a Nero follower the Koch client translates every
pose into Nero joint space (offsets in ``koch_remote_nero_config.py``) before sending; with
any other follower raw leader values go out. Run this with the SAME ``--preset`` as the
robot PC. The leader PC only needs LeRobot importable (see ``_bootstrap.py``) plus the
``dynamixel`` (Koch) or ``feetech`` (SO-101) extra; it never opens the follower or cameras.
"""

from __future__ import annotations

if __package__ in (None, ""):  # allow `python src/leader_client.py`
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "src"

import argparse
import logging

from .devices import load_config, make_parser

logger = logging.getLogger(__name__)

KOCH_REMOTE_TYPES = ("koch_leader_remote",)
SO_REMOTE_TYPES = ("so100_leader_remote", "so101_leader_remote")


def main() -> None:
    # Flags keep the dash style of the repo's koch_leader_remote_client (--ip --port --com --id --fps).
    parser = make_parser(__doc__)
    parser.add_argument("--ip", default=None, help="IP of the robot PC (config leader_client.robot_ip)")
    parser.add_argument("--port", type=int, default=None, help="UDP port (config leader_client.port)")
    parser.add_argument("--com", default=None, help="Serial port of the leader arm (config leader_client.com)")
    parser.add_argument("--id", default=None, help="Leader arm id, selects its calibration file")
    parser.add_argument("--fps", type=int, default=None, help="Read/stream rate")
    parser.add_argument("--gripper-open-pos", type=float, default=None, help="Koch only: gripper spring-back position")
    parser.add_argument(
        "--use-degrees", action=argparse.BooleanOptionalAction, default=None,
        help="SO-101 only: stream degrees (default) or the -100..100 range; must match the follower",
    )
    args, overrides = parser.parse_known_args()

    cfg = load_config(args.preset, overrides)
    lc = cfg.leader_client
    teleop_type = cfg.teleop_type

    from lerobot.utils.utils import init_logging

    init_logging()
    ip = str(args.ip or lc.robot_ip)
    port = int(args.port or lc.port)
    com = str(args.com or lc.com)
    arm_id = str(args.id or lc.id)
    fps = int(args.fps or lc.fps)

    if teleop_type in KOCH_REMOTE_TYPES:
        from lerobot.teleoperators.koch_leader_remote.koch_leader_remote_client import run_remote_client

        # The leader owns the joint mapping: poses go out already in the follower's joint space.
        is_follower_nero = cfg.robot_type == "agx_nero_follower"
        logging.info(
            "Koch leader client: %s -> %s:%s at %s fps (id=%s, nero mapping=%s)",
            com, ip, port, fps, arm_id, is_follower_nero,
        )
        run_remote_client(
            ip=ip,
            port=port,
            com_port=com,
            robot_id=arm_id,
            fps=fps,
            gripper_open_pos=float(args.gripper_open_pos if args.gripper_open_pos is not None else lc.gripper_open_pos),
            is_follower_nero=is_follower_nero,
        )
    elif teleop_type in SO_REMOTE_TYPES:
        from lerobot.teleoperators.so_leader_remote.so_leader_remote_client import run_remote_client

        use_degrees = bool(args.use_degrees if args.use_degrees is not None else lc.use_degrees)
        logging.info(
            "SO-101 leader client: %s -> %s:%s at %s fps (id=%s, use_degrees=%s)",
            com, ip, port, fps, arm_id, use_degrees,
        )
        run_remote_client(ip=ip, port=port, com_port=com, robot_id=arm_id, fps=fps, use_degrees=use_degrees)
    else:
        raise SystemExit(
            f"preset '{cfg.name}' uses teleop type '{teleop_type}', which is not a remote leader. "
            f"Use a preset with koch_leader_remote or so101_leader_remote (e.g. --preset nero, koch, so101_remote)."
        )


if __name__ == "__main__":
    main()

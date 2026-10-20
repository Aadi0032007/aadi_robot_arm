"""EXPERIMENTAL: drive the gripper from a thumb-index pinch seen by a webcam (MediaPipe Hands).

    pip install mediapipe

    python -m src.test_mediapipe --dry_run=true                                   # webcam only, prints the value
    python -m src.test_mediapipe --robot.type=so101_follower --robot.port=COM4 --robot.id=aadi_so101_follower
    python -m src.test_mediapipe --preset so101 --camera_index=1 --pinch_closed=0.25 --pinch_open=1.0

How it works, per frame:

    1. MediaPipe gives 21 hand landmarks. We take thumb tip (4), index tip (8), wrist (0)
       and middle-finger base (9).
    2. pinch = |thumb_tip - index_tip| / |wrist - middle_mcp|   (hand-size normalised, so
       moving the hand closer to the camera does not change it)
    3. gripper.pos = 100 * clip((pinch - pinch_closed) / (pinch_open - pinch_closed), 0, 1)
       (100 = open on SO-101 / Koch; --invert=true flips it), smoothed with an EMA.
    4. The other joints are re-sent at their measured positions so only the gripper moves.
       No hand in view -> hold the last value.

Flags:  --camera_index 0   --show true   --dry_run false   --fps 30   --smooth 0.5
        --pinch_closed 0.25   --pinch_open 1.0   --invert false   --gripper_min 0   --gripper_max 100
        --hand_model <hand_landmarker.task>  (MediaPipe Tasks API; auto-downloaded to models/ if absent)

Watch the on-screen "pinch" number with your hand fully pinched and fully open and set
--pinch_closed / --pinch_open to those values. Press q in the window (or Ctrl-C) to stop.
"""

from __future__ import annotations

if __package__ in (None, ""):  # allow `python src/test_mediapipe.py`
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "src"

import logging
import time
from typing import Any

import cv2
import numpy as np

from .devices import add_bool_flag, describe, load_config, make_parser, make_robot
from .hand_tracking import INDEX_TIP, MIDDLE_MCP, THUMB_TIP, WRIST, HandTracker, pinch_to_gripper

logger = logging.getLogger(__name__)

GRIPPER_KEY = "gripper.pos"


# ── pinch → gripper ──────────────────────────────────────────────────────────────


def pinch_ratio(lm: np.ndarray) -> float:
    pinch = float(np.linalg.norm(lm[THUMB_TIP] - lm[INDEX_TIP]))
    hand = float(np.linalg.norm(lm[WRIST] - lm[MIDDLE_MCP]))
    return pinch / hand if hand > 1e-6 else 0.0


def draw(frame: np.ndarray, lm: np.ndarray | None, ratio: float | None, gripper: float, sending: bool) -> None:
    if lm is not None:
        for p in lm:
            cv2.circle(frame, (int(p[0]), int(p[1])), 3, (0, 255, 0), -1)
        a, b = lm[THUMB_TIP], lm[INDEX_TIP]
        cv2.line(frame, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])), (0, 200, 255), 2)
    text = f"pinch {ratio:.2f}" if ratio is not None else "no hand (holding)"
    cv2.putText(frame, text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(frame, f"gripper {gripper:5.1f}  {'-> robot' if sending else '(dry run)'}", (10, 58),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(frame, "q = quit", (10, frame.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)


# ── main ─────────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = make_parser(__doc__)
    parser.add_argument("--camera_index", type=int, default=0, help="Webcam index for MediaPipe (not the robot cameras)")
    add_bool_flag(parser, "--dry_run", "Webcam + MediaPipe only, no robot")
    add_bool_flag(parser, "--show", "Show the webcam window with landmarks", default=True)
    parser.add_argument("--pinch_closed", type=float, default=0.25, help="pinch ratio that means fully closed")
    parser.add_argument("--pinch_open", type=float, default=1.0, help="pinch ratio that means fully open")
    parser.add_argument("--smooth", type=float, default=0.5, help="EMA factor 0..1 (1 = no smoothing)")
    add_bool_flag(parser, "--invert", "Flip the mapping if the gripper moves the wrong way")
    parser.add_argument("--gripper_min", type=float, default=0.0)
    parser.add_argument("--gripper_max", type=float, default=100.0)
    parser.add_argument("--hand_model", default=None, help="hand_landmarker.task path (Tasks API fallback only)")
    args, overrides = parser.parse_known_args()

    cfg = load_config(args.preset, overrides)
    fps = cfg.fps

    from lerobot.utils.robot_utils import precise_sleep
    from lerobot.utils.utils import init_logging

    init_logging()
    logging.info("\n%s", describe(cfg))

    cap = cv2.VideoCapture(args.camera_index)
    if not cap.isOpened():
        raise SystemExit(f"cannot open webcam {args.camera_index} (try --camera_index=1)")
    tracker = HandTracker(args.hand_model)

    robot = None
    hold: dict[str, Any] = {}
    if not args.dry_run:
        robot = make_robot(cfg)
        if GRIPPER_KEY not in robot.action_features:
            raise SystemExit(f"{cfg.robot_type} has no '{GRIPPER_KEY}' action; keys: {list(robot.action_features)}")
        robot.connect()
        obs = robot.get_observation()
        # Freeze every other joint where it is now; only the gripper will change.
        hold = {k: float(obs[k]) for k in robot.action_features if k in obs and k != GRIPPER_KEY}
        gripper = float(obs.get(GRIPPER_KEY, args.gripper_min))
        logging.info("Holding joints %s; starting gripper at %.1f", sorted(hold), gripper)
    else:
        gripper = args.gripper_min
        logging.info("Dry run: no robot. Tune --pinch_closed / --pinch_open from the on-screen number.")

    period = 1.0 / fps
    try:
        while True:
            t0 = time.perf_counter()
            ok, frame = cap.read()
            if not ok:
                logging.warning("webcam frame missing")
                time.sleep(0.05)
                continue
            frame = cv2.flip(frame, 1)  # mirror so the hand moves the way you expect

            lm = tracker.landmarks(frame)
            ratio = pinch_ratio(lm) if lm is not None else None
            if ratio is not None:
                target = pinch_to_gripper(ratio, args.pinch_closed, args.pinch_open, args.gripper_min, args.gripper_max, args.invert)
                gripper = args.smooth * target + (1.0 - args.smooth) * gripper

            if robot is not None:
                robot.send_action({**hold, GRIPPER_KEY: float(gripper)})

            if args.show:
                draw(frame, lm, ratio, gripper, sending=robot is not None)
                cv2.imshow("test_mediapipe", frame)
                if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                    break
            else:
                print(f"pinch {ratio if ratio is not None else float('nan'):5.2f}  gripper {gripper:6.1f}", end="\r")

            precise_sleep(max(period - (time.perf_counter() - t0), 0.0))
    except KeyboardInterrupt:
        pass
    finally:
        print()
        tracker.close()
        cap.release()
        cv2.destroyAllWindows()
        if robot is not None and robot.is_connected:
            robot.disconnect()


if __name__ == "__main__":
    main()

"""SO-101 inverse kinematics: Cartesian x/y/z + roll/pitch/yaw  ->  joint angles  ->  arm.

Numpy only (see ``kinematics.py``), using the bundled ``urdf/so101/so101_new_calib.urdf``.
Tool point = ``gripper_frame_link``. Frame = URDF ``base_link`` (metres, degrees).

    # 1. no robot: FK/IK round trips and a Jacobian check
    python -m src.so101_ik --self_test

    # 2. where is the tool now?
    python -m src.so101_ik --robot.type=so101_follower --robot.port=COM4 --robot.id=aadi_so101_follower --print_pose

    # 3. move the tool: any of x/y/z (m) and roll/pitch/yaw (deg); omitted values keep the current pose
    python -m src.so101_ik --robot.type=so101_follower --robot.port=COM4 --robot.id=aadi_so101_follower \\
                           --z=0.15 --move_time_s=3
    python -m src.so101_ik ... --x=0.20 --y=0.05 --z=0.12 --pitch=60 --dry_run   # solve + print, do not move

The move is a straight Cartesian line (position) with a slerp (orientation), solved
every control tick, so the tool follows the path rather than swinging through joint
space. Before moving, the end pose is solved once; if it is out of reach nothing moves.

The SO-101 has 5 arm joints: position is always tracked, orientation only as far as the
two remaining DOF allow (pitch and roll about the tool axis are usually reachable, an
arbitrary yaw is not). The residual is printed.

Joint mapping: URDF radians = radians(LeRobot degrees) for every joint (use_degrees=True,
0 = middle of the calibrated range). If ``--print_pose`` disagrees with reality for a
joint, fix it with ``--invert_joints=elbow_flex`` or ``--offset_deg=elbow_flex:5``.
"""

from __future__ import annotations

if __package__ in (None, ""):  # allow `python src/so101_ik.py`
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "src"

import logging
import math
import time

import numpy as np

from ._bootstrap import ARM_ROOT
from .kinematics import (
    ArmKinematics,
    ArmKinematicsConfig,
    SerialChain,
    format_pose,
    matrix_to_pose,
    matrix_to_rotvec,
    pose_to_matrix,
    rotvec_to_matrix,
    solve_ik,
)

logger = logging.getLogger(__name__)

SO101_URDF = ARM_ROOT / "urdf" / "so101" / "so101_new_calib.urdf"
SO101_TIP_LINK = "gripper_frame_link"
SO101_ARM_JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
GRIPPER_KEY = "gripper.pos"


def parse_csv(text: str | None) -> list[str]:
    return [t.strip() for t in (text or "").split(",") if t.strip()]


def parse_offsets(text: str | None) -> dict[str, float]:
    """``"elbow_flex:5,wrist_flex:-3"`` -> ``{"elbow_flex": 5.0, "wrist_flex": -3.0}``."""
    out: dict[str, float] = {}
    for item in parse_csv(text):
        name, _, val = item.partition(":")
        out[name.strip()] = float(val)
    return out


def so101_kinematics_config(
    urdf_path: str | None = None,
    invert_joints: list[str] | None = None,
    offsets_deg: dict[str, float] | None = None,
) -> ArmKinematicsConfig:
    return ArmKinematicsConfig(
        urdf_path=str(urdf_path or SO101_URDF),
        tip_link=SO101_TIP_LINK,
        motors=list(SO101_ARM_JOINTS),
        joint_signs={m: -1.0 for m in (invert_joints or [])},
        joint_offsets_deg=dict(offsets_deg or {}),
    )


def so101_kinematics(**kwargs) -> ArmKinematics:
    return ArmKinematics(so101_kinematics_config(**kwargs))


# ══════════════════════════════════════════════════════════════════════════════
#  Self-test (no robot)
# ══════════════════════════════════════════════════════════════════════════════


def run_self_test(kin: ArmKinematics, trials: int = 300, seed: int = 0) -> bool:
    chain: SerialChain = kin.chain
    rng = np.random.default_rng(seed)
    print(f"URDF chain {chain.base_link} -> {chain.tip_link}, {chain.n} joints:")
    print(kin.describe_limits())
    print(f"\nzero pose (all joints 0 deg):  {format_pose(chain.fk(np.zeros(chain.n)))}")

    lo = np.where(np.isfinite(chain.lower), chain.lower, -math.pi)
    hi = np.where(np.isfinite(chain.upper), chain.upper, math.pi)
    mid, half = 0.5 * (lo + hi), 0.5 * (hi - lo)

    def random_q() -> np.ndarray:
        return mid + 0.8 * half * rng.uniform(-1.0, 1.0, chain.n)

    # 1. analytic Jacobian vs finite differences
    eps = 1e-6
    worst_v = worst_w = 0.0
    for _ in range(25):
        q = random_q()
        T, J = chain.fk_jacobian(q)
        for k in range(chain.n):
            dq = np.zeros(chain.n)
            dq[k] = eps
            T2 = chain.fk(q + dq)
            v_num = (T2[:3, 3] - T[:3, 3]) / eps
            w_num = matrix_to_rotvec(T2[:3, :3] @ T[:3, :3].T) / eps
            worst_v = max(worst_v, float(np.linalg.norm(v_num - J[:3, k])))
            worst_w = max(worst_w, float(np.linalg.norm(w_num - J[3:, k])))
    jac_ok = worst_v < 1e-4 and worst_w < 1e-4
    print(f"\nJacobian vs finite differences: worst linear {worst_v:.2e}, angular {worst_w:.2e}  "
          f"{'OK' if jac_ok else 'MISMATCH'}")

    # 2. FK -> IK round trips from a perturbed seed (orientation is reachable by construction)
    ok = 0
    pos_errs, rot_errs, iters = [], [], []
    t0 = time.perf_counter()
    for _ in range(trials):
        q_true = random_q()
        target = chain.fk(q_true)
        q_seed = np.clip(q_true + rng.normal(0.0, math.radians(15.0), chain.n), lo, hi)
        res = solve_ik(chain, target, q_seed, max_iters=200)
        pos_errs.append(res.pos_err)
        rot_errs.append(res.rot_err)
        iters.append(res.iterations)
        if res.pos_err < 1e-3 and res.rot_err < math.radians(1.0):
            ok += 1
    dt_ms = (time.perf_counter() - t0) * 1e3 / trials
    rate = ok / trials
    print(f"IK round trips ({trials}, seed 15 deg off): {rate * 100:.1f}% within 1 mm / 1 deg, "
          f"median pos err {np.median(pos_errs) * 1e3:.3f} mm, median rot err {math.degrees(np.median(rot_errs)):.3f} deg, "
          f"median {int(np.median(iters))} iterations, {dt_ms:.2f} ms/solve")

    # 3. small jog steps, as the gamepad teleop does them (20 iterations max)
    jog_ok = 0
    for _ in range(trials):
        q = random_q()
        T = chain.fk(q)
        T_t = T.copy()
        T_t[:3, 3] += rng.normal(0.0, 0.003, 3)  # 3 mm step
        res = solve_ik(chain, T_t, q, max_iters=20)
        jog_ok += res.pos_err < 5e-4
    print(f"Jog steps (3 mm, 20 iters): {jog_ok / trials * 100:.1f}% within 0.5 mm")

    # 4. gamepad "arm" mode: tilt (about wrist_flex axis) and roll (about wrist_roll axis) with the
    #    tool point held still, tracking only those two orientation axes.
    i_pitch, i_roll = SO101_ARM_JOINTS.index("wrist_flex"), SO101_ARM_JOINTS.index("wrist_roll")
    tilt_ok = roll_ok = tested = 0
    for _ in range(trials):
        q = mid + 0.6 * half * rng.uniform(-1.0, 1.0, chain.n)  # keep away from limits
        T, _, axes = chain.frames(q)
        for kind, axis in (("tilt", axes[i_pitch]), ("roll", axes[i_roll])):
            T_t = T.copy()
            T_t[:3, :3] = rotvec_to_matrix(math.radians(2.0) * axis) @ T[:3, :3]  # 2 deg step
            res = solve_ik(chain, T_t, q, max_iters=20, orientation_axes=(i_pitch, i_roll))
            good = res.pos_err < 5e-4 and res.rot_err < math.radians(0.2)
            if kind == "tilt":
                tilt_ok += good
            else:
                roll_ok += good
        tested += 1
    print(f"Arm-mode steps (2 deg, tool point fixed): tilt {tilt_ok / tested * 100:.1f}%, "
          f"roll {roll_ok / tested * 100:.1f}% within 0.5 mm / 0.2 deg")

    passed = jac_ok and rate >= 0.90
    print(f"\nself-test {'PASSED' if passed else 'FAILED'}")
    return passed


# ══════════════════════════════════════════════════════════════════════════════
#  Cartesian move on the arm
# ══════════════════════════════════════════════════════════════════════════════


def smoothstep(s: float) -> float:
    return s * s * (3.0 - 2.0 * s)


def main() -> None:
    from .devices import add_bool_flag, describe, load_config, make_parser, make_robot

    parser = make_parser(__doc__)
    add_bool_flag(parser, "--self_test", "No robot: Jacobian check + FK/IK round trips")
    add_bool_flag(parser, "--print_pose", "Print the current joints and tool pose, then exit")
    add_bool_flag(parser, "--dry_run", "Solve and print the target joints, do not move")
    for axis in ("x", "y", "z"):
        parser.add_argument(f"--{axis}", type=float, default=None, help=f"target {axis} in metres (default: current)")
    for axis in ("roll", "pitch", "yaw"):
        parser.add_argument(f"--{axis}", type=float, default=None, help=f"target {axis} in degrees (default: current)")
    parser.add_argument("--move_time_s", type=float, default=3.0, help="Duration of the Cartesian move")
    parser.add_argument("--orientation_weight", type=float, default=1.0, help="0 = position only, 1 = track orientation fully")
    parser.add_argument("--max_pos_error_mm", type=float, default=3.0, help="Abort if IK cannot reach the path within this")
    parser.add_argument("--max_joint_step_deg", type=float, default=10.0, help="Abort if a joint jumps more than this per tick")
    parser.add_argument("--urdf_path", default=None, help="Use another SO-101 URDF (default: bundled)")
    parser.add_argument("--invert_joints", default="", help="Comma list of joints whose sign is flipped, e.g. elbow_flex")
    parser.add_argument("--offset_deg", default="", help="Per-joint offsets, e.g. elbow_flex:5,wrist_flex:-3")
    args, overrides = parser.parse_known_args()

    kin = so101_kinematics(
        urdf_path=args.urdf_path,
        invert_joints=parse_csv(args.invert_joints),
        offsets_deg=parse_offsets(args.offset_deg),
    )

    if args.self_test:
        raise SystemExit(0 if run_self_test(kin) else 1)

    cfg = load_config(args.preset, overrides)

    from lerobot.utils.robot_utils import precise_sleep
    from lerobot.utils.utils import init_logging

    init_logging()
    logging.info("\n%s", describe(cfg))

    robot = make_robot(cfg)
    missing = [k for k in kin.keys if k not in robot.action_features]
    if missing:
        raise SystemExit(f"{cfg.robot_type} does not accept {missing}; this solver is for the SO-101 arm joints.")

    robot.connect()
    try:
        obs = robot.get_observation()
        q0 = kin.motor_to_q(obs)
        T0 = kin.fk(q0)
        print("\ncurrent joints (deg): " + "  ".join(f"{m}={obs[k]:+.1f}" for m, k in zip(kin.motors, kin.keys)))
        print(f"current tool pose   : {format_pose(T0)}")
        outside = kin.out_of_limits(q0)
        if outside:
            print(f"WARNING: {outside} outside the URDF limits. Is the follower in degrees (use_degrees=True)? "
                  "Check --invert_joints / --offset_deg.")

        wants_move = any(getattr(args, a) is not None for a in ("x", "y", "z", "roll", "pitch", "yaw"))
        if args.print_pose or not wants_move:
            if not wants_move and not args.print_pose:
                print("\nNo target given (--x/--y/--z/--roll/--pitch/--yaw); nothing to do.")
            return

        xyz0, rpy0 = matrix_to_pose(T0)
        xyz_t = np.array([v if v is not None else xyz0[i] for i, v in enumerate((args.x, args.y, args.z))])
        rpy_t = np.array([
            math.radians(v) if v is not None else rpy0[i] for i, v in enumerate((args.roll, args.pitch, args.yaw))
        ])
        T_t = pose_to_matrix(xyz_t, rpy_t)
        print(f"target tool pose    : {format_pose(T_t)}")

        tol = args.max_pos_error_mm / 1000.0
        end = kin.solve(T_t, q0, orientation_weight=args.orientation_weight, max_iters=300)
        end_joints = kin.q_to_motor(end.q)
        print("target joints (deg) : " + "  ".join(f"{m}={end_joints[k]:+.1f}" for m, k in zip(kin.motors, kin.keys)))
        print(f"IK residual         : position {end.pos_err * 1e3:.2f} mm, orientation {math.degrees(end.rot_err):.1f} deg "
              f"({end.iterations} iterations)")
        if end.pos_err > tol:
            print(f"\nUNREACHABLE: position error {end.pos_err * 1e3:.1f} mm > {args.max_pos_error_mm} mm "
                  "(outside the workspace or blocked by a joint limit). Not moving.")
            return
        if end.rot_err > math.radians(5.0):
            print("note: orientation only partly reachable at this position (5-DOF arm); position will be exact.")
        if args.dry_run:
            print("\n--dry_run: not moving.")
            return

        fps = cfg.fps
        steps = max(1, int(round(args.move_time_s * fps)))
        p0, R0 = T0[:3, 3], T0[:3, :3]
        rv = matrix_to_rotvec(R0.T @ T_t[:3, :3])
        hold = {GRIPPER_KEY: float(obs[GRIPPER_KEY])} if GRIPPER_KEY in obs and GRIPPER_KEY in robot.action_features else {}
        max_step = math.radians(args.max_joint_step_deg)
        q = q0.copy()
        print(f"\nmoving over {args.move_time_s:.1f} s ({steps} steps) ...")
        for i in range(1, steps + 1):
            t_start = time.perf_counter()
            s = smoothstep(i / steps)
            T = np.eye(4)
            T[:3, 3] = p0 + s * (T_t[:3, 3] - p0)
            T[:3, :3] = R0 @ rotvec_to_matrix(s * rv)
            res = kin.solve(T, q, orientation_weight=args.orientation_weight, max_iters=50)
            if res.pos_err > tol:
                print(f"\nABORT at step {i}/{steps}: path point unreachable (error {res.pos_err * 1e3:.1f} mm). Holding.")
                break
            jump = float(np.max(np.abs(res.q - q)))
            if jump > max_step:
                print(f"\nABORT at step {i}/{steps}: joint jump {math.degrees(jump):.1f} deg > {args.max_joint_step_deg}. Holding.")
                break
            q = res.q
            robot.send_action({**kin.q_to_motor(q), **hold})
            precise_sleep(max(1.0 / fps - (time.perf_counter() - t_start), 0.0))

        time.sleep(0.5)  # let the servos settle before measuring
        obs = robot.get_observation()
        T_end = kin.fk(kin.motor_to_q(obs))
        print(f"reached tool pose   : {format_pose(T_end)}")
        print(f"error vs target     : {np.linalg.norm(T_end[:3, 3] - T_t[:3, 3]) * 1e3:.1f} mm "
              "(includes servo tracking error)")
    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        if robot.is_connected:
            robot.disconnect()


if __name__ == "__main__":
    main()

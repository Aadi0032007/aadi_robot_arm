"""Nero seven-joint Cartesian IK, using the same numpy solver as SO101.

    python -m src.nero_ik --self_test
    python -m src.nero_ik --print_pose
    python -m src.nero_ik --z=0.40 --dry_run
    python -m src.nero_ik --z=0.40 --move_time_s=5
    python -m src.nero_ik --seed_deg=0,20,0,60,0,10,0 --z=0.40 --dry_run

Positions are metres in base_link, RPY angles are degrees. Omitted components
keep the starting pose. Default tip is link7 (SDK wrist/flange frame).
--tip_link=gripper_tcp selects the nominal AGX grasp reference; calibrate it with
--tcp_xyz_m=x,y,z and --tcp_rpy_deg=r,p,y relative to gripper_base.
The nominal reference [0,0,0.138] comes from finger joint frames, not a measured
contact point. Gripper opening is separate from the seven-joint IK chain.
"""
from __future__ import annotations

if __package__ in (None, ''):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = 'src'

import argparse
import math
import time
import numpy as np

from .kinematics import ArmKinematics, ArmKinematicsConfig, format_pose, make_transform, matrix_to_pose, matrix_to_rotvec, pose_to_matrix, rotvec_to_matrix
from .nero_robot import NERO_JOINTS, NERO_URDF, NeroJointRobot, NeroRobotConfig
from .so101_ik import parse_csv, parse_offsets

NERO_ARM_JOINTS = NERO_JOINTS
NERO_TIP_LINK = 'link7'


def vector(text: str, size: int) -> np.ndarray:
    values = np.array([float(v) for v in text.split(',')])
    if values.shape != (size,) or not np.all(np.isfinite(values)):
        raise ValueError(f'Expected {size} finite comma-separated values')
    return values


def nero_kinematics_config(urdf_path=None, tip_link=NERO_TIP_LINK, motors=None,
                          invert_joints=None, offsets_deg=None, limit_margin_deg=1.) -> ArmKinematicsConfig:
    return ArmKinematicsConfig(
        urdf_path=str(urdf_path or NERO_URDF), tip_link=tip_link, base_link='base_link',
        motors=list(motors or NERO_JOINTS), urdf_joints=list(NERO_JOINTS),
        joint_signs={n: -1. for n in (invert_joints or [])},
        joint_offsets_deg=dict(offsets_deg or {}), limit_margin_deg=limit_margin_deg)


class NeroKinematics(ArmKinematics):
    def __init__(self, cfg: ArmKinematicsConfig, tcp_xyz_m=None, tcp_rpy_deg=None):
        super().__init__(cfg)
        if self.chain.n != 7:
            raise ValueError('Nero IK requires all seven arm joints; use link7/flange_link/gripper_tcp as tip')
        if tcp_xyz_m is not None or tcp_rpy_deg is not None:
            if cfg.tip_link != 'gripper_tcp':
                raise ValueError('TCP offsets require tip_link=gripper_tcp')
            xyz = np.asarray(tcp_xyz_m if tcp_xyz_m is not None else [0, 0, .138], dtype=float)
            rpy = np.asarray(tcp_rpy_deg if tcp_rpy_deg is not None else [0, 0, 0], dtype=float)
            if xyz.shape != (3,) or rpy.shape != (3,) or not np.all(np.isfinite(np.r_[xyz, rpy])):
                raise ValueError('TCP translation and RPY must contain three finite values')
            next(j for j in self.chain.joints if j.name == 'gripper_tcp_joint').origin = make_transform(xyz, np.radians(rpy))


def nero_kinematics(tcp_xyz_m=None, tcp_rpy_deg=None, **kwargs):
    return NeroKinematics(nero_kinematics_config(**kwargs), tcp_xyz_m, tcp_rpy_deg)


def reachable(result, pos_error_m, rot_error_rad, orientation_weight=1.):
    return (np.all(np.isfinite(result.q)) and math.isfinite(result.pos_err) and result.pos_err <= pos_error_m
            and (orientation_weight <= 0 or (math.isfinite(result.rot_err) and result.rot_err <= rot_error_rad)))


def plan_cartesian(kin, q0, target, duration_s=5., fps=30., max_pos_error_mm=1.,
                   max_rot_error_deg=1., max_joint_step_deg=5., max_joint_speed_deg_s=30.,
                   max_joint_accel_deg_s2=120., orientation_weight=1.):
    """Preflight every pose before motion; warm-start from each accepted result.

    Returns joint samples excluding the starting pose. Raises on infeasible pose,
    joint limit, discontinuity, excessive velocity or acceleration.
    """
    numeric = [duration_s, fps, max_pos_error_mm, max_rot_error_deg, max_joint_step_deg,
               max_joint_speed_deg_s, max_joint_accel_deg_s2]
    if not all(math.isfinite(v) and v > 0 for v in numeric):
        raise ValueError('Duration, FPS, tolerances and motion limits must be positive and finite')
    if not math.isfinite(orientation_weight) or orientation_weight < 0:
        raise ValueError('orientation_weight must be finite and nonnegative')
    q = np.asarray(q0, dtype=float).copy()
    target = np.asarray(target, dtype=float)
    if q.shape != (7,) or not np.all(np.isfinite(q)) or target.shape != (4,4) or not np.all(np.isfinite(target)):
        raise ValueError('Expected seven finite joints and a finite 4x4 target')
    lo, hi = kin.chain.lower+kin.limit_margin, kin.chain.upper-kin.limit_margin
    if np.any(q < lo) or np.any(q > hi):
        raise ValueError('Starting joints are outside the configured joint-limit margin')
    steps = max(2, math.ceil(duration_s*fps))
    dt = duration_s/steps
    start = kin.fk(q)
    rv = matrix_to_rotvec(start[:3,:3].T @ target[:3,:3])
    path, velocity = [], np.zeros(7)
    for i in range(1, steps+1):
        u = i/steps
        s = u*u*u*(10.-15.*u+6.*u*u)  # zero velocity and acceleration at both ends
        T = np.eye(4)
        T[:3,3] = start[:3,3]+s*(target[:3,3]-start[:3,3])
        T[:3,:3] = start[:3,:3] @ rotvec_to_matrix(s*rv)
        res = kin.solve(T, q, max_iters=80, orientation_weight=orientation_weight)
        if not reachable(res, max_pos_error_mm/1000., math.radians(max_rot_error_deg), orientation_weight):
            raise ValueError(f'Path step {i}/{steps} unreachable: {res.pos_err*1000:.2f} mm, {math.degrees(res.rot_err):.2f} deg')
        if np.any(res.q < lo-1e-10) or np.any(res.q > hi+1e-10):
            raise ValueError(f'Path step {i}/{steps} violates joint limits')
        dq_deg = np.degrees(res.q-q)
        new_velocity = dq_deg/dt
        acceleration = (new_velocity-velocity)/dt
        if np.max(np.abs(dq_deg)) > max_joint_step_deg:
            raise ValueError('Joint discontinuity: reduce target change or increase move_time_s')
        if np.max(np.abs(new_velocity)) > max_joint_speed_deg_s:
            raise ValueError('Joint speed exceeded: increase move_time_s')
        if np.max(np.abs(acceleration)) > max_joint_accel_deg_s2:
            raise ValueError('Joint acceleration exceeded: increase move_time_s')
        path.append(res.q.copy())
        q, velocity = res.q, new_velocity
    if np.max(np.abs(velocity/dt)) > max_joint_accel_deg_s2:
        raise ValueError('Stopping acceleration exceeded: increase move_time_s')
    return np.asarray(path), dt


def run_self_test(kin, trials=300, seed=42):
    rng = np.random.default_rng(seed)
    chain = kin.chain
    lo, hi = chain.lower+kin.limit_margin, chain.upper-kin.limit_margin
    mid, half = (lo+hi)/2, (hi-lo)/2
    print(f'{chain.base_link} -> {chain.tip_link}: {chain.n} joints\n{kin.describe_limits()}')
    worst_v = worst_w = 0.
    success = 0
    times = []
    for i in range(trials):
        truth = mid+.7*half*rng.uniform(-1,1,7)
        T, J = chain.fk_jacobian(truth)
        if i < 25:
            for k in range(7):
                perturbed = truth.copy()
                perturbed[k] += 1e-6
                Tp = chain.fk(perturbed)
                worst_v = max(worst_v, np.linalg.norm((Tp[:3,3]-T[:3,3])/1e-6-J[:3,k]))
                worst_w = max(worst_w, np.linalg.norm(matrix_to_rotvec(Tp[:3,:3] @ T[:3,:3].T)/1e-6-J[3:,k]))
        q0 = np.clip(truth+rng.normal(0, math.radians(10), 7), lo, hi)
        started = time.perf_counter()
        res = kin.solve(T, q0, max_iters=150)
        times.append(1000*(time.perf_counter()-started))
        success += reachable(res, .001, math.radians(1))
    print(f'Jacobian errors: linear {worst_v:.2e}, angular {worst_w:.2e}')
    print(f'FK/IK: {success}/{trials} within 1 mm / 1 deg; median {np.median(times):.2f} ms/solve')
    target = kin.fk(mid)
    target[:3,3] += [.005, -.003, .004]
    try:
        path, dt = plan_cartesian(kin, mid, target, max_joint_accel_deg_s2=300.)
        print(f'Continuous Cartesian path: {len(path)} steps; final {format_pose(kin.fk(path[-1]))}')
        path_ok = True
    except ValueError as e:
        print(f'Continuous Cartesian path FAILED: {e}')
        path_ok = False
    passed = worst_v < 1e-4 and worst_w < 1e-4 and success/trials >= .95 and path_ok
    print('self-test ' + ('PASSED' if passed else 'FAILED'))
    return passed


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for name in ('self_test', 'print_pose', 'dry_run'):
        parser.add_argument('--'+name, action='store_true')
    for axis in ('x','y','z','roll','pitch','yaw'):
        parser.add_argument('--'+axis, type=float)
    parser.add_argument('--seed_deg', help='Seven native joint degrees; enables hardware-free dry runs')
    parser.add_argument('--preset', default='nero_cartesian')
    parser.add_argument('--tip_link', default=NERO_TIP_LINK)
    parser.add_argument('--urdf_path')
    parser.add_argument('--tcp_xyz_m')
    parser.add_argument('--tcp_rpy_deg')
    parser.add_argument('--invert_joints', default='')
    parser.add_argument('--offset_deg', default='')
    parser.add_argument('--move_time_s', type=float, default=5.)
    parser.add_argument('--orientation_weight', type=float, default=1.)
    parser.add_argument('--max_pos_error_mm', type=float, default=1.)
    parser.add_argument('--max_rot_error_deg', type=float, default=1.)
    parser.add_argument('--max_joint_step_deg', type=float, default=5.)
    parser.add_argument('--max_joint_speed_deg_s', type=float, default=30.)
    parser.add_argument('--max_joint_accel_deg_s2', type=float, default=120.)
    args, overrides = parser.parse_known_args()
    kin = nero_kinematics(urdf_path=args.urdf_path, tip_link=args.tip_link,
        invert_joints=parse_csv(args.invert_joints), offsets_deg=parse_offsets(args.offset_deg),
        tcp_xyz_m=vector(args.tcp_xyz_m,3) if args.tcp_xyz_m else None,
        tcp_rpy_deg=vector(args.tcp_rpy_deg,3) if args.tcp_rpy_deg else None)
    if args.self_test:
        raise SystemExit(0 if run_self_test(kin) else 1)
    from .config import load
    from .devices import apply_overrides, parse_overrides
    cfg = load(args.preset)
    apply_overrides(cfg, parse_overrides(overrides))
    if cfg.robot_type != 'agx_nero_native':
        parser.error('Use the nero_cartesian preset/native seven-joint interface')
    spec = dict(cfg.robot)
    spec.pop('type')
    spec['enable_on_connect'] = False
    robot = None
    try:
        if args.seed_deg:
            if not args.dry_run and not args.print_pose:
                parser.error('--seed_deg requires --dry_run or --print_pose')
            obs = dict(zip(kin.keys, vector(args.seed_deg,7)))
        else:
            robot = NeroJointRobot(NeroRobotConfig(**spec))
            robot.connect()
            obs = robot.get_observation()
        q0 = kin.motor_to_q(obs)
        T0 = kin.fk(q0)
        print(f'Current {kin.chain.tip_link}: {format_pose(T0)}')
        if kin.chain.tip_link == 'gripper_tcp':
            print('TCP is relative to gripper_base; default 138 mm reference is nominal, not measured.')
        values = [getattr(args,a) for a in ('x','y','z','roll','pitch','yaw')]
        if args.print_pose or all(v is None for v in values):
            return
        xyz, rpy = matrix_to_pose(T0)
        target = pose_to_matrix([xyz[i] if values[i] is None else values[i] for i in range(3)],
            [rpy[i] if values[i+3] is None else math.radians(values[i+3]) for i in range(3)])
        path, dt = plan_cartesian(kin,q0,target,duration_s=args.move_time_s,fps=cfg.fps,
            max_pos_error_mm=args.max_pos_error_mm,max_rot_error_deg=args.max_rot_error_deg,
            max_joint_step_deg=args.max_joint_step_deg,max_joint_speed_deg_s=args.max_joint_speed_deg_s,
            max_joint_accel_deg_s2=args.max_joint_accel_deg_s2,orientation_weight=args.orientation_weight)
        print(f'Target: {format_pose(target)}\nPreflight passed: {len(path)} steps, {dt:.4f} s/tick')
        print('Target joint degrees: '+str(kin.q_to_motor(path[-1])))
        if args.dry_run:
            print('Dry run complete; no motion commands sent.')
            return
        # Re-read after planning; a moved starting pose invalidates the preflight.
        now = kin.motor_to_q(robot.get_observation())
        if np.max(np.abs(np.degrees(now-q0))) > .5:
            raise ValueError('Starting pose moved during preflight; run the command again')
        robot.enable()
        hold = {'gripper.pos':obs['gripper.pos']} if 'gripper.pos' in obs else {}
        for q in path:
            started = time.perf_counter()
            robot.send_action({**kin.q_to_motor(q),**hold})
            time.sleep(max(0.,dt-(time.perf_counter()-started)))
        time.sleep(.3)
        print('Measured pose: '+format_pose(kin.fk_motors(robot.get_observation())))
    except (ValueError, RuntimeError, TimeoutError) as e:
        raise SystemExit(str(e)) from e
    finally:
        if robot is not None:
            robot.disconnect()


if __name__ == '__main__':
    main()

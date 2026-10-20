"""Pure-numpy kinematics for serial arms described by a URDF: FK, Jacobian, IK.

No placo / pinocchio / scipy: only ``numpy`` and the standard library, so it runs on
Windows, macOS, Linux and the Jetson alike.

    chain = SerialChain.from_urdf("so101_new_calib.urdf", tip_link="gripper_frame_link")
    T = chain.fk(q)                                   # 4x4 base->tip, q in radians
    res = solve_ik(chain, T_target, q_seed)           # res.q, res.pos_err [m], res.rot_err [rad]

``ArmKinematics`` adds the LeRobot side: it maps ``{"shoulder_pan.pos": deg, ...}`` motor
values to URDF joint radians (with optional per-joint sign / offset) and back.

IK method (task priority, damped least squares):
    1. primary task   : tip POSITION        dq1 = Jp+ e_p
    2. secondary task : tip ORIENTATION in the null space of the position task
                        dq = dq1 + Z (Jr Z)+ (w * e_r - Jr dq1)
                        Z = exact position null-space basis from SVD
On a 5-DOF arm such as the SO-101, position is prioritized and orientation tracks
as far as the remaining DOF allow. Backtracking checks the actual tool position.
Joints at bounds are locked when a step points outward and the remaining joints
are re-solved. A joint initially outside its limit may stay but not go further out.

Conventions: lengths in metres, angles in radians, rpy = URDF fixed-axis roll/pitch/yaw
(R = Rz(yaw) @ Ry(pitch) @ Rx(roll)), rotation vectors = axis * angle.
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np

# ══════════════════════════════════════════════════════════════════════════════
#  Rotation helpers
# ══════════════════════════════════════════════════════════════════════════════


def rot_x(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def rot_y(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def rot_z(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def skew(v: Iterable[float]) -> np.ndarray:
    x, y, z = (float(t) for t in v)
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def rpy_to_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """URDF convention: fixed axes X, then Y, then Z  ->  R = Rz(yaw) Ry(pitch) Rx(roll)."""
    return rot_z(yaw) @ rot_y(pitch) @ rot_x(roll)


def matrix_to_rpy(R: np.ndarray) -> np.ndarray:
    """Inverse of :func:`rpy_to_matrix`. Returns ``[roll, pitch, yaw]`` in radians."""
    r20 = float(np.clip(R[2, 0], -1.0, 1.0))
    pitch = math.asin(-r20)
    if abs(r20) < 1.0 - 1e-9:
        roll = math.atan2(R[2, 1], R[2, 2])
        yaw = math.atan2(R[1, 0], R[0, 0])
    else:  # gimbal lock (pitch = +-90 deg): roll and yaw are coupled, put it all in yaw
        roll = 0.0
        yaw = math.atan2(-R[0, 1], R[1, 1])
    return np.array([roll, pitch, yaw])


def axis_angle_to_matrix(axis: Iterable[float], angle: float) -> np.ndarray:
    a = np.asarray(list(axis), dtype=float)
    n = float(np.linalg.norm(a))
    if n < 1e-12 or abs(angle) < 1e-15:
        return np.eye(3)
    K = skew(a / n)
    return np.eye(3) + math.sin(angle) * K + (1.0 - math.cos(angle)) * (K @ K)


def rotvec_to_matrix(v: Iterable[float]) -> np.ndarray:
    v = np.asarray(list(v), dtype=float)
    angle = float(np.linalg.norm(v))
    if angle < 1e-12:
        return np.eye(3) + skew(v)
    return axis_angle_to_matrix(v / angle, angle)


def matrix_to_rotvec(R: np.ndarray) -> np.ndarray:
    """Log map SO(3) -> rotation vector (axis * angle), robust near 0 and near pi."""
    cos_a = float(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0))
    angle = math.acos(cos_a)
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])  # = 2 sin(angle) axis
    if angle < 1e-6:
        return 0.5 * v
    if math.pi - angle < 1e-4:
        i = int(np.argmax(np.diag(R)))
        col = R[:, i] + np.eye(3)[:, i]  # (R + I) = 2 a a^T at angle pi
        axis = col / np.linalg.norm(col)
        if float(np.dot(axis, v)) < 0.0:
            axis = -axis
        return axis * angle
    return (angle / (2.0 * math.sin(angle))) * v


def make_transform(xyz: Iterable[float], rpy: Iterable[float]) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = rpy_to_matrix(*[float(a) for a in rpy])
    T[:3, 3] = np.asarray(list(xyz), dtype=float)
    return T


def pose_to_matrix(xyz: Iterable[float], rpy: Iterable[float]) -> np.ndarray:
    return make_transform(xyz, rpy)


def matrix_to_pose(T: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``(xyz [m], rpy [rad])`` of a 4x4 transform."""
    return T[:3, 3].copy(), matrix_to_rpy(T[:3, :3])


def damped_pinv(J: np.ndarray, damping: float) -> np.ndarray:
    """Damped least-squares pseudo-inverse  J^T (J J^T + lambda^2 I)^-1."""
    m = J.shape[0]
    return J.T @ np.linalg.solve(J @ J.T + (damping**2) * np.eye(m), np.eye(m))


# ══════════════════════════════════════════════════════════════════════════════
#  URDF chain
# ══════════════════════════════════════════════════════════════════════════════

MOVABLE = ("revolute", "continuous", "prismatic")


@dataclass
class ChainJoint:
    name: str
    type: str
    parent: str
    child: str
    origin: np.ndarray  # 4x4 parent -> joint frame
    axis: np.ndarray  # unit axis in the joint frame
    lower: float
    upper: float


def _floats(text: str | None, default: list[float]) -> np.ndarray:
    if not text:
        return np.asarray(default, dtype=float)
    return np.asarray([float(t) for t in text.split()], dtype=float)


def parse_urdf_joints(path: str | Path) -> dict[str, ChainJoint]:
    """All joints of a URDF, keyed by their CHILD link name."""
    root = ET.parse(str(path)).getroot()
    by_child: dict[str, ChainJoint] = {}
    for j in root.findall("joint"):  # direct children only (skips <transmission><joint>)
        jtype = (j.get("type") or "fixed").strip()
        parent_el, child_el = j.find("parent"), j.find("child")
        if parent_el is None or child_el is None:
            continue
        origin_el = j.find("origin")
        xyz = _floats(origin_el.get("xyz") if origin_el is not None else None, [0.0, 0.0, 0.0])
        rpy = _floats(origin_el.get("rpy") if origin_el is not None else None, [0.0, 0.0, 0.0])
        axis_el = j.find("axis")
        axis = _floats(axis_el.get("xyz") if axis_el is not None else None, [1.0, 0.0, 0.0])  # URDF default
        n = float(np.linalg.norm(axis))
        axis = axis / n if n > 1e-12 else axis
        limit_el = j.find("limit")
        if jtype == "continuous" or limit_el is None:
            lower, upper = -math.inf, math.inf
        else:
            lower = float(limit_el.get("lower", "-inf"))
            upper = float(limit_el.get("upper", "inf"))
        by_child[child_el.get("link")] = ChainJoint(
            name=j.get("name", ""),
            type=jtype,
            parent=parent_el.get("link"),
            child=child_el.get("link"),
            origin=make_transform(xyz, rpy),
            axis=axis,
            lower=lower,
            upper=upper,
        )
    return by_child


class SerialChain:
    """The joints from ``base_link`` to ``tip_link`` of a URDF tree."""

    def __init__(self, joints: list[ChainJoint], base_link: str, tip_link: str) -> None:
        self.joints = joints
        self.base_link = base_link
        self.tip_link = tip_link
        self.active = [j for j in joints if j.type in MOVABLE]
        self.joint_names = [j.name for j in self.active]
        self.lower = np.array([j.lower for j in self.active], dtype=float)
        self.upper = np.array([j.upper for j in self.active], dtype=float)

    @classmethod
    def from_urdf(cls, path: str | Path, tip_link: str, base_link: str | None = None) -> "SerialChain":
        by_child = parse_urdf_joints(path)
        if tip_link not in by_child and tip_link != base_link:
            links = sorted(set(by_child) | {j.parent for j in by_child.values()})
            raise ValueError(f"tip link {tip_link!r} not found in {path}; links: {links}")
        chain: list[ChainJoint] = []
        link = tip_link
        while link in by_child and link != base_link:
            joint = by_child[link]
            chain.append(joint)
            link = joint.parent
        if base_link is not None and link != base_link:
            raise ValueError(f"{base_link!r} is not an ancestor of {tip_link!r} in {path}")
        chain.reverse()
        for joint in chain:
            if joint.type not in ("fixed", *MOVABLE):
                raise ValueError(f"joint {joint.name!r} has unsupported type {joint.type!r}")
        return cls(chain, base_link=link, tip_link=tip_link)

    @property
    def n(self) -> int:
        return len(self.active)

    def fk(self, q: Iterable[float]) -> np.ndarray:
        return self.fk_jacobian(q, jacobian=False)[0]

    def frames(self, q: Iterable[float]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Base->tip transform, plus every movable joint's origin and unit axis in the base frame
        (arrays of shape (n, 3), chain order)."""
        q = np.asarray(list(q), dtype=float)
        if q.shape[0] != self.n:
            raise ValueError(f"expected {self.n} joint values {self.joint_names}, got {q.shape[0]}")
        T = np.eye(4)
        origins = np.zeros((self.n, 3))
        axes = np.zeros((self.n, 3))
        i = 0
        for joint in self.joints:
            T = T @ joint.origin
            if joint.type == "fixed":
                continue
            origins[i] = T[:3, 3]
            axes[i] = T[:3, :3] @ joint.axis
            M = np.eye(4)
            if joint.type == "prismatic":
                M[:3, 3] = joint.axis * q[i]
            else:
                M[:3, :3] = axis_angle_to_matrix(joint.axis, q[i])
            T = T @ M
            i += 1
        return T, origins, axes

    def fk_jacobian(self, q: Iterable[float], jacobian: bool = True) -> tuple[np.ndarray, np.ndarray | None]:
        """Base->tip transform and the 6xN geometric Jacobian (rows: v_xyz, w_xyz in the base frame)."""
        T, origins, axes = self.frames(q)
        if not jacobian:
            return T, None
        J = np.zeros((6, self.n))
        p_tip = T[:3, 3]
        for k, joint in enumerate(self.active):
            if joint.type == "prismatic":
                J[:3, k] = axes[k]
            else:
                J[:3, k] = np.cross(axes[k], p_tip - origins[k])
                J[3:, k] = axes[k]
        return T, J

    def mid(self) -> np.ndarray:
        lo = np.where(np.isfinite(self.lower), self.lower, -math.pi)
        hi = np.where(np.isfinite(self.upper), self.upper, math.pi)
        return 0.5 * (lo + hi)


# ══════════════════════════════════════════════════════════════════════════════
#  Inverse kinematics
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class IKResult:
    q: np.ndarray
    pos_err: float  # metres
    rot_err: float  # radians (only along orientation_axes when those were given)
    iterations: int


def _orientation_selector(J: np.ndarray, orientation_axes: Iterable[int] | None) -> np.ndarray | None:
    """Rows = unit world axes of the chosen joints (k x 3), or None for full 3-axis orientation."""
    if orientation_axes is None:
        return None
    S = J[3:, list(orientation_axes)].T
    return S / np.maximum(np.linalg.norm(S, axis=1, keepdims=True), 1e-12)


def solve_ik(
    chain: SerialChain,
    target: np.ndarray,
    q_seed: Iterable[float],
    *,
    orientation_weight: float = 1.0,
    orientation_axes: Iterable[int] | None = None,
    max_iters: int = 30,
    pos_tol: float = 1e-4,
    rot_tol: float = 1e-3,
    damping: float = 0.01,
    max_step: float = 0.2,
    limit_margin: float = 0.0,
) -> IKResult:
    """Task-priority IK: position first, orientation (scaled by ``orientation_weight``) in its
    null space. ``orientation_weight <= 0`` = position only, orientation left free.

    ``orientation_axes``: indices of joints whose current world axes are the ONLY rotation
    axes the orientation task tracks. For a 5-DOF arm pass (pitch joint, roll joint): the
    error about the remaining axis (the gripper's azimuth, which the arm cannot set
    independently of its position) is ignored instead of being fought."""
    q_seed = np.asarray(list(q_seed), dtype=float)
    q = q_seed.copy()
    # A joint already outside its (margin-shrunk) limit may stay there, but not go further out.
    lo = np.minimum(chain.lower + limit_margin, q_seed)
    hi = np.maximum(chain.upper - limit_margin, q_seed)
    p_t = target[:3, 3]
    R_t = target[:3, :3]
    axes_idx = list(orientation_axes) if orientation_axes is not None else None

    iterations = 0
    for iterations in range(1, max_iters + 1):
        T, J = chain.fk_jacobian(q)
        e_p = p_t - T[:3, 3]
        e_r = matrix_to_rotvec(R_t @ T[:3, :3].T)
        Jr = J[3:]
        S = _orientation_selector(J, axes_idx)
        if S is not None:
            e_r = S @ e_r
            Jr = S @ Jr
        if np.linalg.norm(e_p) < pos_tol and (orientation_weight <= 0.0 or np.linalg.norm(e_r) < rot_tol):
            break
        Jp = J[:3]
        # Re-solve with outward-pointing joints locked, rather than clipping a
        # coupled solution and losing the motion supplied by the other joints.
        free = np.ones(chain.n, dtype=bool)
        for _ in range(chain.n + 1):
            A = Jp[:, free]
            dq = np.zeros(chain.n)
            primary = damped_pinv(A, damping) @ e_p
            dq[free] = primary
            if orientation_weight > 0.0 and A.shape[1]:
                # Damping belongs in the solve, not in the null-space projector:
                # I - damped_pinv(A) @ A is not a true null-space projector.
                _, singular, Vt = np.linalg.svd(A, full_matrices=True)
                rank = int(np.sum(singular > max(A.shape) * np.finfo(float).eps * max(singular[0], 1.0))) if singular.size else 0
                Z = Vt[rank:].T
                B = Jr[:, free]
                if Z.shape[1]:
                    dq[free] += Z @ (damped_pinv(B @ Z, damping) @ (orientation_weight * e_r - B @ primary))
            blocked = free & (((q <= lo + 1e-10) & (dq < 0.0)) | ((q >= hi - 1e-10) & (dq > 0.0)))
            if not np.any(blocked):
                break
            free[blocked] = False
        biggest = float(np.max(np.abs(dq))) if dq.size else 0.0
        if biggest > max_step:
            dq *= max_step / biggest
        # Backtrack nonlinear steps. Once position is within tolerance,
        # orientation may improve only while preserving that tolerance.
        old_pos = float(np.linalg.norm(e_p))
        old_rot = float(np.linalg.norm(e_r))
        accepted = False
        for attempt in range(12):
            candidate = np.clip(q + dq * (0.5 ** attempt), lo, hi)
            Tc, Jc = chain.fk_jacobian(candidate)
            pc = float(np.linalg.norm(p_t - Tc[:3, 3]))
            rc = matrix_to_rotvec(R_t @ Tc[:3, :3].T)
            Sc = _orientation_selector(Jc, axes_idx)
            if Sc is not None:
                rc = Sc @ rc
            position_improved = pc < old_pos - 1e-12
            orientation_improved = orientation_weight > 0.0 and pc <= pos_tol and float(np.linalg.norm(rc)) < old_rot - 1e-12
            if position_improved or orientation_improved:
                q = candidate
                accepted = True
                break
        if not accepted:
            break

    T, J = chain.fk_jacobian(q)
    e_r = matrix_to_rotvec(R_t @ T[:3, :3].T)
    S = _orientation_selector(J, axes_idx)
    if S is not None:
        e_r = S @ e_r
    return IKResult(
        q=q,
        pos_err=float(np.linalg.norm(p_t - T[:3, 3])),
        rot_err=float(np.linalg.norm(e_r)),
        iterations=iterations,
    )


# ══════════════════════════════════════════════════════════════════════════════
#  LeRobot motor values <-> URDF joints
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class ArmKinematicsConfig:
    urdf_path: str
    tip_link: str
    # LeRobot motor names, in chain order from base to tip (without ".pos").
    motors: list[str]
    base_link: str | None = None
    # URDF joint name for each motor (same order). None = same names as the motors.
    urdf_joints: list[str] | None = None
    # urdf_rad = sign * radians(motor_deg) + radians(offset_deg). Signs must be +1 or -1.
    joint_signs: dict[str, float] = field(default_factory=dict)
    joint_offsets_deg: dict[str, float] = field(default_factory=dict)
    # IK keeps this far inside the URDF limits.
    limit_margin_deg: float = 1.0


class ArmKinematics:
    """A :class:`SerialChain` plus the mapping from LeRobot motor degrees to URDF radians.

    The follower must report joints in DEGREES (SO-101 ``use_degrees=True``, the default;
    Koch needs ``use_degrees=True`` too). Grippers are not part of the chain.
    """

    def __init__(self, cfg: ArmKinematicsConfig) -> None:
        path = Path(cfg.urdf_path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"URDF not found: {path}")
        self.cfg = cfg
        self.chain = SerialChain.from_urdf(path, cfg.tip_link, cfg.base_link)
        urdf_joints = list(cfg.urdf_joints or cfg.motors)
        if len(urdf_joints) != len(cfg.motors):
            raise ValueError(f"urdf_joints {urdf_joints} and motors {cfg.motors} differ in length")
        if urdf_joints != self.chain.joint_names:
            raise ValueError(
                f"URDF chain {self.chain.base_link} -> {self.chain.tip_link} has movable joints "
                f"{self.chain.joint_names}, the config maps {urdf_joints}. Fix motors / urdf_joints (same order)."
            )
        self.motors = list(cfg.motors)
        self.keys = [f"{m}.pos" for m in self.motors]
        self.signs = np.array([float(cfg.joint_signs.get(m, 1.0)) for m in self.motors])
        if not np.all(np.isin(self.signs, (-1.0, 1.0))):
            raise ValueError(f"joint_signs must be +1 or -1, got {cfg.joint_signs}")
        self.offsets = np.radians([float(cfg.joint_offsets_deg.get(m, 0.0)) for m in self.motors])
        self.limit_margin = math.radians(cfg.limit_margin_deg)

    # -- conversions ------------------------------------------------------------

    def motor_to_q(self, values: dict[str, float]) -> np.ndarray:
        missing = [k for k in self.keys if k not in values]
        if missing:
            raise KeyError(f"motor values lack {missing}")
        deg = np.array([float(values[k]) for k in self.keys])
        return self.signs * np.radians(deg) + self.offsets

    def q_to_motor(self, q: Iterable[float]) -> dict[str, float]:
        deg = np.degrees((np.asarray(list(q), dtype=float) - self.offsets) * self.signs)
        return {k: float(v) for k, v in zip(self.keys, deg)}

    # -- kinematics -------------------------------------------------------------

    def fk(self, q: Iterable[float]) -> np.ndarray:
        return self.chain.fk(q)

    def fk_motors(self, values: dict[str, float]) -> np.ndarray:
        return self.chain.fk(self.motor_to_q(values))

    def solve(self, target: np.ndarray, q_seed: Iterable[float], **kwargs) -> IKResult:
        kwargs.setdefault("limit_margin", self.limit_margin)
        return solve_ik(self.chain, target, q_seed, **kwargs)

    def out_of_limits(self, q: Iterable[float], slack_deg: float = 5.0) -> list[str]:
        q = np.asarray(list(q), dtype=float)
        slack = math.radians(slack_deg)
        bad = (q < self.chain.lower - slack) | (q > self.chain.upper + slack)
        return [m for m, b in zip(self.motors, bad) if b]

    def describe_limits(self) -> str:
        rows = []
        for m, j, lo, hi in zip(self.motors, self.chain.joint_names, self.chain.lower, self.chain.upper):
            rows.append(f"  {m:<14} (urdf {j:<14}) {math.degrees(lo):8.1f} .. {math.degrees(hi):7.1f} deg")
        return "\n".join(rows)


def format_pose(T: np.ndarray) -> str:
    xyz, rpy = matrix_to_pose(T)
    r, p, y = np.degrees(rpy)
    return (
        f"x={xyz[0] * 100:+7.2f} cm  y={xyz[1] * 100:+7.2f} cm  z={xyz[2] * 100:+7.2f} cm   "
        f"roll={r:+7.1f}  pitch={p:+7.1f}  yaw={y:+7.1f} deg"
    )

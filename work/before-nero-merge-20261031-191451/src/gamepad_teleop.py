"""Gamepad Cartesian jog: a LeRobot ``Teleoperator`` that moves the TOOL, not single joints.

Robot-agnostic: the gamepad produces a Cartesian twist, the URDF-based IK in
``kinematics.py`` turns it into joint targets for whatever arm the URDF describes. The
defaults are the SO-101 follower (bundled URDF); any other arm needs its URDF, the chain's
motor names and a follower that reports joints in degrees.

    python -m src.teleop --robot.type=so101_follower --robot.port=COM4 --robot.id=aadi_so101_follower \\
                         --teleop.type=gamepad_ee

Add --port=5001 to listen for UDP gamepad input on 0.0.0.0 instead of local USB.
On the gamepad computer: python -m src.gamepad_remote_client --ip=ROBOT_IP --port=5001
The transport is shared by all robot models using gamepad_ee. Missing/disconnected
remote input holds the current command after udp_timeout_s (default 0.3 seconds).

Mapping (W3C / Xbox layout, see gamepad.py):

    Gamepad input        Function
    Left stick up/down   X  forward / backward
    Left stick L/R       Y  left / right
    Right stick up/down  Z  up / down
    Right stick L/R      Yaw: swing the arm left / right (shoulder_pan)
    D-pad up/down        Pitch: tilt the gripper up / down (wrist_flex only)
    D-pad L/R            Roll: rotate the gripper (wrist_roll only)
    RB / LB              Gripper open / close
    X                    Precision mode on/off (x precision_scale)
    Y                    Toggle base / tool reference frame
    LT                   Deadman only if --teleop.require_deadman=true (off by default)
    A, B, RT, Start      Reserved (ignored); speed is configured with command arguments

Each tick: target = FK(last command) + twist * dt, in the base or the tool frame, then
IK seeded from the last command. Because the target is re-based on the last command
every tick, an unreachable direction (e.g. yaw on the 5-DOF SO-101) cannot wind up: it
simply produces no motion. If the IK cannot hold the position (workspace edge, joint
limit) the step is rejected and the arm holds. Joint steps are capped per tick.
Position is primary, orientation secondary (see kinematics.solve_ik).

Axis directions are the URDF base_link axes. If a stick moves the tool the wrong way,
flip that axis: ``--teleop.invert=x,pitch``.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .gamepad import (
    AXIS_LEFT_X,
    AXIS_LEFT_Y,
    AXIS_RIGHT_X,
    AXIS_RIGHT_Y,
    BTN_DPAD_DOWN,
    BTN_DPAD_LEFT,
    BTN_DPAD_RIGHT,
    BTN_DPAD_UP,
    BTN_LB,
    BTN_LT,
    BTN_RB,
    BTN_RT,
    BTN_X,
    BTN_Y,
    NUM_BUTTONS,
)
from .kinematics import ArmKinematics, ArmKinematicsConfig, format_pose, rotvec_to_matrix
from .so101_ik import SO101_ARM_JOINTS, SO101_TIP_LINK, SO101_URDF, parse_csv, parse_offsets

logger = logging.getLogger(__name__)

AXES = ("x", "y", "z", "roll", "pitch", "yaw")


def _lerobot_base():
    from lerobot.teleoperators import Teleoperator, TeleoperatorConfig

    return Teleoperator, TeleoperatorConfig


_Teleoperator, _TeleoperatorConfig = _lerobot_base()


@dataclass
class GamepadEETeleopConfig(_TeleoperatorConfig):
    udp_port: int | None = None  # None = local USB; otherwise listen on 0.0.0.0
    udp_timeout_s: float = 0.3  # missing remote frames: hold, including the gripper
    max_tracking_error_deg: float | None = None  # pause/re-anchor when measured joints lag commands
    # ── kinematics (defaults = SO-101 follower) ──────────────────────────────
    urdf_path: str = str(SO101_URDF)
    tip_link: str = SO101_TIP_LINK
    base_link: str | None = None
    motors: list[str] = field(default_factory=lambda: list(SO101_ARM_JOINTS))  # chain order, no ".pos"
    urdf_joints: list[str] | None = None  # None = same names as motors
    invert_joints: str = ""  # comma list, e.g. "elbow_flex"
    joint_offsets_deg: str = ""  # e.g. "elbow_flex:5,wrist_flex:-3"
    limit_margin_deg: float = 1.0
    # ── gripper ───────────────────────────────────────────────────────────────
    gripper_key: str | None = "gripper.pos"
    gripper_min: float = 0.0
    gripper_max: float = 100.0
    gripper_speed: float = 60.0  # units/s while A or B is held
    # ── speeds (at speed level x1.0) ─────────────────────────────────────────
    max_linear_speed: float = 0.10  # m/s
    max_angular_speed: float = 1.0  # rad/s (~57 deg/s)
    speed_levels: list[float] = field(default_factory=lambda: [0.1, 0.25, 0.5, 0.75, 1.0, 1.5])
    initial_speed_level: int = 2  # index into speed_levels -> x0.5 = 5 cm/s
    precision_scale: float = 0.2
    # ── input shaping ─────────────────────────────────────────────────────────
    deadzone: float = 0.12
    trigger_deadzone: float = 0.05
    expo: float = 1.5  # >1 = finer control near the centre of the stick
    invert: str = ""  # comma list of axes to flip: x,y,z,roll,pitch,yaw
    # ── orientation handling ─────────────────────────────────────────────────
    # "arm"  (default, 5-DOF arms like SO-101 / Koch):
    #        pitch = tilt the gripper up/down about the pitch_joint axis (the arm's real pitch
    #                axis), tool point held still;
    #        roll  = rotate about the gripper's own axis (roll_joint axis), tool point held;
    #        yaw   = swing the whole arm about yaw_joint: a 5-DOF arm cannot turn the gripper
    #                sideways without moving it, so "face left/right" moves the tool on an arc;
    #        orientation hold tracks pitch + roll; adaptive translation may relax it.
    # "frame": full 3-axis rotations about the base / tool frame axes (6+ DOF arms).
    orientation_mode: str = "arm"
    # Arm-mode translation: prefer the current tilt/roll, but allow the wrist
    # to change when that hold obstructs a reachable Cartesian position.
    translation_orientation: str = 'adaptive'  # adaptive | hold | free
    # arm mode: pitch / roll move ONLY pitch_joint / roll_joint (the gripper swings about the
    # wrist, nothing else moves). False = IK keeps the tool point still and moves the other
    # joints to do it.
    wrist_direct: bool = True
    pitch_joint: str = "wrist_flex"
    roll_joint: str = "wrist_roll"
    yaw_joint: str = "shoulder_pan"
    # ── behaviour ─────────────────────────────────────────────────────────────
    frame: str = "base"  # translation frame "base" | "tool" (Y toggles); "tool" in arm mode = along the gripper
    require_deadman: bool = False  # True = motion only while RB is held
    gripper_needs_deadman: bool = True
    # ── IK / safety ───────────────────────────────────────────────────────────
    orientation_weight: float = 1.0  # 0 = position only
    ik_iters: int = 20
    max_pos_error_m: float = 0.003  # reject a step the IK cannot reach within this
    max_rot_error_deg: float | None = None  # full-pose arms can reject orientation failures too
    max_joint_step_deg: float = 5.0  # per control tick
    z_min_m: float | None = 0.02  # tool may not be driven below this height (None = off)
    debug: bool = False  # log which inputs are active (twice a second) to spot a stuck axis


class GamepadEETeleop(_Teleoperator):
    config_class = GamepadEETeleopConfig
    name = "gamepad_ee"

    def __init__(self, config: GamepadEETeleopConfig):
        super().__init__(config)
        self.config = config
        if config.udp_port is not None and (type(config.udp_port) is not int or not 1 <= config.udp_port <= 65535):
            raise ValueError('teleop.udp_port must be an integer in [1, 65535]')
        if config.max_tracking_error_deg is not None and (not math.isfinite(config.max_tracking_error_deg) or config.max_tracking_error_deg <= 0):
            raise ValueError('max_tracking_error_deg must be positive and finite or None')
        self.kin = ArmKinematics(
            ArmKinematicsConfig(
                urdf_path=config.urdf_path,
                tip_link=config.tip_link,
                base_link=config.base_link,
                motors=list(config.motors),
                urdf_joints=config.urdf_joints,
                joint_signs={m: -1.0 for m in parse_csv(config.invert_joints)},
                joint_offsets_deg=parse_offsets(config.joint_offsets_deg),
                limit_margin_deg=config.limit_margin_deg,
            )
        )
        bad = [a for a in parse_csv(config.invert) if a not in AXES]
        if bad:
            raise ValueError(f"teleop.invert: unknown axes {bad}; use {AXES}")
        self._sign = {a: (-1.0 if a in parse_csv(config.invert) else 1.0) for a in AXES}
        if config.frame not in ("base", "tool"):
            raise ValueError("teleop.frame must be 'base' or 'tool'")
        if not config.speed_levels:
            raise ValueError("teleop.speed_levels must not be empty")
        if config.orientation_mode not in ("arm", "frame"):
            raise ValueError("teleop.orientation_mode must be 'arm' or 'frame'")
        if config.translation_orientation not in ('adaptive', 'hold', 'free'):
            raise ValueError('teleop.translation_orientation must be adaptive, hold or free')
        if config.orientation_mode == "arm":
            for label, joint in (("pitch", config.pitch_joint), ("roll", config.roll_joint), ("yaw", config.yaw_joint)):
                if joint not in self.kin.motors:
                    raise ValueError(f"teleop.{label}_joint={joint!r} is not one of the IK joints {self.kin.motors}")
            self._i_pitch = self.kin.motors.index(config.pitch_joint)
            self._i_roll = self.kin.motors.index(config.roll_joint)
            self._i_yaw = self.kin.motors.index(config.yaw_joint)

        self._reader = None
        self._q: np.ndarray | None = None
        self._gripper: float | None = None
        self.needs_seed = True
        self._speed_idx = int(min(max(config.initial_speed_level, 0), len(config.speed_levels) - 1))
        self._precision = False
        self._frame = config.frame
        self._prev_pressed: set[int] = set()
        self._engaged = False
        self._input_available = False
        self._tracking_paused = False
        self._last_t: float | None = None
        self._warned_at: dict[str, float] = {}

    # ── features ─────────────────────────────────────────────────────────────

    @property
    def action_features(self) -> dict[str, type]:
        feats = {k: float for k in self.kin.keys}
        if self.config.gripper_key:
            feats[self.config.gripper_key] = float
        return feats

    @property
    def feedback_features(self) -> dict[str, type]:
        return {}

    @property
    def speed_scale(self) -> float:
        return self.config.speed_levels[self._speed_idx] * (self.config.precision_scale if self._precision else 1.0)

    @property
    def help_text(self) -> str:
        c = self.config
        lin = c.max_linear_speed * self.speed_scale * 100
        ang = math.degrees(c.max_angular_speed * self.speed_scale)
        deadman = "  HOLD LT      deadman: no motion without it\n" if c.require_deadman else ""
        return (
            f"Gamepad Cartesian jog  (frame={self._frame}, speed x{self.speed_scale:.2f} = {lin:.1f} cm/s, {ang:.0f} deg/s)\n"
            f"{deadman}"
            "  left stick   up/down = X forward/back   left/right = Y left/right\n"
            + (
                "  right stick  up/down = Z up/down        left/right = swing arm left/right\n"
                if c.orientation_mode == "arm"
                else "  right stick  up/down = Z up/down        left/right = yaw\n"
            )
            + "  D-pad        left/right = roll          up/down = pitch up/down\n"
            "  RB / LB      gripper open / close\n"
            "  X            precision mode on/off      Y = base/tool frame"
        )

    # ── lifecycle ────────────────────────────────────────────────────────────

    @property
    def is_connected(self) -> bool:
        return self._reader is not None

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    def connect(self, calibrate: bool = True) -> None:
        if self._reader is not None:
            return
        if self.config.udp_port is not None:
            from .gamepad_udp import UDPGamepadReader
            self._reader = UDPGamepadReader(self.config.udp_port, self.config.udp_timeout_s,
                debug=self.config.debug)
            logger.info('%s; waiting for gamepad_remote_client', self._reader.device_name)
            return
        from .gamepad import GamepadReader

        self._reader = GamepadReader()
        if self._reader.read() is None:
            logger.warning("No gamepad attached yet; the arm holds until one is plugged in.")
        else:
            logger.info("Gamepad: %s", self._reader.device_name)

    def disconnect(self) -> None:
        if self._reader is not None:
            self._reader.close()
            self._reader = None

    # ── seeding (called by teleop.py / record.py with the follower's observation) ─

    def seed_from_observation(self, observation: dict[str, Any]) -> None:
        self._q = self.kin.motor_to_q(observation)
        outside = self.kin.out_of_limits(self._q)
        if outside:
            logger.warning(
                "%s outside the URDF joint limits. Is the follower in degrees (use_degrees=True)? "
                "Check teleop.invert_joints / teleop.joint_offsets_deg.", outside,
            )
        if self.config.gripper_key:
            self._gripper = float(observation.get(self.config.gripper_key, self.config.gripper_min))
        self._last_t = time.perf_counter()
        self.needs_seed = False
        logger.info("Tool pose at start: %s", format_pose(self.kin.fk(self._q)))

    @property
    def command_ready(self) -> bool:
        """The enclosing loop must not transmit stale targets during dropout/pause."""
        return self._input_available and not self._tracking_paused

    def update_observation(self, observation: dict[str, Any]) -> None:
        """Called each control tick before input integration, using live feedback."""
        measured = self.kin.motor_to_q(observation)
        limit = self.config.max_tracking_error_deg
        if (self._q is not None and self._input_available and limit is not None
                and np.max(np.abs(np.degrees(self._q-measured))) > limit):
            if not self._tracking_paused:
                logger.warning('Gamepad tracking error exceeds %.1f deg; release sticks, D-pad and RB/LB to resume.', limit)
            self._tracking_paused = True
        if self._q is None or not self._input_available or self._tracking_paused:
            self._q = measured.copy()
            if self.config.gripper_key:
                self._gripper = float(observation[self.config.gripper_key])

    def on_target_rejected(self, observation: dict[str, Any]) -> None:
        """Handle a tracking fault detected by the driver between loop reads."""
        self._tracking_paused = True
        self.update_observation(observation)

    # ── action ───────────────────────────────────────────────────────────────

    def get_action(self) -> dict[str, float]:
        if self._q is None:
            raise RuntimeError("gamepad_ee: call seed_from_observation(obs) before get_action()")
        c = self.config
        now = time.perf_counter()
        dt = min(0.1, now - (self._last_t or now))  # cap so a stall cannot produce a jump
        self._last_t = now

        state = self._reader.read() if self._reader is not None else None
        if state is None:
            self._input_available = False
            message = 'Waiting for fresh UDP gamepad input; no commands sent.' if c.udp_port is not None else 'No gamepad attached; no commands sent.'
            self._warn("nopad", message)
            self._set_engaged(False)
            self._prev_pressed = set()
            return self._action()

        self._input_available = True

        pressed = {i for i in range(NUM_BUTTONS) if state.pressed(i)}
        if self._tracking_paused:
            v, w = self._twist(state, pressed)
            if not np.any(v) and not np.any(w) and not pressed.intersection((BTN_RB, BTN_LB)):
                self._tracking_paused = False
                logger.info('Gamepad re-armed from measured joints; controls are neutral.')
            self._prev_pressed = pressed
            self._set_engaged(False)
            return self._action()
        edges = pressed - self._prev_pressed
        self._prev_pressed = pressed
        self._handle_toggles(edges)

        deadman = (BTN_LT in pressed) or not c.require_deadman
        self._set_engaged(deadman)

        if c.gripper_key and self._gripper is not None and (deadman or not c.gripper_needs_deadman):
            direction = (1.0 if BTN_RB in pressed else 0.0) - (1.0 if BTN_LB in pressed else 0.0)
            if direction:
                self._gripper = float(np.clip(self._gripper + direction * c.gripper_speed * dt, c.gripper_min, c.gripper_max))

        if not deadman:
            return self._action()

        v, w = self._twist(state, pressed)
        if c.debug and (np.any(v) or np.any(w)):
            names = [n for n, val in zip(AXES, (*v, *w)) if val]
            self._debug(
                f"active: {names}  v={np.round(v, 2).tolist()} w={np.round(w, 2).tolist()}  "
                f"raw axes={np.round(state.axes, 3).tolist()} LT={state.value(BTN_LT):.3f} RT={state.value(BTN_RT):.3f}"
            )
        if np.any(v) or np.any(w):
            scale = self.speed_scale
            self._step(v * c.max_linear_speed * scale * dt, w * c.max_angular_speed * scale * dt)
        return self._action()

    def send_feedback(self, feedback: dict[str, float]) -> None:
        raise NotImplementedError

    # ── internals ────────────────────────────────────────────────────────────

    def _action(self) -> dict[str, float]:
        out = self.kin.q_to_motor(self._q)
        if self.config.gripper_key and self._gripper is not None:
            out[self.config.gripper_key] = float(self._gripper)
        return out

    def _set_engaged(self, engaged: bool) -> None:
        if engaged != self._engaged:
            self._engaged = engaged
            if self.config.require_deadman:
                logger.info("motion %s", "ENABLED (LT held)" if engaged else "disabled (hold LT to move)")

    def _handle_toggles(self, edges: set[int]) -> None:
        changed = False
        if BTN_X in edges:
            self._precision = not self._precision
            changed = True
        if BTN_Y in edges:
            self._frame = "tool" if self._frame == "base" else "base"
            changed = True
        if changed:
            logger.info(
                "frame=%s  speed x%.2f (%.1f cm/s, %.0f deg/s)%s",
                self._frame, self.speed_scale, self.config.max_linear_speed * self.speed_scale * 100,
                math.degrees(self.config.max_angular_speed * self.speed_scale), "  PRECISION" if self._precision else "",
            )

    def _shape(self, value: float, deadzone: float) -> float:
        a = abs(value)
        if a < deadzone:
            return 0.0
        t = min(1.0, (a - deadzone) / (1.0 - deadzone))
        return math.copysign(t**self.config.expo, value)

    def _twist(self, state, pressed: set[int]) -> tuple[np.ndarray, np.ndarray]:
        c, s = self.config, self._sign
        lx = self._shape(state.axis(AXIS_LEFT_X), c.deadzone)
        ly = self._shape(state.axis(AXIS_LEFT_Y), c.deadzone)
        rx = self._shape(state.axis(AXIS_RIGHT_X), c.deadzone)
        ry = self._shape(state.axis(AXIS_RIGHT_Y), c.deadzone)
        roll = (1.0 if BTN_DPAD_RIGHT in pressed else 0.0) - (1.0 if BTN_DPAD_LEFT in pressed else 0.0)
        pitch = (1.0 if BTN_DPAD_UP in pressed else 0.0) - (1.0 if BTN_DPAD_DOWN in pressed else 0.0)
        # Stick axes are W3C: up and left are negative.
        #   left stick  -> X (up = forward), Y (left = +Y)
        #   right stick -> Z (up = up),       yaw (left = counter-clockwise from above)
        #   D-pad U/D   -> pitch (up = tilt gripper up)
        v = np.array([-ly * s["x"], -lx * s["y"], -ry * s["z"]])
        w = np.array([roll * s["roll"], pitch * s["pitch"], -rx * s["yaw"]])
        return v, w

    def _step(self, dp: np.ndarray, dtheta: np.ndarray) -> None:
        """``dp`` = translation this tick (x, y, z); ``dtheta`` = (roll, pitch, yaw) this tick."""
        if self.config.orientation_mode == "arm":
            self._step_arm(dp, float(dtheta[0]), float(dtheta[1]), float(dtheta[2]))
        else:
            self._step_frame(dp, dtheta)

    def _floor(self, p: np.ndarray, p_t: np.ndarray) -> np.ndarray:
        c = self.config
        if c.z_min_m is not None and p_t[2] < c.z_min_m and p_t[2] < p[2]:
            p_t = p_t.copy()
            p_t[2] = p[2]  # no further descent below the floor, other axes still move
            self._warn("floor", f"tool at the z_min_m floor ({c.z_min_m * 100:.1f} cm); not moving lower.")
        return p_t

    def _solve(self, p_t: np.ndarray, R_t: np.ndarray, q_seed: np.ndarray, orientation_axes=None,
               translation_only: bool = False) -> np.ndarray | None:
        """IK for one jog step; None (with a warning) when the position cannot be reached."""
        c = self.config
        target = np.eye(4)
        target[:3, :3] = R_t
        target[:3, 3] = p_t
        weight = c.orientation_weight
        if translation_only and c.translation_orientation == 'free':
            weight = 0.
        res = self.kin.solve(
            target, q_seed, orientation_weight=weight, max_iters=c.ik_iters,
            orientation_axes=orientation_axes,
        )
        if translation_only and c.translation_orientation == 'adaptive' and weight > 0:
            requested_step = float(np.linalg.norm(p_t - self.kin.fk(q_seed)[:3, 3]))
            # The 3 mm acceptance tolerance alone can hide a stalled 1-2 mm jog.
            # Retry when orientation holding consumes a significant part of the
            # requested displacement, even if the original solve was accepted.
            if res.pos_err > max(1e-4, .2 * requested_step):
                relaxed = self.kin.solve(target, q_seed, orientation_weight=0.,
                    max_iters=c.ik_iters, orientation_axes=orientation_axes)
                if (np.all(np.isfinite(relaxed.q)) and math.isfinite(relaxed.pos_err)
                        and relaxed.pos_err < res.pos_err - 1e-6):
                    res, weight = relaxed, 0.
                    if c.debug:
                        self._debug('Translation: relaxing wrist orientation to preserve Cartesian progress')
        rot_bad = (c.max_rot_error_deg is not None and weight > 0
                   and res.rot_err > math.radians(c.max_rot_error_deg))
        if not np.all(np.isfinite(res.q)) or res.pos_err > c.max_pos_error_m or rot_bad:
            self._warn(
                "reach",
                f"target pose not reached ({res.pos_err * 1e3:.1f} mm, {math.degrees(res.rot_err):.1f} deg): holding.",
            )
            return None
        return res.q

    def _commit(self, q_new: np.ndarray) -> None:
        dq = q_new - self._q
        biggest = float(np.max(np.abs(dq))) if dq.size else 0.0
        limit = math.radians(self.config.max_joint_step_deg)
        if biggest > limit:
            dq *= limit / biggest
        self._q = self._q + dq

    # -- "frame" mode: rotations about base / tool axes (6+ DOF arms) --------------

    def _step_frame(self, dp: np.ndarray, dtheta: np.ndarray) -> None:
        T = self.kin.fk(self._q)
        p, R = T[:3, 3], T[:3, :3]
        dR = rotvec_to_matrix(dtheta)
        if self._frame == "tool":
            p_t = p + R @ dp
            R_t = R @ dR
        else:
            p_t = p + dp
            R_t = dR @ R
        q = self._solve(self._floor(p, p_t), R_t, self._q)
        if q is not None:
            self._commit(q)

    # -- "arm" mode: 5-DOF aware ----------------------------------------------------

    def _arm_axes(self, origins: np.ndarray, axes: np.ndarray, p_tool: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """``(approach, left)``: unit vector out of the gripper (roll axis, signed towards the tool
        point) and the arm's pitch axis signed to point to the arm's left (so a rotation of -angle
        about it tips the gripper UP)."""
        up = np.array([0.0, 0.0, 1.0])
        z_roll = axes[self._i_roll]
        approach = z_roll if float(np.dot(z_roll, p_tool - origins[self._i_roll])) >= 0.0 else -z_roll
        # "Outward" = horizontal direction from the pan axis to the wrist; fall back to the gripper's
        # own horizontal direction when the wrist is right above the pan axis.
        out = origins[self._i_roll] - origins[self._i_yaw]
        out = out - float(np.dot(out, up)) * up
        if np.linalg.norm(out) < 1e-3:
            out = approach - float(np.dot(approach, up)) * up
        n = axes[self._i_pitch]
        if np.linalg.norm(out) < 1e-6:
            return approach, n
        left_ref = np.cross(up, out / np.linalg.norm(out))
        left = n if float(np.dot(n, left_ref)) >= 0.0 else -n
        return approach, left

    def _joint_step(self, q: np.ndarray, k: int, delta: float, label: str) -> None:
        """Move one joint by ``delta`` rad, respecting its limit (a joint already past it may stay)."""
        chain = self.kin.chain
        lo = min(chain.lower[k] + self.kin.limit_margin, q[k])
        hi = max(chain.upper[k] - self.kin.limit_margin, q[k])
        new = float(np.clip(q[k] + delta, lo, hi))
        if new == q[k]:
            self._warn(f"limit_{k}", f"{label} at its joint limit.")
        q[k] = new

    def _step_arm(self, dp: np.ndarray, droll: float, dpitch: float, dyaw: float) -> None:
        c = self.config
        chain = self.kin.chain
        q = self._q.copy()
        # A deliberate rotation request keeps its orientation target. Adaptive
        # relaxation applies only when the operator asks for translation alone.
        translation_only = np.any(dp) and not (droll or dpitch or dyaw)

        # Yaw = swing the whole arm about the pan joint (exact, no IK needed).
        if dyaw:
            _, _, axes = chain.frames(q)
            sign = 1.0 if axes[self._i_yaw][2] >= 0.0 else -1.0  # positive dyaw = counter-clockwise from above
            self._joint_step(q, self._i_yaw, sign * dyaw, c.yaw_joint)

        if c.wrist_direct and (dpitch or droll):
            # Pitch / roll = that ONE wrist joint only; nothing else moves.
            T, origins, axes = chain.frames(q)
            approach, left = self._arm_axes(origins, axes, T[:3, 3])
            if dpitch:  # nose up = rotation of -angle about the left-pointing pitch axis
                sign = 1.0 if float(np.dot(axes[self._i_pitch], left)) >= 0.0 else -1.0
                self._joint_step(q, self._i_pitch, -sign * dpitch, c.pitch_joint)
            if droll:
                sign = 1.0 if float(np.dot(axes[self._i_roll], approach)) >= 0.0 else -1.0
                self._joint_step(q, self._i_roll, sign * droll, c.roll_joint)
            droll = dpitch = 0.0

        # Translation (and, with wrist_direct=False, pitch/roll about the tool point) through IK.
        # Prefer tilt + roll while moving; adaptive translation can release that
        # preference when it obstructs position. Explicit rotations keep their target.
        if np.any(dp) or droll or dpitch:
            T, origins, axes = chain.frames(q)
            p, R = T[:3, 3].copy(), T[:3, :3]
            approach, left = self._arm_axes(origins, axes, p)
            if self._frame == "tool":
                dp = dp[0] * approach + dp[1] * left + dp[2] * np.cross(approach, left)
            p_t = self._floor(p, p + dp)
            R_t = rotvec_to_matrix(droll * approach - dpitch * left) @ R
            q_ik = self._solve(p_t, R_t, q, orientation_axes=(self._i_pitch, self._i_roll),
                translation_only=bool(translation_only))
            if q_ik is not None:
                q = q_ik

        self._commit(q)

    def _debug(self, msg: str, period_s: float = 0.5) -> None:
        now = time.monotonic()
        if now - self._warned_at.get("debug", -math.inf) >= period_s:
            self._warned_at["debug"] = now
            logger.info(msg)

    def _warn(self, key: str, msg: str, period_s: float = 2.0) -> None:
        now = time.monotonic()
        if now - self._warned_at.get(key, -math.inf) >= period_s:
            self._warned_at[key] = now
            logger.warning(msg)

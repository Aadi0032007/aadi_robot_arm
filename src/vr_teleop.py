"""VR Cartesian teleop: a Meta Quest controller moves the TOOL, through the Quest browser (WebXR).

A LeRobot ``Teleoperator`` (type ``vr_ee``) for ``teleop.py`` / ``record.py``, on the same URDF
IK as ``gamepad_ee`` and ``hand_ee``. Defaults = SO-101 follower (bundled URDF). Nothing is
installed on the headset: ``vr_server.py`` serves ``vr_page.html``, the Quest browser opens it
and streams both controllers back over Wi-Fi.

    python -m src.vr_teleop                       # preview: headset only, simulated arm, no robot
    python -m src.teleop --preset so101_vr        # the real arm (= --teleop.type=vr_ee)

In the headset: open the printed https://<pc-ip>:8443 in the Quest browser, accept the
certificate warning once (Advanced -> Proceed), press "Start (passthrough)". After that nobody
has to wear the headset (see "Without wearing the headset" below).

Nothing moves until you squeeze GRIP. Right controller (``hand=right``; with ``hand=left`` the
same buttons on the left one):

    hold GRIP (side button)     engage: the tool follows the controller; release = the arm stops
    move                        the tool moves the same way (x position_scale), from where you squeezed
    tilt nose down / up         gripper tilts down / up        (wrist_flex only)
    twist about where it points gripper rolls the same way     (wrist_roll only)
    TRIGGER                     gripper closes as far as you pull (released = open)
    A                           precision mode on / off (x precision_scale)
    thumbstick click            arm to the home pose (GRIP released)

The other controller: X = start / stop recording (Space in teleop.py), Y = redo (Left / r),
thumbstick click = status panel on / off.

Every squeeze of GRIP is a new reference: the controller pose at that moment is matched to where
the arm is at that moment, and only the controller's motion from there moves the arm. The arm
never jumps: let go, move your hand somewhere comfortable, squeeze again (like lifting a mouse).
The squeeze sets the directions too: forward = the way the controller points when you squeeze
(its horizontal heading), up = up. Point it the way the robot reaches, squeeze, push forward to
reach further. Nothing depends on where the headset is or looks. ``--teleop.yaw_offset_deg``
turns the mapping (180 = you face the robot and point at it).

A 5-DOF arm cannot turn the gripper sideways without moving it, so turning the controller left /
right is ignored (the arm's heading follows the tool position). Tilt and twist are measured about
the controller's own axes and move only their wrist motor; while the tool translates the IK
keeps the gripper's tilt and roll, as in ``hand_ee``.

Without wearing the headset: the Quest only has to see the controllers. Press Start once, then
put the headset down (stand or table) with its front cameras facing where your hands will be.
It must not fall asleep: turn the proximity sensor off (Meta Quest Developer Hub -> device ->
Proximity sensor, or ``adb shell am broadcast -a com.oculus.vrpowermanager.prox_close``) or cover
the sensor between the lenses. If it sleeps anyway the arm stops; wake it and press Start again.
The VR state (READY / MOVING / CONTROLLER NOT TRACKED ...) is also logged in the terminal, and the
controller buzzes on every squeeze. With no headset connected at all, teleop keeps running and
the arm holds; the keys, AI control and recording work as usual.

Safety: the arm only follows while GRIP is held and the controller is optically tracked (the
Quest flags a controller it only dead-reckons), headset data older than ``timeout_s`` or jumping
more than ``jump_max_m`` in one tick stops it, the target never goes below
``min(z_min_m, height at the squeeze)``, joints stay inside their URDF limits (right up to them:
``limit_margin_deg=0`` for the full reach; a joint already past a limit may stay there but not go
further), an unreachable target is followed only as far as the arm reaches, and every joint
follows its target with speed / acceleration limits. ``home_on_start=true`` glides the arm
to ``home_deg`` when teleop starts (off by default: the arm stays where it is until you squeeze).
"""

from __future__ import annotations

if __package__ in (None, ""):  # allow `python src/vr_teleop.py`
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "src"

import logging
import math
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ._bootstrap import ensure_lerobot_importable
from .kinematics import ArmKinematics, ArmKinematicsConfig, IKResult, axis_angle_to_matrix, format_pose, matrix_to_rotvec, rot_z, rotvec_to_matrix
from .so101_ik import SO101_ARM_JOINTS, SO101_TIP_LINK, SO101_URDF, parse_csv, parse_offsets, smoothstep
from .vr_server import VRLink

logger = logging.getLogger(__name__)

AXES = ("x", "y", "z", "roll", "pitch", "yaw")
# WebXR axes (x right, y up, z backward) -> robot base axes (x forward, y left, z up). det = +1.
XR_TO_ROBOT = np.array([[0.0, 0.0, -1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
UP = np.array([0.0, 0.0, 1.0])
JOINT_LABELS = {"shoulder_pan": "pan", "shoulder_lift": "lift", "elbow_flex": "elbow", "wrist_flex": "w.flex",
                "wrist_roll": "w.roll"}


def _lerobot_base():
    ensure_lerobot_importable()  # also when run as `python -m src.vr_teleop`
    from lerobot.teleoperators import Teleoperator, TeleoperatorConfig

    return Teleoperator, TeleoperatorConfig


_Teleoperator, _TeleoperatorConfig = _lerobot_base()


@dataclass
class VREETeleopConfig(_TeleoperatorConfig):
    # ── headset link (vr_server.py) ──────────────────────────────────────────
    host: str = "0.0.0.0"
    port: int = 8443
    https: bool = True  # False = plain http, for USB + `adb reverse tcp:8443 tcp:8443` (http://localhost:8443)
    cert_file: str | None = None  # None = self-signed, made once in revobots_arm/certs/
    key_file: str | None = None
    hand: str = "right"  # controller that drives the arm; the other one has the recording buttons
    timeout_s: float = 0.25  # headset data older than this: the arm stops
    haptics: bool = True
    # ── kinematics (defaults = SO-101 follower) ──────────────────────────────
    urdf_path: str = str(SO101_URDF)
    tip_link: str = SO101_TIP_LINK
    base_link: str | None = None
    motors: list[str] = field(default_factory=lambda: list(SO101_ARM_JOINTS))  # chain order, no ".pos"
    urdf_joints: list[str] | None = None  # None = same names as motors
    invert_joints: str = ""  # comma list, e.g. "elbow_flex"
    joint_offsets_deg: str = ""  # e.g. "elbow_flex:5,wrist_flex:-3"
    limit_margin_deg: float = 0.0  # how far inside the URDF joint limits the IK stays; 0 = full range (max reach)
    pitch_joint: str = "wrist_flex"
    roll_joint: str = "wrist_roll"
    yaw_joint: str = "shoulder_pan"
    orientation_mode: str = 'arm'  # arm = SO101 pitch/roll; frame = full controller SE(3)
    orientation_scale: float = 1.0
    # ── start-up ──────────────────────────────────────────────────────────────
    home_on_start: bool = False  # True = glide to home_deg when teleop starts (thumbstick click does it any time)
    home_deg: list[float] | None = None  # motor degrees per arm joint; None = every joint at 0
    home_speed_deg_s: float = 30.0
    home_min_time_s: float = 2.0
    # ── controller -> tool mapping ───────────────────────────────────────────
    engage_threshold: float = 0.6  # GRIP value that engages ...
    release_threshold: float = 0.4  # ... and the one that releases (hysteresis)
    position_scale: float = 1.0  # tool metres per hand metre
    precision_scale: float = 0.3  # A button: position_scale x this
    roll_scale: float = 1.0  # 0 = twisting the controller does not roll the gripper
    pitch_scale: float = 1.0  # 0 = tilting the controller does not tilt the gripper
    yaw_offset_deg: float = 0.0  # turns "forward" (= where the controller points at the squeeze); 180 = you face the robot
    invert: str = ""  # comma list of axes to flip: x,y,z,roll,pitch
    jump_max_m: float = 0.15  # controller jump per control tick = tracking glitch, the arm stops
    z_min_m: float | None = 0.03  # tool target never below min(this, its height at engage); None = off
    # ── gripper from the trigger ─────────────────────────────────────────────
    gripper_key: str | None = "gripper.pos"  # None = leave the gripper out of the action
    gripper_min: float = 0.0  # trigger fully pulled
    gripper_max: float = 100.0  # trigger released (100 = open on SO-101)
    gripper_invert: bool = False
    gripper_needs_engage: bool = True  # False = the trigger works with GRIP released too
    # ── IK / arm motion ───────────────────────────────────────────────────────
    orientation_weight: float = 1.0  # keep the gripper's tilt + roll while translating; 0 = position only
    ik_iters: int = 30
    max_pos_error_m: float = 0.005  # IK result farther than this from the target = out of reach
    max_rot_error_deg: float = 1.5  # enforced in full frame mode
    reach_search_steps: int = 4  # out of reach: bisection steps toward the target
    max_joint_speed_deg_s: float = 150.0
    max_joint_accel_deg_s2: float = 800.0
    follow_gain: float = 12.0  # 1/s: how fast a joint closes the gap to its target (lower = smoother)
    joint_deadband_deg: float = 0.2  # a joint closer than this to its target stays put


@dataclass
class _Sample:
    p: np.ndarray  # (3,) grip position, robot axes (not yet turned to "forward") [m]
    H: np.ndarray  # (3, 3) controller frame, robot axes; columns = where it points, its left, its top
    grip: float
    trigger: float


@dataclass
class _Input:
    sample: _Sample | None  # None = the arm must not follow
    problem: str  # why sample is None (shown in the HUD)
    pressed: set[str]  # buttons held on the driving controller
    other_pressed: set[str]  # ... and on the other one


def _quat_to_matrix(q: Any) -> np.ndarray:
    x, y, z, w = (float(v) for v in q)
    n = math.sqrt(x * x + y * y + z * z + w * w)
    if not n > 1e-9:
        raise ValueError("zero quaternion")
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
        [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
        [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
    ])


def _buttons(ctrl: Any) -> set[str]:
    if not isinstance(ctrl, dict):
        return set()
    return {name for name in ("a", "b", "stick_click") if ctrl.get(name)}


def _heading(H: np.ndarray) -> np.ndarray | None:
    """Horizontal direction the controller points (robot axes). Pointed steeply down or up, its top
    takes over (``f - f_z * top``), so the heading stays defined and continuous."""
    f, top = H[:, 0], H[:, 2]
    g = f - float(f[2]) * top
    g = np.array([g[0], g[1], 0.0])
    n = float(np.linalg.norm(g))
    return g / n if n > 1e-3 else None


def _swing(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Smallest rotation that turns unit vector ``a`` onto unit vector ``b``."""
    axis = np.cross(a, b)
    s = float(np.linalg.norm(axis))
    c = float(np.dot(a, b))
    if s < 1e-9:
        if c > 0.0:
            return np.eye(3)
        perp = np.cross(a, [1.0, 0.0, 0.0])
        if np.linalg.norm(perp) < 1e-6:
            perp = np.cross(a, [0.0, 1.0, 0.0])
        return axis_angle_to_matrix(perp, math.pi)
    return axis_angle_to_matrix(axis / s, math.atan2(s, c))


def _wrap(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def _sign(x: float) -> float:
    return -1.0 if x < 0.0 else 1.0


class VREETeleop(_Teleoperator):
    config_class = VREETeleopConfig
    name = "vr_ee"

    def __init__(self, config: VREETeleopConfig):
        super().__init__(config)
        self.config = config
        c = config
        if c.orientation_mode not in ('arm', 'frame'):
            raise ValueError('orientation_mode must be arm or frame')
        self.kin = ArmKinematics(
            ArmKinematicsConfig(
                urdf_path=c.urdf_path,
                tip_link=c.tip_link,
                base_link=c.base_link,
                motors=list(c.motors),
                urdf_joints=c.urdf_joints,
                joint_signs={m: -1.0 for m in parse_csv(c.invert_joints)},
                joint_offsets_deg=parse_offsets(c.joint_offsets_deg),
                limit_margin_deg=c.limit_margin_deg,
            )
        )
        inverted = parse_csv(c.invert)
        bad = [a for a in inverted if a not in AXES]
        if bad:
            raise ValueError(f"teleop.invert: unknown axes {bad}; use {AXES}")
        sign = {a: (-1.0 if a in inverted else 1.0) for a in AXES}
        self._sign_xyz = np.array([sign["x"], sign["y"], sign["z"]])
        self._sign_roll, self._sign_pitch = sign["roll"], sign["pitch"]
        self._sign_rotation = np.array([sign['roll'], sign['pitch'], sign['yaw']])
        if c.hand not in ("right", "left"):
            raise ValueError("teleop.hand must be 'right' or 'left'")
        self._other = "left" if c.hand == "right" else "right"
        if not 0.0 < c.release_threshold <= c.engage_threshold < 1.0:
            raise ValueError("need 0 < teleop.release_threshold <= teleop.engage_threshold < 1")
        for label, joint in (("pitch", c.pitch_joint), ("roll", c.roll_joint), ("yaw", c.yaw_joint)):
            if joint not in self.kin.motors:
                raise ValueError(f"teleop.{label}_joint={joint!r} is not one of the IK joints {self.kin.motors}")
        self._i_pitch = self.kin.motors.index(c.pitch_joint)
        self._i_roll = self.kin.motors.index(c.roll_joint)
        self._i_yaw = self.kin.motors.index(c.yaw_joint)
        home_deg = list(c.home_deg) if c.home_deg is not None else [0.0] * len(self.kin.motors)
        if len(home_deg) != len(self.kin.motors):
            raise ValueError(f"teleop.home_deg needs {len(self.kin.motors)} values ({self.kin.motors})")
        self._q_home = self.kin.motor_to_q(dict(zip(self.kin.keys, home_deg)))
        self._lo = self.kin.chain.lower + self.kin.limit_margin
        self._hi = self.kin.chain.upper - self.kin.limit_margin

        self._link: VRLink | None = None
        self._urls: list[str] = []
        self.needs_seed = True
        self._seeded_once = False
        self._phase = "idle"  # idle | homing | ready
        self._q_cmd: np.ndarray | None = None  # what is sent
        self._qd = np.zeros(len(self.kin.motors))  # its velocity (rad/s), for the acceleration limit
        self._q_goal: np.ndarray | None = None  # where the controller wants the joints
        self._q_ik: np.ndarray | None = None  # last IK solution, the next solve's seed
        self._gripper: float | None = None
        self._home_from: np.ndarray | None = None
        self._home_t0 = 0.0
        self._home_T = 1.0
        self._last_t: float | None = None
        # engage (clutch) state, latched in _engage()
        self._engaged = False
        self._precision = False
        self._align: np.ndarray | None = None  # turns robot axes so that "forward" = controller heading at the squeeze
        self._p0 = np.zeros(3)  # tool position at engage
        self._R0 = np.eye(3)  # tool orientation at engage
        self._c0 = np.zeros(3)  # controller position at engage (aligned)
        self._H0 = np.eye(3)  # controller frame at engage
        self._z_floor: float | None = None
        self._pan_origin = np.zeros(3)
        self._dir0: np.ndarray | None = None  # tool heading at engage (horizontal, from the pan axis)
        self._s_roll = 1.0
        self._s_pitch = 1.0
        self._roll_prev = 0.0
        self._lo_eng = self._lo.copy()  # joint limits for this engage: a joint already past
        self._hi_eng = self._hi.copy()  # its URDF limit may stay there, not go further
        # input bookkeeping
        self._last_raw_p: np.ndarray | None = None
        self._prev_pressed: set[str] = set()
        self._prev_other: set[str] = set()
        self._events: list[str] = []
        self._problem = ""
        self._at_edge = False
        self._teleop_status = ""
        self._warned_at: dict[str, float] = {}
        # terminal log of the VR state (nobody may be looking at the headset's panel)
        self._state_logged = ""
        self._state_pending = ""
        self._state_since = 0.0

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
    def state(self) -> str:
        if self._phase == "homing":
            return "homing"
        if self._engaged:
            return "moving"
        return self._problem.lower() if self._problem else self._phase

    @property
    def help_text(self) -> str:
        c = self.config
        b1 = "A" if c.hand == "right" else "X"
        o1, o2 = ("X", "Y") if c.hand == "right" else ("A", "B")
        urls = "  or  ".join(self._urls) or "(not started)"
        return (
            f"VR teleop, {c.hand} controller. In the Quest browser open:  {urls}\n"
            + ("  first time: certificate warning -> Advanced -> Proceed\n" if c.https else "")
            + "  then 'Start (passthrough)'; the headset may then be put down (proximity sensor off).\n"
            "  hold GRIP = new reference here: the tool follows the controller (release = it stops)\n"
            "  forward = where the controller points when you squeeze   TRIGGER = close the gripper\n"
            f"  tilt / twist = gripper tilt / roll   {b1} = precision   thumbstick click = home\n"
            f"  other controller: {o1} = record start/stop, {o2} = redo"
        )

    # ── lifecycle ────────────────────────────────────────────────────────────

    @property
    def is_connected(self) -> bool:
        return self._link is not None

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    def connect(self, calibrate: bool = True) -> None:
        if self._link is not None:
            return
        c = self.config
        link = VRLink(host=c.host, port=c.port, https=c.https, cert_file=c.cert_file, key_file=c.key_file, hand=c.hand)
        self._urls = link.start()
        self._link = link
        logger.info("vr_ee: in the Quest browser open  %s", "  or  ".join(self._urls))

    def disconnect(self) -> None:
        if self._link is not None:
            self._link.stop()
            self._link = None

    # ── called by teleop.py ──────────────────────────────────────────────────

    def seed_from_observation(self, observation: dict[str, Any]) -> None:
        """Start from the follower's measured joints (first start, and after AI control)."""
        q = self.kin.motor_to_q(observation)
        outside = self.kin.out_of_limits(q)
        if outside:
            logger.warning(
                "%s outside the URDF joint limits. Is the follower in degrees (use_degrees=True)? "
                "Check teleop.invert_joints / teleop.joint_offsets_deg.", outside,
            )
        c = self.config
        self._gripper = float(observation.get(c.gripper_key, c.gripper_max)) if c.gripper_key else None
        self._q_cmd = q
        self._qd = np.zeros_like(q)
        self._q_goal = None
        self._q_ik = None
        self._engaged = False
        self._last_t = time.perf_counter()
        self.needs_seed = False
        logger.info("vr_ee: tool at %s", format_pose(self.kin.fk(q)))
        if not self._seeded_once and c.home_on_start:
            self._start_homing()
        else:
            self._phase = "ready"
        self._seeded_once = True

    def pop_events(self) -> list[str]:
        """Teleop events from the other controller's buttons ("toggle_record", "redo")."""
        events, self._events = self._events, []
        return events

    def show_status(self, text: str) -> None:
        """teleop.py's status (recording / idle, HUMAN / AI), shown on the headset's third line."""
        self._teleop_status = text

    def _start_homing(self) -> None:
        c = self.config
        self._phase = "homing"
        self._engaged = False
        self._q_goal = None
        self._home_from = self._q_cmd.copy()
        self._qd = np.zeros_like(self._q_cmd)
        dist_deg = math.degrees(float(np.max(np.abs(self._q_home - self._home_from)))) if self._q_home.size else 0.0
        self._home_T = max(c.home_min_time_s, dist_deg / max(c.home_speed_deg_s, 1e-3))
        self._home_t0 = time.perf_counter()
        logger.info("vr_ee: homing, every arm joint -> its home angle over %.1f s. Keep clear.", self._home_T)

    # ── action ───────────────────────────────────────────────────────────────

    def get_action(self) -> dict[str, float]:
        if self._q_cmd is None:
            raise RuntimeError("vr_ee: call seed_from_observation(obs) before get_action()")
        if self._link is None:
            raise RuntimeError("vr_ee: not connected")
        c = self.config
        now = time.perf_counter()
        dt = min(0.1, now - (self._last_t or now))  # cap so a stall cannot produce a jump
        self._last_t = now

        inp = self._read_input()
        self._problem = inp.problem
        edges = inp.pressed - self._prev_pressed
        self._prev_pressed = inp.pressed
        other_edges = inp.other_pressed - self._prev_other
        self._prev_other = inp.other_pressed
        if "a" in other_edges:
            self._events.append("toggle_record")
        if "b" in other_edges:
            self._events.append("redo")

        if self._phase == "homing":
            s = min(1.0, (now - self._home_t0) / self._home_T)
            self._q_cmd = self._home_from + smoothstep(s) * (self._q_home - self._home_from)
            if s >= 1.0:
                self._phase = "ready"
                self._qd = np.zeros_like(self._q_cmd)
                logger.info("vr_ee: arm at home. Hold GRIP to move it.")
        else:
            sample = inp.sample
            self._handle_buttons(edges, sample)
            engaged = sample is not None and self._grip_held(sample)
            if sample is not None and engaged:
                if not self._engaged:
                    self._engage(sample)  # this squeeze = the new reference, where the arm is now
                self._engaged = True
                self._q_goal = self._goal(sample)
            else:
                self._engaged = False
            if sample is not None and c.gripper_key and (engaged or not c.gripper_needs_engage):
                self._gripper = self._trigger_to_gripper(sample.trigger)
            if self._q_goal is not None:  # released: the arm settles on its last target and holds
                self._q_cmd = self._follow(self._q_cmd, self._q_goal, dt)

        self._publish_status(now)
        self._log_state(now)
        if not self._link.connected:
            self._warn("link", f"vr_ee: waiting for the headset: open {self._urls[0]} in the Quest browser", 10.0)
        return self._action()

    def send_feedback(self, feedback: dict[str, float]) -> None:
        raise NotImplementedError

    # ── input ────────────────────────────────────────────────────────────────

    def _read_input(self) -> _Input:
        c = self.config
        msg = self._link.latest(c.timeout_s)
        if msg is None:
            self._last_raw_p = None
            # page open but no XR frames: Start not pressed, or the headset fell asleep
            problem = "PRESS START IN THE HEADSET (ASLEEP?)" if self._link.connected else "HEADSET NOT CONNECTED"
            return _Input(None, problem, set(), set())
        ctrls = msg.get("controllers")
        ctrls = ctrls if isinstance(ctrls, dict) else {}
        mine = ctrls.get(c.hand)
        pressed, other_pressed = _buttons(mine), _buttons(ctrls.get(self._other))
        if not isinstance(mine, dict):
            self._last_raw_p = None
            return _Input(None, f"NO {c.hand.upper()} CONTROLLER", pressed, other_pressed)
        try:
            if not mine.get("tracked"):
                raise ValueError("not tracked")
            p = XR_TO_ROBOT @ np.asarray(mine["p"], dtype=float).reshape(3)
            R = _quat_to_matrix(mine["q"])
            grip = float(mine.get("grip", 0.0))
            trigger = float(mine.get("trigger", 0.0))
            if not np.all(np.isfinite(p)):
                raise ValueError("non-finite position")
        except (KeyError, TypeError, ValueError):
            self._last_raw_p = None
            return _Input(None, "CONTROLLER NOT TRACKED", pressed, other_pressed)
        jump = self._last_raw_p is not None and float(np.linalg.norm(p - self._last_raw_p)) > c.jump_max_m
        self._last_raw_p = p
        if jump:  # the next tick compares against this sample, so a real relocation is accepted then
            return _Input(None, "TRACKING JUMP", pressed, other_pressed)
        H = XR_TO_ROBOT @ R @ XR_TO_ROBOT.T
        sample = _Sample(p=p, H=H, grip=min(max(grip, 0.0), 1.0), trigger=min(max(trigger, 0.0), 1.0))
        return _Input(sample, "", pressed, other_pressed)

    def _grip_held(self, sample: _Sample | None) -> bool:
        if sample is None:
            return False
        c = self.config
        return sample.grip >= (c.release_threshold if self._engaged else c.engage_threshold)

    def _handle_buttons(self, edges: set[str], sample: _Sample | None) -> None:
        if "a" in edges:
            self._precision = not self._precision
            logger.info("vr_ee: precision %s (x%.2f)", "ON" if self._precision else "off", self._scale())
            self._haptic(0.4, 40)
            if self._engaged and sample is not None:
                self._engage(sample, new_heading=False)  # re-latch at the new scale: no jump, same directions
        if "stick_click" in edges and not self._engaged:
            self._start_homing()

    def _scale(self) -> float:
        c = self.config
        return c.position_scale * (c.precision_scale if self._precision else 1.0)

    # ── controller -> joints ─────────────────────────────────────────────────

    def _engage(self, sample: _Sample, new_heading: bool = True) -> None:
        """This squeeze is the new reference: the controller pose now is matched to where the arm is
        now, and from here on only the controller's motion counts. Forward = where it points now."""
        c = self.config
        heading = _heading(sample.H) if new_heading or self._align is None else None
        if heading is not None:
            psi = math.atan2(float(heading[1]), float(heading[0]))
            self._align = rot_z(math.radians(c.yaw_offset_deg) - psi)
        elif self._align is None:  # pointed straight up / down with no usable top: the XR space's forward
            self._align = rot_z(math.radians(c.yaw_offset_deg))
        # Where the arm is: the target it holds since the last release (right after start-up or AI
        # control: its measured pose; after homing: the home pose).
        base = self._q_goal if self._q_goal is not None else self._q_cmd
        self._lo_eng = np.minimum(self._lo, base)
        self._hi_eng = np.maximum(self._hi, base)
        self._q_ik = base.copy()
        T, origins, axes = self.kin.chain.frames(base)
        self._p0 = T[:3, 3].copy()
        self._R0 = T[:3, :3].copy()
        self._z_floor = None if c.z_min_m is None else min(c.z_min_m, float(self._p0[2]))
        self._c0 = self._align @ sample.p
        self._H0 = sample.H.copy()
        self._roll_prev = 0.0
        self._pan_origin = origins[self._i_yaw].copy()
        self._dir0 = self._horizontal(self._p0 - self._pan_origin)
        # Joint directions: + on the roll joint = + roll about the gripper's own axis (pointing out of
        # the jaws); + on the pitch joint = + pitch about the arm's left = nose down.
        self._s_roll = _sign(float(np.dot(axes[self._i_roll], self._p0 - origins[self._i_roll])))
        left = np.cross(UP, self._dir0) if self._dir0 is not None else axes[self._i_pitch]
        self._s_pitch = _sign(float(np.dot(axes[self._i_pitch], left)))
        self._at_edge = False
        self._haptic(0.25, 30)

    def _goal(self, sample: _Sample) -> np.ndarray:
        """Coupled Cartesian target: SO101 tilt/roll, or full SE(3) in frame mode."""
        c = self.config
        p_t = self._p0 + self._scale() * self._sign_xyz * (self._align @ sample.p - self._c0)
        if self._z_floor is not None:
            p_t[2] = max(p_t[2], self._z_floor)
        if c.orientation_mode == 'frame':
            # Controller rotation since the clutch engage, expressed in the same
            # aligned base axes used for translation; retain all three DOF.
            delta = self._align @ sample.H @ self._H0.T @ self._align.T
            R_t = rotvec_to_matrix(c.orientation_scale * self._sign_rotation * matrix_to_rotvec(delta)) @ self._R0
            return self._solve_position(p_t, R_t).copy()
        roll, pitch = self._rot_deltas(sample.H)
        # Express wrist input as a tool orientation target and solve it together
        # with position. Adding wrist angles after IK moves the tool point.
        _, _, axes = self.kin.chain.frames(self._q_ik)
        R_t = self._hold_rotation(p_t)
        R_t = rotvec_to_matrix(axes[self._i_roll] * self._s_roll * self._sign_roll * c.roll_scale * roll) @ R_t
        R_t = rotvec_to_matrix(axes[self._i_pitch] * self._s_pitch * self._sign_pitch * c.pitch_scale * pitch) @ R_t
        return self._solve_position(p_t, R_t).copy()

    def _rot_deltas(self, H: np.ndarray) -> tuple[float, float]:
        """(roll, pitch) of the controller since the engage, about its own axes: pitch = elevation of
        where it points (+ = nose down), roll = twist about where it points. Left / right is left out."""
        f0, u0 = self._H0[:, 0], self._H0[:, 2]
        f, u = H[:, 0], H[:, 2]
        f_loc = self._H0.T @ f
        pitch = math.atan2(-float(f_loc[2]), math.hypot(float(f_loc[0]), float(f_loc[1])))
        u_swing = _swing(f0, f) @ u0  # its top, had it only been pointed, not twisted
        roll = math.atan2(float(np.dot(np.cross(u_swing, u), f)), float(np.dot(u_swing, u)))
        roll = self._roll_prev + _wrap(roll - self._roll_prev)  # past 180 deg keeps counting
        self._roll_prev = roll
        return roll, pitch

    def _solve_position(self, p_t: np.ndarray, R_t: np.ndarray | None = None) -> np.ndarray:
        c = self.config
        q_seed = self._q_ik
        res = self._ik(p_t, q_seed, R_t)
        def good(result):
            return (np.all(np.isfinite(result.q)) and result.pos_err <= c.max_pos_error_m
                    and (c.orientation_mode != 'frame' or c.orientation_weight <= 0
                         or result.rot_err <= math.radians(c.max_rot_error_deg)))
        if good(res):
            self._q_ik = res.q
            self._at_edge = False
            return self._q_ik
        # Out of reach: go as far as the arm can along the line from the last reached point.
        p_prev = self.kin.fk(q_seed)[:3, 3]
        R_prev = self.kin.fk(q_seed)[:3, :3]
        rv = matrix_to_rotvec(R_prev.T @ R_t) if c.orientation_mode == 'frame' and R_t is not None else None
        lo, hi, best = 0.0, 1.0, None
        for _ in range(c.reach_search_steps):
            mid = 0.5 * (lo + hi)
            rotation = R_prev @ rotvec_to_matrix(mid * rv) if rv is not None else R_t
            r = self._ik(p_prev + mid * (p_t - p_prev), q_seed, rotation)
            if good(r):
                lo, best = mid, r.q
            else:
                hi = mid
        if best is not None:
            self._q_ik = best
        if not self._at_edge:
            self._haptic(0.7, 80, key="edge", period_s=0.5)
        self._at_edge = True
        self._warn("reach", f"vr_ee: target out of reach ({res.pos_err * 1e3:.0f} mm short): the arm follows as far as it goes.")
        return self._q_ik

    def _ik(self, p: np.ndarray, q_seed: np.ndarray, R_t: np.ndarray | None = None) -> IKResult:
        c = self.config
        T = np.eye(4)
        T[:3, :3] = self._hold_rotation(p) if R_t is None else R_t
        T[:3, 3] = p
        return self.kin.solve(
            T, q_seed, orientation_weight=c.orientation_weight, max_iters=c.ik_iters,
            orientation_axes=None if c.orientation_mode == 'frame' else (self._i_pitch, self._i_roll),
        )

    def _horizontal(self, v: np.ndarray) -> np.ndarray | None:
        h = v - float(np.dot(v, UP)) * UP
        n = float(np.linalg.norm(h))
        return h / n if n > 0.03 else None

    def _hold_rotation(self, p: np.ndarray) -> np.ndarray:
        """Engage orientation turned with the arm toward ``p`` (the pan swing a position move needs)."""
        d = self._horizontal(p - self._pan_origin)
        if d is None or self._dir0 is None:
            return self._R0
        phi = math.atan2(float(np.dot(np.cross(self._dir0, d), UP)), float(np.dot(self._dir0, d)))
        return rotvec_to_matrix(UP * phi) @ self._R0

    def _follow(self, q: np.ndarray, goal: np.ndarray, dt: float) -> np.ndarray:
        """Move every joint toward ``goal`` with a speed limit, an acceleration limit and a braking
        curve (never faster than it can stop in the remaining distance), so it does not overshoot."""
        c = self.config
        if dt <= 0.0:
            return q
        v_max = math.radians(c.max_joint_speed_deg_s)
        a_max = math.radians(c.max_joint_accel_deg_s2)
        err = goal - q
        dist = np.abs(err)
        speed = np.minimum(np.minimum(c.follow_gain * dist, v_max), np.sqrt(2.0 * a_max * dist))
        speed[dist < math.radians(c.joint_deadband_deg)] = 0.0
        v_want = np.sign(err) * speed
        self._qd = self._qd + np.clip(v_want - self._qd, -a_max * dt, a_max * dt)
        q_new = q + self._qd * dt
        crossed = (np.sign(goal - q_new) != np.sign(err)) & (err != 0.0)  # stepped past the goal: land on it
        q_new[crossed] = goal[crossed]
        self._qd[crossed] = 0.0
        return q_new

    def _trigger_to_gripper(self, trigger: float) -> float:
        c = self.config
        t = 1.0 - trigger if c.gripper_invert else trigger
        return c.gripper_max - t * (c.gripper_max - c.gripper_min)

    def _action(self) -> dict[str, float]:
        out = self.kin.q_to_motor(self._q_cmd)
        if self.config.gripper_key and self._gripper is not None:
            out[self.config.gripper_key] = float(self._gripper)
        return out

    # ── feedback to the headset ──────────────────────────────────────────────

    def _publish_status(self, now: float) -> None:
        if self._phase == "homing":
            pct = 100.0 * min(1.0, (now - self._home_t0) / self._home_T)
            head, color = f"HOMING {pct:3.0f}%  keep clear", "#ffa040"
        elif self._engaged:
            head = f"MOVING  precision x{self._scale():.2f}" if self._precision else "MOVING"
            color = "#50e050"
        elif self._problem:
            head, color = self._problem, "#ff6060"
        else:
            head = "READY  hold GRIP to move" + ("  (precision)" if self._precision else "")
            color = "#ffffff"
        p = self.kin.fk(self._q_cmd)[:3, 3]
        grip = f"gripper {self._gripper:3.0f}   " if self._gripper is not None else ""
        line2 = f"{grip}tool x {p[0] * 100:4.1f}  y {p[1] * 100:+5.1f}  z {p[2] * 100:4.1f} cm"
        line3 = "EDGE OF REACH: the arm stops short" if self._engaged and self._at_edge else self._teleop_status
        self._link.set_status({"lines": [head, line2, line3], "color": color})

    def _log_state(self, now: float) -> None:
        """Log the VR state in the terminal when it changes and stays for 0.3 s (one-tick blips are skipped)."""
        state = self.state
        if state != self._state_pending:
            self._state_pending, self._state_since = state, now
        elif state != self._state_logged and now - self._state_since >= 0.3:
            self._state_logged = state
            logger.info("vr_ee: %s", state.upper())

    def _haptic(self, intensity: float, duration_ms: float, key: str | None = None, period_s: float = 0.0) -> None:
        if not self.config.haptics or self._link is None:
            return
        if key is not None:
            now = time.monotonic()
            if now - self._warned_at.get(f"haptic_{key}", -math.inf) < period_s:
                return
            self._warned_at[f"haptic_{key}"] = now
        self._link.haptic(intensity, duration_ms)

    def _warn(self, key: str, msg: str, period_s: float = 2.0) -> None:
        now = time.monotonic()
        if now - self._warned_at.get(key, -math.inf) >= period_s:
            self._warned_at[key] = now
            logger.warning(msg)


# ── preview: headset only, no robot ────────────────────────────────────────────


def main() -> None:
    from .devices import load_config, make_parser, make_teleop

    parser = make_parser(
        "Preview of the VR teleop with NO robot: the arm is simulated (it follows the targets with\n"
        "the same speed / acceleration limits). Use it to connect the headset, check the directions,\n"
        "the gripper and the buttons; drive the real arm with src.teleop --preset so101_vr.\n\n"
        + (__doc__ or "")
    )
    args, overrides = parser.parse_known_args()
    cfg = load_config(args.preset or "so101_vr", overrides)
    if cfg.teleop_type != "vr_ee":
        raise SystemExit(f"preset {cfg.name!r} uses teleop {cfg.teleop_type!r}; the preview needs --teleop.type=vr_ee")

    from lerobot.utils.robot_utils import precise_sleep
    from lerobot.utils.utils import init_logging

    init_logging()
    teleop = make_teleop(cfg)
    sim = {k: 0.0 for k in teleop.kin.keys}  # simulated arm, at zero
    if teleop.config.gripper_key:
        sim[teleop.config.gripper_key] = teleop.config.gripper_max
    teleop.connect()
    logging.info("PREVIEW, no robot.\n%s", teleop.help_text)
    teleop.show_status("PREVIEW: no robot")
    period = 1.0 / cfg.fps
    try:
        teleop.seed_from_observation(sim)
        while True:
            t0 = time.perf_counter()
            action = teleop.get_action()
            sim.update(action)
            for event in teleop.pop_events():
                logging.info("button -> teleop event %r (Space / Left in src.teleop)", event)
            joints = "  ".join(f"{JOINT_LABELS.get(k[:-4], k[:-4])} {v:+6.1f}" for k, v in action.items())
            print(f"[{teleop.state[:24]:<24}] {joints}   ", end="\r")
            precise_sleep(max(period - (time.perf_counter() - t0), 0.0))
    except KeyboardInterrupt:
        pass
    finally:
        print()
        teleop.disconnect()


if __name__ == "__main__":
    main()

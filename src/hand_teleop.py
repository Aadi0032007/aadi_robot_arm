"""Hand Cartesian teleop: your thumb and index finger, seen by a webcam (MediaPipe), move the TOOL.

A LeRobot ``Teleoperator`` (type ``hand_ee``) for ``teleop.py`` / ``record.py``, on the same
URDF IK as ``gamepad_ee``. Defaults = SO-101 follower (bundled URDF).

    pip install mediapipe
    python -m src.hand_teleop --teleop.type=hand_ee              # preview: webcam only, no robot
    python -m src.teleop --robot.type=so101_follower --robot.port=COM4 --robot.id=aadi_so101_follower \\
                         --teleop.type=hand_ee --teleop.camera_index=0

Every start:

    1. HOME       the arm glides to every arm joint at 0 deg (the gripper stays where it is).
    2. CALIBRATE  the window asks for your hand: hold it where you want "zero" to be, at the
                  distance from the webcam you want as zero, and keep it still (~1.5 s).
                  That hand pose becomes the reference: hand at reference = arm at zero.
    3. TRACK      tool = zero pose + (hand - reference).
                  c in the window = back to 1 + 2 (re-zero). No hand in view = the arm holds.

Mapping (you stand behind the robot, facing where it reaches; the webcam faces you and the
picture is mirrored, so the window behaves like a mirror):

    Hand                                        Robot
    move right / left                           Y  tool right / left
    move up / down                              Z  tool up / down
    move toward / away from the webcam          X  tool forward / back
    tilt the fingertips down / up               pitch: gripper nose down / up   (wrist_flex only)
    turn about the axis pointing at the webcam  roll: gripper rolls the same way (wrist_roll only)
    turn left / right about the vertical        yaw: the whole arm swings        (shoulder_pan)
    pinch thumb + index together / apart        gripper closes / opens

A 5-DOF arm cannot turn the gripper sideways without moving it, so yaw swings the arm, as in
``gamepad_ee``'s arm mode; pitch and roll move only their wrist motor, and while the tool
translates the IK keeps the gripper's tilt and roll. Facing the robot instead of standing
behind it: ``--teleop.invert=x,y,roll,pitch``.

Only thumb and index landmarks are used (0 wrist, 1-4 thumb, 5-8 index):

    position, rotation  rigid triangle wrist(0) / thumb base(1) / index knuckle(5); pinching
                        does not move it, so closing the gripper does not move the arm
    distance (X)        hand scale from the rigid points 0, 1, 2, 5 (image size vs MediaPipe's
                        metric world landmarks) with an assumed webcam field of view
    gripper             |thumb tip(4) - index tip(8)| / |wrist(0) - index knuckle(5)|, in 3D

Stabilisation, in order (all in HandEETeleopConfig):

    frame gate      drop frames with low MediaPipe confidence, a thumb/index point at the image
                    edge, or the other hand (lock_hand); after a dropout, reacquire_frames good
                    frames in a row are needed before the arm follows again
    glitch gate     drop a frame that jumps more than jump_max_m / jump_max_deg from the last
                    good one, unless the jump persists jump_accept_frames frames (a real move)
    distance        the hand scale gets its own median + slow filter (scale_min_cutoff_hz):
                    it drives X and the cm-per-pixel of Y/Z, so its noise used to reach all three
    median          per-axis median over median_window frames (kills one-frame spikes)
    One Euro        adaptive low-pass: filter_min_cutoff_hz / _rot_hz at rest (lower = steadier),
                    filter_beta_pos / _rot to cut the lag on fast moves
    hold bands      play of hold_band_m (hold_band_depth_m for X) / hold_band_deg: jitter smaller
                    than this never reaches the arm, anywhere in the workspace
    arm motion      every joint follows its target with a speed limit, an acceleration limit and
                    a braking curve (max_joint_speed_deg_s, max_joint_accel_deg_s2, follow_gain):
                    no jerks when the target jumps, no overshoot

The window shows the measured jitter before and after (``jitter raw ... -> out ...``) and why
a frame was dropped. Steadier still: lower filter_min_cutoff_*, raise hold_band_*, light the
hand evenly (webcam blur and auto-exposure flicker are the main noise sources).

Safety: the arm only follows a calibrated hand; targets are clamped to max_offset_m /
max_angle_deg around zero and above z_min_m; a target out of reach is followed as far as the
arm can go. The hand webcam must not be one of the robot cameras (pick another camera_index).
"""

from __future__ import annotations

if __package__ in (None, ""):  # allow `python src/hand_teleop.py`
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "src"

import logging
import math
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable

import cv2
import numpy as np

from ._bootstrap import ensure_lerobot_importable
from .hand_tracking import (
    INDEX_CHAIN,
    INDEX_MCP,
    INDEX_TIP,
    THUMB_CHAIN,
    THUMB_CMC,
    THUMB_MCP,
    THUMB_TIP,
    WRIST,
    Backlash,
    HandDetection,
    HandTracker,
    MedianFilter,
    OneEuroFilter,
    pinch_to_gripper,
)
from .kinematics import ArmKinematics, ArmKinematicsConfig, IKResult, format_pose, matrix_to_rpy, rotvec_to_matrix
from .so101_ik import SO101_ARM_JOINTS, SO101_TIP_LINK, SO101_URDF, parse_csv, parse_offsets, smoothstep

logger = logging.getLogger(__name__)

AXES = ("x", "y", "z", "roll", "pitch", "yaw")
USED_POINTS = tuple(range(9))  # wrist + thumb 1-4 + index 5-8
ANCHOR = (WRIST, THUMB_CMC, INDEX_MCP)  # rigid base triangle: position + orientation
RIGID_POINTS = (WRIST, THUMB_CMC, THUMB_MCP, INDEX_MCP)  # hand scale (distance); no fingertips, so pinching cannot change it
# Mirrored webcam axes (x right on screen, y down, z away from the webcam) -> robot base axes
# (X forward, Y left, Z up) for a user behind the robot facing the webcam: toward the webcam
# = forward, your right = the robot's right (-Y), up = up. det = -1: it also undoes the mirror.
CAM_TO_ROBOT = np.array([[0.0, 0.0, -1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])
GAP_RESET_S = 0.25  # hand missing longer than this: the filters start over
WINDOW = "hand_ee"

GREEN, CYAN, ORANGE, YELLOW, RED, WHITE, GREY = (
    (0, 220, 0), (255, 200, 0), (0, 165, 255), (0, 230, 255), (0, 0, 255), (255, 255, 255), (200, 200, 200),
)
JOINT_LABELS = {"shoulder_pan": "pan", "shoulder_lift": "lift", "elbow_flex": "elbow", "wrist_flex": "w.flex",
                "wrist_roll": "w.roll"}


def _lerobot_base():
    ensure_lerobot_importable()  # also when run as `python -m src.hand_teleop`
    from lerobot.teleoperators import Teleoperator, TeleoperatorConfig

    return Teleoperator, TeleoperatorConfig


_Teleoperator, _TeleoperatorConfig = _lerobot_base()


@dataclass
class HandEETeleopConfig(_TeleoperatorConfig):
    # ── webcam + MediaPipe ────────────────────────────────────────────────────
    camera_index: int = 0
    camera_width: int = 640
    camera_height: int = 480
    camera_hfov_deg: float = 65.0  # webcam horizontal field of view; only scales the X (distance) axis
    mirror: bool = True  # selfie view; also what MediaPipe's left/right labels assume
    show: bool = True  # window with landmarks, state and keys (c = re-zero, q = quit)
    hand: str = "any"  # "any" | "right" | "left"
    model_complexity: int = 1  # legacy MediaPipe API: 0 = faster, 1 = steadier landmarks
    hand_model: str | None = None  # Tasks API model file (auto-downloaded to models/)
    min_detection_confidence: float = 0.7
    min_tracking_confidence: float = 0.7  # higher = MediaPipe re-detects instead of drifting
    # ── frame gate ────────────────────────────────────────────────────────────
    min_hand_score: float = 0.8  # MediaPipe hand confidence; frames below are dropped
    border_margin: float = 0.03  # drop frames with a thumb/index point this close to the edge (x image width)
    lock_hand: bool = True  # after calibration follow only the hand (left/right) that was calibrated
    reacquire_frames: int = 4  # after a dropout, good frames in a row before the arm follows again
    jump_max_m: float = 0.08  # one-frame jump bigger than this = glitch, dropped ...
    jump_max_deg: float = 30.0
    jump_accept_frames: int = 4  # ... unless it persists this many frames (then it is a real move)
    # ── kinematics (defaults = SO-101 follower) ──────────────────────────────
    urdf_path: str = str(SO101_URDF)
    tip_link: str = SO101_TIP_LINK
    base_link: str | None = None
    motors: list[str] = field(default_factory=lambda: list(SO101_ARM_JOINTS))  # chain order, no ".pos"
    urdf_joints: list[str] | None = None  # None = same names as motors
    invert_joints: str = ""  # comma list, e.g. "elbow_flex"
    joint_offsets_deg: str = ""  # e.g. "elbow_flex:5,wrist_flex:-3"
    limit_margin_deg: float = 1.0
    pitch_joint: str = "wrist_flex"
    roll_joint: str = "wrist_roll"
    yaw_joint: str = "shoulder_pan"
    # ── start-up: home, then the hand reference ──────────────────────────────
    home_deg: list[float] | None = None  # motor degrees per arm joint; None = every joint at 0
    home_speed_deg_s: float = 30.0
    home_min_time_s: float = 2.0
    calib_hold_s: float = 1.5  # hold the hand this long to set the reference
    calib_still_m: float = 0.02  # "still" = within this distance ...
    calib_still_deg: float = 10.0  # ... and this rotation of where the countdown started
    # ── hand -> tool mapping ──────────────────────────────────────────────────
    position_scale: list[float] = field(default_factory=lambda: [1.0, 1.0, 1.0])  # robot m per hand m: x, y, z
    rotation_scale: list[float] = field(default_factory=lambda: [1.0, 1.0, 1.0])  # roll, pitch, yaw (0 = off)
    invert: str = ""  # comma list of axes to flip: x,y,z,roll,pitch,yaw
    deadband_m: float = 0.01  # offsets from the reference below this count as zero
    deadband_deg: float = 4.0
    max_offset_m: float = 0.30  # per axis, around the zero pose
    max_angle_deg: float = 90.0
    # ── smoothing ─────────────────────────────────────────────────────────────
    median_window: int = 5  # frames; 1 = off
    scale_min_cutoff_hz: float = 0.4  # hand-distance estimate
    filter_min_cutoff_hz: float = 0.5  # position at rest: lower = steadier, more lag
    filter_min_cutoff_rot_hz: float = 0.3  # rotation at rest
    filter_beta_pos: float = 1.5  # cutoff rise per m/s of hand speed (less lag on fast moves)
    filter_beta_rot: float = 0.2  # cutoff rise per rad/s
    filter_beta_pinch: float = 0.5
    hold_band_m: float = 0.005  # jitter smaller than this (Y, Z) never moves the arm
    hold_band_depth_m: float = 0.01  # same for X (distance is the noisiest axis)
    hold_band_deg: float = 3.0
    hand_timeout_s: float = 0.3  # hand data older than this: the arm settles on its last target
    # ── gripper from the pinch ────────────────────────────────────────────────
    gripper_key: str | None = "gripper.pos"  # None = leave the gripper out of the action
    pinch_closed: float = 0.3  # pinch value (shown in the window) that means fully closed
    pinch_open: float = 1.2  # ... and fully open
    gripper_min: float = 0.0
    gripper_max: float = 100.0  # 100 = open on SO-101 / Koch
    gripper_invert: bool = False
    gripper_band: float = 2.0  # gripper changes smaller than this are ignored
    # ── IK / arm motion ───────────────────────────────────────────────────────
    orientation_weight: float = 1.0  # keep the gripper's tilt + roll while translating; 0 = position only
    ik_iters: int = 30
    max_pos_error_m: float = 0.005  # IK result farther than this from the target = out of reach
    reach_search_steps: int = 4  # out of reach: bisection steps toward the target
    max_joint_speed_deg_s: float = 90.0
    max_joint_accel_deg_s2: float = 300.0
    follow_gain: float = 6.0  # 1/s: how fast a joint closes the gap to its target (lower = smoother)
    joint_deadband_deg: float = 0.3  # a joint closer than this to its target stays put
    z_min_m: float | None = 0.03  # tool target never below this height (None = off)


@dataclass
class HandMeasurement:
    """Raw per-frame values from the thumb + index landmarks."""

    anchor_px: np.ndarray  # (2,) image position of the rigid triangle's centre, pixels
    px_per_m: float  # hand scale: image pixels per metre at the hand
    frame: np.ndarray  # (3, 3) hand orientation in webcam axes (columns = hand axes)
    pinch: float  # |thumb tip - index tip| / |wrist - index knuckle|, both in 3D


@dataclass
class HandSample:
    point: np.ndarray  # (3,) hand anchor in metres, webcam axes (x right on screen, y down, z away from the webcam)
    frame: np.ndarray
    pinch: float


@dataclass
class HandCommand:
    v: np.ndarray  # (6,) offsets from the zero pose: x, y, z [m] in the robot base frame, roll, pitch, yaw [rad]
    gripper: float | None
    pinch: float
    stamp: float  # perf_counter() of the webcam frame


# ── hand geometry ────────────────────────────────────────────────────────────────


def measure_hand(det: HandDetection) -> HandMeasurement | None:
    """Anchor, hand scale, orientation and pinch from the thumb + index landmarks, or None."""
    if det.world is None:
        return None
    world = det.world.astype(float)
    img = det.image.astype(float)

    # Scale: MediaPipe's world landmarks are metric and aligned with the image axes, so the image
    # spread over the world spread of the same rigid points is pixels per metre at the hand.
    idx = list(RIGID_POINTS)
    di = img[idx] - img[idx].mean(axis=0)
    dw = world[idx, :2] - world[idx, :2].mean(axis=0)
    denom = float(np.sum(dw * dw))
    if denom < 1e-8:
        return None
    px_per_m = float(np.sum(di * dw)) / denom
    if px_per_m <= 1.0:
        return None

    # Orientation: the index metacarpal (wrist -> index knuckle) and the normal of the rigid
    # wrist / thumb-base / index-knuckle triangle.
    fwd = world[INDEX_MCP] - world[WRIST]
    normal = np.cross(fwd, world[THUMB_CMC] - world[WRIST])
    span = float(np.linalg.norm(fwd))
    if span < 1e-4 or float(np.linalg.norm(normal)) < 1e-8:
        return None
    a1 = fwd / span
    a2 = normal / float(np.linalg.norm(normal))
    frame = np.column_stack([a1, a2, np.cross(a1, a2)])
    pinch = float(np.linalg.norm(world[THUMB_TIP] - world[INDEX_TIP])) / span
    return HandMeasurement(anchor_px=img[list(ANCHOR)].mean(axis=0), px_per_m=px_per_m, frame=frame, pinch=pinch)


def rotation_angle(R: np.ndarray) -> float:
    return math.acos(float(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)))


def mean_rotation(rotations: list[np.ndarray]) -> np.ndarray:
    """Chordal mean: the rotation closest to the average matrix."""
    U, _, Vt = np.linalg.svd(np.sum(rotations, axis=0))
    if np.linalg.det(U @ Vt) < 0.0:
        U[:, -1] *= -1.0
    return U @ Vt


def soft_deadband(v: np.ndarray, band: np.ndarray) -> np.ndarray:
    return np.sign(v) * np.maximum(np.abs(v) - band, 0.0)


def jitter(history: deque) -> tuple[float, float] | None:
    """Frame-to-frame RMS change of (position [m], rotation [rad]) / sqrt(2): ~ the noise level."""
    if len(history) < 5:
        return None
    d = np.diff(np.array(history), axis=0)
    return float(np.sqrt(np.mean(d[:, :3] ** 2)) / math.sqrt(2.0)), float(np.sqrt(np.mean(d[:, 3:] ** 2)) / math.sqrt(2.0))


def _sign(x: float) -> float:
    return -1.0 if x < 0.0 else 1.0


def _px(p: np.ndarray) -> tuple[int, int]:
    return int(round(float(p[0]))), int(round(float(p[1])))


def _text(img: np.ndarray, text: str, org: tuple[int, int], scale: float, color, thickness: int = 1) -> None:
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


# ── webcam thread ────────────────────────────────────────────────────────────────


class _HandInput:
    """Webcam + MediaPipe on a background thread. Holds the hand reference (calibration) and
    turns every good frame into a gated, filtered :class:`HandCommand` while tracking. The
    control thread switches modes with ``begin_calibration`` / ``reset`` and reads ``latest``."""

    def __init__(self, config: HandEETeleopConfig, to_robot: Callable[[np.ndarray, np.ndarray], np.ndarray]) -> None:
        c = config
        self.c = c
        self._to_robot = to_robot
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._cap = None
        self._tracker: HandTracker | None = None
        self.error: BaseException | None = None
        # shared with the control thread, under _lock
        self._mode = "wait"  # wait | calibrate | track
        self._gen = 0  # bumped by every mode change from the control thread
        self._ref: tuple[np.ndarray, np.ndarray] | None = None
        self._ref_hand = ""  # "Left" / "Right" of the calibrated hand
        self._cmd: HandCommand | None = None
        self._progress = 0.0
        self._requests: set[str] = set()
        self._status: dict[str, Any] = {}
        # webcam thread only
        self._seen_gen = -1
        self._calib: list[tuple[float, HandSample]] = []
        self._last_seen = -math.inf
        self._streak = 0  # good frames in a row
        self._glitches = 0  # consecutive frames rejected by the glitch gate
        self._last_ok: HandSample | None = None
        self._dropped = ""  # why the last frame was not used (shown in the window)
        self._fps = 0.0
        self._t_frame: float | None = None
        self._scale_median = MedianFilter(c.median_window)
        self._scale_filter = OneEuroFilter(c.scale_min_cutoff_hz, 0.0)
        self._median = MedianFilter(c.median_window)
        self._filter = OneEuroFilter(
            [c.filter_min_cutoff_hz] * 3 + [c.filter_min_cutoff_rot_hz] * 3,
            [c.filter_beta_pos] * 3 + [c.filter_beta_rot] * 3,
        )
        self._hold = Backlash([c.hold_band_depth_m, c.hold_band_m, c.hold_band_m] + [math.radians(c.hold_band_deg)] * 3)
        self._pinch_filter = OneEuroFilter(c.filter_min_cutoff_hz, c.filter_beta_pinch)
        self._grip_hold = Backlash(c.gripper_band)
        self._deadband = np.array([c.deadband_m] * 3 + [math.radians(c.deadband_deg)] * 3)
        self._limit = np.array([c.max_offset_m] * 3 + [math.radians(c.max_angle_deg)] * 3)
        self._raw_hist: deque[np.ndarray] = deque(maxlen=20)
        self._out_hist: deque[np.ndarray] = deque(maxlen=20)

    # -- lifecycle (control thread) -----------------------------------------------

    def start(self) -> None:
        c = self.c
        cap = cv2.VideoCapture(c.camera_index)
        if not cap.isOpened():
            raise ConnectionError(f"hand_ee: cannot open webcam {c.camera_index} (try --teleop.camera_index=1)")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, c.camera_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, c.camera_height)
        try:
            self._tracker = HandTracker(
                c.hand_model,
                max_num_hands=2 if (c.hand != "any" or c.lock_hand) else 1,
                model_complexity=c.model_complexity,
                min_detection_confidence=c.min_detection_confidence,
                min_tracking_confidence=c.min_tracking_confidence,
            )
        except BaseException:
            cap.release()
            raise
        self._cap = cap
        self._thread = threading.Thread(target=self._run, name="hand_ee-webcam", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        alive = False
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            alive = self._thread.is_alive()
            self._thread = None
        if alive:  # stuck in a driver call; the daemon thread still owns the tracker
            logger.warning("hand_ee: webcam thread did not stop in time")
            return
        if self._tracker is not None:
            self._tracker.close()
            self._tracker = None
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    # -- control-thread API -------------------------------------------------------

    def begin_calibration(self) -> None:
        self._set_mode("calibrate")

    def reset(self) -> None:
        self._set_mode("wait")

    def _set_mode(self, mode: str) -> None:
        with self._lock:
            self._mode = mode
            self._gen += 1
            self._ref = None
            self._ref_hand = ""
            self._cmd = None
            self._progress = 0.0

    @property
    def calibrated(self) -> bool:
        with self._lock:
            return self._ref is not None

    def latest(self, max_age_s: float) -> HandCommand | None:
        with self._lock:
            cmd = self._cmd
        if cmd is None or time.perf_counter() - cmd.stamp > max_age_s:
            return None
        return cmd

    def pop_request(self, name: str) -> bool:
        with self._lock:
            if name in self._requests:
                self._requests.discard(name)
                return True
            return False

    def set_status(self, status: dict[str, Any]) -> None:
        with self._lock:
            self._status = status

    # -- webcam thread ------------------------------------------------------------

    def _run(self) -> None:
        c = self.c
        failing_since: float | None = None
        try:
            while not self._stop.is_set():
                ok, frame = self._cap.read()
                now = time.perf_counter()
                if not ok or frame is None:
                    failing_since = failing_since or now
                    if now - failing_since > 3.0:
                        raise ConnectionError(f"webcam {c.camera_index} stopped delivering frames")
                    time.sleep(0.02)
                    continue
                failing_since = None
                if self._t_frame is not None and now > self._t_frame:
                    self._fps = 0.9 * self._fps + 0.1 / (now - self._t_frame)
                self._t_frame = now
                if c.mirror:
                    frame = cv2.flip(frame, 1)
                h, w = frame.shape[:2]
                det = self._pick(self._tracker.detect(frame))
                self._process(det, w, h, now)
                if c.show:
                    self._draw(frame, det, now)
                    cv2.imshow(WINDOW, frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord("c"), ord("C")):
                        with self._lock:
                            self._requests.add("recalibrate")
                    elif key in (27, ord("q"), ord("Q")):
                        with self._lock:
                            self._requests.add("quit")
        except Exception as e:  # noqa: BLE001
            logger.exception("hand_ee: webcam / MediaPipe thread stopped")
            self.error = e
        finally:
            if c.show:
                try:
                    cv2.destroyWindow(WINDOW)
                except cv2.error:
                    pass

    def _pick(self, hands: list[HandDetection]) -> HandDetection | None:
        if not hands:
            return None
        want = "" if self.c.hand == "any" else self.c.hand.lower()
        if not want and self.c.lock_hand:
            with self._lock:
                want = self._ref_hand.lower()
        if not want:
            return max(hands, key=lambda d: d.score)
        return next((d for d in hands if d.handedness.lower() == want), None)

    def _reject_reason(self, det: HandDetection, w: int, h: int) -> str:
        c = self.c
        if det.world is None:
            return "no 3D landmarks from MediaPipe"
        if det.score < c.min_hand_score:
            return f"low confidence {det.score:.2f}"
        m = c.border_margin * w
        pts = det.image[list(USED_POINTS)]
        if (pts[:, 0] < m).any() or (pts[:, 0] > w - m).any() or (pts[:, 1] < m).any() or (pts[:, 1] > h - m).any():
            return "hand at the image edge"
        return ""

    def _reset_filters(self, scale: bool = True) -> None:
        """Restart the output filters; ``scale=False`` keeps the (continuous) hand-distance filter."""
        if scale:
            self._scale_median.reset()
            self._scale_filter.reset()
        for f in (self._median, self._filter, self._hold, self._pinch_filter, self._grip_hold):
            f.reset()
        self._raw_hist.clear()
        self._out_hist.clear()
        self._last_ok = None
        self._glitches = 0

    def _process(self, det: HandDetection | None, w: int, h: int, now: float) -> None:
        with self._lock:
            mode, gen, ref = self._mode, self._gen, self._ref
        if gen != self._seen_gen:  # the control thread changed mode: start clean
            self._seen_gen = gen
            self._calib = []
            self._reset_filters()

        meas = None
        if det is None:
            self._dropped = "no hand" if self.c.hand == "any" and not self.c.lock_hand else "no (matching) hand"
        else:
            self._dropped = self._reject_reason(det, w, h)
            if not self._dropped:
                meas = measure_hand(det)
                if meas is None:
                    self._dropped = "degenerate landmarks"
        if meas is None:
            self._streak = 0
            if mode == "calibrate" and self._calib:
                self._calib = []
                with self._lock:
                    self._progress = 0.0
            return

        if now - self._last_seen > GAP_RESET_S:
            self._reset_filters()
        self._last_seen = now
        self._streak += 1
        sample = self._to_sample(meas, w, h, now)
        if mode == "calibrate":
            self._calibrate(sample, det.handedness, now, gen)
        elif mode == "track" and ref is not None:
            self._track(sample, now, gen, ref)

    def _to_sample(self, meas: HandMeasurement, w: int, h: int, now: float) -> HandSample:
        """Webcam-frame 3D point from the anchor pixel and the SMOOTHED hand scale: the scale is the
        noisiest quantity and sets X (distance) and the metres-per-pixel of Y / Z."""
        c = self.c
        log_s = self._scale_filter(self._scale_median([math.log(meas.px_per_m)]), now)
        s = math.exp(float(log_s[0]))
        focal = 0.5 * w / math.tan(math.radians(c.camera_hfov_deg) / 2.0)
        return HandSample(
            point=np.array([(meas.anchor_px[0] - 0.5 * w) / s, (meas.anchor_px[1] - 0.5 * h) / s, focal / s]),
            frame=meas.frame,
            pinch=meas.pinch,
        )

    def _calibrate(self, sample: HandSample, label: str, now: float, gen: int) -> None:
        c = self.c
        buf = self._calib
        buf.append((now, sample))
        pts = np.array([s.point for _, s in buf])
        moved = float(np.max(np.linalg.norm(pts - pts.mean(axis=0), axis=1)))
        turned = max(rotation_angle(s.frame @ buf[0][1].frame.T) for _, s in buf)
        if moved > c.calib_still_m or turned > math.radians(c.calib_still_deg):
            buf[:] = [(now, sample)]  # the hand moved: the countdown starts again from here
        held = now - buf[0][0]
        done = held >= c.calib_hold_s and len(buf) >= 5 and self._streak >= c.reacquire_frames
        ref = None
        if done:
            ref = (np.mean([s.point for _, s in buf], axis=0), mean_rotation([s.frame for _, s in buf]))
        with self._lock:
            if gen != self._gen:
                return
            self._progress = min(1.0, held / max(c.calib_hold_s, 1e-3))
            if done:
                self._ref = ref
                self._ref_hand = label if c.lock_hand else ""
                self._mode = "track"
                self._cmd = None
        if ref is not None:
            self._calib = []
            self._reset_filters(scale=False)  # the reference was measured with this distance filter
            logger.info(
                "hand_ee: %s hand reference set (~%.0f cm from the webcam). Tracking.", label or "the", ref[0][2] * 100
            )

    def _track(self, sample: HandSample, now: float, gen: int, ref: tuple[np.ndarray, np.ndarray]) -> None:
        c = self.c
        # Glitch gate: a one-frame teleport is a misdetection, unless it persists.
        if self._last_ok is not None:
            jump = float(np.linalg.norm(sample.point - self._last_ok.point))
            turn = rotation_angle(sample.frame @ self._last_ok.frame.T)
            if jump > c.jump_max_m or turn > math.radians(c.jump_max_deg):
                self._glitches += 1
                if self._glitches < c.jump_accept_frames:
                    self._dropped = f"glitch: {jump * 100:.0f} cm / {math.degrees(turn):.0f} deg in one frame"
                    return
        self._glitches = 0
        self._last_ok = sample

        ref_point, ref_frame = ref
        raw = self._to_robot(sample.point - ref_point, sample.frame @ ref_frame.T)
        self._raw_hist.append(raw)
        v = self._filter(self._median(raw), now)
        v = self._hold(v)
        v = np.clip(soft_deadband(v, self._deadband), -self._limit, self._limit)
        self._out_hist.append(v)

        pinch = float(self._pinch_filter(sample.pinch, now))
        gripper = None
        if c.gripper_key:
            g = pinch_to_gripper(pinch, c.pinch_closed, c.pinch_open, c.gripper_min, c.gripper_max, c.gripper_invert)
            gripper = float(self._grip_hold(g))

        if self._streak < c.reacquire_frames:  # just back in view: let the filters settle first
            self._dropped = "reacquiring"
            return
        cmd = HandCommand(v=v, gripper=gripper, pinch=pinch, stamp=now)
        with self._lock:
            if gen == self._gen:
                self._cmd = cmd

    def _draw(self, frame: np.ndarray, det: HandDetection | None, now: float) -> None:
        c = self.c
        with self._lock:
            mode, progress, cmd, status = self._mode, self._progress, self._cmd, dict(self._status)
        h, w = frame.shape[:2]
        if det is not None:
            pts = det.image
            used = not self._dropped or self._dropped == "reacquiring"
            for chain in (THUMB_CHAIN, INDEX_CHAIN):
                for a, b in zip(chain, chain[1:]):
                    cv2.line(frame, _px(pts[a]), _px(pts[b]), GREEN if used else GREY, 2)
            for i in USED_POINTS:
                cv2.circle(frame, _px(pts[i]), 4, GREEN if used else GREY, -1)
            cv2.polylines(frame, [np.array([_px(pts[i]) for i in ANCHOR], dtype=np.int32)], True, CYAN, 2)
            cv2.line(frame, _px(pts[THUMB_TIP]), _px(pts[INDEX_TIP]), ORANGE, 2)

        fresh = cmd is not None and now - cmd.stamp <= c.hand_timeout_s
        if status.get("phase") == "homing":
            banner, color = f"HOMING: robot -> all joints 0 deg  {100 * status.get('home', 0.0):3.0f}%", ORANGE
        elif mode == "calibrate":
            banner, color = ("HOLD STILL ..." if self._streak else "SHOW YOUR HAND where you want ZERO"), YELLOW
        elif mode == "track":
            banner, color = ("TRACKING", GREEN) if fresh else ("NO HAND - arm settles and holds", RED)
        else:
            banner, color = "starting ...", WHITE
        cv2.rectangle(frame, (0, 0), (w, 34), (0, 0, 0), -1)
        _text(frame, banner, (10, 24), 0.7, color, 2)
        _text(frame, f"{self._fps:4.1f} fps", (w - 90, 24), 0.5, GREY, 1)

        y = 56
        if mode == "calibrate":
            cv2.rectangle(frame, (10, 40), (w - 10, 52), WHITE, 1)
            cv2.rectangle(frame, (10, 40), (10 + int((w - 20) * progress), 52), YELLOW, -1)
            y = 72
        lines: list[tuple[str, tuple[int, int, int]]] = []
        if self._last_ok is not None or self._calib:
            last = self._last_ok if self._last_ok is not None else self._calib[-1][1]
            lines.append((f"hand ~{last.point[2] * 100:.0f} cm from webcam   pinch {last.pinch:.2f}", WHITE))
        if mode == "track" and cmd is not None:
            v = cmd.v
            lines.append((f"dX {v[0] * 100:+5.1f}  dY {v[1] * 100:+5.1f}  dZ {v[2] * 100:+5.1f} cm", WHITE))
            grip = f"   gripper {cmd.gripper:3.0f}" if cmd.gripper is not None else ""
            lines.append((
                f"roll {math.degrees(v[3]):+4.0f}  pitch {math.degrees(v[4]):+4.0f}  yaw {math.degrees(v[5]):+4.0f} deg{grip}",
                WHITE,
            ))
            raw_j, out_j = jitter(self._raw_hist), jitter(self._out_hist)
            if raw_j is not None and out_j is not None:
                lines.append((
                    f"jitter raw {raw_j[0] * 1e3:4.1f} mm {math.degrees(raw_j[1]):3.1f} deg"
                    f"  ->  out {out_j[0] * 1e3:4.1f} mm {math.degrees(out_j[1]):3.1f} deg",
                    GREY,
                ))
        for text in status.get("lines", []):
            lines.append((text, RED if text.startswith("EDGE") else WHITE))
        if self._dropped and mode in ("calibrate", "track"):
            lines.append((f"frame dropped: {self._dropped}", ORANGE))
        for text, col in lines:
            _text(frame, text, (10, y), 0.5, col, 1)
            y += 20
        _text(frame, "c = re-zero (arm goes home)   q / Esc = quit", (10, h - 12), 0.5, GREY, 1)


# ── teleoperator ─────────────────────────────────────────────────────────────────


class HandEETeleop(_Teleoperator):
    config_class = HandEETeleopConfig
    name = "hand_ee"

    def __init__(self, config: HandEETeleopConfig):
        super().__init__(config)
        self.config = config
        c = config
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
        if c.hand not in ("any", "right", "left"):
            raise ValueError("teleop.hand must be 'any', 'right' or 'left'")
        if len(c.position_scale) != 3 or len(c.rotation_scale) != 3:
            raise ValueError("teleop.position_scale and teleop.rotation_scale take 3 values each")
        for label, joint in (("pitch", c.pitch_joint), ("roll", c.roll_joint), ("yaw", c.yaw_joint)):
            if joint not in self.kin.motors:
                raise ValueError(f"teleop.{label}_joint={joint!r} is not one of the IK joints {self.kin.motors}")
        self._i_pitch = self.kin.motors.index(c.pitch_joint)
        self._i_roll = self.kin.motors.index(c.roll_joint)
        self._i_yaw = self.kin.motors.index(c.yaw_joint)
        self._gain = np.array([*c.position_scale, *c.rotation_scale], dtype=float)
        self._gain *= np.array([-1.0 if a in inverted else 1.0 for a in AXES])
        self._cam_to_robot = CAM_TO_ROBOT if c.mirror else CAM_TO_ROBOT @ np.diag([-1.0, 1.0, 1.0])

        # Home pose (every arm joint at 0 deg unless home_deg says otherwise) and the axes the
        # hand's roll / pitch / yaw map onto: the gripper axis, the arm's pitch axis, and up.
        home_deg = list(c.home_deg) if c.home_deg is not None else [0.0] * len(self.kin.motors)
        if len(home_deg) != len(self.kin.motors):
            raise ValueError(f"teleop.home_deg needs {len(self.kin.motors)} values ({self.kin.motors})")
        self._q_home = self.kin.motor_to_q(dict(zip(self.kin.keys, home_deg)))
        T, origins, axes = self.kin.chain.frames(self._q_home)
        self._p_home = T[:3, 3].copy()
        self._R_home = T[:3, :3].copy()
        self._up = np.array([0.0, 0.0, 1.0])
        approach = axes[self._i_roll]
        if float(np.dot(approach, self._p_home - origins[self._i_roll])) < 0.0:
            approach = -approach
        left = np.cross(self._up, approach)
        if np.linalg.norm(left) < 1e-3:  # gripper vertical at home: use the pitch joint's axis
            left = axes[self._i_pitch]
        left = left / np.linalg.norm(left)
        self._B = np.column_stack([approach, left, np.cross(approach, left)])  # roll, pitch, yaw axes
        self._joint_sign = (
            _sign(float(np.dot(axes[self._i_roll], self._B[:, 0]))),
            _sign(float(np.dot(axes[self._i_pitch], self._B[:, 1]))),
            _sign(float(np.dot(axes[self._i_yaw], self._B[:, 2]))),
        )
        self._pan_origin = origins[self._i_yaw].copy()
        self._home_dir = self._horizontal(self._p_home - self._pan_origin)
        self._lo = self.kin.chain.lower + self.kin.limit_margin
        self._hi = self.kin.chain.upper - self.kin.limit_margin

        self._input: _HandInput | None = None
        self.needs_seed = True
        self._phase = "idle"  # idle | homing | calibrating | tracking
        self._q_cmd: np.ndarray | None = None  # what is sent
        self._qd = np.zeros(len(self.kin.motors))  # its velocity (rad/s), for the acceleration limit
        self._q_goal: np.ndarray | None = None  # where the hand wants the joints
        self._q_ik: np.ndarray | None = None  # last IK solution, the next solve's seed
        self._gripper: float | None = None
        self._home_from: np.ndarray | None = None
        self._home_t0 = 0.0
        self._home_T = 1.0
        self._last_t: float | None = None
        self._at_edge = False
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
    def phase(self) -> str:
        return self._phase

    @property
    def quit_requested(self) -> bool:
        """q / Esc pressed in the hand window (teleop.py quits on its own global q / Esc)."""
        return self._input is not None and self._input.pop_request("quit")

    @property
    def help_text(self) -> str:
        c = self.config
        return (
            f"Hand teleop (webcam {c.camera_index}, thumb + index only)\n"
            "  1. the arm goes to all joints = 0 deg\n"
            f"  2. show your hand where you want zero and hold it still ({c.calib_hold_s:.1f} s)\n"
            "  3. move: right/left = Y, up/down = Z, toward/away from the webcam = X\n"
            "     tilt fingertips down/up = pitch (wrist_flex), turn about the axis to the webcam = roll\n"
            "     (wrist_roll), turn left/right = yaw (swings the arm), pinch = gripper\n"
            "  hand window: c = re-zero (arm goes home, show the hand again)   q / Esc = quit\n"
            "  no hand in view = the arm settles on its last target and holds"
        )

    # ── lifecycle ────────────────────────────────────────────────────────────

    @property
    def is_connected(self) -> bool:
        return self._input is not None

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    def connect(self, calibrate: bool = True) -> None:
        if self._input is not None:
            return
        hand_input = _HandInput(self.config, self._hand_to_robot)
        hand_input.start()
        self._input = hand_input
        logger.info("hand_ee: webcam %d open, home pose %s", self.config.camera_index, format_pose(self.kin.fk(self._q_home)))

    def disconnect(self) -> None:
        if self._input is not None:
            self._input.stop()
            self._input = None

    # ── seeding (called by teleop.py / record.py with the follower's observation) ─

    def seed_from_observation(self, observation: dict[str, Any]) -> None:
        q = self.kin.motor_to_q(observation)
        outside = self.kin.out_of_limits(q)
        if outside:
            logger.warning(
                "%s outside the URDF joint limits. Is the follower in degrees (use_degrees=True)? "
                "Check teleop.invert_joints / teleop.joint_offsets_deg.", outside,
            )
        c = self.config
        self._gripper = float(observation.get(c.gripper_key, c.gripper_min)) if c.gripper_key else None
        self._q_cmd = q
        self._qd = np.zeros_like(q)
        self._last_t = time.perf_counter()
        self.needs_seed = False
        if self._phase == "tracking" and self._input is not None and self._input.calibrated:
            # e.g. after AI control: keep the hand reference, glide back to the hand
            logger.info("hand_ee: re-seeded; the arm moves back to your hand (max %.0f deg/s).", c.max_joint_speed_deg_s)
        else:
            self._start_homing()

    def _start_homing(self) -> None:
        c = self.config
        self._phase = "homing"
        self._home_from = self._q_cmd.copy()
        self._qd = np.zeros_like(self._q_cmd)
        self._q_goal = None
        dist_deg = math.degrees(float(np.max(np.abs(self._q_home - self._home_from)))) if self._q_home.size else 0.0
        self._home_T = max(c.home_min_time_s, dist_deg / max(c.home_speed_deg_s, 1e-3))
        self._home_t0 = time.perf_counter()
        self._at_edge = False
        if self._input is not None:
            self._input.reset()
        logger.info("hand_ee: homing, every arm joint -> its home angle over %.1f s.", self._home_T)

    # ── action ───────────────────────────────────────────────────────────────

    def get_action(self) -> dict[str, float]:
        if self._q_cmd is None:
            raise RuntimeError("hand_ee: call seed_from_observation(obs) before get_action()")
        if self._input is None:
            raise RuntimeError("hand_ee: not connected")
        if self._input.error is not None:
            raise RuntimeError(f"hand_ee: webcam thread stopped: {self._input.error}") from self._input.error
        c = self.config
        now = time.perf_counter()
        dt = min(0.1, now - (self._last_t or now))  # cap so a stall cannot produce a jump
        self._last_t = now

        if self._input.pop_request("recalibrate"):
            logger.info("hand_ee: re-zero requested.")
            self._start_homing()

        if self._phase == "homing":
            s = min(1.0, (now - self._home_t0) / self._home_T)
            self._q_cmd = self._home_from + smoothstep(s) * (self._q_home - self._home_from)
            if s >= 1.0:
                self._phase = "calibrating"
                self._q_ik = self._q_home.copy()
                self._qd = np.zeros_like(self._q_cmd)
                self._input.begin_calibration()
                logger.info(
                    "hand_ee: arm at home. Show your hand to the webcam where you want zero and hold it still for %.1f s.",
                    c.calib_hold_s,
                )
        elif self._phase == "calibrating" and self._input.calibrated:
            self._phase = "tracking"
            self._at_edge = False

        if self._phase == "tracking":
            cmd = self._input.latest(c.hand_timeout_s)
            if cmd is not None:
                self._q_goal = self._goal(cmd.v)
                if cmd.gripper is not None:
                    self._gripper = cmd.gripper
            if self._q_goal is not None:  # no fresh hand: settle on the last target
                self._q_cmd = self._follow(self._q_cmd, self._q_goal, dt)

        self._publish_status(now)
        return self._action()

    def send_feedback(self, feedback: dict[str, float]) -> None:
        raise NotImplementedError

    # ── internals ────────────────────────────────────────────────────────────

    def _action(self) -> dict[str, float]:
        out = self.kin.q_to_motor(self._q_cmd)
        if self.config.gripper_key and self._gripper is not None:
            out[self.config.gripper_key] = float(self._gripper)
        return out

    def _hand_to_robot(self, dp_cam: np.ndarray, R_cam: np.ndarray) -> np.ndarray:
        """Hand offset from the reference (webcam axes) -> (x, y, z, roll, pitch, yaw) for the robot.
        Called on the webcam thread; uses only values fixed in __init__."""
        M = self._cam_to_robot
        dp = M @ dp_cam
        rpy = matrix_to_rpy(self._B.T @ (M @ R_cam @ M.T) @ self._B)
        return np.concatenate([dp, rpy]) * self._gain

    def _goal(self, v: np.ndarray) -> np.ndarray:
        """Joint target: IK for the position (gripper keeps its home tilt + roll), then the hand's
        pitch / roll / yaw on their own joints."""
        c = self.config
        p_t = self._p_home + v[:3]
        if c.z_min_m is not None:
            p_t[2] = max(p_t[2], c.z_min_m)
        q = self._solve_position(p_t).copy()
        s_roll, s_pitch, s_yaw = self._joint_sign
        q[self._i_roll] += s_roll * v[3]
        q[self._i_pitch] += s_pitch * v[4]
        q[self._i_yaw] += s_yaw * v[5]
        return np.clip(q, self._lo, self._hi)

    def _solve_position(self, p_t: np.ndarray) -> np.ndarray:
        c = self.config
        q_seed = self._q_ik
        res = self._ik(p_t, q_seed)
        if res.pos_err <= c.max_pos_error_m:
            self._q_ik = res.q
            self._at_edge = False
            return self._q_ik
        # Out of reach: go as far as the arm can along the line from the last reached point.
        p_prev = self.kin.fk(q_seed)[:3, 3]
        lo, hi, best = 0.0, 1.0, None
        for _ in range(c.reach_search_steps):
            mid = 0.5 * (lo + hi)
            r = self._ik(p_prev + mid * (p_t - p_prev), q_seed)
            if r.pos_err <= c.max_pos_error_m:
                lo, best = mid, r.q
            else:
                hi = mid
        if best is not None:
            self._q_ik = best
        self._at_edge = True
        self._warn("reach", f"hand target out of reach ({res.pos_err * 1e3:.0f} mm short): the arm follows as far as it goes.")
        return self._q_ik

    def _ik(self, p: np.ndarray, q_seed: np.ndarray) -> IKResult:
        c = self.config
        T = np.eye(4)
        T[:3, :3] = self._hold_rotation(p)
        T[:3, 3] = p
        return self.kin.solve(
            T, q_seed, orientation_weight=c.orientation_weight, max_iters=c.ik_iters,
            orientation_axes=(self._i_pitch, self._i_roll),
        )

    def _horizontal(self, v: np.ndarray) -> np.ndarray | None:
        h = v - float(np.dot(v, self._up)) * self._up
        n = float(np.linalg.norm(h))
        return h / n if n > 0.03 else None

    def _hold_rotation(self, p: np.ndarray) -> np.ndarray:
        """Home orientation turned with the arm toward ``p`` (the pan swing a position move needs)."""
        d = self._horizontal(p - self._pan_origin)
        if d is None or self._home_dir is None:
            return self._R_home
        phi = math.atan2(float(np.dot(np.cross(self._home_dir, d), self._up)), float(np.dot(self._home_dir, d)))
        return rotvec_to_matrix(self._up * phi) @ self._R_home

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

    def _publish_status(self, now: float) -> None:
        p = self.kin.fk(self._q_cmd)[:3, 3]
        motors = self.kin.q_to_motor(self._q_cmd)
        joints = "  ".join(f"{JOINT_LABELS.get(m, m)} {motors[k]:+.0f}" for m, k in zip(self.kin.motors, self.kin.keys))
        lines = [f"tool x {p[0] * 100:5.1f}  y {p[1] * 100:+5.1f}  z {p[2] * 100:5.1f} cm", f"deg: {joints}"]
        if self._phase == "tracking" and self._at_edge:
            lines.append("EDGE OF REACH: the arm stops short of your hand")
        home = min(1.0, (now - self._home_t0) / self._home_T) if self._phase == "homing" else 1.0
        self._input.set_status({"phase": self._phase, "home": home, "lines": lines})

    def _warn(self, key: str, msg: str, period_s: float = 2.0) -> None:
        now = time.monotonic()
        if now - self._warned_at.get(key, -math.inf) >= period_s:
            self._warned_at[key] = now
            logger.warning(msg)


# ── preview: webcam only, no robot ─────────────────────────────────────────────


def main() -> None:
    from .devices import load_config, make_parser, make_teleop

    parser = make_parser(
        "Preview of the hand teleop with NO robot: the arm is simulated (it follows the targets with\n"
        "the same speed / acceleration limits). Use it to try the start-up, the mapping, the jitter\n"
        "numbers and the pinch values; drive the real arm with src.teleop --teleop.type=hand_ee.\n\n"
        + (__doc__ or "")
    )
    args, overrides = parser.parse_known_args()
    cfg = load_config(args.preset or "so101_hand", overrides)
    if cfg.teleop_type != "hand_ee":
        raise SystemExit(
            f"preset {cfg.name!r} uses teleop {cfg.teleop_type!r}; the preview needs --teleop.type=hand_ee"
        )

    from lerobot.utils.robot_utils import precise_sleep
    from lerobot.utils.utils import init_logging

    init_logging()
    teleop = make_teleop(cfg)
    sim = {k: 0.0 for k in teleop.kin.keys}  # simulated arm, at zero
    if teleop.config.gripper_key:
        sim[teleop.config.gripper_key] = teleop.config.gripper_min
    teleop.connect()
    logging.info("PREVIEW, no robot.\n%s", teleop.help_text)
    period = 1.0 / cfg.fps
    try:
        teleop.seed_from_observation(sim)
        while not teleop.quit_requested:
            t0 = time.perf_counter()
            action = teleop.get_action()
            sim.update(action)
            joints = "  ".join(f"{JOINT_LABELS.get(k[:-4], k[:-4])} {v:+6.1f}" for k, v in action.items())
            print(f"[{teleop.phase:<11}] {joints}   ", end="\r")
            precise_sleep(max(period - (time.perf_counter() - t0), 0.0))
    except KeyboardInterrupt:
        pass
    finally:
        print()
        teleop.disconnect()


if __name__ == "__main__":
    main()

"""Native seven-joint Nero interface, independent of Koch leader retargeting.

LeRobot-style observation/action values are degrees; pyAgxArm receives radians.
Connecting does not send joint/gripper targets. Read-only clients do not enable.
"""
from dataclasses import dataclass
import json
import logging
import math
import platform
import time
from pathlib import Path
import numpy as np

logger = logging.getLogger(__name__)

from ._bootstrap import ARM_ROOT
from .kinematics import SerialChain

NERO_JOINTS = [f'joint{i}' for i in range(1, 8)]
NERO_URDF = ARM_ROOT / 'urdf' / 'nero' / 'nero_with_gripper.urdf'


class NeroTrackingError(ValueError):
    """A valid target is too far from feedback; no motion command was sent."""
    def __init__(self, measured_action, target_deg, limit):
        self.measured_action = dict(measured_action)
        measured = np.array([measured_action[f'{n}.pos'] for n in NERO_JOINTS])
        errors = np.abs(target_deg-measured)
        i = int(np.argmax(errors))
        super().__init__(f'Nero tracking limit: {NERO_JOINTS[i]} target {target_deg[i]:.2f} deg, '
            f'measured {measured[i]:.2f} deg, difference {errors[i]:.2f} deg > {limit:.2f} deg')


@dataclass
class NeroRobotConfig:
    id: str = 'revobots_nero'
    interface: str | None = None
    channel: str | None = None
    firmware_version: str = 'v111'
    bitrate: int = 1_000_000
    speed_percent: int = 20
    enable_on_connect: bool = True
    disable_on_disconnect: bool = False
    connect_timeout_s: float = 5.0
    feedback_timeout_s: float = 0.5
    max_relative_target: float | None = 5.0
    with_gripper: bool = False
    gripper_mode: str = 'width'  # width (m), angle (deg), auto (0..100% of reported mode)
    gripper_force: float = 0.5
    rest_on_start: bool = False
    rest_deg: list[float] | None = None  # seven joint degrees; file wins if present
    rest_pose_file: str | None = None
    rest_step_deg: float = 4.0
    rest_settle_deg: float = 1.0
    rest_timeout_s: float = 45.0


class NeroJointRobot:
    name = 'agx_nero_native'
    robot_type = name

    def __init__(self, config: NeroRobotConfig):
        if config.gripper_mode not in ('width', 'angle', 'auto'):
            raise ValueError('gripper_mode must be width, angle or auto')
        if not 1 <= config.speed_percent <= 100:
            raise ValueError('speed_percent must be in [1, 100]')
        if any(not math.isfinite(v) or v <= 0 for v in (config.connect_timeout_s, config.feedback_timeout_s)):
            raise ValueError('Connection and feedback timeouts must be positive and finite')
        if config.max_relative_target is not None and (not math.isfinite(config.max_relative_target) or config.max_relative_target <= 0):
            raise ValueError('max_relative_target must be positive or None')
        self.config = config
        self.id = config.id
        self._arm = self._gripper = None
        self._gripper_feedback_mode = None
        self._connected = self._enabled = False
        self._last_feedback_stamp = None
        self._feedback_changed_at = None
        self._clip_warn_t = 0.0
        chain = SerialChain.from_urdf(NERO_URDF, 'link7', 'base_link')
        self._lower, self._upper = np.degrees(chain.lower), np.degrees(chain.upper)
        self.logs = {}

    @property
    def action_features(self):
        names = NERO_JOINTS + (['gripper'] if self.config.with_gripper else [])
        return {f'{name}.pos': float for name in names}

    @property
    def observation_features(self):
        return self.action_features

    @property
    def is_connected(self):
        return self._connected

    @property
    def is_calibrated(self):
        return True

    def calibrate(self):
        pass  # Manufacturer joint zeros; no leader calibration or retargeting.

    def connect(self, calibrate=True):
        if self.is_connected:
            return
        from pyAgxArm import AgxArmFactory, create_agx_arm_config
        defaults = {'Windows': ('agx_cando', '0'), 'Linux': ('socketcan', 'can0'), 'Darwin': ('slcan', '/dev/ttyACM0')}
        interface, channel = defaults.get(platform.system(), (None, None))
        c = self.config
        self._arm = AgxArmFactory.create_arm(create_agx_arm_config(
            robot='nero', firmeware_version=c.firmware_version,
            interface=c.interface or interface, channel=c.channel or channel, bitrate=c.bitrate))
        try:
            if c.with_gripper:
                self._gripper = self._arm.init_effector(self._arm.OPTIONS.EFFECTOR.AGX_GRIPPER)
            self._arm.connect()
            self._connected = True
            self._last_feedback_stamp = self._feedback_changed_at = None
            deadline = time.monotonic() + c.connect_timeout_s
            while True:
                try:
                    self.get_observation()
                    break
                except RuntimeError:
                    if time.monotonic() >= deadline:
                        raise TimeoutError('Nero did not provide joint/gripper feedback')
                    time.sleep(.01)
            if c.enable_on_connect:
                self.enable()
            if c.rest_on_start:
                self.goto_rest()
        except Exception:
            self.disconnect()
            raise

    def enable(self):
        if not self.is_connected:
            raise RuntimeError('Nero is not connected')
        deadline = time.monotonic() + self.config.connect_timeout_s
        while not self._arm.enable():
            if time.monotonic() >= deadline:
                raise TimeoutError('Nero joint enable was not confirmed')
            time.sleep(.01)
        self._enabled = True
        self.configure()

    def configure(self):
        if self._enabled:
            self._arm.set_joint_limits_enabled(True)
            self._arm.set_auto_set_motion_mode_enabled(True)
            self._arm.set_speed_percent(self.config.speed_percent)

    def _rest_target_deg(self) -> np.ndarray | None:
        c = self.config
        values = None
        path = (c.rest_pose_file or '').strip()
        if path:
            p = Path(path).expanduser()
            if p.is_file():
                data = json.loads(p.read_text(encoding='utf-8'))
                values = data.get('joints_deg', data.get('rest_deg'))
                logger.info('Nero rest pose from %s', p)
            else:
                logger.warning('Nero rest pose file missing: %s', p)
        if values is None and c.rest_deg is not None:
            values = c.rest_deg
        if values is None:
            return None
        q = np.array([float(v) for v in values], dtype=float)
        if q.shape != (7,) or not np.all(np.isfinite(q)):
            raise ValueError('rest pose must be seven finite joint degrees')
        return np.clip(q, self._lower, self._upper)

    def goto_rest(self) -> None:
        """Glide every joint to the saved rest pose. Does not flash firmware."""
        if not self._enabled:
            raise RuntimeError('Enable Nero before going to rest')
        target = self._rest_target_deg()
        if target is None:
            logger.warning('rest_on_start is set but no rest pose is configured')
            return
        c = self.config
        step = min(float(c.rest_step_deg), float(c.max_relative_target or c.rest_step_deg))
        step = max(0.2, step)
        settle = max(0.2, float(c.rest_settle_deg))
        timeout = max(1.0, float(c.rest_timeout_s))
        dt = 1.0 / 30.0
        t0 = time.monotonic()
        last_log = 0.0
        logger.info(
            'Nero rest: %s',
            {NERO_JOINTS[i]: round(float(target[i]), 3) for i in range(7)},
        )
        while time.monotonic() - t0 < timeout:
            obs = self.get_observation()
            measured = np.array([obs[f'{n}.pos'] for n in NERO_JOINTS], dtype=float)
            err = target - measured
            remaining = float(np.max(np.abs(err)))
            now = time.monotonic()
            if now - last_log >= 1.0:
                logger.info('Nero rest remaining %.2f deg', remaining)
                last_log = now
            if remaining <= settle:
                action = {f'{n}.pos': float(target[i]) for i, n in enumerate(NERO_JOINTS)}
                self.send_action(action)
                logger.info('Nero rest reached (err %.2f deg)', remaining)
                return
            cmd = measured + np.clip(err, -step, step)
            action = {f'{n}.pos': float(cmd[i]) for i, n in enumerate(NERO_JOINTS)}
            self.send_action(action)
            time.sleep(dt)
        raise TimeoutError(f'Nero rest pose not reached within {timeout:.0f}s')

    def get_observation(self):
        if not self.is_connected:
            raise RuntimeError('Nero is not connected')
        feedback = self._arm.get_joint_angles()
        if feedback is None:
            raise RuntimeError('Nero joint feedback is missing')
        q = np.asarray(feedback.msg, dtype=float)
        if q.shape != (7,) or not np.all(np.isfinite(q)):
            raise ValueError('Nero feedback must contain seven finite joint angles')
        # Detect a frozen SDK cache without assuming its timestamp clock/units.
        stamp = getattr(feedback, 'timestamp', None)
        now = time.monotonic()
        if stamp is not None:
            if self._feedback_changed_at is None or stamp != self._last_feedback_stamp:
                self._last_feedback_stamp, self._feedback_changed_at = stamp, now
            elif now - self._feedback_changed_at > self.config.feedback_timeout_s:
                raise TimeoutError('Nero joint feedback stopped updating')
        obs = {f'{name}.pos': float(value) for name, value in zip(NERO_JOINTS, np.degrees(q))}
        if self._gripper is not None:
            status = self._gripper.get_gripper_status()
            if status is None or not math.isfinite(float(status.msg.value)):
                raise RuntimeError('AGX gripper feedback is missing or invalid')
            mode = status.msg.mode
            if mode not in ('width', 'angle'):
                raise ValueError(f'Unknown gripper feedback mode: {mode}')
            if self.config.gripper_mode != 'auto' and mode != self.config.gripper_mode:
                raise ValueError(f'Gripper feedback mode is {status.msg.mode}; configure gripper_mode to match')
            self._gripper_feedback_mode = mode
            value = float(status.msg.value)
            # Preserve the reported mode. Auto means percentage of its range,
            # not a conversion between mechanical width and motor angle.
            obs['gripper.pos'] = value * 1000. if self.config.gripper_mode == 'auto' and mode == 'width' else value
        return obs

    def send_action(self, action):
        if not self._enabled:
            raise RuntimeError('Enable Nero before sending joint targets')
        unknown = set(action) - set(self.action_features)
        if unknown:
            raise ValueError(f'Unknown Nero action keys: {sorted(unknown)}')
        obs = self.get_observation()
        # Missing joints hold their measured angle; order is never dictionary order.
        q = np.array([float(action.get(f'{n}.pos', obs[f'{n}.pos'])) for n in NERO_JOINTS])
        if not np.all(np.isfinite(q)):
            raise ValueError('Nero targets must be finite')
        # After a power cycle the measured pose can sit a few degrees past the
        # URDF box (J4 especially). Clipping keeps teleop alive instead of
        # exiting; it does not flash firmware.
        clipped = np.clip(q, self._lower, self._upper)
        if np.any(clipped != q):
            now = time.monotonic()
            if now - self._clip_warn_t >= 5.0:
                logger.warning(
                    'Nero target clipped to URDF limits: %s',
                    {NERO_JOINTS[i]: (float(q[i]), float(clipped[i]))
                     for i in range(len(NERO_JOINTS)) if clipped[i] != q[i]},
                )
                self._clip_warn_t = now
            q = clipped
        c = self.config
        if c.max_relative_target is not None:
            measured = np.array([obs[f'{n}.pos'] for n in NERO_JOINTS])
            if np.any(np.abs(q-measured) > c.max_relative_target):
                raise NeroTrackingError(obs, q, c.max_relative_target)
        gripper = action.get('gripper.pos')
        maximum = .1 if c.gripper_mode == 'width' else 100.
        if gripper is not None and (not math.isfinite(float(gripper)) or not 0 <= float(gripper) <= maximum):
            raise ValueError(f'AGX gripper target must be finite and in [0, {maximum}] ({c.gripper_mode})')
        self._arm.move_j(np.radians(q).tolist())
        if gripper is not None:
            mode = self._gripper_feedback_mode if c.gripper_mode == 'auto' else c.gripper_mode
            value = float(gripper) / 1000. if c.gripper_mode == 'auto' and mode == 'width' else float(gripper)
            move = self._gripper.move_gripper_m if mode == 'width' else self._gripper.move_gripper_deg
            move(value=value, force=c.gripper_force)
        return {**dict(zip([f'{n}.pos' for n in NERO_JOINTS], q.tolist())),
                **({'gripper.pos': float(gripper) if gripper is not None else obs['gripper.pos']}
                   if self._gripper is not None else {})}

    def disconnect(self):
        try:
            if self._arm is not None:
                if self._enabled and self.config.disable_on_disconnect:
                    self._arm.disable()
                self._arm.disconnect()
        finally:
            self._connected = self._enabled = False
            self._arm = self._gripper = None

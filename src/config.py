"""All settings for the Revobots arm, as plain Python. No JSON.

Same idea as the AGV repo's ``LAB/config.py``: dataclasses with sensible defaults,
edited in place. Three presets are defined at the bottom; pick one with
``--preset nero|koch|so101|so101_remote`` on any script, or ``REVOBOTS_ARM_PRESET``.
The default is ``DEFAULT_PRESET``.

``robot`` and ``teleop`` are dicts because they map 1:1 onto LeRobot's own config
dataclasses (``KochFollowerConfig``, ``AgxNeroFollowerConfig``, ``KochLeaderRemoteConfig``
...): ``type`` is the registered name, every other key is a field of that dataclass.
Cameras are NOT part of the robot dict: they are owned by ``cameras.py`` (frame bus,
hardware decode) and merged into the observation by ``RobotWithCameras``.
"""

from __future__ import annotations

import os
import platform
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

IS_LINUX = platform.system() == "Linux"
IS_JETSON = IS_LINUX and Path("/etc/nv_tegra_release").exists()


def video(n: int) -> int | str:
    """Camera source for index ``n``: ``/dev/videoN`` on Linux, the bare index elsewhere."""
    return f"/dev/video{n}" if IS_LINUX else n


def serial_port(windows: str, linux: str) -> str:
    return linux if IS_LINUX else windows


# ── cameras ──────────────────────────────────────────────────────────────────


@dataclass
class CameraConfig:
    """One USB camera. ``name`` becomes ``observation.images.<name>`` and the frame
    bus region ``lab_<name>``."""

    name: str
    source: int | str
    width: int = 640
    height: int = 480
    fps: int = 30
    # Jetson: v4l2src → nvv4l2decoder/nvvidconv GStreamer pipeline through OpenCV.
    # If that pipeline fails to open, capture falls back to plain OpenCV V4L2.
    hw_decode: bool = IS_JETSON
    pixel_format: str = "MJPG"          # MJPG | YUYV (used by both paths)
    # Publish every frame to shared memory (/dev/shm/lab_<name>) so inference and
    # any other process can read frames while teleop owns the device.
    publish_frames: bool = True
    rotation: int = 0                   # 0 | 90 | 180 | 270, applied after capture
    enabled: bool = True
    open_timeout_sec: float = 12.0

    @property
    def frame_shape(self) -> tuple[int, int, int]:
        """(height, width, 3) of the frames this camera delivers, after rotation."""
        if self.rotation in (90, 270):
            return (self.width, self.height, 3)
        return (self.height, self.width, 3)


def default_cameras() -> list[CameraConfig]:
    """The two-camera rig. Not used by any preset by default: cameras are OFF unless given
    on the command line (``--cameras.wrist.source=0 --cameras.eye.source=2``) or a preset
    sets ``cameras=default_cameras()``."""
    return [
        CameraConfig(name="wrist", source=video(0)),
        CameraConfig(name="eye", source=video(2)),
    ]


# ── sub-configs ──────────────────────────────────────────────────────────────


@dataclass
class LeaderClientConfig:
    """Defaults for ``leader_client.py`` on the PC the leader arm is plugged into.

    Which client runs (Koch or SO-101) follows the preset's ``teleop["type"]``:
    ``koch_leader_remote`` → Koch client, ``so101_leader_remote`` → SO client.
    """

    robot_ip: str = "192.168.1.42"
    port: int = 5001                    # must match teleop["port"] on the robot PC
    com: str = serial_port("COM3", "/dev/ttyACM1")
    id: str = "aadi"                    # calibration file key of the leader arm
    fps: int = 60
    gripper_open_pos: float = 45.0      # Koch leader only: gripper spring-back position
    use_degrees: bool = True            # SO leader only: must match the follower's use_degrees


@dataclass
class RecordConfig:
    # CLI: --display_data=true --display_mode=rerun --display_ip=... --display_port=... --play_sounds=false
    display_data: bool = False
    display_mode: str = "rerun"         # rerun | foxglove
    display_ip: str | None = None       # rerun: remote viewer ip; foxglove: bind interface
    display_port: int | None = None
    display_compressed_images: bool = False
    play_sounds: bool = False           # spoken prompts off; --play_sounds=true to enable
    codec: str = "mp4v"                 # OpenCV FOURCC for the raw session mp4s
    redo_starts_recording: bool = True  # teleop: after Left/r start the next take at once
    frame_stale_s: float = 0.5          # warn when a camera frame is older than this
    # record.py (timed, lerobot-record style). CLI: --dataset.num_episodes / episode_time_s / reset_time_s
    num_episodes: int = 50
    episode_time_s: float = 30.0
    reset_time_s: float = 10.0


@dataclass
class DatasetConfig:
    """Same field names as lerobot-record's --dataset.* where they overlap."""

    repo_id: str = "Aadi/revobots_arm_v1"
    root: str | None = None             # None → $HF_LEROBOT_HOME/<repo_id>
    push_to_hub: bool = False
    private: bool | None = None
    tags: list[str] = field(default_factory=lambda: ["revobots", "arm"])
    video: bool = True                  # convert.py: encode videos (False → PNG images)
    num_image_writer_threads_per_camera: int = 4
    video_encoding_batch_size: int = 1
    episode: int = 0                    # replay.py: --dataset.episode


@dataclass
class ReplayConfig:
    ramp_s: float = 1.5                 # glide from the current pose to the first action


@dataclass
class PolicyConfig:
    path: str = "outputs/train/act_arm/checkpoints/last/pretrained_model"
    dataset_repo_id: str | None = None  # None → dataset.repo_id
    dataset_root: str | None = None
    device: str = "cuda"
    duration_s: float = 60.0
    fps: int | None = None              # None → ArmConfig.fps


@dataclass
class AiConfig:
    """Teleop ↔ inference link. Teleop owns the arm and the cameras; inference is a
    separate process that reads frames from the frame bus and joint state from
    ``state_port``, and sends actions to ``action_port``. Teleop only applies them
    while AI control is on (``a`` key)."""

    # OFF by default: no UDP sockets are opened and the `a` key does nothing.
    # Turn on with --ai.enabled=true (required on the teleop side before running inference.py).
    enabled: bool = False
    host: str = "127.0.0.1"
    action_port: int = 5010             # inference → teleop   {"seq","t","action":{...}}
    state_port: int = 5011              # teleop → inference   {"seq","t","observation":{...},...}
    action_ttl_s: float = 0.5           # older AI actions are ignored (arm holds)
    bus_timeout_s: float = 5.0          # inference waits this long for the first frame/state
    # While AI drives, keep reading the leader so a human can take over by moving it.
    # With koch_leader_remote this blocks when the leader client is not streaming.
    read_leader_while_ai: bool = True
    # Leader movement (max joint delta from its pose when AI was enabled) that hands
    # control back to the human. Units follow the leader: Koch ±100 range, SO-101 degrees.
    human_takeover_threshold: float = 8.0


# ── top level ────────────────────────────────────────────────────────────────


@dataclass
class ArmConfig:
    name: str
    robot: dict[str, Any]
    teleop: dict[str, Any]
    # No cameras unless given: --cameras.wrist.source=0 --cameras.eye.source=2
    cameras: list[CameraConfig] = field(default_factory=list)
    fps: int = 30
    # Empty unless given: --dataset.single_task="Pick the cube"
    task: str = ""
    sessions_root: Path = Path("~/.cache/revobots/arm/sessions").expanduser()
    leader_client: LeaderClientConfig = field(default_factory=LeaderClientConfig)
    record: RecordConfig = field(default_factory=RecordConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    replay: ReplayConfig = field(default_factory=ReplayConfig)
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    ai: AiConfig = field(default_factory=AiConfig)
    # Path to <lerobot repo>/src when LeRobot is neither installed nor next to this folder.
    lerobot_src: str | None = None

    @property
    def robot_type(self) -> str:
        return str(self.robot["type"])

    @property
    def teleop_type(self) -> str:
        return str(self.teleop["type"])

    @property
    def enabled_cameras(self) -> list[CameraConfig]:
        return [c for c in self.cameras if c.enabled]

    @property
    def camera_names(self) -> list[str]:
        return [c.name for c in self.enabled_cameras]

    def camera(self, name: str) -> CameraConfig:
        for c in self.cameras:
            if c.name == name:
                return c
        raise KeyError(f"no camera named {name!r}; have {[c.name for c in self.cameras]}")


# ── presets ──────────────────────────────────────────────────────────────────

KOCH_LEADER_JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_roll", "wrist_flex", "gripper"]


def koch_leader_remote(port: int = 5001) -> dict[str, Any]:
    return dict(type="koch_leader_remote", host="0.0.0.0", port=port, max_action_age_s=0.5, waiting_log_period_s=2.0)


def so101_leader_remote(port: int = 5002) -> dict[str, Any]:
    return dict(type="so101_leader_remote", host="0.0.0.0", port=port, max_action_age_s=0.5, waiting_log_period_s=2.0)


def so101_follower_robot() -> dict[str, Any]:
    return dict(
        type="so101_follower",
        port=serial_port("COM4", "/dev/ttyACM0"),
        id="aadi_so101_follower",          # calibration file key (lerobot-calibrate --robot.id)
        use_degrees=True,
        max_relative_target=None,
        disable_torque_on_disconnect=True,
    )


def gamepad_ee_teleop(**fields: Any) -> dict[str, Any]:
    """Gamepad Cartesian jog through URDF IK (see gamepad_teleop.py). Defaults = SO-101.
    Any field of GamepadEETeleopConfig can be passed, e.g. max_linear_speed=0.05, invert="pitch"."""
    return dict(type="gamepad_ee", **fields)


def hand_ee_teleop(**fields: Any) -> dict[str, Any]:
    """Webcam hand (MediaPipe, thumb + index) Cartesian teleop through URDF IK (see hand_teleop.py).
    Defaults = SO-101. Any field of HandEETeleopConfig can be passed, e.g. camera_index=1, invert="y"."""
    return dict(type="hand_ee", **fields)


def vr_ee_teleop(**fields: Any) -> dict[str, Any]:
    """Meta Quest controller (Quest browser page served by teleop, WebXR) Cartesian teleop through
    URDF IK (see vr_teleop.py). Defaults = SO-101. Any field of VREETeleopConfig can be passed,
    e.g. port=8443, hand="left", position_scale=0.7, yaw_offset_deg=180."""
    return dict(type="vr_ee", **fields)


def keyboard_joints_teleop(speed: float = 30.0, gripper_speed: float = 60.0) -> dict[str, Any]:
    """Hold-a-key joint jog, no leader arm and no IK (see keyboard_teleop.py for the bindings).
    ``speed`` is in the follower's joint units per second: degrees for SO-101/Nero, % for Koch."""
    return dict(type="keyboard_joints", speed=speed, gripper_speed=gripper_speed)


def nero() -> ArmConfig:
    """AgileX Nero follower over CAN, driven by a Koch leader on another PC (default)."""
    return ArmConfig(
        name="nero",
        robot=dict(
            type="agx_nero_follower",
            id="revobots_nero",
            interface=None,                 # None → auto: agx_cando/0 (Windows), socketcan/can0 (Linux)
            channel=None,
            firmware_version="v111",        # "default" (<=1.10) | "v111" (>=1.11)
            bitrate=1_000_000,
            speed_percent=20,
            max_relative_target=None,       # degrees per step, None = unlimited
            disable_on_disconnect=True,
            teleop_action_joints=list(KOCH_LEADER_JOINTS),
        ),
        teleop=koch_leader_remote(),
        dataset=DatasetConfig(repo_id="Aadi/revobots_nero_v1", tags=["revobots", "nero", "agilex", "arm"]),
        policy=PolicyConfig(path="outputs/train/act_nero/checkpoints/last/pretrained_model"),
        replay=ReplayConfig(ramp_s=2.0),
    )


def nero_cartesian() -> ArmConfig:
    """Native seven-joint interface for host IK; no Koch leader retargeting."""
    from .nero_robot import NERO_JOINTS, NERO_URDF
    cfg = nero()
    cfg.name = 'nero_cartesian'
    cfg.robot = dict(type='agx_nero_native', id='revobots_nero', interface=None, channel=None,
        firmware_version='v111', speed_percent=20, max_relative_target=5.,
        disable_on_disconnect=False, with_gripper=False)
    cfg.teleop = dict(type='gamepad_ee', urdf_path=str(NERO_URDF), tip_link='link7',
        base_link='base_link', motors=list(NERO_JOINTS), orientation_mode='frame',
        orientation_weight=1., max_pos_error_m=.001, max_rot_error_deg=1.5,
        max_joint_step_deg=1., max_linear_speed=.04, max_angular_speed=.5,
        require_deadman=True, gripper_key=None)
    # Native seven-joint datasets/policies have a different action schema.
    cfg.dataset.repo_id = 'Aadi/revobots_nero_cartesian_v1'
    cfg.policy.path = ''
    return cfg


def nero_gamepad() -> ArmConfig:
    cfg = nero_cartesian()
    cfg.name = 'nero_gamepad'
    # Same button behavior as SO101: sticks are live, A opens, B closes.
    # Auto gripper mode exposes 0..100% in whichever SDK mode is reported.
    cfg.robot.update(with_gripper=True, gripper_mode='auto')
    cfg.teleop.update(require_deadman=False, gripper_needs_deadman=False,
        gripper_key='gripper.pos', gripper_min=0., gripper_max=100., gripper_speed=60.)
    return cfg


def nero_vr() -> ArmConfig:
    from .nero_robot import NERO_JOINTS, NERO_URDF
    cfg = nero_cartesian()
    cfg.name = 'nero_vr'
    cfg.teleop = dict(type='vr_ee', urdf_path=str(NERO_URDF), tip_link='link7',
        base_link='base_link', motors=list(NERO_JOINTS), orientation_mode='frame',
        pitch_joint='joint6', roll_joint='joint7', yaw_joint='joint1',
        orientation_weight=1., max_pos_error_m=.001, max_rot_error_deg=1.5,
        max_joint_speed_deg_s=30., max_joint_accel_deg_s2=120.,
        home_on_start=False, gripper_key=None, limit_margin_deg=1.)
    return cfg


def koch() -> ArmConfig:
    """Koch follower (Dynamixel, serial) driven by a Koch leader on another PC."""
    return ArmConfig(
        name="koch",
        robot=dict(
            type="koch_follower",
            port=serial_port("COM4", "/dev/ttyACM0"),
            id="revobots_koch_follower",
            use_degrees=False,              # False → body joints in -100..100, gripper 0..100
            max_relative_target=None,
            disable_torque_on_disconnect=True,
        ),
        teleop=koch_leader_remote(),
        dataset=DatasetConfig(repo_id="Aadi/revobots_koch_v1", tags=["revobots", "koch", "arm"]),
        policy=PolicyConfig(path="outputs/train/act_koch/checkpoints/last/pretrained_model"),
    )


def so101() -> ArmConfig:
    """SO-101 follower + SO-101 leader, both plugged into the robot PC."""
    return ArmConfig(
        name="so101",
        robot=so101_follower_robot(),
        teleop=dict(
            type="so101_leader",
            port=serial_port("COM5", "/dev/ttyACM1"),
            id="aadi_so101_leader",         # calibration file key (lerobot-calibrate --teleop.id)
            use_degrees=True,
        ),
        dataset=DatasetConfig(repo_id="Aadi/revobots_so101_v1", tags=["revobots", "so101", "arm"]),
        policy=PolicyConfig(path="outputs/train/act_so101/checkpoints/last/pretrained_model"),
    )


def so101_remote() -> ArmConfig:
    """SO-101 follower on the robot PC, SO-101 leader on another PC (so_leader_remote, UDP 5002)."""
    return ArmConfig(
        name="so101_remote",
        robot=so101_follower_robot(),
        teleop=so101_leader_remote(),
        leader_client=LeaderClientConfig(port=5002, com=serial_port("COM5", "/dev/ttyACM1"), id="aadi_so101_leader", use_degrees=True),
        dataset=DatasetConfig(repo_id="Aadi/revobots_so101_v1", tags=["revobots", "so101", "arm"]),
        policy=PolicyConfig(path="outputs/train/act_so101/checkpoints/last/pretrained_model"),
    )


def so101_keyboard() -> ArmConfig:
    """SO-101 follower jogged from the keyboard (no leader arm, no IK)."""
    return ArmConfig(
        name="so101_keyboard",
        robot=so101_follower_robot(),
        teleop=keyboard_joints_teleop(speed=30.0, gripper_speed=60.0),
        dataset=DatasetConfig(repo_id="Aadi/revobots_so101_v1", tags=["revobots", "so101", "arm"]),
        policy=PolicyConfig(path="outputs/train/act_so101/checkpoints/last/pretrained_model"),
    )


def so101_gamepad() -> ArmConfig:
    """SO-101 follower driven in Cartesian space from a USB gamepad (URDF IK, no leader arm)."""
    return ArmConfig(
        name="so101_gamepad",
        robot=so101_follower_robot(),
        teleop=gamepad_ee_teleop(),
        dataset=DatasetConfig(repo_id="Aadi/revobots_so101_v1", tags=["revobots", "so101", "arm"]),
        policy=PolicyConfig(path="outputs/train/act_so101/checkpoints/last/pretrained_model"),
    )


def so101_hand() -> ArmConfig:
    """SO-101 follower driven by your hand in front of a webcam (MediaPipe thumb + index, URDF IK).
    Starts by homing every arm joint to 0 deg, then takes your hand pose as the zero reference."""
    return ArmConfig(
        name="so101_hand",
        robot=so101_follower_robot(),
        teleop=hand_ee_teleop(camera_index=0),
        dataset=DatasetConfig(repo_id="Aadi/revobots_so101_v1", tags=["revobots", "so101", "arm"]),
        policy=PolicyConfig(path="outputs/train/act_so101/checkpoints/last/pretrained_model"),
    )


def so101_vr() -> ArmConfig:
    """SO-101 follower driven by a Meta Quest controller (Quest browser, WebXR, URDF IK).
    Nothing moves until GRIP is held; every squeeze is a new reference at the arm's current pose.
    TRIGGER closes the gripper. The headset does not have to be worn once the page is started."""
    return ArmConfig(
        name="so101_vr",
        robot=so101_follower_robot(),
        teleop=vr_ee_teleop(),
        dataset=DatasetConfig(repo_id="Aadi/revobots_so101_v1", tags=["revobots", "so101", "arm"]),
        policy=PolicyConfig(path="outputs/train/act_so101/checkpoints/last/pretrained_model"),
    )


PRESETS = {
    "nero": nero,
    "nero_cartesian": nero_cartesian,
    "nero_gamepad": nero_gamepad,
    "nero_vr": nero_vr,
    "koch": koch,
    "so101": so101,
    "so101_remote": so101_remote,
    "so101_keyboard": so101_keyboard,
    "so101_gamepad": so101_gamepad,
    "so101_hand": so101_hand,
    "so101_vr": so101_vr,
}
DEFAULT_PRESET = "nero"
PRESET_ENV_VAR = "REVOBOTS_ARM_PRESET"


def load(preset: str | None = None) -> ArmConfig:
    name = preset or os.environ.get(PRESET_ENV_VAR) or DEFAULT_PRESET
    try:
        return PRESETS[name]()
    except KeyError:
        raise ValueError(f"unknown preset {name!r}; available: {sorted(PRESETS)}") from None

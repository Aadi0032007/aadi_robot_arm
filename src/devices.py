"""Glue between ``config.py`` and LeRobot: build the follower and the leader objects.

The LeRobot robot is always built WITHOUT cameras. Cameras are owned by
``cameras.py`` (frame bus, hardware decode) and merged into the observation by
``cameras.RobotWithCameras``, so the dataset features end up identical to what
``lerobot-record`` would produce with the cameras inside the robot.
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib
import json
import logging
from pathlib import Path
from typing import Any

from . import config as _config
from ._bootstrap import ensure_lerobot_importable
from .config import ArmConfig

logger = logging.getLogger(__name__)

# Importing these packages runs their ``@register_subclass`` decorators, which is what
# makes ``get_choice_class("koch_follower")`` & co. resolve. A missing optional
# dependency only disables that one device type.
_DEVICE_PACKAGES = (
    "lerobot.robots.koch_follower",
    "lerobot.robots.so_follower",
    "lerobot.robots.agx_nero_follower",
    "lerobot.teleoperators.koch_leader",
    "lerobot.teleoperators.koch_leader_remote",
    "lerobot.teleoperators.so_leader",
    "lerobot.teleoperators.so_leader_remote",
)


# ── config loading ────────────────────────────────────────────────────────────


OVERRIDES_HELP = """\
Configuration arguments use the same names as the lerobot CLI (lerobot-teleoperate / -record / -replay):
  --robot.type=so101_follower --robot.port=COM4 --robot.id=aadi_so101_follower
  --teleop.type=so101_leader --teleop.port=COM5 --teleop.id=aadi_so101_leader
  --teleop.type=keyboard_joints --teleop.speed=30
  --teleop.type=gamepad_ee --port=5001   UDP gamepad on 0.0.0.0; omit --port for local USB
  --debug=true   Print received UDP gamepad packets and active mapped inputs (default: false)
  --teleop.type=hand_ee --teleop.camera_index=0 --teleop.invert=y,yaw
  --teleop.type=vr_ee --teleop.port=8443 --teleop.hand=right --teleop.position_scale=1.0
  --fps=30 --display_data=true --display_mode=rerun --play_sounds=false
  --dataset.repo_id=Aadi/so101_v1 --dataset.single_task="Pick the cube" --dataset.root=...
  --dataset.num_episodes=10 --dataset.episode_time_s=30 --dataset.reset_time_s=10
  --dataset.episode=0 --dataset.push_to_hub=true --dataset.video=false
  --policy.path=<ckpt> --policy.device=cuda --dataset_repo_id=... --single_task=... --control_time_s=60
Extra sections of this folder, same style:
  --cameras.wrist.source=0 --cameras.eye.enabled=false   or   --cameras=none
  --sessions_root=<dir> --ai.action_port=5010 --record.codec=mp4v --replay.ramp_s=0
Anything not given comes from the preset (--preset, default 'nero'). Setting --robot.type or
--teleop.type discards the preset's other keys for that section. Values are parsed as
int/float/bool/none/JSON when they look like one, else kept as text."""

# Repo-style names → where they live in ArmConfig.
ALIASES = {
    "port": "teleop.udp_port",
    "debug": "teleop.debug",
    "dataset.single_task": "task",
    "single_task": "task",
    "dataset.fps": "fps",
    "dataset.num_episodes": "record.num_episodes",
    "dataset.episode_time_s": "record.episode_time_s",
    "dataset.reset_time_s": "record.reset_time_s",
    "display_data": "record.display_data",
    "display_mode": "record.display_mode",
    "display_ip": "record.display_ip",
    "display_port": "record.display_port",
    "display_compressed_images": "record.display_compressed_images",
    "play_sounds": "record.play_sounds",
    "dataset_repo_id": "policy.dataset_repo_id",
    "root": "policy.dataset_root",
    "control_time_s": "policy.duration_s",
}


def str2bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    low = str(value).strip().lower()
    if low in ("true", "1", "yes", "on", ""):
        return True
    if low in ("false", "0", "no", "off"):
        return False
    raise argparse.ArgumentTypeError(f"expected true/false, got {value!r}")


def make_parser(description: str | None) -> argparse.ArgumentParser:
    """Parser with --preset, lerobot-style epilog, and no prefix matching (so unknown
    --robot.* style args are never mistaken for abbreviations of known flags)."""
    parser = argparse.ArgumentParser(
        description=description,
        epilog=OVERRIDES_HELP,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        allow_abbrev=False,
    )
    add_preset_arg(parser)
    return parser


def add_bool_flag(parser: argparse.ArgumentParser, name: str, help: str, default: bool = False) -> None:
    """``--resume``, ``--resume=true`` and ``--resume false`` all work (lerobot style)."""
    parser.add_argument(name, nargs="?", const=True, default=default, type=str2bool, help=help)


def add_preset_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--preset",
        default=None,
        choices=sorted(_config.PRESETS),
        help=f"Config preset from src/config.py (default: ${_config.PRESET_ENV_VAR} or '{_config.DEFAULT_PRESET}')",
    )
    if not parser.epilog:
        parser.epilog = OVERRIDES_HELP
        parser.formatter_class = argparse.RawDescriptionHelpFormatter


def load_config(preset: str | None = None, overrides: list[str] | dict[str, str] | None = None) -> ArmConfig:
    """Pick a preset from ``config.py``, apply CLI overrides, make sure ``lerobot`` is importable.

    ``overrides`` is the unknown-args list from ``parser.parse_known_args()`` (or an
    already parsed ``{"robot.port": "COM4"}`` dict).
    """
    cfg = _config.load(preset)
    if overrides:
        parsed = parse_overrides(overrides) if isinstance(overrides, list) else dict(overrides)
        apply_overrides(cfg, parsed)
    ensure_lerobot_importable(cfg.lerobot_src)
    return cfg


# ── CLI overrides ─────────────────────────────────────────────────────────────

_SECTION_DATACLASSES = ("leader_client", "record", "dataset", "replay", "policy", "ai")
_SCALARS = ("fps", "task", "sessions_root", "lerobot_src")


def parse_overrides(tokens: list[str]) -> dict[str, str]:
    """``["--robot.port=COM4", "--fps", "30"]`` → ``{"robot.port": "COM4", "fps": "30"}``."""
    out: dict[str, str] = {}
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if not tok.startswith("--"):
            raise SystemExit(f"unexpected argument {tok!r}. Overrides look like --robot.port=COM4.\n\n{OVERRIDES_HELP}")
        body = tok[2:]
        if "=" in body:
            key, val = body.split("=", 1)
            i += 1
        else:
            key = body
            if i + 1 < len(tokens) and not tokens[i + 1].startswith("--"):
                val = tokens[i + 1]
                i += 2
            else:
                val = "true"
                i += 1
        out[key.strip()] = val
    return out


def coerce(raw: str) -> Any:
    s = raw.strip()
    low = s.lower()
    if low in ("none", "null"):
        return None
    if low in ("true", "yes", "on"):
        return True
    if low in ("false", "no", "off"):
        return False
    for cast in (int, float):
        try:
            return cast(s)
        except ValueError:
            pass
    if s[:1] in "[{":
        try:
            return json.loads(s)
        except json.JSONDecodeError:
            pass
    return s


def overridden(overrides: list[str] | dict[str, str] | None, *keys: str) -> bool:
    """Whether any of ``keys`` (after alias resolution) was given on the command line."""
    if not overrides:
        return False
    parsed = parse_overrides(overrides) if isinstance(overrides, list) else dict(overrides)
    given = {ALIASES.get(k, k) for k in parsed}
    return any(ALIASES.get(k, k) in given for k in keys)


def apply_overrides(cfg: ArmConfig, overrides: dict[str, str]) -> ArmConfig:
    # Type changes first: they reset their section, then the remaining keys fill it.
    ordered = sorted(overrides.items(), key=lambda kv: 0 if kv[0] in ("robot.type", "teleop.type") else 1)
    for raw_key, raw in ordered:
        key = ALIASES.get(raw_key, raw_key)
        parts = key.split(".")
        head = parts[0]
        val = coerce(raw)
        if key == 'teleop.debug':
            val = str2bool(raw)

        if head in ("robot", "teleop"):
            if len(parts) != 2:
                raise SystemExit(f"--{key}: expected --{head}.<field>=<value>")
            section = getattr(cfg, head)
            if parts[1] == "type":
                section.clear()
                section["type"] = str(val)
            else:
                section[parts[1]] = val

        elif head == "cameras":
            if len(parts) == 1:
                if val is None or val is False or str(val).lower() in ("none", "off", ""):
                    cfg.cameras = []
                else:
                    raise SystemExit("use --cameras=none, or --cameras.<name>.<field>=<value>")
            elif len(parts) == 3:
                name, field_name = parts[1], parts[2]
                cam = next((c for c in cfg.cameras if c.name == name), None)
                if cam is None:
                    cam = _config.CameraConfig(name=name, source=val if field_name == "source" else 0)
                    cfg.cameras.append(cam)
                if not hasattr(cam, field_name):
                    raise SystemExit(f"--{key}: CameraConfig has no field '{field_name}'")
                setattr(cam, field_name, val)
            else:
                raise SystemExit(f"--{key}: expected --cameras.<name>.<field>=<value> or --cameras=none")

        elif head in _SCALARS and len(parts) == 1:
            if head == "sessions_root":
                val = Path(str(val)).expanduser()
            setattr(cfg, head, val)

        elif head in _SECTION_DATACLASSES and len(parts) == 2:
            sub = getattr(cfg, head)
            if not hasattr(sub, parts[1]):
                raise SystemExit(f"--{key}: {type(sub).__name__} has no field '{parts[1]}'")
            setattr(sub, parts[1], val)

        else:
            raise SystemExit(f"unknown override --{key}\n\n{OVERRIDES_HELP}")
    if 'port' in overrides and cfg.teleop_type != 'gamepad_ee':
        raise SystemExit('--port selects UDP input for --teleop.type=gamepad_ee only')
    return cfg


# ── config dict → LeRobot config dataclass ───────────────────────────────────


def _register_device_configs() -> None:
    for name in _DEVICE_PACKAGES:
        try:
            importlib.import_module(name)
        except ImportError as e:
            logger.debug("Could not import %s: %s", name, e)


def _choice_class(registry: Any, type_name: str, what: str) -> type:
    try:
        return registry.get_choice_class(type_name)
    except KeyError:
        known = sorted(registry.get_known_choices())
        raise ValueError(f"Unknown {what} type '{type_name}'. Registered types: {known}") from None


def _instantiate(cls: type, spec: dict[str, Any], what: str) -> Any:
    accepted = {f.name for f in dataclasses.fields(cls)}
    unknown = sorted(set(spec) - accepted)
    if unknown:
        raise ValueError(
            f"{what}: '{cls.__name__}' does not accept {unknown}. Accepted keys: {sorted(accepted)}"
        )
    return cls(**spec)


def _normalise_common(spec: dict[str, Any]) -> None:
    if spec.get("calibration_dir"):
        spec["calibration_dir"] = Path(spec["calibration_dir"]).expanduser()
    else:
        spec.pop("calibration_dir", None)
    # `ensure_safe_goal_position` checks `isinstance(x, float)`, so an int would fail.
    mrt = spec.get("max_relative_target")
    if isinstance(mrt, int) and not isinstance(mrt, bool):
        spec["max_relative_target"] = float(mrt)
    elif isinstance(mrt, dict):
        spec["max_relative_target"] = {k: float(v) for k, v in mrt.items()}


def build_robot_config(cfg: ArmConfig) -> Any:
    if cfg.robot_type == 'agx_nero_native':
        from .nero_robot import NeroRobotConfig
        spec = dict(cfg.robot)
        spec.pop('type')
        return _instantiate(NeroRobotConfig, spec, 'robot')
    _register_device_configs()
    from lerobot.robots import RobotConfig

    spec = dict(cfg.robot)
    cls = _choice_class(RobotConfig, str(spec.pop("type")), "robot")
    _normalise_common(spec)
    if "cameras" in {f.name for f in dataclasses.fields(cls)}:
        spec["cameras"] = {}  # cameras live in cameras.py, never inside the LeRobot robot
    return _instantiate(cls, spec, "robot")


def build_teleop_config(cfg: ArmConfig) -> Any:
    _register_device_configs()
    from lerobot.teleoperators import TeleoperatorConfig

    spec = dict(cfg.teleop)
    cls = _choice_class(TeleoperatorConfig, str(spec.pop("type")), "teleoperator")
    _normalise_common(spec)
    return _instantiate(cls, spec, "teleop")


# ── factories ─────────────────────────────────────────────────────────────────


def make_robot(cfg: ArmConfig):
    """The bare LeRobot follower (motors only, no cameras)."""
    if cfg.robot_type == 'agx_nero_native':
        from .nero_robot import NeroJointRobot
        return NeroJointRobot(build_robot_config(cfg))
    from lerobot.robots import make_robot_from_config

    return make_robot_from_config(build_robot_config(cfg))


def make_teleop(cfg: ArmConfig):
    """LeRobot leader/teleoperator, or one of this folder's own input devices."""
    if cfg.teleop_type == "keyboard_joints":
        from .keyboard_teleop import KeyboardJointTeleop, KeyboardJointTeleopConfig

        spec = dict(cfg.teleop)
        spec.pop("type")
        return KeyboardJointTeleop(_instantiate(KeyboardJointTeleopConfig, spec, "teleop"))

    if cfg.teleop_type == "gamepad_ee":
        from .gamepad_teleop import GamepadEETeleop, GamepadEETeleopConfig

        spec = dict(cfg.teleop)
        spec.pop("type")
        return GamepadEETeleop(_instantiate(GamepadEETeleopConfig, spec, "teleop"))

    if cfg.teleop_type == "hand_ee":
        from .hand_teleop import HandEETeleop, HandEETeleopConfig

        spec = dict(cfg.teleop)
        spec.pop("type")
        return HandEETeleop(_instantiate(HandEETeleopConfig, spec, "teleop"))

    if cfg.teleop_type == "vr_ee":
        from .vr_teleop import VREETeleop, VREETeleopConfig

        spec = dict(cfg.teleop)
        spec.pop("type")
        return VREETeleop(_instantiate(VREETeleopConfig, spec, "teleop"))

    from lerobot.teleoperators import make_teleoperator_from_config

    return make_teleoperator_from_config(build_teleop_config(cfg))


def describe(cfg: ArmConfig) -> str:
    cams = ", ".join(
        f"{c.name}({c.width}x{c.height}@{c.fps}{', hw' if c.hw_decode else ''}{', bus' if c.publish_frames else ''})"
        for c in cfg.enabled_cameras
    ) or "none  (add --cameras.wrist.source=0 --cameras.eye.source=2)"
    ai = (
        f"on  action udp:{cfg.ai.action_port}  state udp:{cfg.ai.state_port}"
        if cfg.ai.enabled
        else "off (--ai.enabled=true to allow inference.py / the `a` key)"
    )
    return (
        f"preset={cfg.name}\n"
        f"  robot : {cfg.robot_type}  {_short(cfg.robot)}\n"
        f"  teleop: {cfg.teleop_type}  {_short(cfg.teleop)}\n"
        f"  cams  : {cams}\n"
        f"  fps   : {cfg.fps}\n"
        f"  ai    : {ai}\n"
        f"  task  : {cfg.task!r}" + ("  (--dataset.single_task=... to set)" if not cfg.task else "")
    )


def _short(section: dict[str, Any]) -> str:
    return " ".join(f"{k}={v}" for k, v in section.items() if k != "type" and v is not None)

"""Keyboard joint jog: a LeRobot ``Teleoperator`` that moves one joint per key, no IK.

Hold a key to move the joint at ``speed`` units per second; release to stop. The target
starts from the follower's measured pose (``seed_from_observation``), so nothing jumps,
and it is re-seeded whenever the arm was driven by something else (AI mode).

Default bindings (left key decreases, right key increases):

    shoulder_pan   j / l          wrist_flex   h / y
    shoulder_lift  k / i          wrist_roll   g / t
    elbow_flex     u / o          gripper      [ (close) / ] (open)
    +  /  -        speed x2 / speed /2

They avoid the keys ``teleop.py`` already uses (Space, Left, r, a, Esc, q). Keys are
global (pynput), like every other LeRobot keyboard device: the terminal does not need
focus, and typing in another window while this runs will move the arm.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_BINDINGS: dict[str, tuple[str, str]] = {
    "shoulder_pan": ("j", "l"),
    "shoulder_lift": ("k", "i"),
    "elbow_flex": ("u", "o"),
    "wrist_flex": ("h", "y"),
    "wrist_roll": ("g", "t"),
    "gripper": ("[", "]"),
}
SPEED_UP_KEY = "+"
SPEED_DOWN_KEY = "-"


def _lerobot_base():
    from lerobot.teleoperators import Teleoperator, TeleoperatorConfig

    return Teleoperator, TeleoperatorConfig


_Teleoperator, _TeleoperatorConfig = _lerobot_base()


@dataclass
class KeyboardJointTeleopConfig(_TeleoperatorConfig):
    # Joints to expose, in action order. Must be joints the follower accepts.
    joints: list[str] = field(default_factory=lambda: list(DEFAULT_BINDINGS))
    # joint -> (decrease key, increase key)
    bindings: dict[str, tuple[str, str]] = field(default_factory=lambda: dict(DEFAULT_BINDINGS))
    # Units per second while a key is held (degrees for SO-101/Nero, % of range for Koch).
    speed: float = 30.0
    gripper_speed: float = 60.0
    # Optional clamp per joint, e.g. {"gripper": (0.0, 100.0)}. None = no clamp (the motor
    # bus still enforces the calibrated range).
    limits: dict[str, tuple[float, float]] | None = None


class KeyboardJointTeleop(_Teleoperator):
    config_class = KeyboardJointTeleopConfig
    name = "keyboard_joints"

    def __init__(self, config: KeyboardJointTeleopConfig):
        super().__init__(config)
        self.config = config
        self._pressed: set[str] = set()
        self._lock = threading.Lock()
        self._listener = None
        self._target: dict[str, float] | None = None
        self._last_t: float | None = None
        self._speed_scale = 1.0
        self._speed_keys_seen: set[str] = set()
        self.needs_seed = True

    # ── features ─────────────────────────────────────────────────────────────

    @property
    def action_features(self) -> dict[str, type]:
        return {f"{j}.pos": float for j in self.config.joints}

    @property
    def feedback_features(self) -> dict[str, type]:
        return {}

    @property
    def help_text(self) -> str:
        rows = [f"{j}: {dec}/{inc}" for j, (dec, inc) in self.config.bindings.items() if j in self.config.joints]
        return "Keyboard jog  " + "  ".join(rows) + f"  speed {SPEED_UP_KEY}/{SPEED_DOWN_KEY}"

    # ── lifecycle ────────────────────────────────────────────────────────────

    @property
    def is_connected(self) -> bool:
        return self._listener is not None

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    def connect(self, calibrate: bool = True) -> None:
        if self._listener is not None:
            return
        try:
            from pynput import keyboard
        except ImportError as e:
            raise ImportError("keyboard_joints needs pynput (pip install pynput)") from e

        def name_of(key) -> str | None:
            char = getattr(key, "char", None)
            return char.lower() if isinstance(char, str) and char else None

        def on_press(key) -> None:
            name = name_of(key)
            if name is None:
                return
            with self._lock:
                self._pressed.add(name)

        def on_release(key) -> None:
            name = name_of(key)
            if name is None:
                return
            with self._lock:
                self._pressed.discard(name)
                self._speed_keys_seen.discard(name)

        self._listener = keyboard.Listener(on_press=on_press, on_release=on_release)
        self._listener.start()
        logger.info("%s connected. %s", self, self.help_text)

    def disconnect(self) -> None:
        if self._listener is not None:
            self._listener.stop()
            self._listener = None
        with self._lock:
            self._pressed.clear()

    # ── seeding ──────────────────────────────────────────────────────────────

    def seed_from_observation(self, observation: dict[str, Any]) -> None:
        """Start (or restart) the jog target at the follower's measured pose."""
        target = {}
        for joint in self.config.joints:
            key = f"{joint}.pos"
            if key in observation:
                target[key] = float(observation[key])
        missing = [j for j in self.config.joints if f"{j}.pos" not in target]
        if missing:
            raise KeyError(f"observation lacks joints {missing}; cannot seed the keyboard jog")
        self._target = target
        self._last_t = time.perf_counter()
        self.needs_seed = False

    # ── action ───────────────────────────────────────────────────────────────

    def get_action(self) -> dict[str, float]:
        if self._target is None:
            raise RuntimeError("keyboard_joints: call seed_from_observation(obs) before get_action()")

        now = time.perf_counter()
        dt = min(0.1, now - (self._last_t or now))  # cap so a stall cannot produce a jump
        self._last_t = now

        with self._lock:
            pressed = set(self._pressed)
            # speed keys act once per press, not per tick
            new_speed_keys = {k for k in (SPEED_UP_KEY, SPEED_DOWN_KEY) if k in pressed and k not in self._speed_keys_seen}
            self._speed_keys_seen |= new_speed_keys
        if SPEED_UP_KEY in new_speed_keys:
            self._speed_scale = min(8.0, self._speed_scale * 2)
            logger.info("jog speed x%.2f", self._speed_scale)
        if SPEED_DOWN_KEY in new_speed_keys:
            self._speed_scale = max(0.125, self._speed_scale / 2)
            logger.info("jog speed x%.2f", self._speed_scale)

        limits = self.config.limits or {}
        for joint in self.config.joints:
            dec, inc = self.config.bindings.get(joint, (None, None))
            direction = (1.0 if inc in pressed else 0.0) - (1.0 if dec in pressed else 0.0)
            if direction == 0.0:
                continue
            speed = self.config.gripper_speed if joint == "gripper" else self.config.speed
            key = f"{joint}.pos"
            value = self._target[key] + direction * speed * self._speed_scale * dt
            if joint in limits:
                lo, hi = limits[joint]
                value = min(hi, max(lo, value))
            self._target[key] = value

        return dict(self._target)

    def send_feedback(self, feedback: dict[str, float]) -> None:
        raise NotImplementedError

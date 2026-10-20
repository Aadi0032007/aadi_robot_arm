"""Gamepad reader: physical USB gamepad → normalised W3C "standard" Xbox layout.

Lifted from the AGV repo's ``LAB/pilot/pilot_bridge.py`` (gamepad half only; the
WebSocket/UDP bridge for the AGV is not here). Same mapping, so a controller that works
for the AGV reports identical button indices and axes for the arm.

    pip install pygame

    python -m src.gamepad --list_devices       # what pygame / SDL2 can see
    python -m src.gamepad                      # live print of pressed buttons and axes

Button indices (W3C standard, ``GamepadState.buttons[i]``):

    0-3   A, B, X, Y                 8-11  Back, Start, L3, R3
    4-5   LB, RB                     12-15 D-pad up, down, left, right
    6-7   LT, RT (analog "value")    16    Guide / Home

Axes (``GamepadState.axes``): 0 left X, 1 left Y, 2 right X, 3 right Y, each in [-1, 1]
(up/left negative, as the browser Gamepad API reports them).

Usage from a control loop::

    reader = GamepadReader()
    ...
    state = reader.read()                 # None while no gamepad is attached
    if state and state.pressed(BTN_A): ...
    lx = state.axis(AXIS_LEFT_X)
"""

from __future__ import annotations

if __package__ in (None, ""):  # allow `python src/gamepad.py`
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "src"

import argparse
import logging
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)

# ── W3C standard layout ────────────────────────────────────────────────────────

BTN_A, BTN_B, BTN_X, BTN_Y = 0, 1, 2, 3
BTN_LB, BTN_RB = 4, 5
BTN_LT, BTN_RT = 6, 7            # analog: use .value
BTN_BACK, BTN_START, BTN_L3, BTN_R3 = 8, 9, 10, 11
BTN_DPAD_UP, BTN_DPAD_DOWN, BTN_DPAD_LEFT, BTN_DPAD_RIGHT = 12, 13, 14, 15
BTN_GUIDE = 16

AXIS_LEFT_X, AXIS_LEFT_Y, AXIS_RIGHT_X, AXIS_RIGHT_Y = 0, 1, 2, 3

BUTTON_NAMES = {
    BTN_A: "A", BTN_B: "B", BTN_X: "X", BTN_Y: "Y",
    BTN_LB: "LB", BTN_RB: "RB", BTN_LT: "LT", BTN_RT: "RT",
    BTN_BACK: "Back", BTN_START: "Start", BTN_L3: "L3", BTN_R3: "R3",
    BTN_DPAD_UP: "Up", BTN_DPAD_DOWN: "Down", BTN_DPAD_LEFT: "Left", BTN_DPAD_RIGHT: "Right",
    BTN_GUIDE: "Guide",
}

NUM_BUTTONS = 17
NUM_AXES = 4
AXIS_EPS = 0.01
TRIGGER_EPS = 0.02

# SDL2 game-controller button indices → W3C indices.
SDL_BTN_TO_GP: dict[int, int] = {
    0: BTN_A,
    1: BTN_B,
    2: BTN_X,
    3: BTN_Y,
    4: BTN_BACK,
    5: BTN_GUIDE,
    6: BTN_START,
    7: BTN_L3,
    8: BTN_R3,
    9: BTN_LB,
    10: BTN_RB,
    11: BTN_DPAD_UP,
    12: BTN_DPAD_DOWN,
    13: BTN_DPAD_LEFT,
    14: BTN_DPAD_RIGHT,
}

# Generic pygame joystick (XInput-style) buttons 4..10 → W3C indices.
JOY_BTN_TO_GP: dict[int, int] = {4: BTN_LB, 5: BTN_RB, 6: BTN_BACK, 7: BTN_START, 8: BTN_L3, 9: BTN_R3, 10: BTN_GUIDE}


# ── state ──────────────────────────────────────────────────────────────────────


def empty_buttons() -> list[dict[str, float | bool]]:
    return [{"pressed": False, "value": 0.0} for _ in range(NUM_BUTTONS)]


def empty_axes() -> list[float]:
    return [0.0] * NUM_AXES


@dataclass
class GamepadState:
    buttons: list[dict[str, float | bool]] = field(default_factory=empty_buttons)
    axes: list[float] = field(default_factory=empty_axes)

    def pressed(self, index: int) -> bool:
        if index < 0 or index >= len(self.buttons):
            return False
        b = self.buttons[index]
        return bool(b["pressed"]) or float(b["value"]) > 0.5

    def value(self, index: int) -> float:
        """Analog value in [0, 1] (triggers); 0/1 for digital buttons."""
        if index < 0 or index >= len(self.buttons):
            return 0.0
        return float(self.buttons[index]["value"])

    def axis(self, index: int, deadzone: float = 0.0) -> float:
        if index < 0 or index >= len(self.axes):
            return 0.0
        v = float(self.axes[index])
        return 0.0 if abs(v) < deadzone else v

    def pressed_names(self) -> list[str]:
        return [BUTTON_NAMES.get(i, str(i)) for i in range(NUM_BUTTONS) if self.pressed(i)]

    def copy(self) -> "GamepadState":
        return GamepadState(buttons=[dict(b) for b in self.buttons], axes=list(self.axes))

    def equals(self, other: "GamepadState") -> bool:
        for a, b in zip(self.axes, other.axes):
            if abs(a - b) > AXIS_EPS:
                return False
        for a, b in zip(self.buttons, other.buttons):
            if bool(a["pressed"]) != bool(b["pressed"]):
                return False
            if abs(float(a["value"]) - float(b["value"])) > TRIGGER_EPS:
                return False
        return True


# ── normalisation ──────────────────────────────────────────────────────────────


def _clamp(value: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


def _is_raw_int(raw: float | int) -> bool:
    return isinstance(raw, int) and not isinstance(raw, bool)


def _norm_axis(raw: float | int) -> float:
    """Stick axis → [-1, 1].

    The source format is decided by TYPE, not by magnitude: SDL game-controller axes are raw
    int16 (-32768..32767), pygame joystick axes are floats in [-1, 1]. (The AGV bridge decided
    by magnitude, so a stick resting at raw 1 or -1 read as FULL deflection and kept moving.)
    """
    if _is_raw_int(raw):
        v = int(raw)
        return _clamp(v / 32767.0 if v >= 0 else v / 32768.0)
    return _clamp(float(raw))


def _norm_trigger(raw: float | int) -> float:
    """Trigger → [0, 1]. Raw int (SDL controller): 0..32767, rest 0.
    Float (pygame joystick): [-1, 1] with rest -1, or [0, 1] with rest 0."""
    if _is_raw_int(raw):
        return _clamp(int(raw) / 32767.0, 0.0, 1.0)
    v = float(raw)
    if v < 0:
        return _clamp((v + 1.0) / 2.0, 0.0, 1.0)
    return _clamp(v, 0.0, 1.0)


def _set_button(buttons: list[dict[str, float | bool]], index: int, value: float) -> None:
    if index < 0 or index >= NUM_BUTTONS:
        return
    buttons[index] = {"pressed": value > 0.5, "value": float(value)}


def _apply_hat(buttons: list[dict[str, float | bool]], hat: tuple[int, int]) -> None:
    x, y = hat
    _set_button(buttons, BTN_DPAD_UP, 1.0 if y > 0 else 0.0)
    _set_button(buttons, BTN_DPAD_DOWN, 1.0 if y < 0 else 0.0)
    _set_button(buttons, BTN_DPAD_LEFT, 1.0 if x < 0 else 0.0)
    _set_button(buttons, BTN_DPAD_RIGHT, 1.0 if x > 0 else 0.0)


# ── reader ─────────────────────────────────────────────────────────────────────


class GamepadReader:
    """Poll the first physical gamepad and emit normalised W3C state.

    Prefers the SDL2 game-controller API (correct mapping for any controller SDL knows),
    falls back to the generic pygame joystick with an XInput-style mapping. Handles
    hot-plug: ``read()`` returns ``None`` while nothing is attached and picks the pad up
    again when it comes back.
    """

    def __init__(self) -> None:
        try:
            import pygame
        except ImportError as e:
            raise ImportError("gamepad support needs pygame (pip install pygame)") from e
        self._pygame = pygame
        self._sdl_controller_mod: Any = None
        self._controller: Any = None
        self._joystick: Any = None
        self._use_sdl_controller = False
        self._device_id = ""
        self._device_name = ""
        self._instance_id: Optional[int] = None
        self._connected = False
        self._last_state = GamepadState()
        self._init_pygame()

    # -- properties -------------------------------------------------------------

    @property
    def connected(self) -> bool:
        return self._connected

    @property
    def device_id(self) -> str:
        return self._device_id

    @property
    def device_name(self) -> str:
        return self._device_name

    @property
    def last_state(self) -> GamepadState:
        return self._last_state

    # -- simple API --------------------------------------------------------------

    def read(self) -> Optional[GamepadState]:
        """Poll once; the current state, or None while no gamepad is attached."""
        self.poll()
        return self._last_state if self._connected else None

    def close(self) -> None:
        self._close_device()
        try:
            self._pygame.quit()
        except Exception:  # noqa: BLE001
            pass

    # -- pygame / SDL plumbing ----------------------------------------------------

    def _init_pygame(self) -> None:
        pygame = self._pygame
        pygame.init()
        pygame.joystick.init()
        try:
            from pygame._sdl2 import controller as sdl_controller

            sdl_controller.init()
            self._sdl_controller_mod = sdl_controller
            try:
                sdl_controller.set_eventstate(True)
            except (AttributeError, pygame.error):
                pass
        except (ImportError, AttributeError):
            self._sdl_controller_mod = None

    def _open_first_device(self) -> bool:
        pygame = self._pygame
        self._close_device()

        if self._sdl_controller_mod is not None:
            for idx in range(self._sdl_controller_mod.get_count()):
                if self._sdl_controller_mod.is_controller(idx):
                    try:
                        ctrl = self._sdl_controller_mod.Controller(idx)
                        ctrl.init()
                    except pygame.error as exc:
                        logger.debug("SDL controller %d init failed: %s", idx, exc)
                        continue
                    self._controller = ctrl
                    self._use_sdl_controller = True
                    self._device_id = f"sdl-{idx}"
                    self._device_name = ctrl.name or f"Controller {idx}"
                    self._instance_id = int(ctrl.id)
                    self._connected = True
                    logger.info("Opened SDL game controller: %s", self._device_name)
                    return True

        for idx in range(pygame.joystick.get_count()):
            try:
                js = pygame.joystick.Joystick(idx)
                js.init()
            except pygame.error as exc:
                logger.debug("Joystick %d init failed: %s", idx, exc)
                continue
            self._joystick = js
            self._use_sdl_controller = False
            self._device_id = f"joy-{idx}"
            self._device_name = js.get_name() or f"Joystick {idx}"
            self._instance_id = js.get_instance_id()
            self._connected = True
            logger.info(
                "Opened joystick (generic mapping): %s (axes=%d buttons=%d hats=%d)",
                self._device_name, js.get_numaxes(), js.get_numbuttons(), js.get_numhats(),
            )
            return True

        return False

    def _close_device(self) -> None:
        pygame = self._pygame
        if self._controller is not None:
            try:
                self._controller.quit()
            except pygame.error:
                pass
            self._controller = None
        if self._joystick is not None:
            try:
                self._joystick.quit()
            except pygame.error:
                pass
            self._joystick = None
        self._use_sdl_controller = False
        self._connected = False
        self._device_id = ""
        self._device_name = ""
        self._instance_id = None
        self._last_state = GamepadState()

    def _controller_attached(self) -> bool:
        if self._controller is None:
            return False
        attached_fn = getattr(self._controller, "attached", None)
        if attached_fn is None:
            return True
        return bool(attached_fn()) if callable(attached_fn) else bool(attached_fn)

    def _sdl_has_game_controller(self) -> bool:
        if self._sdl_controller_mod is None:
            return self._pygame.joystick.get_count() > 0
        return any(self._sdl_controller_mod.is_controller(i) for i in range(self._sdl_controller_mod.get_count()))

    def _disconnect_event_seen(self) -> bool:
        pygame = self._pygame
        for event in pygame.event.get():
            if event.type not in (pygame.JOYDEVICEREMOVED, pygame.CONTROLLERDEVICEREMOVED):
                continue
            inst = getattr(event, "instance_id", None)
            if self._instance_id is None or inst == self._instance_id:
                logger.info("Gamepad removed (instance_id=%s)", inst)
                return True
        return False

    def _is_sdl_controller_alive(self) -> bool:
        if self._controller is None:
            return False
        if self._disconnect_event_seen():
            return False
        if not self._controller_attached():
            logger.info("Gamepad detached (attached() returned False)")
            return False
        if self._pygame.joystick.get_count() == 0:
            logger.info("Gamepad detached (joystick count=0)")
            return False
        if not self._sdl_has_game_controller():
            logger.info("Gamepad detached (no SDL game controllers present)")
            return False
        try:
            self._controller.get_axis(0)
            return True
        except self._pygame.error:
            logger.info("Gamepad detached (axis read failed)")
            return False

    def _disconnect_gamepad_event(self) -> dict[str, Any]:
        old_id, old_name = self._device_id, self._device_name
        self._close_device()
        return {"type": "gamepad", "connected": False, "id": old_id, "name": old_name}

    # -- reading ---------------------------------------------------------------------

    def _read_sdl_controller(self) -> GamepadState:
        assert self._controller is not None
        buttons = empty_buttons()
        axes = empty_axes()
        for sdl_idx, gp_idx in SDL_BTN_TO_GP.items():
            _set_button(buttons, gp_idx, 1.0 if self._controller.get_button(sdl_idx) else 0.0)
        _set_button(buttons, BTN_LT, _norm_trigger(self._controller.get_axis(4)))
        _set_button(buttons, BTN_RT, _norm_trigger(self._controller.get_axis(5)))
        for axis_idx in range(NUM_AXES):
            axes[axis_idx] = _norm_axis(self._controller.get_axis(axis_idx))
        return GamepadState(buttons=buttons, axes=axes)

    def _read_joystick_xinput(self, js: Any) -> GamepadState:
        buttons = empty_buttons()
        axes = empty_axes()
        name = (js.get_name() or "").lower()

        for gp_idx in range(min(js.get_numbuttons(), 4)):
            _set_button(buttons, gp_idx, 1.0 if js.get_button(gp_idx) else 0.0)

        if js.get_numbuttons() >= 11:
            for src, dst in JOY_BTN_TO_GP.items():
                if src < js.get_numbuttons():
                    _set_button(buttons, dst, 1.0 if js.get_button(src) else 0.0)

        if js.get_numaxes() >= 6:
            _set_button(buttons, BTN_LT, _norm_trigger(js.get_axis(2)))
            _set_button(buttons, BTN_RT, _norm_trigger(js.get_axis(5)))
        elif js.get_numbuttons() >= 8:
            _set_button(buttons, BTN_LT, 1.0 if js.get_button(6) else 0.0)
            _set_button(buttons, BTN_RT, 1.0 if js.get_button(7) else 0.0)

        if js.get_numhats() > 0:
            _apply_hat(buttons, js.get_hat(0))
        elif js.get_numbuttons() >= 16:
            for src, dst in ((11, BTN_DPAD_UP), (12, BTN_DPAD_DOWN), (13, BTN_DPAD_LEFT), (14, BTN_DPAD_RIGHT)):
                _set_button(buttons, dst, 1.0 if js.get_button(src) else 0.0)

        if js.get_numaxes() >= 2:
            axes[AXIS_LEFT_X] = _clamp(js.get_axis(0))
            axes[AXIS_LEFT_Y] = _clamp(js.get_axis(1))
        if js.get_numaxes() >= 4:
            # 8BitDo pads report the right stick on axes 3/4 instead of 2/3.
            axes[AXIS_RIGHT_X] = _clamp(js.get_axis(3) if "8bitdo" in name else js.get_axis(2))
            axes[AXIS_RIGHT_Y] = _clamp(js.get_axis(4) if "8bitdo" in name else js.get_axis(3))

        return GamepadState(buttons=buttons, axes=axes)

    def _read_current(self) -> GamepadState:
        if self._use_sdl_controller and self._controller is not None:
            return self._read_sdl_controller()
        if self._joystick is not None:
            return self._read_joystick_xinput(self._joystick)
        return GamepadState()

    def poll(self) -> tuple[Optional[dict[str, Any]], Optional[dict[str, Any]]]:
        """One poll. Returns ``(gamepad_event, state_event)``: a connect/disconnect notice
        and a state message, each ``None`` when nothing changed (same contract as the AGV bridge)."""
        pygame = self._pygame
        pygame.event.pump()

        gamepad_event: Optional[dict[str, Any]] = None
        state_event: Optional[dict[str, Any]] = None

        if not self._connected:
            if self._open_first_device():
                gamepad_event = {"type": "gamepad", "connected": True, "id": self._device_id, "name": self._device_name}
                self._last_state = self._read_current()
                state_event = self._state_message(self._last_state)
            else:
                return None, None
        else:
            if self._use_sdl_controller:
                if self._sdl_controller_mod is None or not self._is_sdl_controller_alive():
                    return self._disconnect_gamepad_event(), None
            elif self._joystick is None or pygame.joystick.get_count() == 0:
                return self._disconnect_gamepad_event(), None
            elif not self._joystick.get_attached():
                logger.info("Gamepad detached (joystick.get_attached()=False)")
                return self._disconnect_gamepad_event(), None
            elif self._disconnect_event_seen():
                return self._disconnect_gamepad_event(), None

        current = self._read_current()
        if not self._last_state.equals(current):
            self._last_state = current
            state_event = self._state_message(current)

        return gamepad_event, state_event

    @staticmethod
    def _state_message(state: GamepadState) -> dict[str, Any]:
        return {"type": "state", "ts": time.time(), "buttons": state.buttons, "axes": state.axes}


# ── CLI ────────────────────────────────────────────────────────────────────────


def list_devices() -> None:
    import pygame

    pygame.init()
    pygame.joystick.init()
    print("Pygame joysticks:")
    if pygame.joystick.get_count() == 0:
        print("  (none)")
    for idx in range(pygame.joystick.get_count()):
        js = pygame.joystick.Joystick(idx)
        js.init()
        print(f"  [{idx}] {js.get_name()} (axes={js.get_numaxes()}, buttons={js.get_numbuttons()}, hats={js.get_numhats()})")
        js.quit()
    try:
        from pygame._sdl2 import controller as sdl_controller

        sdl_controller.init()
        print("\nSDL2 game controllers:")
        if sdl_controller.get_count() == 0:
            print("  (none)")
        for idx in range(sdl_controller.get_count()):
            label = "game controller" if sdl_controller.is_controller(idx) else "joystick only"
            name = sdl_controller.name_forindex(idx) or f"device-{idx}"
            print(f"  [{idx}] {name} ({label})")
    except (ImportError, AttributeError):
        print("\nSDL2 game controller API not available in this pygame build.")
    if sys.platform.startswith("linux"):
        try:
            import evdev
            from evdev import ecodes

            print("\nevdev input devices (gamepad-like):")
            found = False
            for path in evdev.list_devices():
                try:
                    dev = evdev.InputDevice(path)
                except OSError:
                    continue
                caps = dev.capabilities()
                keys = caps.get(ecodes.EV_KEY, [])
                gamepad_keys = {ecodes.BTN_A, ecodes.BTN_B, ecodes.BTN_X, ecodes.BTN_Y, ecodes.BTN_TL, ecodes.BTN_TR}
                if gamepad_keys.intersection(keys) or caps.get(ecodes.EV_ABS, []):
                    found = True
                    print(f"  {path} — {dev.name}")
            if not found:
                print("  (none)")
        except ImportError:
            print("\nevdev not installed (pip install evdev) — skipping Linux device scan.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--list_devices", action="store_true", help="List detected gamepads and exit")
    parser.add_argument("--hz", type=float, default=20.0, help="Print rate")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    if args.list_devices:
        list_devices()
        return

    reader = GamepadReader()
    print("Waiting for a gamepad... (Ctrl-C to stop)")
    try:
        while True:
            state = reader.read()
            if state is None:
                print("no gamepad".ljust(100), end="\r")
            else:
                axes = " ".join(f"{a:+.2f}" for a in state.axes)
                print(
                    f"{reader.device_name[:24]:<24} axes[LX LY RX RY]={axes}  "
                    f"LT={state.value(BTN_LT):.2f} RT={state.value(BTN_RT):.2f}  "
                    f"pressed={state.pressed_names()}".ljust(100),
                    end="\r",
                )
            time.sleep(1.0 / max(1.0, args.hz))
    except KeyboardInterrupt:
        print()
    finally:
        reader.close()


if __name__ == "__main__":
    main()

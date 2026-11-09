"""Robot-independent UDP transport for the existing normalized GamepadState.

Accepts both the original gamepad_state v1 packets and Porter/RevoPilot
protocol v2 (named 8BitDo axes/buttons, ``protocol="v2"``). The control loop
always sees a W3C ``GamepadState``.
"""
from __future__ import annotations

import json
import logging
import math
import socket
import time

from .gamepad import (
    BTN_A,
    BTN_B,
    BTN_BACK,
    BTN_DPAD_DOWN,
    BTN_DPAD_LEFT,
    BTN_DPAD_RIGHT,
    BTN_DPAD_UP,
    BTN_GUIDE,
    BTN_L3,
    BTN_LB,
    BTN_LT,
    BTN_R3,
    BTN_RB,
    BTN_RT,
    BTN_START,
    BTN_X,
    BTN_Y,
    GamepadState,
    NUM_AXES,
    NUM_BUTTONS,
    empty_axes,
    empty_buttons,
)

MAX_PACKET_BYTES = 8192
logger = logging.getLogger(__name__)

V2_DIGITAL = {
    "A": BTN_A,
    "B": BTN_B,
    "X": BTN_X,
    "Y": BTN_Y,
    "L1": BTN_LB,
    "R1": BTN_RB,
    "Options": BTN_BACK,
    "Menu": BTN_START,
    "L3": BTN_L3,
    "R3": BTN_R3,
    "Home": BTN_GUIDE,
}


def _finite_number(value, lo, hi, name="value"):
    if type(value) not in (int, float) or not math.isfinite(value) or not lo <= value <= hi:
        raise ValueError(f"Gamepad {name} outside normalized range")
    return float(value)


def _as_float(value, default=0.0):
    if value is None:
        return default
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError("Gamepad value outside normalized range")
    return float(value)


def encode_state(state: GamepadState | None, session: str, seq: int) -> bytes:
    payload = dict(type="gamepad_state", version=1, session=session, seq=seq, connected=state is not None)
    if state is not None:
        payload.update(axes=state.axes, buttons=state.buttons)
    return json.dumps(payload, separators=(",", ":"), allow_nan=False).encode("utf-8")


def encode_v2_state(state: GamepadState | None, seq: int, brain: str = "HI") -> bytes:
    """Porter/RevoPilot v2: named buttons and axes, no W3C arrays."""
    brake = 1 if state is None else 0
    st = state or GamepadState()

    def digital(index: int) -> float:
        return 1.0 if st.pressed(index) else 0.0

    dx = digital(BTN_DPAD_RIGHT) - digital(BTN_DPAD_LEFT)
    dy = digital(BTN_DPAD_UP) - digital(BTN_DPAD_DOWN)
    payload = {
        "seq": seq,
        "t": int(time.time()),
        "brain": brain,
        "protocol": "v2",
        "brake": brake,
        "A": digital(BTN_A),
        "B": digital(BTN_B),
        "X": digital(BTN_X),
        "Y": digital(BTN_Y),
        "L1": digital(BTN_LB),
        "R1": digital(BTN_RB),
        "L3": digital(BTN_L3),
        "R3": digital(BTN_R3),
        "Options": digital(BTN_BACK),
        "Menu": digital(BTN_START),
        "Home": digital(BTN_GUIDE),
        "L4": 0.0,
        "R4": 0.0,
        "M1": 0.0,
        "M2": 0.0,
        "lx": float(st.axis(0)),
        "ly": float(st.axis(1)),
        "rx": float(st.axis(2)),
        "ry": float(st.axis(3)),
        "dx": dx,
        "dy": dy,
        "l2": float(st.value(BTN_LT)),
        "r2": float(st.value(BTN_RT)),
    }
    return json.dumps(payload, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _decode_v1(msg: dict):
    if msg.get("type") != "gamepad_state" or msg.get("version") != 1:
        raise ValueError("Unsupported gamepad packet")
    session, seq, connected = msg.get("session"), msg.get("seq"), msg.get("connected")
    if not isinstance(session, str) or not 1 <= len(session) <= 64:
        raise ValueError("Invalid gamepad session")
    if type(seq) is not int or seq < 0 or type(connected) is not bool:
        raise ValueError("Invalid gamepad sequence/connection state")
    if not connected:
        return session, seq, None
    axes, buttons = msg.get("axes"), msg.get("buttons")
    if not isinstance(axes, list) or len(axes) != NUM_AXES:
        raise ValueError("Expected four normalized axes")
    if not isinstance(buttons, list) or len(buttons) != NUM_BUTTONS:
        raise ValueError("Expected seventeen W3C buttons")
    axes = [_finite_number(value, -1.0, 1.0, "axis") for value in axes]
    result_buttons = []
    for button in buttons:
        if not isinstance(button, dict) or type(button.get("pressed")) is not bool:
            raise ValueError("Invalid gamepad button")
        result_buttons.append(
            dict(pressed=button["pressed"], value=_finite_number(button.get("value"), 0.0, 1.0, "button"))
        )
    return session, seq, GamepadState(axes=axes, buttons=result_buttons)


def _decode_v2(msg: dict):
    seq = msg.get("seq")
    if type(seq) is not int or seq < 0:
        raise ValueError("Invalid gamepad sequence/connection state")
    brake = msg.get("brake", 0)
    if type(brake) not in (int, float) or not math.isfinite(brake):
        raise ValueError("Invalid v2 brake")
    session = "v2"
    if int(brake) != 0:
        return session, seq, None
    axes = empty_axes()
    for i, name in enumerate(("lx", "ly", "rx", "ry")):
        raw = msg.get(name, 0.0)
        axes[i] = _finite_number(_as_float(raw), -1.0, 1.0, name)
    buttons = empty_buttons()
    for name, index in V2_DIGITAL.items():
        pressed = _as_float(msg.get(name, 0.0)) > 0.5
        buttons[index] = dict(pressed=pressed, value=1.0 if pressed else 0.0)
    l2_raw = msg.get("l2", msg.get("L2", 0.0))
    r2_raw = msg.get("r2", msg.get("R2", 0.0))
    l2 = _finite_number(_as_float(l2_raw), 0.0, 1.0, "l2")
    r2 = _finite_number(_as_float(r2_raw), 0.0, 1.0, "r2")
    buttons[BTN_LT] = dict(pressed=l2 > 0.5, value=l2)
    buttons[BTN_RT] = dict(pressed=r2 > 0.5, value=r2)
    dx = _finite_number(_as_float(msg.get("dx", 0.0)), -1.0, 1.0, "dx")
    dy = _finite_number(_as_float(msg.get("dy", 0.0)), -1.0, 1.0, "dy")
    buttons[BTN_DPAD_UP] = dict(pressed=dy > 0.5, value=1.0 if dy > 0.5 else 0.0)
    buttons[BTN_DPAD_DOWN] = dict(pressed=dy < -0.5, value=1.0 if dy < -0.5 else 0.0)
    buttons[BTN_DPAD_LEFT] = dict(pressed=dx < -0.5, value=1.0 if dx < -0.5 else 0.0)
    buttons[BTN_DPAD_RIGHT] = dict(pressed=dx > 0.5, value=1.0 if dx > 0.5 else 0.0)
    return session, seq, GamepadState(axes=axes, buttons=buttons)


def decode_packet(data: bytes):
    """v1 gamepad_state or Porter protocol v2 → (session, seq, GamepadState|None)."""
    if len(data) > MAX_PACKET_BYTES:
        raise ValueError("Gamepad packet too large")
    try:
        msg = json.loads(data)
    except (ValueError, UnicodeError, RecursionError) as e:
        raise ValueError("Invalid gamepad JSON") from e
    if not isinstance(msg, dict):
        raise ValueError("Invalid gamepad JSON")
    if str(msg.get("protocol") or "") == "v2":
        return _decode_v2(msg)
    return _decode_v1(msg)


def decode_state(data: bytes):
    """Validate packets before any axes/buttons reach robot control."""
    return decode_packet(data)


class UDPGamepadReader:
    """Nonblocking latest-state reader; absent/disconnected/stale input returns None.

    One active sender/session owns the input until it disconnects or times out.
    Duplicate and reordered sequence numbers do not refresh the watchdog.
    """

    def __init__(self, port: int, timeout_s: float = 0.3, host: str = "0.0.0.0", debug: bool = False):
        if type(port) is not int or not 0 <= port <= 65535:
            raise ValueError("UDP port must be an integer in [0, 65535]")
        if not math.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError("UDP timeout must be positive and finite")
        self.timeout_s = timeout_s
        self.debug = debug
        self._timeout_logged = False
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            self._sock.bind((host, port))
            self._sock.setblocking(False)
        except BaseException:
            self._sock.close()
            raise
        self.address = self._sock.getsockname()
        self._peer = None
        self._last_seq = -1
        self._received_at = None
        self._state = None

    @property
    def device_name(self):
        return f"UDP gamepad on {self.address[0]}:{self.address[1]}"

    def _log_packet(self, address, status, data):
        if self.debug:
            logger.info("GAMEPAD RX from=%s:%s status=%s payload=%r", *address, status, data)

    def read(self) -> GamepadState | None:
        if self._sock is None:
            return None
        # Bound draining so a flooded socket cannot monopolize the control tick.
        for _ in range(256):
            try:
                data, address = self._sock.recvfrom(MAX_PACKET_BYTES + 1)
            except BlockingIOError:
                break
            except OSError as e:
                if getattr(e, "winerror", None) == 10040:  # oversized UDP datagram
                    continue
                raise
            try:
                session, seq, state = decode_packet(data)
            except (ValueError, OverflowError) as e:
                self._log_packet(address, f"rejected: {e}", data)
                continue
            now = time.monotonic()
            peer = (address, session)
            live = self._state is not None and self._received_at is not None and now - self._received_at <= self.timeout_s
            if self._peer == peer:
                if seq <= self._last_seq:
                    self._log_packet(address, "ignored: duplicate/out-of-order", data)
                    continue
            elif live:
                self._log_packet(address, "ignored: another sender is active", data)
                continue
            self._peer, self._last_seq = peer, seq
            self._state, self._received_at = state, now
            self._timeout_logged = False
            self._log_packet(address, "accepted", data)
        if self._received_at is None or time.monotonic() - self._received_at > self.timeout_s:
            if self.debug and self._received_at is not None and not self._timeout_logged:
                logger.info(
                    "GAMEPAD RX timeout: no fresh packet for %.3f s; input unavailable",
                    self.timeout_s,
                )
                self._timeout_logged = True
            return None
        return self._state.copy() if self._state is not None else None

    def close(self):
        if self._sock is not None:
            self._sock.close()
            self._sock = None
        self._state = None

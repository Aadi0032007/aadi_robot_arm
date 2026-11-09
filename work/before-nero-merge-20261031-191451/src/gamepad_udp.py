"""Robot-independent UDP transport for the existing normalized GamepadState."""
from __future__ import annotations

import json
import logging
import math
import socket
import time

from .gamepad import GamepadState, NUM_AXES, NUM_BUTTONS

MAX_PACKET_BYTES = 8192
logger = logging.getLogger(__name__)


def encode_state(state: GamepadState | None, session: str, seq: int) -> bytes:
    payload = dict(type='gamepad_state', version=1, session=session, seq=seq, connected=state is not None)
    if state is not None:
        payload.update(axes=state.axes, buttons=state.buttons)
    return json.dumps(payload, separators=(',', ':'), allow_nan=False).encode('utf-8')


def decode_state(data: bytes):
    """Validate packets before any axes/buttons reach robot control."""
    if len(data) > MAX_PACKET_BYTES:
        raise ValueError('Gamepad packet too large')
    try:
        msg = json.loads(data)
    except (ValueError, UnicodeError, RecursionError) as e:
        raise ValueError('Invalid gamepad JSON') from e
    if not isinstance(msg, dict) or msg.get('type') != 'gamepad_state' or msg.get('version') != 1:
        raise ValueError('Unsupported gamepad packet')
    session, seq, connected = msg.get('session'), msg.get('seq'), msg.get('connected')
    if not isinstance(session, str) or not 1 <= len(session) <= 64:
        raise ValueError('Invalid gamepad session')
    if type(seq) is not int or seq < 0 or type(connected) is not bool:
        raise ValueError('Invalid gamepad sequence/connection state')
    if not connected:
        return session, seq, None
    axes, buttons = msg.get('axes'), msg.get('buttons')
    if not isinstance(axes, list) or len(axes) != NUM_AXES:
        raise ValueError('Expected four normalized axes')
    if not isinstance(buttons, list) or len(buttons) != NUM_BUTTONS:
        raise ValueError('Expected seventeen W3C buttons')

    def number(value, lo, hi):
        if type(value) not in (int, float) or not math.isfinite(value) or not lo <= value <= hi:
            raise ValueError('Gamepad value outside normalized range')
        return float(value)

    axes = [number(value, -1., 1.) for value in axes]
    result_buttons = []
    for button in buttons:
        if not isinstance(button, dict) or type(button.get('pressed')) is not bool:
            raise ValueError('Invalid gamepad button')
        result_buttons.append(dict(pressed=button['pressed'], value=number(button.get('value'), 0., 1.)))
    return session, seq, GamepadState(axes=axes, buttons=result_buttons)


class UDPGamepadReader:
    """Nonblocking latest-state reader; absent/disconnected/stale input returns None.

    One active sender/session owns the input until it disconnects or times out.
    Duplicate and reordered sequence numbers do not refresh the watchdog.
    """
    def __init__(self, port: int, timeout_s: float = .3, host: str = '0.0.0.0', debug: bool = False):
        if type(port) is not int or not 0 <= port <= 65535:
            raise ValueError('UDP port must be an integer in [0, 65535]')
        if not math.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError('UDP timeout must be positive and finite')
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
        return f'UDP gamepad on {self.address[0]}:{self.address[1]}'

    def _log_packet(self, address, status, data):
        if self.debug:
            logger.info('GAMEPAD RX from=%s:%s status=%s payload=%r', *address, status, data)

    def read(self) -> GamepadState | None:
        if self._sock is None:
            return None
        # Bound draining so a flooded socket cannot monopolize the control tick.
        for _ in range(256):
            try:
                data, address = self._sock.recvfrom(MAX_PACKET_BYTES+1)
            except BlockingIOError:
                break
            except OSError as e:
                if getattr(e, 'winerror', None) == 10040:  # oversized UDP datagram
                    continue
                raise
            try:
                session, seq, state = decode_state(data)
            except (ValueError, OverflowError) as e:
                self._log_packet(address, f'rejected: {e}', data)
                continue
            now = time.monotonic()
            peer = (address, session)
            live = self._state is not None and self._received_at is not None and now-self._received_at <= self.timeout_s
            if self._peer == peer:
                if seq <= self._last_seq:
                    self._log_packet(address, 'ignored: duplicate/out-of-order', data)
                    continue
            elif live:
                self._log_packet(address, 'ignored: another sender is active', data)
                continue
            self._peer, self._last_seq = peer, seq
            self._state, self._received_at = state, now
            self._timeout_logged = False
            self._log_packet(address, 'accepted', data)
        if self._received_at is None or time.monotonic()-self._received_at > self.timeout_s:
            if self.debug and self._received_at is not None and not self._timeout_logged:
                logger.info('GAMEPAD RX timeout: no fresh packet for %.3f s; input unavailable', self.timeout_s)
                self._timeout_logged = True
            return None
        return self._state.copy() if self._state is not None else None

    def close(self):
        if self._sock is not None:
            self._sock.close()
            self._sock = None
        self._state = None

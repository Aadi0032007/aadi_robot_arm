"""UDP JSON link between teleop and inference (same pattern as the AGV's AI port).

    teleop   binds  ai.action_port  ← inference sends {"seq","t","action":{...}}
    inference binds ai.state_port   ← teleop sends   {"seq","t","observation":{...},"action":{...},"ai":bool,...}

Connectionless on purpose: either side can start first, and a dropped packet is
simply superseded by the next one. Freshness is judged on arrival time.
"""

from __future__ import annotations

import json
import logging
import socket
import threading
import time
from typing import Any, Optional

logger = logging.getLogger(__name__)

MAX_PACKET_BYTES = 65535


class UdpJsonListener:
    """Keeps the most recent JSON object received on ``host:port``."""

    def __init__(self, host: str, port: int, label: str = "udp") -> None:
        self._host = host
        self._port = port
        self._label = label
        self._sock: Optional[socket.socket] = None
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._latest: Optional[dict[str, Any]] = None
        self._rx_t: float = 0.0
        self._last_seq: int = -1
        self.received = 0

    def start(self) -> None:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((self._host, self._port))
        sock.settimeout(0.5)
        self._sock = sock
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True, name=f"udp-{self._label}")
        self._thread.start()
        logger.info("%s: listening on udp://%s:%s", self._label, self._host, self._port)

    def stop(self) -> None:
        self._stop.set()
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        self._sock = None

    def latest(self, max_age_s: float | None = None) -> Optional[dict[str, Any]]:
        """The newest packet, or None if nothing arrived (or it is older than ``max_age_s``)."""
        with self._lock:
            if self._latest is None:
                return None
            if max_age_s is not None and (time.monotonic() - self._rx_t) > max_age_s:
                return None
            return dict(self._latest)

    def age(self) -> float:
        with self._lock:
            return float("inf") if self._rx_t <= 0.0 else time.monotonic() - self._rx_t

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                data, _ = self._sock.recvfrom(MAX_PACKET_BYTES)
            except socket.timeout:
                continue
            except ConnectionResetError:
                continue  # Windows: ICMP unreachable from an earlier send on this socket
            except OSError:
                break
            try:
                pkt = json.loads(data.decode("utf-8"))
            except (ValueError, UnicodeDecodeError) as exc:
                logger.debug("%s: dropped malformed packet: %s", self._label, exc)
                continue
            if not isinstance(pkt, dict):
                continue
            seq = pkt.get("seq")
            with self._lock:
                if isinstance(seq, int) and seq <= self._last_seq and seq > 0:
                    continue  # reordered datagram
                if isinstance(seq, int):
                    self._last_seq = seq
                self._latest = pkt
                self._rx_t = time.monotonic()
                self.received += 1


class UdpJsonSender:
    """Fire-and-forget JSON datagrams to ``host:port``, with ``seq`` and ``t`` stamped in."""

    def __init__(self, host: str, port: int) -> None:
        self._addr = (host, port)
        self._sock: Optional[socket.socket] = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._seq = 0

    def send(self, payload: dict[str, Any]) -> None:
        if self._sock is None:
            return
        self._seq += 1
        body = {"seq": self._seq, "t": time.time(), **payload}
        try:
            self._sock.sendto(json.dumps(body, separators=(",", ":")).encode("utf-8"), self._addr)
        except OSError as exc:  # nobody listening yet is not an error
            logger.debug("udp send to %s failed: %s", self._addr, exc)

    def close(self) -> None:
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass
            self._sock = None

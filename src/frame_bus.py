# -*- coding: utf-8 -*-
"""
Frame bus — shared-memory fan-out for camera frames.

Copied from the AGV repo (``LAB/utils/frame_bus.py``) so the arm uses the same
mechanism: teleop is the sole writer for each named bus; any number of external
processes can attach as readers without ever touching V4L2. The kernel only lets
one process own a /dev/videoN node, so this is how downstream consumers (AI
inference, debug tools, anything else) get frames without fighting teleop for
the camera.

Memory lives in /dev/shm, which is a tmpfs (pure RAM). A 640x480x3 BGR
frame is ~900 KB; a memcpy at that size is ~1 ms on a Jetson — about
3% of one 30 fps frame budget. Publishing is effectively free.

Layout (one shared-memory region per camera):
    offset      size        field        notes
    0           8           magic        b"LABFBUS" + layout version byte
    8           4           n_slots      uint32
    12          4           dtype_code   uint32 — 0 = uint8 (BGR)
    16          12          shape        3 x uint32: height, width, channels
    28          4           reserved
    32          8           pub_count    uint64 — frames published so far
    40          24          padding      align the slot table to 64 bytes
    64          n*64        slot table   per slot: float64 ts, uint64 frame_idx
    64+n*64     n*h*w*c     payloads     n contiguous BGR frames

Concurrency:
    A ring of n slots, not a seqlock. The writer fills the slot that the
    *next* frame will occupy, then bumps pub_count — it never mutates the
    slot that is currently published, so a reader is never locked out and
    there is no mid-publish state to wait on.

    Frame f lives in slot (f-1) % n. The writer next touches that slot
    while producing frame f+n, which it begins only after publishing
    f+n-1. So a reader that sampled pub_count == f has a valid copy iff
    pub_count has advanced by no more than n-2 by the time it finishes.
    With n=4 that tolerates the reader being stalled for two whole frame
    periods (~66 ms at 30 fps) against a ~1 ms copy — the margin is there
    for GC pauses and CUDA syncs, not for the memcpy.

    This also keeps the payload stores a full n-1 frame periods away from
    being recycled, which is what makes the design safe on aarch64 where
    plain stores are not ordered. (The GIL orders nothing here: writer and
    readers are separate processes on separate cores.)

Usage (writer, inside teleop):
    pub = FrameBusPublisher("wrist", height=480, width=640, channels=3)
    pub.publish(frame_bgr, timestamp)
    pub.close()        # closes the region; unlinks /dev/shm/lab_wrist

Usage (reader, in any other process):
    rdr = FrameBusReader("wrist")
    ts, frame = rdr.read_latest()       # (None, None) if no frame yet
    rdr.close()

The layout version in the magic is checked on attach. Bumping it means
teleop and every reader must be restarted together; a mismatch raises
rather than misparsing someone else's bytes as pixels.
"""

from __future__ import annotations

from multiprocessing import shared_memory
from typing import Optional

import numpy as np


# ── Layout constants ──────────────────────────────────────────────────────────

_SHM_PREFIX = "lab_"          # /dev/shm/lab_<camera_name>

_MAGIC = b"LABFBUS"           # 7 bytes
_LAYOUT_VERSION = 2           # bumped when the layout below changes
_MAGIC_V = _MAGIC + bytes([_LAYOUT_VERSION])

_HEADER_SIZE = 64             # slot table starts here
_SLOT_DESC_SIZE = 64          # one cache line per slot descriptor

# Field offsets within the header (in bytes)
_OFF_MAGIC     = 0
_OFF_N_SLOTS   = 8
_OFF_DTYPE     = 12
_OFF_SHAPE     = 16    # 3 x uint32: height, width, channels
_OFF_PUB_COUNT = 32

# Field offsets within one slot descriptor
_OFF_SLOT_TS  = 0      # float64
_OFF_SLOT_IDX = 8      # uint64

# 4 slots lets a reader stall for two frame periods and still hold a valid
# copy. 3 is the minimum the validation bound allows; 4 costs 900 KB more.
DEFAULT_N_SLOTS = 4


def _region_name(camera_name: str) -> str:
    return f"{_SHM_PREFIX}{camera_name}"


def _region_size(n_slots: int, height: int, width: int, channels: int) -> int:
    return (_HEADER_SIZE
            + n_slots * _SLOT_DESC_SIZE
            + n_slots * height * width * channels)


def _untrack(shm) -> None:
    """Stop the resource tracker from unlinking this region behind our back.

    The tracker calls shm_unlink() when any Python process that touched the
    object exits (forks and children included). That deletes the /dev/shm
    name while the fd stays open — the publisher keeps writing but readers
    can no longer attach by name. We own the lifetime explicitly instead.
    """
    try:
        from multiprocessing import resource_tracker
        resource_tracker.unregister(shm._name, "shared_memory")
    except Exception:
        pass


def _close_quietly(shm) -> None:
    try:
        shm.close()
    except Exception:
        pass


# ── Publisher ────────────────────────────────────────────────────────────────

class FrameBusPublisher:
    """Writer side. Owns a /dev/shm region for one camera."""

    def __init__(
        self,
        camera_name: str,
        height: int,
        width: int,
        channels: int = 3,
        n_slots: int = DEFAULT_N_SLOTS,
    ) -> None:
        self._name     = _region_name(camera_name)
        self._height   = int(height)
        self._width    = int(width)
        self._channels = int(channels)
        self._n_slots  = int(n_slots)

        if self._n_slots < 3:
            # The reader's validity bound is (pub_count advanced <= n-2), so
            # n=2 would leave zero tolerance and n=1 would be unreadable.
            raise ValueError(f"n_slots must be >= 3, got {self._n_slots}")

        size = _region_size(self._n_slots, self._height, self._width, self._channels)

        # Clean up any stale region left behind by a previous crashed teleop.
        try:
            stale = shared_memory.SharedMemory(name=self._name, create=False)
            stale.close()
            stale.unlink()
        except FileNotFoundError:
            pass

        self._shm = shared_memory.SharedMemory(name=self._name, create=True, size=size)
        _untrack(self._shm)

        buf = self._shm.buf

        # Scalar views into the header — direct memory, no copies.
        self._n_slots_v = np.ndarray((1,), dtype=np.uint32, buffer=buf, offset=_OFF_N_SLOTS)
        self._dtype_v   = np.ndarray((1,), dtype=np.uint32, buffer=buf, offset=_OFF_DTYPE)
        self._shape_v   = np.ndarray((3,), dtype=np.uint32, buffer=buf, offset=_OFF_SHAPE)
        self._pub_count = np.ndarray((1,), dtype=np.uint64, buffer=buf, offset=_OFF_PUB_COUNT)

        # Slot table — strided views, one descriptor per cache line.
        self._slot_ts = np.ndarray(
            (self._n_slots,), dtype=np.float64, buffer=buf,
            offset=_HEADER_SIZE + _OFF_SLOT_TS, strides=(_SLOT_DESC_SIZE,),
        )
        self._slot_idx = np.ndarray(
            (self._n_slots,), dtype=np.uint64, buffer=buf,
            offset=_HEADER_SIZE + _OFF_SLOT_IDX, strides=(_SLOT_DESC_SIZE,),
        )

        # Payload ring — n contiguous frames.
        self._payload = np.ndarray(
            (self._n_slots, self._height, self._width, self._channels),
            dtype=np.uint8,
            buffer=buf,
            offset=_HEADER_SIZE + self._n_slots * _SLOT_DESC_SIZE,
        )

        # Initialize the body before stamping the magic. A reader can open
        # this region by name the instant SharedMemory() returns, so the
        # magic is what says "the header behind me is real".
        self._pub_count[0]  = 0
        self._slot_ts[:]    = 0.0
        self._slot_idx[:]   = 0
        self._n_slots_v[0]  = self._n_slots
        self._dtype_v[0]    = 0          # 0 = uint8
        self._shape_v[0]    = self._height
        self._shape_v[1]    = self._width
        self._shape_v[2]    = self._channels
        buf[_OFF_MAGIC:_OFF_MAGIC + 8] = _MAGIC_V

        self._count  = 0   # frames published (cheaper than reading back)
        self._closed = False

    @property
    def name(self) -> str:
        return self._name

    @property
    def shape(self) -> tuple[int, int, int]:
        return (self._height, self._width, self._channels)

    @property
    def n_slots(self) -> int:
        return self._n_slots

    def publish(self, frame: np.ndarray, timestamp: float) -> None:
        """Publish a frame. Caller must pass an array with the configured shape & dtype."""
        if self._closed:
            return
        if frame.shape != (self._height, self._width, self._channels):
            raise ValueError(
                f"frame shape {frame.shape} != region shape "
                f"({self._height}, {self._width}, {self._channels})"
            )
        if frame.dtype != np.uint8:
            raise ValueError(f"frame dtype {frame.dtype} != uint8")

        # Fill the next slot. Readers are looking at slot (pub_count-1) % n,
        # which is not this one, so nothing below is observable until the
        # pub_count store at the end.
        slot = self._count % self._n_slots

        np.copyto(self._payload[slot], frame)          # straight memcpy
        self._slot_ts[slot]  = float(timestamp)
        self._count += 1
        self._slot_idx[slot] = np.uint64(self._count)

        self._pub_count[0] = np.uint64(self._count)    # publish

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True

        # Park the bus before tearing it down: a reader that still has this
        # region mapped (unlink does not unmap it) then sees "no frames"
        # instead of the last one, forever.
        try:
            self._pub_count[0] = 0
        except Exception:
            pass

        # Drop numpy views before closing the buffer — otherwise the BufferError
        # "memoryview has N exported buffers" fires on .close().
        self._n_slots_v = None  # type: ignore[assignment]
        self._dtype_v   = None  # type: ignore[assignment]
        self._shape_v   = None  # type: ignore[assignment]
        self._pub_count = None  # type: ignore[assignment]
        self._slot_ts   = None  # type: ignore[assignment]
        self._slot_idx  = None  # type: ignore[assignment]
        self._payload   = None  # type: ignore[assignment]

        _close_quietly(self._shm)
        try:
            self._shm.unlink()
        except Exception:
            pass


# ── Reader ───────────────────────────────────────────────────────────────────

def _probe_header(shm):
    """Read header fields as plain ints, releasing every buffer view first.

    Views exported from shm.buf block shm.close() with a BufferError, so the
    probe has to hand back plain values and drop its own references before
    the caller may decide to close.
    """
    buf = shm.buf
    size = len(buf)
    if size < _HEADER_SIZE:
        del buf
        return None

    magic  = bytes(buf[_OFF_MAGIC:_OFF_MAGIC + 8])
    n_view = np.ndarray((1,), dtype=np.uint32, buffer=buf, offset=_OFF_N_SLOTS)
    d_view = np.ndarray((1,), dtype=np.uint32, buffer=buf, offset=_OFF_DTYPE)
    s_view = np.ndarray((3,), dtype=np.uint32, buffer=buf, offset=_OFF_SHAPE)

    out = {
        "magic":    magic,
        "n_slots":  int(n_view[0]),
        "dtype":    int(d_view[0]),
        "height":   int(s_view[0]),
        "width":    int(s_view[1]),
        "channels": int(s_view[2]),
        "size":     size,
    }

    del n_view, d_view, s_view, buf
    return out


class FrameBusReader:
    """Reader side. Attaches to a region written by teleop."""

    def __init__(self, camera_name: str) -> None:
        self._name = _region_name(camera_name)
        self._shm: Optional[shared_memory.SharedMemory] = None
        self._pub_count = None
        self._slot_ts   = None
        self._slot_idx  = None
        self._payload   = None
        self._n_slots   = 0
        self._shape: Optional[tuple] = None
        self._last_count = 0

        # Try to attach now; if teleop hasn't started yet, read_latest()
        # will retry transparently.
        self._try_attach()

    # ── attach/detach ────────────────────────────────────────────────────────

    def _try_attach(self) -> bool:
        if self._shm is not None:
            return True
        try:
            shm = shared_memory.SharedMemory(name=self._name, create=False)
        except FileNotFoundError:
            return False

        _untrack(shm)

        hdr = _probe_header(shm)
        if hdr is None:
            _close_quietly(shm)
            return False

        magic = hdr["magic"]
        if magic != _MAGIC_V:
            if magic[:len(_MAGIC)] == _MAGIC:
                # Right bus, wrong layout — a deploy skew, not a race.
                version = magic[len(_MAGIC)] if len(magic) > len(_MAGIC) else 0
                _close_quietly(shm)
                raise RuntimeError(
                    f"frame bus {self._name!r} has layout version {version}, "
                    f"this build expects {_LAYOUT_VERSION}. Restart teleop and "
                    f"every frame-bus reader together."
                )
            # All-zero or foreign bytes: the writer is probably still
            # initializing. Back off and let the caller retry.
            _close_quietly(shm)
            return False

        n = hdr["n_slots"]
        h, w, c = hdr["height"], hdr["width"], hdr["channels"]
        if n < 3 or h <= 0 or w <= 0 or c <= 0:
            _close_quietly(shm)
            return False
        if hdr["dtype"] != 0:
            _close_quietly(shm)
            raise RuntimeError(
                f"frame bus {self._name!r} has dtype_code {hdr['dtype']}, "
                f"expected 0 (uint8)."
            )

        need = _region_size(n, h, w, c)
        if hdr["size"] < need:
            _close_quietly(shm)
            raise RuntimeError(
                f"frame bus {self._name!r} is {hdr['size']} bytes but its header "
                f"describes {n} x {h}x{w}x{c} needing {need}. Refusing to read."
            )

        buf = shm.buf
        self._pub_count = np.ndarray((1,), dtype=np.uint64, buffer=buf, offset=_OFF_PUB_COUNT)
        self._slot_ts = np.ndarray(
            (n,), dtype=np.float64, buffer=buf,
            offset=_HEADER_SIZE + _OFF_SLOT_TS, strides=(_SLOT_DESC_SIZE,),
        )
        self._slot_idx = np.ndarray(
            (n,), dtype=np.uint64, buffer=buf,
            offset=_HEADER_SIZE + _OFF_SLOT_IDX, strides=(_SLOT_DESC_SIZE,),
        )
        self._payload = np.ndarray(
            (n, h, w, c), dtype=np.uint8, buffer=buf,
            offset=_HEADER_SIZE + n * _SLOT_DESC_SIZE,
        )

        self._n_slots = n
        self._shape = (h, w, c)
        self._shm = shm
        return True

    # ── public API ───────────────────────────────────────────────────────────

    @property
    def attached(self) -> bool:
        return self._shm is not None

    @property
    def n_slots(self) -> int:
        return self._n_slots

    @property
    def last_frame_idx(self) -> int:
        return self._last_count

    def shape(self) -> Optional[tuple]:
        return self._shape

    def read_latest(self, max_attempts: int = 8) -> tuple[Optional[float], Optional[np.ndarray]]:
        """Return (timestamp, frame_copy), or (None, None) if nothing is published.

        The copy is validated by how far pub_count moved while we were
        copying: if the writer stayed more than n-2 frames behind us it
        cannot have reached our slot, so the bytes we took are intact.
        A failed attempt just re-reads the (now newer) latest frame — there
        is no writer window to wait out.
        """
        if not self._try_attach():
            return None, None

        n = self._n_slots
        for _ in range(max_attempts):
            c1 = int(self._pub_count[0])
            if c1 == 0:
                return None, None              # writer has published nothing

            slot  = (c1 - 1) % n
            ts    = float(self._slot_ts[slot])
            frame = self._payload[slot].copy()  # snapshot out of shared memory

            c2 = int(self._pub_count[0])
            if c2 - c1 <= n - 2:                # writer has not lapped our slot
                self._last_count = c1
                return ts, frame
            # Writer ran away with us mid-copy — take the newer frame instead.

        return None, None

    def read_new(self, max_attempts: int = 8) -> tuple[Optional[float], Optional[np.ndarray]]:
        """Like read_latest but returns (None, None) if the frame hasn't advanced."""
        if not self._try_attach():
            return None, None
        if int(self._pub_count[0]) == self._last_count:
            return None, None
        return self.read_latest(max_attempts=max_attempts)

    def close(self) -> None:
        self._pub_count = None
        self._slot_ts   = None
        self._slot_idx  = None
        self._payload   = None
        self._n_slots   = 0
        self._shape     = None
        if self._shm is not None:
            _close_quietly(self._shm)
            self._shm = None
        # Reader never calls unlink — only the writer owns the region's lifetime.

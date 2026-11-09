"""Cameras: single-owner capture, frame bus publishing, observation merge.

Two capture backends, chosen per camera with ``CameraConfig.type``:

    opencv           UsbCameraCapture   V4L2 / DirectShow through OpenCV, or the Jetson
                                        GStreamer pipeline when ``hw_decode=True`` (falls
                                        back to plain OpenCV if that pipeline fails to open).
    intelrealsense   RealSenseCapture   pyrealsense2 directly (same as the AGV LAB code),
                                        colour and optionally depth (``use_depth=True``);
                                        arrays identical to LeRobot's ``RealSenseCamera``.

Both follow the AGV ``LAB/cameras.py`` design: one background reader per device (the
only process that touches it), a 1-slot latest-frame holder for in-process readers
(teleop, record) and, when ``publish_frames=True``, every frame also goes to the
shared-memory frame bus so any other process (inference, debug tools) can read it
without opening the device:

    lab_<name>          BGR colour   uint8   (H, W, 3)
    lab_<name>_depth    depth in mm  uint16  (H, W, 1)     RealSense with use_depth=True

Observation keys follow LeRobot's robots (``so_follower`` & co.): ``<name>`` is the RGB
image and ``<name>_depth`` the depth map, so ``hw_to_dataset_features`` produces
``observation.images.<name>`` and ``observation.images.<name>_depth`` (flagged
``is_depth_map``) exactly as ``lerobot-record`` would with the cameras inside the robot.

With ``stream.enabled`` the same process also runs the AGV's in-process GstRtspServer
(port 8556) for the dashboard: ``rtsp://<robot pc>:8556/ai`` (``stream.camera``, default
the first camera) and ``/<name>`` for every other colour camera, each
``appsrc → (x264enc | nvv4l2h264enc) → rtph264pay``. MediaMTX pulls it, as on the AGV.
The feeders read the frame bus; a dropped or skipped frame is bridged with the last one
(``stream.max_repeated_frames`` in a row at most), after which the mount stops and logs
which camera and frame bus stalled. Capture never waits on the encoder or a client.

    CamerasManager     starts every enabled camera of the config, and the RTSP server.
    RobotWithCameras   wraps a LeRobot robot built WITHOUT cameras and merges the latest
                       frames into ``get_observation()`` / ``observation_features``.
    RtspServer         dashboard stream, fed from the frame bus.

CLI (same jobs as lerobot-find-cameras):
    python -m src.cameras find                       # OpenCV devices and RealSense serials
    python -m src.cameras find realsense
    python -m src.cameras test --record_time_s=5 --cameras.wrist.source=0 --cameras.eye.source=2
    python -m src.cameras preview --cameras.eye.type=intelrealsense --cameras.eye.source=<serial> --cameras.eye.use_depth=true
    python -m src.cameras preview --cameras.eye.source=0 --stream.enabled=true   # + rtsp://<this pc>:8556/ai
"""

from __future__ import annotations

if __package__ in (None, ""):  # allow `python src/cameras.py`
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "src"

import glob
import logging
import platform
import threading
import time
from typing import Any, Optional

import cv2
import numpy as np

from .config import ArmConfig, CameraConfig, depth_key

logger = logging.getLogger(__name__)

_ROTATIONS = {
    90: cv2.ROTATE_90_CLOCKWISE,
    180: cv2.ROTATE_180,
    270: cv2.ROTATE_90_COUNTERCLOCKWISE,
}

Frame = tuple[Optional[float], Optional[np.ndarray]]
# (timestamp, BGR colour, depth) taken under one lock: always the same frameset.
FramePair = tuple[Optional[float], Optional[np.ndarray], Optional[np.ndarray]]


# ══════════════════════════════════════════════════════════════════════════════
#  Shared capture machinery: 1-slot holder + frame bus publishing
# ══════════════════════════════════════════════════════════════════════════════


class _CaptureBase:
    """Common part of both backends: open on a helper thread (so a wedged device
    cannot hang startup), run the reader on a daemon thread, keep the newest colour
    and depth frame in a lock-protected 1-slot holder and mirror them to the frame
    bus. Consumers never block capture."""

    backend = "none"

    def __init__(self, cfg: CameraConfig) -> None:
        cfg.validate()
        self.name = cfg.name
        self.cfg = cfg
        self.frames_read = 0

        self._frame: Optional[np.ndarray] = None   # BGR uint8
        self._depth: Optional[np.ndarray] = None   # uint16 mm (H, W, 1)
        self._ts: float = 0.0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._initial: Any = None                   # device handle from start()

        self._publish_enabled = bool(cfg.publish_frames)
        self._publishers: dict[str, Any] = {}       # bus name -> FrameBusPublisher
        self._bus_disabled: set[str] = set()

    # ── what this camera contributes to the observation ──────────────────────

    @property
    def use_rgb(self) -> bool:
        return self.cfg.use_rgb

    @property
    def use_depth(self) -> bool:
        return self.cfg.use_depth

    def observation_shapes(self) -> dict[str, tuple[int, int, int]]:
        return self.cfg.observation_shapes()

    @property
    def bus_names(self) -> list[str]:
        return list(self._publishers)

    # ── reading ──────────────────────────────────────────────────────────────

    def read_latest(self) -> Frame:
        """(timestamp, BGR frame) from the 1-slot holder. Non-blocking, not copied."""
        with self._lock:
            if self._frame is None:
                return None, None
            return self._ts, self._frame

    def read_latest_depth(self) -> Frame:
        """(timestamp, depth uint16 (H, W, 1) in mm). Non-blocking, not copied."""
        with self._lock:
            if self._depth is None:
                return None, None
            return self._ts, self._depth

    def read_latest_pair(self) -> FramePair:
        """(timestamp, BGR frame, depth) under ONE lock acquisition, so colour and depth
        come from the same frameset. Either array is None when that stream is off.
        Non-blocking, not copied. Use this, not two separate reads, when both are needed."""
        with self._lock:
            if self._frame is None and self._depth is None:
                return None, None, None
            return self._ts, self._frame, self._depth

    def has_frame(self) -> bool:
        with self._lock:
            return (not self.use_rgb or self._frame is not None) and (
                not self.use_depth or self._depth is not None
            )

    def wait_for_frame(self, timeout_s: float) -> bool:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if self.has_frame():
                return True
            time.sleep(0.02)
        return self.has_frame()

    # ── lifecycle ────────────────────────────────────────────────────────────

    def _open(self) -> Any:
        """Open the device and return a handle. Raise or return None on failure."""
        raise NotImplementedError

    def _run(self) -> None:
        """Reader loop; must poll ``self._stop`` and call ``_store`` per frame."""
        raise NotImplementedError

    def start(self, open_timeout_sec: float | None = None) -> bool:
        """Open the device (on a helper thread) and start the reader. False if unusable."""
        timeout = float(open_timeout_sec or self.cfg.open_timeout_sec)
        result: dict[str, Any] = {}

        def _open() -> None:
            try:
                result["handle"] = self._open()
            except Exception as exc:  # noqa: BLE001
                result["err"] = exc

        opener = threading.Thread(target=_open, daemon=True, name=f"cam-open-{self.name}")
        opener.start()
        opener.join(timeout=max(1.0, timeout))
        if opener.is_alive():
            logger.error("%s: open timed out after %.0fs (%r)", self.name, timeout, self.cfg.source)
            return False
        if "err" in result:
            logger.error("%s: open error: %s", self.name, result["err"])
            return False
        handle = result.get("handle")
        if handle is None:
            logger.error("%s: cannot open %r", self.name, self.cfg.source)
            return False

        self._initial = handle
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True, name=f"cam-{self.name}")
        self._thread.start()
        streams = "+".join(s for s, on in (("rgb", self.use_rgb), ("depth", self.use_depth)) if on)
        logger.info(
            "%s: started %sx%s@%s %s via %s%s",
            self.name, self.cfg.width, self.cfg.height, self.cfg.fps, streams, self.backend,
            " [bus]" if self._publish_enabled else "",
        )
        return True

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None
        with self._lock:
            self._frame = None
            self._depth = None
        for bus, pub in self._publishers.items():
            try:
                pub.close()
            except Exception as exc:  # noqa: BLE001
                logger.warning("%s: publisher %s close error: %s", self.name, bus, exc)
        self._publishers.clear()

    # ── storing + publishing ─────────────────────────────────────────────────

    def _store(self, frame: Optional[np.ndarray], depth: Optional[np.ndarray], now: float) -> None:
        with self._lock:  # in-process consumers (teleop / record)
            if frame is not None:
                self._frame = frame
            if depth is not None:
                self._depth = depth
            self._ts = now
            self.frames_read += 1

        if not self._publish_enabled:  # out-of-process consumers (inference)
            return
        if frame is not None:
            self._publish(self.name, frame, now)
        if depth is not None:
            self._publish(depth_key(self.name), depth, now)

    def _publish(self, bus: str, array: np.ndarray, now: float) -> None:
        if bus in self._bus_disabled:
            return
        pub = self._publishers.get(bus)
        if pub is None:
            if array.ndim != 3:
                logger.warning("%s: bus %s disabled, unexpected ndim=%s", self.name, bus, array.ndim)
                self._bus_disabled.add(bus)
                return
            try:
                from .frame_bus import FrameBusPublisher

                h, w, c = array.shape
                pub = FrameBusPublisher(bus, height=h, width=w, channels=c, dtype=array.dtype)
                self._publishers[bus] = pub
                logger.info("%s: frame bus → %s (%sx%sx%s %s)", self.name, pub.name, w, h, c, array.dtype)
            except Exception as exc:  # noqa: BLE001
                logger.error("%s: frame bus %s init failed: %s", self.name, bus, exc)
                self._bus_disabled.add(bus)
                return
        try:
            pub.publish(np.ascontiguousarray(array), now)
        except Exception as exc:  # noqa: BLE001
            logger.error("%s: bus %s publish error: %s (bus disabled)", self.name, bus, exc)
            try:
                pub.close()
            except Exception:  # noqa: BLE001
                pass
            self._publishers.pop(bus, None)
            self._bus_disabled.add(bus)


# ══════════════════════════════════════════════════════════════════════════════
#  OpenCV / GStreamer capture — sole owner of the device
# ══════════════════════════════════════════════════════════════════════════════


class UsbCameraCapture(_CaptureBase):
    """Background reader for one OpenCV-accessible camera (USB/UVC)."""

    def _gst_pipeline(self) -> str:
        cfg = self.cfg
        fmt = cfg.pixel_format.upper()
        # io-mode=2 = GST_V4L2_IO_MMAP, the explicit, reliable choice for UVC devices.
        if fmt == "YUYV":
            return (
                f"v4l2src device={cfg.source} io-mode=2 ! "
                f"video/x-raw,format=YUY2,width={cfg.width},height={cfg.height},framerate={cfg.fps}/1 ! "
                "nvvidconv ! video/x-raw,format=BGRx ! videoconvert ! video/x-raw,format=BGR ! "
                "appsink drop=true max-buffers=1 sync=false"
            )
        return (
            f"v4l2src device={cfg.source} io-mode=2 ! "
            f"image/jpeg,width={cfg.width},height={cfg.height},framerate={cfg.fps}/1 ! "
            "nvv4l2decoder mjpeg=1 ! nvvidconv ! video/x-raw,format=BGRx ! "
            "videoconvert ! video/x-raw,format=BGR ! "
            "appsink drop=true max-buffers=1 sync=false"
        )

    def _open(self) -> Optional[cv2.VideoCapture]:
        cfg = self.cfg
        if cfg.hw_decode:
            pipeline = self._gst_pipeline()
            logger.info("%s: gst → %s", self.name, pipeline)
            cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
            if cap.isOpened():
                self.backend = "gstreamer"
                return cap
            cap.release()
            logger.warning("%s: hardware-decode pipeline failed to open, falling back to OpenCV", self.name)

        api = cv2.CAP_V4L2 if platform.system() == "Linux" else cv2.CAP_ANY
        cap = cv2.VideoCapture(cfg.source, api)
        if not cap.isOpened():
            cap.release()
            return None
        if cfg.pixel_format:
            try:
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*cfg.pixel_format.upper()[:4]))
            except Exception:  # noqa: BLE001
                pass
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.height)
        cap.set(cv2.CAP_PROP_FPS, cfg.fps)
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:  # noqa: BLE001
            pass
        self.backend = "opencv"
        return cap

    def _run(self) -> None:
        backoff = 1.0
        rotate = _ROTATIONS.get(int(self.cfg.rotation or 0))
        while not self._stop.is_set():
            cap = self._initial
            self._initial = None
            if cap is None:
                try:
                    cap = self._open()
                except Exception as exc:  # noqa: BLE001
                    logger.warning("%s: reopen failed: %s", self.name, exc)
                    cap = None
            if cap is None:
                self._stop.wait(timeout=backoff)
                backoff = min(backoff * 2, 10.0)
                continue
            backoff = 1.0

            while not self._stop.is_set():
                ok, frame = cap.read()
                if not ok or frame is None:
                    break
                if rotate is not None:
                    frame = cv2.rotate(frame, rotate)
                self._store(frame, None, time.time())

            cap.release()
            if not self._stop.is_set():
                logger.warning("%s: stream lost, reconnecting in %.1fs", self.name, backoff)
                self._stop.wait(timeout=backoff)
                backoff = min(backoff * 2, 10.0)


# ══════════════════════════════════════════════════════════════════════════════
#  Intel RealSense capture — pyrealsense2 directly (AGV LAB style), colour + depth
# ══════════════════════════════════════════════════════════════════════════════

# LeRobot's depth contract: uint16 where one unit is 1 mm (z16 at depth_units = 0.001 m).
_DEPTH_UNIT_M = 0.001


def _import_rs() -> Any:
    try:
        import pyrealsense2 as rs  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError(
            "pyrealsense2 is not installed (pip install pyrealsense2; pyrealsense2-macosx on macOS)"
        ) from exc
    return rs


def _rs_valid(frame: Any) -> bool:
    """pyrealsense2 frames are falsy when that stream is missing from the frameset."""
    try:
        return bool(frame)
    except Exception:  # noqa: BLE001
        return frame is not None


def resolve_realsense_serial(rs: Any, source: int | str) -> str:
    """Serial number for ``source``: digits are a serial number, anything else a unique
    device name (same rule as LeRobot's ``RealSenseCamera``)."""
    src = str(source).strip()
    if src.isdigit():
        return src
    found = [(d.get_info(rs.camera_info.name), d.get_info(rs.camera_info.serial_number))
             for d in rs.context().query_devices()]
    matches = [sn for name, sn in found if name == src]
    if not matches:
        raise RuntimeError(f"no RealSense named {src!r}; connected: {found}")
    if len(matches) > 1:
        raise RuntimeError(f"several RealSense named {src!r} ({matches}); use the serial number")
    return matches[0]


class RealSenseCapture(_CaptureBase):
    """Background reader for one Intel RealSense (D4xx / D5xx), straight on pyrealsense2.

    Same design as the AGV ``LAB/cameras.py`` RealSenseCapture and the OpenCV path above:
    this thread is the only owner of the device; every frameset goes to the 1-slot holder
    (teleop / record, read with ``read_latest_pair``) and, with ``publish_frames=True``, to
    the shared-memory frame bus so any other process reads it without opening the camera:

        lab_<name>          BGR colour     uint8   (H, W, 3)
        lab_<name>_depth    depth in mm    uint16  (H, W, 1)

    The arrays match LeRobot's ``RealSenseCamera`` so the dataset is the one
    ``lerobot-record`` would write: colour and depth are enabled at the same
    width/height/fps and taken from ONE frameset (colour N and depth N belong together);
    depth is the z16 map, NOT aligned to colour (LeRobot does not align), uint16 mm,
    shape (H, W, 1); a warm-up of at least 1 s runs before the first frame counts;
    rotation is applied after capture to both streams. Colour stays BGR on the bus;
    ``RobotWithCameras`` converts it to RGB for the observation, as LeRobot does.

    Depth units: LeRobot passes the raw z16 values through and calls them millimetres,
    true at the D4xx default of 1 mm. A D405 ships at 0.1 mm, so the sensor is asked for
    1 mm units before streaming, and if it refuses the values are rescaled here. Either
    way ``<name>_depth`` really is millimetres.
    """

    backend = "realsense"
    lost_after_s = 10.0                  # no complete frameset for this long → reconnect

    def __init__(self, cfg: CameraConfig) -> None:
        super().__init__(cfg)
        self.serial: Optional[str] = None
        self._mm_per_unit = 1.0          # raw z16 → mm; 1.0 when the sensor runs at 1 mm units

    # ── device setup ─────────────────────────────────────────────────────────

    def _request_mm_units(self, rs: Any, serial: str) -> None:
        """Ask the depth sensor for 1 mm units before streaming. Best effort."""
        for dev in rs.context().query_devices():
            if dev.get_info(rs.camera_info.serial_number) != serial:
                continue
            try:
                sensor = dev.first_depth_sensor()
                opt = rs.option.depth_units
                if sensor.supports(opt) and not sensor.is_option_read_only(opt):
                    if abs(float(sensor.get_option(opt)) - _DEPTH_UNIT_M) > 1e-9:
                        sensor.set_option(opt, _DEPTH_UNIT_M)
            except Exception as exc:  # noqa: BLE001
                logger.warning("%s: could not set depth units to 1 mm: %s", self.name, exc)
            return

    def _complete(self, frames: Any) -> bool:
        """True when every enabled stream is present in the frameset."""
        if frames is None:
            return False
        if self.use_rgb and not _rs_valid(frames.get_color_frame()):
            return False
        if self.use_depth and not _rs_valid(frames.get_depth_frame()):
            return False
        return True

    def _open(self) -> Any:
        """Start the pipeline, check the negotiated modes and depth unit, run the warm-up.
        Returns the running ``rs.pipeline``; raises on any failure."""
        rs = _import_rs()
        cfg = self.cfg
        serial = resolve_realsense_serial(rs, cfg.source)
        if cfg.use_depth:
            self._request_mm_units(rs, serial)

        rs_cfg = rs.config()
        rs_cfg.enable_device(serial)
        # cfg.width x cfg.height is the capture size; rotation is applied afterwards.
        if cfg.use_rgb:
            rs_cfg.enable_stream(rs.stream.color, cfg.width, cfg.height, rs.format.bgr8, cfg.fps)
        if cfg.use_depth:
            rs_cfg.enable_stream(rs.stream.depth, cfg.width, cfg.height, rs.format.z16, cfg.fps)

        pipe = rs.pipeline()
        profile = pipe.start(rs_cfg)
        try:
            for stream, on in ((rs.stream.color, cfg.use_rgb), (rs.stream.depth, cfg.use_depth)):
                if not on:
                    continue
                vsp = profile.get_stream(stream).as_video_stream_profile()
                if (vsp.width(), vsp.height(), vsp.fps()) != (cfg.width, cfg.height, cfg.fps):
                    raise RuntimeError(
                        f"{stream} negotiated {vsp.width()}x{vsp.height()}@{vsp.fps()}, "
                        f"config asks {cfg.width}x{cfg.height}@{cfg.fps}"
                    )

            self._mm_per_unit = 1.0
            if cfg.use_depth:
                scale_m = float(profile.get_device().first_depth_sensor().get_depth_scale())
                factor = scale_m / _DEPTH_UNIT_M
                if abs(factor - 1.0) > 1e-6:
                    self._mm_per_unit = factor
                    logger.warning("%s: depth unit is %.4g mm, rescaling to mm in software",
                                   self.name, factor)

            # Warm-up like LeRobot (>= 1 s): the first framesets after start are often incomplete.
            warm = max(1.0, float(cfg.warmup_s))
            deadline = time.monotonic() + warm
            got = False
            while time.monotonic() < deadline:
                ok, frames = pipe.try_wait_for_frames(timeout_ms=int(warm * 1000) + 1000)
                got = got or (ok and self._complete(frames))
            if not got:
                raise RuntimeError(f"no complete frameset during the {warm:.0f}s warm-up")
        except Exception:
            try:
                pipe.stop()
            except Exception:  # noqa: BLE001
                pass
            raise

        self.serial = serial
        return pipe

    def _depth_mm(self, raw: np.ndarray) -> np.ndarray:
        if self._mm_per_unit == 1.0:
            return raw
        return np.rint(raw.astype(np.float32) * self._mm_per_unit).clip(0, 65535).astype(np.uint16)

    # ── reader loop ──────────────────────────────────────────────────────────

    def _run(self) -> None:
        backoff = 1.0
        rotate = _ROTATIONS.get(int(self.cfg.rotation or 0))
        while not self._stop.is_set():
            pipe = self._initial
            self._initial = None
            if pipe is None:
                try:
                    pipe = self._open()
                except Exception as exc:  # noqa: BLE001
                    logger.warning("%s: RealSense reopen failed: %s", self.name, exc)
                    pipe = None
            if pipe is None:
                self._stop.wait(timeout=backoff)
                backoff = min(backoff * 2, 10.0)
                continue
            backoff = 1.0

            last_ok = time.monotonic()
            while not self._stop.is_set():
                try:
                    ok, frames = pipe.try_wait_for_frames(timeout_ms=500)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("%s: wait_for_frames raised: %s", self.name, exc)
                    break
                if not ok or not self._complete(frames):
                    if time.monotonic() - last_ok > self.lost_after_s:
                        break
                    continue
                last_ok = time.monotonic()

                # Copy out: librealsense recycles these buffers once the frameset is released.
                frame = depth = None
                if self.use_rgb:
                    frame = np.array(frames.get_color_frame().get_data(), dtype=np.uint8, copy=True)
                    if rotate is not None:
                        frame = cv2.rotate(frame, rotate)
                if self.use_depth:
                    depth = np.array(frames.get_depth_frame().get_data(), dtype=np.uint16, copy=True)
                    depth = self._depth_mm(depth)
                    if rotate is not None:
                        depth = cv2.rotate(depth, rotate)
                    depth = depth[..., np.newaxis]            # (H, W) → (H, W, 1), LeRobot layout
                # One _store per frameset: holder and frame bus get colour and depth together.
                self._store(frame, depth, time.time())

            try:
                pipe.stop()
            except Exception:  # noqa: BLE001
                pass
            if not self._stop.is_set():
                logger.warning("%s: RealSense stream lost, reconnecting in %.1fs", self.name, backoff)
                self._stop.wait(timeout=backoff)
                backoff = min(backoff * 2, 10.0)


def make_capture(cfg: CameraConfig) -> _CaptureBase:
    if cfg.is_realsense:
        return RealSenseCapture(cfg)
    return UsbCameraCapture(cfg)


# ══════════════════════════════════════════════════════════════════════════════
#  RTSP server for the dashboard — the AGV GstRtspServer, fed from the frame bus
# ══════════════════════════════════════════════════════════════════════════════


class RtspServer:
    """GstRtspServer on a background GLib thread (the AGV ``LAB/cameras.py`` ``RtspServer``).

    Mounts come from ``ArmConfig.stream_mounts()``:

        /<stream.mount>   ``stream.camera`` (default the first camera)   ← MediaMTX pulls this
        /<name>           every other colour camera

    Each mount is ``appsrc → (x264enc | nvv4l2h264enc) → rtph264pay``, shared by all
    clients and only built while a client pulls it. Its feeder (``_BusAppsrcFeeder``)
    reads the camera's frame bus, so streaming never touches the capture thread.
    """

    def __init__(self, cfg: ArmConfig) -> None:
        self._cfg = cfg
        self._stream = cfg.stream
        self._mounts: dict[str, CameraConfig] = {}
        self._feeders: set[_BusAppsrcFeeder] = set()
        self._feeders_lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._loop: Any = None                      # GLib.MainLoop
        self._server: Any = None                    # GstRtspServer.RTSPServer
        self._stop = threading.Event()

    @property
    def mounts(self) -> dict[str, CameraConfig]:
        return dict(self._mounts)

    # ── lifecycle ────────────────────────────────────────────────────────────

    def start(self) -> bool:
        """Start the server thread and the CLOSE_WAIT monitor. False when the config
        cannot be streamed (logged); a missing GStreamer is logged by the server thread."""
        try:
            self._mounts = self._cfg.stream_mounts()
        except ValueError as exc:
            logger.error("rtsp: %s; stream disabled", exc)
            return False
        self._thread = threading.Thread(target=self._run, daemon=True, name="rtsp-server")
        self._thread.start()
        threading.Thread(target=self._watchdog_loop, daemon=True, name="rtsp-watchdog").start()
        return True

    def stop(self) -> None:
        self._stop.set()
        with self._feeders_lock:
            feeders = list(self._feeders)
            self._feeders.clear()
        for feeder in feeders:
            feeder.stop()
        if self._loop is not None:
            try:
                self._loop.quit()
            except Exception:  # noqa: BLE001
                pass

    # ── GLib main loop ───────────────────────────────────────────────────────

    def _run(self) -> None:
        try:
            import gi

            gi.require_version("Gst", "1.0")
            gi.require_version("GstRtspServer", "1.0")
            from gi.repository import GLib, Gst, GstRtspServer
        except Exception as exc:  # noqa: BLE001
            logger.error("rtsp: GStreamer RTSP server unavailable (%s); stream disabled. "
                         "It needs PyGObject and gst-rtsp-server (Linux / Jetson).", exc)
            return

        s = self._stream
        Gst.init(None)
        # RTSPServer(), not RTSPServer.new(): the AGV construction known to work with MediaMTX pulls.
        self._server = GstRtspServer.RTSPServer()
        if s.bind and s.bind not in ("0.0.0.0", ""):
            self._server.set_address(s.bind)
        self._server.set_service(str(s.port))
        mount_points = self._server.get_mount_points()

        encoder = "nvv4l2h264enc" if s.hw_encode else "x264enc"
        for mount, cam in self._mounts.items():
            factory = GstRtspServer.RTSPMediaFactory()
            factory.set_launch(self._pipeline(cam))
            factory.set_shared(True)
            factory.connect("media-configure", self._on_media_configure, mount, cam)
            mount_points.add_factory(f"/{mount}", factory)
            logger.info("rtsp: mount /%s ← camera %s (frame bus lab_%s, %s %s kbps)",
                        mount, cam.name, cam.name, encoder, s.bitrate_kbps)

        if not self._server.attach(None):
            logger.error("rtsp: cannot listen on %s:%s (port in use?); stream disabled", s.bind, s.port)
            return
        logger.info("rtsp: listening on %s:%s", s.bind, s.port)

        # GstRtspServer does not expire dead sessions on its own.
        GLib.timeout_add_seconds(20, self._cleanup_sessions)

        self._loop = GLib.MainLoop()
        if self._stop.is_set():
            return
        try:
            self._loop.run()
        except Exception as exc:  # noqa: BLE001
            logger.error("rtsp: main loop crashed: %s", exc)

    def _cleanup_sessions(self) -> bool:
        try:
            self._server.get_session_pool().cleanup()
        except Exception:  # noqa: BLE001
            pass
        return True

    # ── pipeline ─────────────────────────────────────────────────────────────

    @staticmethod
    def _caps(cam: CameraConfig) -> str:
        h, w, _ = cam.frame_shape
        return f"video/x-raw,format=BGR,width={w},height={h},framerate={cam.fps}/1"

    def _pipeline(self, cam: CameraConfig) -> str:
        """The AGV USB pipeline: SPS/PPS every second (``config-interval=1``) so MediaMTX
        can join mid-stream, byte-stream caps so h264parse / rtph264pay never renegotiate,
        and a leaky queue so a slow client cannot back-pressure the encoder."""
        s = self._stream
        if s.hw_encode:
            # Jetson NVENC: BGR → NV12 → NVMM. insert-sps-pps=1 puts SPS/PPS on every IDR.
            encoder = (
                "videoconvert ! video/x-raw,format=NV12 ! "
                "nvvidconv ! video/x-raw(memory:NVMM),format=NV12 ! "
                f"nvv4l2h264enc bitrate={int(s.bitrate_kbps) * 1000} "
                "preset-level=1 profile=0 insert-sps-pps=1 iframeinterval=30 "
            )
        else:
            encoder = (
                "videoconvert ! "
                "x264enc tune=zerolatency speed-preset=superfast "
                f"bitrate={int(s.bitrate_kbps)} key-int-max=30 byte-stream=true ! "
                "video/x-h264,profile=baseline,stream-format=byte-stream,alignment=au "
            )
        return (
            "appsrc name=bus_src is-live=true format=time do-timestamp=true block=false "
            f"! {self._caps(cam)} "
            f"! {encoder}"
            "! h264parse config-interval=1 "
            "! queue leaky=downstream max-size-buffers=4 "
            "! rtph264pay name=pay0 pt=96 config-interval=1"
        )

    def _on_media_configure(self, _factory: Any, media: Any, mount: str, cam: CameraConfig) -> None:
        """Start a feeder whenever a media pipeline is built (shared: one per pull, rebuilt
        after the last client leaves) and stop it when that pipeline is torn down. The caps
        also go on the appsrc element: some GStreamer versions need them there."""
        appsrc = media.get_element().get_by_name("bus_src")
        if appsrc is None:
            logger.error("rtsp /%s: appsrc not found in the pipeline", mount)
            return
        try:
            from gi.repository import Gst

            appsrc.set_property("caps", Gst.Caps.from_string(self._caps(cam)))
            appsrc.set_property("format", Gst.Format.TIME)
            appsrc.set_property("is-live", True)
            appsrc.set_property("do-timestamp", True)
            appsrc.set_property("block", False)
        except Exception as exc:  # noqa: BLE001
            logger.warning("rtsp /%s: appsrc property set failed: %s", mount, exc)

        feeder = _BusAppsrcFeeder(mount, cam, appsrc, self._stream.max_repeated_frames)
        with self._feeders_lock:
            if self._stop.is_set():
                return
            self._feeders.add(feeder)
        feeder.start()

        def _unprepared(_media: Any) -> None:
            with self._feeders_lock:
                self._feeders.discard(feeder)
            feeder.stop()

        media.connect("unprepared", _unprepared)

    # ── CLOSE_WAIT monitor ───────────────────────────────────────────────────

    def _watchdog_loop(self) -> None:
        """The AGV CLOSE_WAIT check, logging only (see ``StreamConfig.close_wait_max``)."""
        s = self._stream
        warned = False
        while not self._stop.wait(timeout=max(1.0, float(s.close_wait_interval_s))):
            stuck = _count_close_wait(int(s.port))
            over = stuck > s.close_wait_max
            if over and not warned:
                logger.error("rtsp: %d CLOSE_WAIT connections on :%s, the server may be wedged. "
                             "Restart teleop between sessions to clear it.", stuck, s.port)
            warned = over


class _BusAppsrcFeeder:
    """Pushes one camera's frame bus into a mount's appsrc at the camera's fps (the AGV
    ``_UsbAppsrcFeeder``, reading ``lab_<name>`` instead of the in-process holder).

    Every tick sends the newest frame. When the camera dropped or skipped one,
    ``HoldLastReader`` hands back the previous frame so the stream keeps its rate; after
    ``max_repeats`` repeats in a row without a fresh frame the feeder stops and logs the
    camera and frame bus. appsrc is non-blocking: when the encoder falls behind the frame
    is dropped there, and capture never waits.
    """

    def __init__(self, mount: str, cam: CameraConfig, appsrc: Any, max_repeats: int) -> None:
        self.mount = mount
        self._cam = cam
        self._appsrc = appsrc
        self._max_repeats = int(max_repeats)
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True, name=f"rtsp-feed-{self.mount}")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        try:
            self._appsrc.emit("end-of-stream")
        except Exception:  # noqa: BLE001
            pass
        thread, self._thread = self._thread, None
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=1.0)

    def _run(self) -> None:
        try:
            from gi.repository import Gst
        except Exception:  # noqa: BLE001
            return
        from .frame_bus import FrameBusStalled, HoldLastReader

        cam = self._cam
        h, w, _ = cam.frame_shape
        period = 1.0 / max(1.0, float(cam.fps))
        bus = HoldLastReader(cam.name, self._max_repeats)
        started = time.monotonic()
        waiting_logged = pushed_logged = push_error_logged = resize_logged = False
        try:
            while not self._stop.is_set():
                t0 = time.monotonic()
                try:
                    _, frame = bus.read()
                except FrameBusStalled as exc:
                    logger.error("rtsp /%s: %s Stream stopped.", self.mount, exc)
                    return
                if frame is None:
                    if not waiting_logged and t0 - started > 5.0:
                        logger.warning("rtsp /%s: no frame on frame bus %s yet (camera %s)",
                                       self.mount, bus.region, cam.name)
                        waiting_logged = True
                else:
                    if frame.shape[:2] != (h, w):
                        if not resize_logged:
                            logger.warning("rtsp /%s: frame bus %s is %sx%s, stream is %sx%s; resizing",
                                           self.mount, bus.region, frame.shape[1], frame.shape[0], w, h)
                            resize_logged = True
                        frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
                    try:
                        ret = self._appsrc.emit("push-buffer", Gst.Buffer.new_wrapped(frame.tobytes()))
                        if ret == Gst.FlowReturn.OK:
                            if not pushed_logged:
                                logger.info("rtsp /%s: first frame pushed", self.mount)
                                pushed_logged = True
                        elif not push_error_logged and not self._stop.is_set():
                            # Encoder or client in trouble: the frame is dropped; say so once.
                            logger.warning("rtsp /%s: push-buffer returned %s", self.mount, ret.value_nick)
                            push_error_logged = True
                    except Exception as exc:  # noqa: BLE001
                        if not push_error_logged:
                            logger.warning("rtsp /%s: push-buffer failed: %s", self.mount, exc)
                            push_error_logged = True
                self._stop.wait(timeout=max(0.0, period - (time.monotonic() - t0)))
        finally:
            bus.close()


def _count_close_wait(port: int) -> int:
    """Sockets in CLOSE_WAIT on ``port`` (Linux /proc; 0 elsewhere). Same as the AGV."""
    port_hex = f"{port:04X}"
    count = 0
    for proc_file in ("/proc/net/tcp", "/proc/net/tcp6"):
        try:
            with open(proc_file) as f:
                next(f)
                for line in f:
                    fields = line.split()
                    local_port = fields[1].split(":")[1]
                    state = fields[3]
                    if local_port == port_hex and state == "08":
                        count += 1
        except Exception:  # noqa: BLE001
            continue
    return count


# ══════════════════════════════════════════════════════════════════════════════
#  CamerasManager
# ══════════════════════════════════════════════════════════════════════════════


class CamerasManager:
    """Starts and owns every enabled camera of the config, and the RTSP server when
    ``stream.enabled``."""

    def __init__(self, cfg: ArmConfig) -> None:
        self._cfg = cfg
        self.captures: dict[str, _CaptureBase] = {}
        self.failed: list[str] = []
        self.rtsp: Optional[RtspServer] = None

    @property
    def expected_names(self) -> list[str]:
        return self._cfg.camera_names

    def start(self) -> list[str]:
        for cam in self._cfg.enabled_cameras:
            try:
                cap = make_capture(cam)
            except Exception as exc:  # noqa: BLE001 (bad config)
                logger.error("%s: %s", cam.name, exc)
                self.failed.append(cam.name)
                continue
            if cap.start():
                self.captures[cam.name] = cap
            else:
                self.failed.append(cam.name)
        if self._cfg.stream.enabled:
            server = RtspServer(self._cfg)
            if server.start():
                self.rtsp = server
        return list(self.captures)

    def stop(self) -> None:
        if self.rtsp is not None:  # feeders first: they read the buses the captures own
            self.rtsp.stop()
            self.rtsp = None
        for cap in self.captures.values():
            cap.stop()
        self.captures.clear()

    def read(self, name: str) -> Frame:
        cap = self.captures.get(name)
        return (None, None) if cap is None else cap.read_latest()

    def read_depth(self, name: str) -> Frame:
        cap = self.captures.get(name)
        return (None, None) if cap is None else cap.read_latest_depth()

    def read_pair(self, name: str) -> FramePair:
        """(ts, colour, depth) from one snapshot of the camera's holder."""
        cap = self.captures.get(name)
        return (None, None, None) if cap is None else cap.read_latest_pair()

    def names(self) -> list[str]:
        return list(self.captures)

    def shapes(self) -> dict[str, tuple[int, int, int]]:
        """Configured observation shapes per running camera: ``<name>`` → (H, W, 3) and,
        with depth, ``<name>_depth`` → (H, W, 1). This is what goes into the features."""
        out: dict[str, tuple[int, int, int]] = {}
        for cap in self.captures.values():
            out.update(cap.observation_shapes())
        return out

    def wait_ready(self, timeout_s: float) -> list[str]:
        """Block until every camera delivered its frames; return the names that did not."""
        deadline = time.monotonic() + timeout_s
        missing = []
        for name, cap in self.captures.items():
            if not cap.wait_for_frame(max(0.0, deadline - time.monotonic())):
                missing.append(name)
        return missing


# ══════════════════════════════════════════════════════════════════════════════
#  RobotWithCameras — merge frames into a LeRobot robot's observation
# ══════════════════════════════════════════════════════════════════════════════


class RobotWithCameras:
    """A LeRobot robot (built without cameras) plus this module's cameras.

    Exposes the members the LeRobot scripts touch (``observation_features``,
    ``action_features``, ``cameras``, ``name``, ``robot_type``, ``connect`` ...) and
    forwards everything else to the wrapped robot. Observation keys match what a
    LeRobot robot with the same cameras would produce: ``<name>`` RGB uint8 (H, W, 3)
    and ``<name>_depth`` uint16 mm (H, W, 1).
    """

    def __init__(
        self,
        robot,
        cameras: CamerasManager,
        stale_s: float = 0.5,
        ready_timeout_s: float = 10.0,
    ) -> None:
        self._robot = robot
        self._cams = cameras
        self._stale_s = float(stale_s)
        self._ready_timeout_s = float(ready_timeout_s)
        self._warned_stale: set[str] = set()
        self._warned_resize: set[str] = set()

    # -- LeRobot-facing surface ---------------------------------------------

    @property
    def cameras(self) -> dict[str, _CaptureBase]:
        return dict(self._cams.captures)

    @property
    def observation_features(self) -> dict[str, Any]:
        return {**self._robot.observation_features, **self._cams.shapes()}

    @property
    def action_features(self) -> dict[str, Any]:
        return self._robot.action_features

    @property
    def name(self) -> str:
        return self._robot.name

    @property
    def robot_type(self) -> str:
        return self._robot.robot_type

    @property
    def id(self):
        return self._robot.id

    @property
    def is_connected(self) -> bool:
        return self._robot.is_connected

    @property
    def is_calibrated(self) -> bool:
        return self._robot.is_calibrated

    def connect(self, calibrate: bool = True) -> None:
        missing = [n for n in self._cams.expected_names if n not in self._cams.captures]
        if missing:
            raise RuntimeError(f"cameras failed to open: {missing}. Fix the config or disable them.")
        self._robot.connect(calibrate)
        not_ready = self._cams.wait_ready(self._ready_timeout_s)
        if not_ready:
            self._robot.disconnect()
            raise RuntimeError(f"cameras delivered no frame within {self._ready_timeout_s:.0f}s: {not_ready}")

    def disconnect(self) -> None:
        self._robot.disconnect()

    def calibrate(self) -> None:
        self._robot.calibrate()

    def configure(self) -> None:
        self._robot.configure()

    def _check_age(self, key: str, ts: Optional[float], now: float) -> None:
        age = now - float(ts or 0.0)
        if age > self._stale_s:
            if key not in self._warned_stale:
                logger.warning("camera %s frame is %.0f ms old", key, age * 1e3)
                self._warned_stale.add(key)
        else:
            self._warned_stale.discard(key)

    def _fit(self, key: str, image: np.ndarray, shape: tuple[int, int, int]) -> np.ndarray:
        h, w, _ = shape
        if image.shape[0] == h and image.shape[1] == w:
            return image
        if key not in self._warned_resize:
            logger.warning("camera %s delivers %sx%s but config says %sx%s; resizing",
                           key, image.shape[1], image.shape[0], w, h)
            self._warned_resize.add(key)
        depth = image.ndim == 3 and image.shape[2] == 1
        interp = cv2.INTER_NEAREST if depth else cv2.INTER_AREA
        out = cv2.resize(image, (w, h), interpolation=interp)
        return out[..., np.newaxis] if depth and out.ndim == 2 else out

    def get_observation(self) -> dict[str, Any]:
        obs = self._robot.get_observation()
        now = time.time()
        for name, cap in self._cams.captures.items():
            shapes = cap.observation_shapes()
            # One snapshot of the holder: colour N and depth N from the same frameset,
            # so a recorded frame never mixes two capture instants.
            ts, bgr, dmap = cap.read_latest_pair()
            if cap.use_rgb:
                if bgr is None:
                    raise RuntimeError(f"camera {name!r} has no frame")
                self._check_age(name, ts, now)
                bgr = self._fit(name, bgr, shapes[name])
                obs[name] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)  # LeRobot convention is RGB
            if cap.use_depth:
                key = depth_key(name)
                if dmap is None:
                    raise RuntimeError(f"camera {name!r} has no depth frame")
                self._check_age(key, ts, now)
                obs[key] = self._fit(key, dmap, shapes[key])  # uint16 mm, (H, W, 1)
        return obs

    def send_action(self, action: dict[str, float]) -> dict[str, float]:
        return self._robot.send_action(action)

    def __getattr__(self, item: str) -> Any:
        return getattr(self._robot, item)


# ══════════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════════


def colorize_depth(depth: np.ndarray, max_mm: float = 4000.0) -> np.ndarray:
    """uint16 mm depth → BGR colour map for display (0 .. max_mm)."""
    d = depth[..., 0] if depth.ndim == 3 else depth
    scaled = cv2.convertScaleAbs(d, alpha=255.0 / max_mm)
    return cv2.applyColorMap(scaled, cv2.COLORMAP_JET)


def find_opencv_cameras() -> None:
    """Probe candidate OpenCV devices and print their default modes."""
    if platform.system() == "Linux":
        targets: list[int | str] = sorted(glob.glob("/dev/video*"))
        api = cv2.CAP_V4L2
    else:
        targets = list(range(10))
        api = cv2.CAP_ANY
    print(f"--- OpenCV cameras (type=opencv) ---\n{'source':<14} {'default mode':<16} note")
    found = 0
    for target in targets:
        cap = cv2.VideoCapture(target, api)
        if not cap.isOpened():
            cap.release()
            continue
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        ok, _ = cap.read()
        cap.release()
        found += 1
        print(f"{str(target):<14} {f'{w}x{h}@{fps:.0f}':<16} {'ok' if ok else 'opened but no frame (metadata node?)'}")
    if not found:
        print("none")
    print()


def _rs_info(dev: Any, key: Any) -> str:
    try:
        return str(dev.get_info(key)) if dev.supports(key) else "?"
    except Exception:  # noqa: BLE001
        return "?"


def find_realsense_cameras() -> None:
    """List RealSense devices: serial, name, firmware, USB type and the default colour /
    depth modes (what ``lerobot-find-cameras realsense`` prints), straight from pyrealsense2."""
    print("--- RealSense cameras (type=intelrealsense, source=<serial number>) ---")
    try:
        rs = _import_rs()
    except RuntimeError as exc:
        print(f"{exc}; skipping.\n")
        return
    try:
        devices = list(rs.context().query_devices())
    except Exception as exc:  # noqa: BLE001
        print(f"RealSense search failed: {exc}\n")
        return
    if not devices:
        print("none\n")
        return
    for dev in devices:
        modes = []
        try:
            for sensor in dev.query_sensors():
                for prof in sensor.get_stream_profiles():
                    if prof.is_video_stream_profile() and prof.is_default():
                        v = prof.as_video_stream_profile()
                        modes.append(f"{v.stream_name()} {v.width()}x{v.height()}@{v.fps()} {v.format().name}")
        except Exception as exc:  # noqa: BLE001
            modes.append(f"profiles unavailable ({exc})")
        print(f"serial {_rs_info(dev, rs.camera_info.serial_number)}  "
              f"name {_rs_info(dev, rs.camera_info.name)!r}  "
              f"fw {_rs_info(dev, rs.camera_info.firmware_version)}  "
              f"usb {_rs_info(dev, rs.camera_info.usb_type_descriptor)}")
        print(f"    default: {', '.join(modes) or '?'}")
    print("\nUse: --cameras.<name>.type=intelrealsense --cameras.<name>.source=<serial> [--cameras.<name>.use_depth=true]\n")


def find_cameras(camera_type: str | None = None) -> None:
    if camera_type in (None, "opencv"):
        find_opencv_cameras()
    if camera_type in (None, "realsense"):
        find_realsense_cameras()


def test_cameras(cfg: ArmConfig, seconds: float) -> None:
    cams = CamerasManager(cfg)
    cams.start()
    if cams.failed:
        print(f"failed to open: {cams.failed}")
    if not cams.captures:
        return
    missing = cams.wait_ready(5.0)
    if missing:
        print(f"no frames within 5s from: {missing}")
    start_counts = {n: c.frames_read for n, c in cams.captures.items()}
    t0 = time.perf_counter()
    time.sleep(seconds)
    elapsed = time.perf_counter() - t0
    print(f"\n{'camera':<8} {'backend':<10} {'fps':>6}  {'colour (BGR)':<16} {'depth (mm)':<16} bus")
    for name, cap in cams.captures.items():
        fps = (cap.frames_read - start_counts[name]) / elapsed
        _, frame = cap.read_latest()
        _, dmap = cap.read_latest_depth()
        shape = str(tuple(frame.shape)) if frame is not None else "-"
        dshape = str(tuple(dmap.shape)) if dmap is not None else "-"
        flag = "" if fps >= 0.9 * cap.cfg.fps else "   <- below configured fps"
        bus = ", ".join(cap.bus_names) or "no"
        print(f"{name:<8} {cap.backend:<10} {fps:>6.1f}  {shape:<16} {dshape:<16} {bus}{flag}")
    cams.stop()


def preview(cfg: ArmConfig) -> None:
    cams = CamerasManager(cfg)
    cams.start()
    if not cams.captures:
        print("No cameras opened.")
        return
    try:
        while True:
            for name, cap in cams.captures.items():
                _, bgr = cap.read_latest()
                if bgr is not None:
                    cv2.imshow(name, bgr)
                _, dmap = cap.read_latest_depth()
                if dmap is not None:
                    cv2.imshow(depth_key(name), colorize_depth(dmap))
            if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                break
    finally:
        cams.stop()
        cv2.destroyAllWindows()


def main() -> None:
    from .devices import load_config, make_parser

    parser = make_parser(__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p_find = sub.add_parser("find", help="List OpenCV devices and RealSense serial numbers")
    p_find.add_argument("camera_type", nargs="?", default=None, choices=["opencv", "realsense"])
    p_test = sub.add_parser("test", help="Start the configured cameras and measure delivered fps")
    p_test.add_argument("--record_time_s", type=float, default=5.0, help="How long to measure (like lerobot-find-cameras)")
    sub.add_parser("preview", help="Live windows of the configured cameras (colour + depth), q to quit")
    args, overrides = parser.parse_known_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    if args.command == "find":
        find_cameras(args.camera_type)
        return
    cfg = load_config(args.preset, overrides)
    if args.command == "test":
        test_cameras(cfg, args.record_time_s)
    elif args.command == "preview":
        preview(cfg)


if __name__ == "__main__":
    main()

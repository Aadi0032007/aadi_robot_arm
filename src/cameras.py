"""Cameras: single-owner USB capture, frame bus publishing, observation merge.

Same design as the AGV repo's ``LAB/cameras.py``:

    UsbCameraCapture   one background reader per device (the only process that
                       touches it). Each BGR frame goes to a 1-slot latest-frame
                       holder for in-process readers and, when
                       ``publish_frames=True``, to the shared-memory frame bus
                       ``lab_<name>`` for other processes (inference).
                       ``hw_decode=True`` opens a Jetson GStreamer pipeline
                       (v4l2src → nvv4l2decoder/nvvidconv); if that fails to open
                       the capture falls back to plain OpenCV automatically.
    CamerasManager     starts every enabled camera in the config (``wrist``, ``eye``).
    RobotWithCameras   wraps a LeRobot robot built WITHOUT cameras and merges the
                       latest frames (RGB, resized to the configured shape) into
                       ``get_observation()`` and ``observation_features``, so the
                       dataset features are identical to a robot that owned the
                       cameras itself.

CLI:
    python -m src.cameras find
    python -m src.cameras test --record_time_s=5 --cameras.wrist.source=0 --cameras.eye.source=2
    python -m src.cameras preview --preset so101
"""

from __future__ import annotations

if __package__ in (None, ""):  # allow `python src/cameras.py`
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "src"

import argparse
import glob
import logging
import platform
import threading
import time
from typing import Any, Optional

import cv2
import numpy as np

from .config import ArmConfig, CameraConfig

logger = logging.getLogger(__name__)

_ROTATIONS = {
    90: cv2.ROTATE_90_CLOCKWISE,
    180: cv2.ROTATE_180,
    270: cv2.ROTATE_90_COUNTERCLOCKWISE,
}


# ══════════════════════════════════════════════════════════════════════════════
#  USB capture — sole owner of the device
# ══════════════════════════════════════════════════════════════════════════════


class UsbCameraCapture:
    """Background reader for one camera. Never blocks on any consumer."""

    def __init__(self, cfg: CameraConfig) -> None:
        self.name = cfg.name
        self.cfg = cfg
        self.backend = "none"  # "gstreamer" | "opencv"
        self.frames_read = 0

        self._frame: Optional[np.ndarray] = None
        self._ts: float = 0.0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._initial_cap: Optional[cv2.VideoCapture] = None

        self._publish_enabled = bool(cfg.publish_frames)
        self._publisher = None

    # ── public API ──────────────────────────────────────────────────────────

    def start(self, open_timeout_sec: float | None = None) -> bool:
        """Open the device (on a helper thread, so a wedged UVC device cannot hang
        startup) and start the reader. Returns False if the camera is unusable."""
        timeout = float(open_timeout_sec or self.cfg.open_timeout_sec)
        result: dict[str, Any] = {}

        def _open() -> None:
            try:
                result["cap"] = self._open_capture()
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
        cap = result.get("cap")
        if cap is None or not cap.isOpened():
            if cap is not None:
                cap.release()
            logger.error("%s: cannot open %r", self.name, self.cfg.source)
            return False

        self._initial_cap = cap
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True, name=f"cam-{self.name}")
        self._thread.start()
        logger.info(
            "%s: started %sx%s@%s via %s%s",
            self.name, self.cfg.width, self.cfg.height, self.cfg.fps, self.backend,
            " [bus]" if self._publish_enabled else "",
        )
        return True

    def read_latest(self) -> tuple[Optional[float], Optional[np.ndarray]]:
        """(timestamp, BGR frame) from the 1-slot holder. Non-blocking, not copied."""
        with self._lock:
            if self._frame is None:
                return None, None
            return self._ts, self._frame

    def wait_for_frame(self, timeout_s: float) -> bool:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if self._frame is not None:
                return True
            time.sleep(0.02)
        return self._frame is not None

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        with self._lock:
            self._frame = None
        if self._publisher is not None:
            try:
                self._publisher.close()
            except Exception as exc:  # noqa: BLE001
                logger.warning("%s: publisher close error: %s", self.name, exc)
            self._publisher = None

    # ── opening ─────────────────────────────────────────────────────────────

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

    def _open_capture(self) -> cv2.VideoCapture:
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
        if cap.isOpened():
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

    # ── capture loop ────────────────────────────────────────────────────────

    def _ensure_publisher(self, frame: np.ndarray) -> None:
        if not self._publish_enabled or self._publisher is not None:
            return
        if frame.ndim != 3:
            logger.warning("%s: bus disabled, unexpected frame.ndim=%s", self.name, frame.ndim)
            self._publish_enabled = False
            return
        try:
            from .frame_bus import FrameBusPublisher

            h, w, c = frame.shape
            self._publisher = FrameBusPublisher(self.name, height=h, width=w, channels=c)
            logger.info("%s: frame bus → %s (%sx%sx%s)", self.name, self._publisher.name, w, h, c)
        except Exception as exc:  # noqa: BLE001
            logger.error("%s: frame bus init failed: %s", self.name, exc)
            self._publish_enabled = False

    def _run(self) -> None:
        backoff = 1.0
        rotate = _ROTATIONS.get(int(self.cfg.rotation or 0))
        while not self._stop.is_set():
            cap = self._initial_cap if self._initial_cap is not None else self._open_capture()
            self._initial_cap = None
            if not cap.isOpened():
                cap.release()
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
                now = time.time()

                with self._lock:  # in-process consumers (teleop / record)
                    self._frame = frame
                    self._ts = now
                    self.frames_read += 1

                if self._publish_enabled:  # out-of-process consumers (inference)
                    self._ensure_publisher(frame)
                    if self._publisher is not None:
                        try:
                            self._publisher.publish(np.ascontiguousarray(frame), now)
                        except Exception as exc:  # noqa: BLE001
                            logger.error("%s: publish error: %s (bus disabled)", self.name, exc)
                            try:
                                self._publisher.close()
                            except Exception:  # noqa: BLE001
                                pass
                            self._publisher = None
                            self._publish_enabled = False

            cap.release()
            if not self._stop.is_set():
                logger.warning("%s: stream lost, reconnecting in %.1fs", self.name, backoff)
                self._stop.wait(timeout=backoff)
                backoff = min(backoff * 2, 10.0)


# ══════════════════════════════════════════════════════════════════════════════
#  CamerasManager
# ══════════════════════════════════════════════════════════════════════════════


class CamerasManager:
    """Starts and owns every enabled camera of the config."""

    def __init__(self, cfg: ArmConfig) -> None:
        self._cfg = cfg
        self.captures: dict[str, UsbCameraCapture] = {}
        self.failed: list[str] = []

    @property
    def expected_names(self) -> list[str]:
        return self._cfg.camera_names

    def start(self) -> list[str]:
        for cam in self._cfg.enabled_cameras:
            cap = UsbCameraCapture(cam)
            if cap.start():
                self.captures[cam.name] = cap
            else:
                self.failed.append(cam.name)
        return list(self.captures)

    def stop(self) -> None:
        for cap in self.captures.values():
            cap.stop()
        self.captures.clear()

    def read(self, name: str) -> tuple[Optional[float], Optional[np.ndarray]]:
        cap = self.captures.get(name)
        return (None, None) if cap is None else cap.read_latest()

    def names(self) -> list[str]:
        return list(self.captures)

    def shapes(self) -> dict[str, tuple[int, int, int]]:
        """Configured (height, width, 3) per running camera. This is what goes into the features."""
        return {name: cap.cfg.frame_shape for name, cap in self.captures.items()}

    def wait_ready(self, timeout_s: float) -> list[str]:
        """Block until every camera delivered a frame; return the names that did not."""
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
    forwards everything else to the wrapped robot.
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
    def cameras(self) -> dict[str, UsbCameraCapture]:
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

    def get_observation(self) -> dict[str, Any]:
        obs = self._robot.get_observation()
        now = time.time()
        for name, cap in self._cams.captures.items():
            ts, bgr = cap.read_latest()
            if bgr is None:
                raise RuntimeError(f"camera {name!r} has no frame")
            age = now - float(ts or 0.0)
            if age > self._stale_s:
                if name not in self._warned_stale:
                    logger.warning("camera %s frame is %.0f ms old", name, age * 1e3)
                    self._warned_stale.add(name)
            else:
                self._warned_stale.discard(name)

            h, w, _ = cap.cfg.frame_shape
            if bgr.shape[0] != h or bgr.shape[1] != w:
                if name not in self._warned_resize:
                    logger.warning(
                        "camera %s delivers %sx%s but config says %sx%s; resizing",
                        name, bgr.shape[1], bgr.shape[0], w, h,
                    )
                    self._warned_resize.add(name)
                bgr = cv2.resize(bgr, (w, h), interpolation=cv2.INTER_AREA)
            obs[name] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)  # LeRobot convention is RGB
        return obs

    def send_action(self, action: dict[str, float]) -> dict[str, float]:
        return self._robot.send_action(action)

    def __getattr__(self, item: str) -> Any:
        return getattr(self._robot, item)


# ══════════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════════


def find_cameras() -> None:
    """Probe candidate devices and print their default modes."""
    if platform.system() == "Linux":
        targets: list[int | str] = sorted(glob.glob("/dev/video*"))
        api = cv2.CAP_V4L2
    else:
        targets = list(range(10))
        api = cv2.CAP_ANY
    print(f"{'source':<14} {'default mode':<16} note")
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
        print("No cameras found.")


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
    print(f"\n{'camera':<8} {'backend':<10} {'fps':>6}  frame shape (BGR)   bus")
    for name, cap in cams.captures.items():
        fps = (cap.frames_read - start_counts[name]) / elapsed
        _, frame = cap.read_latest()
        shape = tuple(frame.shape) if frame is not None else "none"
        flag = "" if fps >= 0.9 * cap.cfg.fps else "   <- below configured fps"
        print(f"{name:<8} {cap.backend:<10} {fps:>6.1f}  {str(shape):<18}  {'yes' if cap._publisher is not None else 'no'}{flag}")
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
            if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                break
    finally:
        cams.stop()
        cv2.destroyAllWindows()


def main() -> None:
    from .devices import load_config, make_parser

    parser = make_parser(__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("find", help="Probe devices and print their default modes")
    p_test = sub.add_parser("test", help="Start the configured cameras and measure delivered fps")
    p_test.add_argument("--record_time_s", type=float, default=5.0, help="How long to measure (like lerobot-find-cameras)")
    sub.add_parser("preview", help="Live windows of the configured cameras (q to quit)")
    args, overrides = parser.parse_known_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    if args.command == "find":
        find_cameras()
        return
    cfg = load_config(args.preset, overrides)
    if args.command == "test":
        test_cameras(cfg, args.record_time_s)
    elif args.command == "preview":
        preview(cfg)


if __name__ == "__main__":
    main()

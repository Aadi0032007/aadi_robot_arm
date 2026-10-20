"""Raw session recorder and reader shared by record.py, convert.py and replay.py.

::

    <run>/session_<i>/
        data.jsonl          one row per frame (see README §2)
        meta.json           schema, fps, camera shapes, frame count
        video_<cam>.mp4     one file per camera key, frame i == data.jsonl row i

``SessionWriter`` is passed to LeRobot's ``record_loop`` in place of a
``LeRobotDataset``. The loop only touches three members, ``fps``, ``features`` and
``add_frame(frame)``, and the frames it hands over are already packed by
``build_dataset_frame`` (``observation.state`` / ``observation.images.<cam>`` /
``action`` / ``task``). The writer unpacks them back into named joint values for the
JSONL and streams the images to one mp4 per camera on a background thread.

``state_names`` and ``action_names`` are kept separately, so robots whose observation
has more joints than their action (the Nero follower observes 8, a Koch leader
commands 6) round-trip without special cases.
"""

from __future__ import annotations

import json
import logging
import queue
import re
import shutil
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

import numpy as np

logger = logging.getLogger(__name__)

DATA_FILENAME = "data.jsonl"
META_FILENAME = "meta.json"
SESSION_PREFIX = "session_"
_SESSION_RE = re.compile(rf"^{SESSION_PREFIX}(\d+)$")

OBS_PREFIX = "observation"
ACTION_KEY = "action"
STATE_KEY = f"{OBS_PREFIX}.state"
IMAGE_PREFIX = f"{OBS_PREFIX}.images."

_STOP = object()


def video_filename(camera_key: str) -> str:
    return f"video_{camera_key}.mp4"


# ---------------------------------------------------------------------------
# Directory helpers
# ---------------------------------------------------------------------------


def new_run_dir(sessions_root: Path) -> Path:
    """``<sessions_root>/<YYYYmmdd_HHMMSS>`` (mirrors the AGV ``split_session_<datetime>`` idea)."""
    run_dir = Path(sessions_root).expanduser() / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def latest_run_dir(sessions_root: Path) -> Path | None:
    """Most recently modified run folder under ``sessions_root`` (for ``--resume``)."""
    root = Path(sessions_root).expanduser()
    if not root.is_dir():
        return None
    runs = [p for p in root.iterdir() if p.is_dir()]
    return max(runs, key=lambda p: p.stat().st_mtime) if runs else None


def _session_index(path: Path) -> int | None:
    match = _SESSION_RE.match(path.name)
    return int(match.group(1)) if match else None


def next_session_dir(run_dir: Path) -> Path:
    """First unused ``session_<i>`` under ``run_dir`` (continues numbering on append)."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    used = [i for i in (_session_index(p) for p in run_dir.iterdir() if p.is_dir()) if i is not None]
    return run_dir / f"{SESSION_PREFIX}{max(used) + 1 if used else 0}"


def is_session_dir(path: Path) -> bool:
    path = Path(path)
    return (path / DATA_FILENAME).is_file() and (path / META_FILENAME).is_file()


def find_sessions(root: Path) -> list[Path]:
    """All complete ``session_<i>`` folders under ``root``, in index order.

    ``root`` may itself be a single session folder.
    """
    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(root)
    if is_session_dir(root):
        return [root]
    sessions = [p for p in root.iterdir() if p.is_dir() and is_session_dir(p)]
    return sorted(sessions, key=lambda p: (_session_index(p) is None, _session_index(p) or 0, p.name))


def load_meta(session_dir: Path) -> dict[str, Any]:
    return json.loads((Path(session_dir) / META_FILENAME).read_text(encoding="utf-8"))


def load_rows(session_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(Path(session_dir) / DATA_FILENAME, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------


class SessionWriter:
    """Write one session. Duck-types ``LeRobotDataset`` for ``record_loop``."""

    def __init__(
        self,
        session_dir: Path,
        fps: int,
        features: dict[str, dict[str, Any]],
        meta: dict[str, Any] | None = None,
        codec: str = "mp4v",
        queue_size: int = 900,
    ):
        self.session_dir = Path(session_dir)
        if self.session_dir.exists() and any(self.session_dir.iterdir()):
            raise FileExistsError(f"Refusing to write into a non-empty session folder: {self.session_dir}")
        self.session_dir.mkdir(parents=True, exist_ok=True)

        self.fps = int(fps)
        self.features = features
        self.codec = codec
        self.num_frames = 0

        self._state_names: list[str] = list(features.get(STATE_KEY, {}).get("names", []))
        self._action_names: list[str] = list(features.get(ACTION_KEY, {}).get("names", []))
        self._image_keys: dict[str, str] = {  # dataset key -> camera key
            key: key.removeprefix(IMAGE_PREFIX)
            for key, ft in features.items()
            if ft.get("dtype") in ("image", "video") and key.startswith(IMAGE_PREFIX)
        }

        self._meta: dict[str, Any] = dict(meta or {})
        self._meta.setdefault("session_name", self.session_dir.name)
        self._meta["fps"] = self.fps
        self._meta["codec"] = codec
        self._meta["state_names"] = self._state_names
        self._meta["action_names"] = self._action_names
        self._meta["started_at"] = datetime.now().isoformat(timespec="seconds")
        self._meta["lerobot_features"] = _jsonable(features)
        self._camera_shapes: dict[str, tuple[int, ...]] = {}

        self._jsonl = open(self.session_dir / DATA_FILENAME, "w", encoding="utf-8")
        self._t0: float | None = None
        self._closed = False

        self._queue: queue.Queue = queue.Queue(maxsize=queue_size)
        self._writers: dict[str, Any] = {}
        self._writer_error: BaseException | None = None
        self._thread = threading.Thread(target=self._drain, name=f"{self.session_dir.name}-video", daemon=True)
        self._thread.start()

    # -- LeRobotDataset-compatible surface ---------------------------------

    def add_frame(self, frame: dict[str, Any], extra: dict[str, Any] | None = None) -> None:
        """``frame`` is a ``build_dataset_frame`` output plus ``task``. ``extra`` keys
        (e.g. ``{"source": "ai"}``) are written into the JSONL row as-is."""
        if self._closed:
            raise RuntimeError("SessionWriter is closed.")
        if self._writer_error is not None:
            raise RuntimeError("Video writer thread failed") from self._writer_error

        now = time.perf_counter()
        if self._t0 is None:
            self._t0 = now

        row = {
            "frame_index": self.num_frames,
            "timestamp": round(now - self._t0, 6),
            "wall_time": time.time(),
            "observation": _named(self._state_names, frame.get(STATE_KEY)),
            "action": _named(self._action_names, frame.get(ACTION_KEY)),
            "task": frame.get("task"),
        }
        if extra:
            row.update(extra)

        for ds_key, cam_key in self._image_keys.items():
            image = frame.get(ds_key)
            if image is None:
                raise KeyError(f"Frame is missing image '{ds_key}'. Keys: {sorted(frame)}")
            image = np.asarray(image)
            if cam_key not in self._camera_shapes:
                self._camera_shapes[cam_key] = tuple(int(x) for x in image.shape)
            # Blocks when the encoder falls behind: same back-pressure LeRobot's streaming encoder applies.
            self._queue.put((cam_key, image))

        self._jsonl.write(json.dumps(row) + "\n")
        self.num_frames += 1

    # -- lifecycle ----------------------------------------------------------

    def close(self, discard: bool = False) -> Path | None:
        """Flush everything and write ``meta.json``. With ``discard`` the folder is deleted."""
        if self._closed:
            return None if discard else self.session_dir
        self._closed = True

        self._queue.put(_STOP)
        self._thread.join()
        for writer in self._writers.values():
            writer.release()
        self._writers.clear()
        self._jsonl.close()

        if discard:
            shutil.rmtree(self.session_dir, ignore_errors=True)
            logger.info("Discarded %s", self.session_dir)
            return None

        if self._writer_error is not None:
            logger.error("Video writer failed for %s: %s", self.session_dir, self._writer_error)

        duration = 0.0 if self._t0 is None else max(0.0, time.perf_counter() - self._t0)
        self._meta["num_frames"] = self.num_frames
        self._meta["duration_s"] = round(duration, 3)
        self._meta["cameras"] = {
            cam: {
                "height": shape[0],
                "width": shape[1],
                "channels": shape[2] if len(shape) > 2 else 1,
                "video": video_filename(cam),
            }
            for cam, shape in self._camera_shapes.items()
        }
        # Cameras that never delivered a frame still need an entry so convert.py can complain clearly.
        for cam in self._image_keys.values():
            self._meta["cameras"].setdefault(cam, {"video": video_filename(cam)})

        (self.session_dir / META_FILENAME).write_text(json.dumps(self._meta, indent=2), encoding="utf-8")
        return self.session_dir

    def __enter__(self) -> "SessionWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close(discard=exc_type is not None)

    # -- background video encoding ------------------------------------------

    def _drain(self) -> None:
        import cv2

        try:
            while True:
                item = self._queue.get()
                if item is _STOP:
                    return
                cam_key, image = item
                writer = self._writers.get(cam_key)
                if writer is None:
                    writer = self._open_writer(cv2, cam_key, image.shape)
                    self._writers[cam_key] = writer

                expected = self._camera_shapes[cam_key]
                if tuple(image.shape[:2]) != tuple(expected[:2]):
                    image = cv2.resize(image, (expected[1], expected[0]))
                if image.ndim == 2:
                    bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
                else:
                    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # LeRobot cameras hand out RGB
                writer.write(bgr)
        except BaseException as e:  # surfaced to the control loop on the next add_frame
            self._writer_error = e
            logger.exception("Video writer thread for %s crashed", self.session_dir)

    def _open_writer(self, cv2, cam_key: str, shape: tuple[int, ...]):
        height, width = int(shape[0]), int(shape[1])
        path = self.session_dir / video_filename(cam_key)
        fourcc = cv2.VideoWriter_fourcc(*self.codec)
        writer = cv2.VideoWriter(str(path), fourcc, float(self.fps), (width, height))
        if not writer.isOpened():
            raise RuntimeError(f"cv2.VideoWriter could not open {path} with codec '{self.codec}'")
        return writer


# ---------------------------------------------------------------------------
# Reader: video frames and JSONL rows in lock-step
# ---------------------------------------------------------------------------


class SessionFrames:
    """Iterate ``(row, {camera_key: rgb_uint8})`` over one session.

    Stops at the shortest of the JSONL and every video, exactly like the AGV converter.
    """

    def __init__(self, session_dir: Path):
        self.session_dir = Path(session_dir)
        self.meta = load_meta(self.session_dir)
        self.rows = load_rows(self.session_dir)
        self._caps: dict[str, Any] = {}
        self._warned_resize: set[str] = set()

    @property
    def camera_keys(self) -> list[str]:
        return list(self.meta.get("cameras", {}))

    def __enter__(self) -> "SessionFrames":
        import cv2

        for cam, info in self.meta.get("cameras", {}).items():
            path = self.session_dir / info.get("video", video_filename(cam))
            cap = cv2.VideoCapture(str(path))
            if not cap.isOpened():
                self.__exit__(None, None, None)
                raise RuntimeError(f"Could not open {path}")
            self._caps[cam] = cap
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        for cap in self._caps.values():
            cap.release()
        self._caps.clear()

    def __len__(self) -> int:
        import cv2

        n = len(self.rows)
        for cap in self._caps.values():
            n = min(n, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        return n

    def __iter__(self) -> Iterator[tuple[dict[str, Any], dict[str, np.ndarray]]]:
        import cv2

        for i in range(len(self)):
            images: dict[str, np.ndarray] = {}
            for cam, cap in self._caps.items():
                ok, bgr = cap.read()
                if not ok:
                    return
                info = self.meta["cameras"][cam]
                if "height" in info and (bgr.shape[0], bgr.shape[1]) != (info["height"], info["width"]):
                    if cam not in self._warned_resize:
                        logger.warning(
                            "%s/%s frames are %sx%s but meta says %sx%s; resizing",
                            self.session_dir.name, cam, bgr.shape[1], bgr.shape[0], info["width"], info["height"],
                        )
                        self._warned_resize.add(cam)
                    bgr = cv2.resize(bgr, (info["width"], info["height"]))
                images[cam] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            yield self.rows[i], images


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------


def _named(names: list[str], vector: Any) -> dict[str, float]:
    if vector is None:
        return {}
    values = np.asarray(vector).reshape(-1)
    if len(values) != len(names):
        raise ValueError(f"Expected {len(names)} values for {names}, got {len(values)}")
    return {name: float(v) for name, v in zip(names, values)}


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)

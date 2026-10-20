"""MediaPipe Hands wrapper shared by ``test_mediapipe.py`` and ``hand_teleop.py``.

    pip install mediapipe

Uses the legacy ``mediapipe.solutions.hands`` API when the installed MediaPipe still has
it, otherwise the Tasks ``HandLandmarker`` (model file downloaded once into
``revobots_arm/models/``). Both give, per hand, the 21 image landmarks, the 21 metric
"world" landmarks and the handedness.

MediaPipe landmark indices:

    0 wrist
    thumb   1 CMC (base)   2 MCP   3 IP    4 tip
    index   5 MCP (knuckle) 6 PIP  7 DIP   8 tip
    middle 9-12 · ring 13-16 · pinky 17-20
"""

from __future__ import annotations

import logging
import math
import time
import urllib.request
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from ._bootstrap import ARM_ROOT

logger = logging.getLogger(__name__)

WRIST = 0
THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP = 1, 2, 3, 4
INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP = 5, 6, 7, 8
MIDDLE_MCP = 9
THUMB_CHAIN = (WRIST, THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP)
INDEX_CHAIN = (WRIST, INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP)

HAND_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
)
DEFAULT_HAND_MODEL = ARM_ROOT / "models" / "hand_landmarker.task"


@dataclass
class HandDetection:
    image: np.ndarray  # (21, 2) pixel coordinates in the frame that was passed in
    # (21, 3) metres, origin near the hand centre, axes as the image (x right, y down, z away
    # from the camera). None if this MediaPipe build does not return world landmarks.
    world: np.ndarray | None
    handedness: str  # "Left" | "Right" | "" ; MediaPipe assumes a mirrored (selfie) image
    score: float


# ── MediaPipe wrapper: legacy Solutions API first, Tasks API as fallback ─────────


class HandTracker:
    """Runs MediaPipe Hands on BGR frames. ``detect`` returns every hand found,
    ``landmarks`` only the first hand's (21, 2) pixel landmarks (or None)."""

    def __init__(
        self,
        hand_model: str | None = None,
        max_num_hands: int = 1,
        model_complexity: int = 0,
        min_detection_confidence: float = 0.6,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        try:
            import mediapipe as mp
        except ImportError as e:
            raise SystemExit("mediapipe is not installed: pip install mediapipe") from e

        self._mode = None
        solutions = getattr(mp, "solutions", None)
        if solutions is not None and hasattr(solutions, "hands"):
            self._hands = solutions.hands.Hands(
                static_image_mode=False,
                max_num_hands=max_num_hands,
                model_complexity=model_complexity,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence,
            )
            self._mode = "solutions"
            return

        # Newer MediaPipe without the legacy API: Tasks HandLandmarker needs a model file,
        # downloaded once into revobots_arm/models/ unless hand_model points elsewhere.
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        model_path = Path(hand_model).expanduser() if hand_model else DEFAULT_HAND_MODEL
        if not model_path.is_file():
            model_path.parent.mkdir(parents=True, exist_ok=True)
            logger.info("Downloading MediaPipe hand model (~7 MB) to %s", model_path)
            try:
                urllib.request.urlretrieve(HAND_MODEL_URL, model_path)
            except Exception as e:  # noqa: BLE001
                raise SystemExit(
                    f"could not download {HAND_MODEL_URL}: {e}\nDownload it manually and pass --hand_model=<path>."
                ) from e
        options = vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=max_num_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._landmarker = vision.HandLandmarker.create_from_options(options)
        self._mp = mp
        self._mode = "tasks"
        self._t0 = time.monotonic()
        self._last_ts_ms = -1

    def detect(self, frame_bgr: np.ndarray) -> list[HandDetection]:
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        out: list[HandDetection] = []

        if self._mode == "solutions":
            res = self._hands.process(rgb)
            worlds = getattr(res, "multi_hand_world_landmarks", None) or []
            labels = res.multi_handedness or []
            for i, lm in enumerate(res.multi_hand_landmarks or []):
                image = np.array([[p.x * w, p.y * h] for p in lm.landmark], dtype=np.float32)
                world = (
                    np.array([[p.x, p.y, p.z] for p in worlds[i].landmark], dtype=np.float32) if i < len(worlds) else None
                )
                label, score = "", 1.0
                if i < len(labels) and labels[i].classification:
                    label = labels[i].classification[0].label
                    score = float(labels[i].classification[0].score)
                out.append(HandDetection(image, world, label, score))
            return out

        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        # VIDEO mode rejects a timestamp that does not increase.
        ts_ms = max(int((time.monotonic() - self._t0) * 1000), self._last_ts_ms + 1)
        self._last_ts_ms = ts_ms
        res = self._landmarker.detect_for_video(mp_image, ts_ms)
        worlds = res.hand_world_landmarks or []
        labels = res.handedness or []
        for i, lm in enumerate(res.hand_landmarks or []):
            image = np.array([[p.x * w, p.y * h] for p in lm], dtype=np.float32)
            world = np.array([[p.x, p.y, p.z] for p in worlds[i]], dtype=np.float32) if i < len(worlds) else None
            label, score = "", 1.0
            if i < len(labels) and labels[i]:
                label = labels[i][0].category_name
                score = float(labels[i][0].score)
            out.append(HandDetection(image, world, label, score))
        return out

    def landmarks(self, frame_bgr: np.ndarray) -> np.ndarray | None:
        """The first hand's 21 landmarks as an (21, 2) array in pixels, or None."""
        hands = self.detect(frame_bgr)
        return hands[0].image if hands else None

    def close(self) -> None:
        if self._mode == "solutions":
            self._hands.close()
        elif self._mode == "tasks":
            self._landmarker.close()


# ── helpers ──────────────────────────────────────────────────────────────────────


def pinch_to_gripper(ratio: float, closed: float, opened: float, lo: float, hi: float, invert: bool) -> float:
    t = (ratio - closed) / (opened - closed) if opened != closed else 0.0
    t = min(1.0, max(0.0, t))
    if invert:
        t = 1.0 - t
    return lo + t * (hi - lo)


class MedianFilter:
    """Per-component median over the last ``window`` samples: removes single-frame spikes
    (a landmark that jumps for one frame) without smearing real motion much."""

    def __init__(self, window: int = 5) -> None:
        self._buf: deque[np.ndarray] = deque(maxlen=max(1, int(window)))

    def reset(self) -> None:
        self._buf.clear()

    def __call__(self, x: float | np.ndarray) -> np.ndarray:
        self._buf.append(np.array(x, dtype=float))
        return np.median(np.stack(self._buf), axis=0)


class Backlash:
    """Play (hysteresis) per component: the output only moves once the input leaves a band of
    +-``band`` around it, then follows it at that distance. Small jitter is absorbed anywhere,
    not only around zero, so motors do not hum while the hand is held still."""

    def __init__(self, band: float | np.ndarray) -> None:
        self.band = np.asarray(band, dtype=float)
        self._y: np.ndarray | None = None

    def reset(self) -> None:
        self._y = None

    def __call__(self, x: float | np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if self._y is None:
            self._y = x.copy()
        else:
            d = x - self._y
            self._y = self._y + np.sign(d) * np.maximum(np.abs(d) - self.band, 0.0)
        return self._y.copy()


class OneEuroFilter:
    """One Euro filter (Casiez et al. 2012) on a scalar or a vector, per component.

    Low ``min_cutoff`` = steadier when the hand is still; higher ``beta`` = less lag when it
    moves fast (the cutoff rises with speed). Both may be one value or one per component.
    """

    def __init__(
        self, min_cutoff: float | np.ndarray = 1.0, beta: float | np.ndarray = 0.0, d_cutoff: float = 1.0
    ) -> None:
        self.min_cutoff = np.asarray(min_cutoff, dtype=float)
        self.beta = np.asarray(beta, dtype=float)
        self.d_cutoff = float(d_cutoff)
        self.reset()

    def reset(self) -> None:
        self._x: np.ndarray | None = None
        self._dx: np.ndarray | None = None
        self._t: float | None = None

    @staticmethod
    def _alpha(cutoff: float | np.ndarray, dt: float) -> float | np.ndarray:
        tau = 1.0 / (2.0 * math.pi * np.maximum(cutoff, 1e-6))
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x: float | np.ndarray, t: float) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if self._x is None or self._t is None:
            self._x, self._dx, self._t = x.copy(), np.zeros_like(x), t
            return x.copy()
        dt = t - self._t
        if dt <= 0.0:
            return self._x.copy()
        dx = (x - self._x) / dt
        a_d = self._alpha(self.d_cutoff, dt)
        self._dx = a_d * dx + (1.0 - a_d) * self._dx
        a = self._alpha(self.min_cutoff + self.beta * np.abs(self._dx), dt)
        self._x = a * x + (1.0 - a) * self._x
        self._t = t
        return self._x.copy()

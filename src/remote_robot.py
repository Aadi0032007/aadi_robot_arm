"""``ArmProxyRobot``: a LeRobot ``Robot`` that talks to a running ``teleop.py``.

Used by ``inference.py``. Teleop owns the arm and the cameras, so the policy
process cannot open either. Instead:

    observation.images.<cam>   ← frame bus  ``lab_<cam>``  (shared memory, BGR → RGB)
    observation.state          ← UDP        ``ai.state_port``  published by teleop each tick
    action                     → UDP        ``ai.action_port`` applied by teleop while AI is on

Because it implements the ``Robot`` interface, LeRobot's own
``lerobot_inference_api.run_inference`` drives it unchanged.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from .ai_link import UdpJsonListener, UdpJsonSender
from .config import AiConfig
from .frame_bus import FrameBusReader

logger = logging.getLogger(__name__)

# Imported lazily so this module can be imported before _bootstrap ran.


def _lerobot_base():
    from lerobot.robots import Robot, RobotConfig

    return Robot, RobotConfig


_Robot, _RobotConfig = _lerobot_base()


@dataclass
class ArmProxyRobotConfig(_RobotConfig):
    pass


class ArmProxyRobot(_Robot):
    config_class = ArmProxyRobotConfig
    name = "arm_proxy"

    def __init__(self, features: dict[str, dict[str, Any]], robot_type: str, ai: AiConfig) -> None:
        super().__init__(ArmProxyRobotConfig(id="arm_proxy"))
        # predict_action() receives this; keep it equal to the dataset's robot_type.
        self.robot_type = robot_type
        self._ai = ai

        from lerobot.utils.constants import ACTION, OBS_STR

        image_prefix = f"{OBS_STR}.images."
        self._state_names: list[str] = list(features.get(f"{OBS_STR}.state", {}).get("names", []))
        self._action_names: list[str] = list(features.get(ACTION, {}).get("names", []))
        self._camera_shapes: dict[str, tuple[int, int, int]] = {
            key.removeprefix(image_prefix): tuple(int(x) for x in ft["shape"])  # type: ignore[misc]
            for key, ft in features.items()
            if key.startswith(image_prefix) and ft.get("dtype") in ("image", "video")
        }

        self._readers: dict[str, FrameBusReader] = {}
        self._state_rx: UdpJsonListener | None = None
        self._action_tx: UdpJsonSender | None = None
        self._connected = False
        self._warned_missing_state = False
        self._warned_resize: set[str] = set()

    # ── features ─────────────────────────────────────────────────────────────

    @property
    def observation_features(self) -> dict[str, Any]:
        return {**{n: float for n in self._state_names}, **self._camera_shapes}

    @property
    def action_features(self) -> dict[str, Any]:
        return {n: float for n in self._action_names}

    @property
    def cameras(self) -> dict[str, FrameBusReader]:
        return dict(self._readers) if self._readers else dict.fromkeys(self._camera_shapes)  # type: ignore[return-value]

    # ── lifecycle ────────────────────────────────────────────────────────────

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    def connect(self, calibrate: bool = True) -> None:
        if self._connected:
            return
        ai = self._ai
        self._state_rx = UdpJsonListener(ai.host, ai.state_port, "state")
        self._state_rx.start()
        self._action_tx = UdpJsonSender(ai.host, ai.action_port)
        for cam in self._camera_shapes:
            self._readers[cam] = FrameBusReader(cam)

        deadline = time.monotonic() + ai.bus_timeout_s
        pending = set(self._camera_shapes) | {"<state>"}
        while pending and time.monotonic() < deadline:
            for cam in list(pending):
                if cam == "<state>":
                    if self._state_rx.latest() is not None:
                        pending.discard(cam)
                elif self._readers[cam].read_latest()[1] is not None:
                    pending.discard(cam)
            if pending:
                time.sleep(0.1)
        if pending:
            self.disconnect()
            raise RuntimeError(
                f"nothing received from teleop within {ai.bus_timeout_s:.0f}s for {sorted(pending)}. "
                f"Start `python -m src.teleop` first (cameras must have publish_frames=True, "
                f"state goes to udp://{ai.host}:{ai.state_port})."
            )
        self._connected = True
        logger.info("%s connected: cams=%s state=udp:%s action=udp:%s",
                    self, sorted(self._camera_shapes), ai.state_port, ai.action_port)

    def disconnect(self) -> None:
        for rdr in self._readers.values():
            rdr.close()
        self._readers.clear()
        if self._state_rx is not None:
            self._state_rx.stop()
            self._state_rx = None
        if self._action_tx is not None:
            self._action_tx.close()
            self._action_tx = None
        self._connected = False

    # ── observation / action ─────────────────────────────────────────────────

    def get_observation(self) -> dict[str, Any]:
        pkt = self._state_rx.latest() if self._state_rx is not None else None
        state = (pkt or {}).get("observation") or {}
        obs: dict[str, Any] = {}
        missing = []
        for name in self._state_names:
            if name in state:
                obs[name] = float(state[name])
            else:
                obs[name] = 0.0
                missing.append(name)
        if missing and not self._warned_missing_state:
            logger.warning("teleop state lacks %s; feeding 0.0 (dataset/robot mismatch?)", missing)
            self._warned_missing_state = True

        for cam, (h, w, _) in self._camera_shapes.items():
            _ts, bgr = self._readers[cam].read_latest()
            if bgr is None:
                raise RuntimeError(f"frame bus lab_{cam} is empty. Is teleop still running?")
            if bgr.shape[0] != h or bgr.shape[1] != w:
                if cam not in self._warned_resize:
                    logger.warning("bus %s is %sx%s, policy expects %sx%s; resizing", cam, bgr.shape[1], bgr.shape[0], w, h)
                    self._warned_resize.add(cam)
                bgr = cv2.resize(bgr, (w, h), interpolation=cv2.INTER_AREA)
            obs[cam] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return obs

    def send_action(self, action: dict[str, float]) -> dict[str, float]:
        out = {k: float(v) for k, v in action.items() if k in self.action_features}
        if self._action_tx is not None:
            self._action_tx.send({"action": out})
        return out

# Revobots Arm

Stand-alone control, recording, conversion, replay and inference for the Revobots
robot arm. Default hardware: **AgileX Nero follower** (CAN) driven by a **Koch leader
on another PC** over UDP, with two cameras, **wrist** and **eye**. Koch-follower and
SO-101 presets are one flag away.

Nero now also has native seven-joint Cartesian IK, gamepad and VR presets.
See [NERO_IK.md](NERO_IK.md) for self-tests, dry runs, moves and TCP configuration.
Gamepad control supports local USB or UDP for every arm: add `--port=5001` on
the robot and run `src.gamepad_remote_client` on the gamepad computer.
See [GAMEPAD_REMOTE.md](GAMEPAD_REMOTE.md).

Everything here **imports** LeRobot from the repository next to this folder and never
modifies it. When this folder becomes its own repository, point it at a LeRobot
checkout with `LEROBOT_SRC` or `ArmConfig.lerobot_src` (or `pip install -e` LeRobot).

```
revobots_arm/
├── README.md
└── src/
    ├── config.py          ← ALL settings, plain Python dataclasses + presets (nero | koch | so101 | so101_remote | so101_keyboard)
    ├── keyboard_teleop.py ← hold-a-key joint jog teleoperator (no leader arm, no IK)
    ├── gamepad.py         ← USB gamepad → W3C Xbox layout (mapping lifted from the AGV pilot bridge)
    ├── gamepad_teleop.py  ← gamepad Cartesian jog (teleop type gamepad_ee), any arm with a URDF
    ├── kinematics.py      ← numpy-only URDF FK / Jacobian / task-priority IK (no placo)
    ├── so101_ik.py        ← SO-101 IK profile + CLI: print pose, Cartesian goto, self-test
    ├── test_mediapipe.py  ← experimental: gripper from a thumb-index pinch (webcam + MediaPipe)
    ├── devices.py         ← config → LeRobot follower / leader objects (robot built without cameras)
    ├── _bootstrap.py      ← finds the LeRobot source tree, adds it to sys.path
    ├── cameras.py         ← UsbCameraCapture (hw decode + fallback, frame bus), CamerasManager, RobotWithCameras
    ├── frame_bus.py       ← shared-memory frame fan-out (copied from the AGV repo)
    ├── ai_link.py         ← UDP JSON listener/sender for the teleop ↔ inference link
    ├── recorder.py        ← raw session format: video_<cam>.mp4 + data.jsonl + meta.json
    ├── teleop.py          ← MAIN: teleop + on-demand recording + AI hand-over (Space / Left / a / Esc)
    ├── leader_client.py   ← runs on the leader PC, streams the Koch or SO-101 leader to the robot PC
    ├── inference.py       ← policy process; talks to a running teleop through the bus + UDP
    ├── remote_robot.py    ← ArmProxyRobot: LeRobot Robot backed by the frame bus + UDP
    ├── record.py          ← timed, lerobot-record-style recording (alternative to teleop.py)
    ├── convert.py         ← sessions → LeRobotDataset (mirrors data_convert_agv.py)
    └── replay.py          ← replay a raw session or a dataset episode on the arm
```

---

## 1. Process layout (same shape as the AGV robot)

```
 leader PC                       robot PC
 ─────────                       ────────────────────────────────────────────────────────
 leader_client.py  ──UDP 5001──▶ teleop.py ─────── owns the arm (CAN / serial)
                                    │  │  ├── owns the cameras  ──▶ frame bus  lab_wrist, lab_eye
                                    │  │  ├── writes sessions   ──▶ <sessions_root>/<run>/session_<i>/
                                    │  │  └── publishes joint state ──UDP 5011──▶ ┐
                                    │  ◀───── applies AI actions ◀──UDP 5010───── inference.py
                                    │                                  ▲            ├── reads frame bus
                                    └── `a` key: HUMAN ⇄ AI            └────────────┘   (LeRobot policy)
```

Teleop is the only process that opens hardware, exactly like the AGV's `LAB/teleop.py`.
Inference never touches a camera or the arm: images come from the frame bus, joint
state from a UDP feed, and its actions are only applied while teleop is in AI mode.

---

## 2. Cameras (`cameras.py`)

Mirrors the AGV `LAB/cameras.py`:

- `UsbCameraCapture` is the sole owner of a device. It keeps the newest BGR frame in a
  1-slot holder for in-process readers and publishes every frame to `lab_<name>` when
  `publish_frames=True`. Consumers never block capture.
- `hw_decode=True` opens the Jetson GStreamer pipeline
  (`v4l2src → nvv4l2decoder mjpeg=1 → nvvidconv`, or the YUY2 variant). If that
  pipeline fails to open, capture falls back to plain OpenCV V4L2 by itself. The default
  is `IS_JETSON` (true when `/etc/nv_tegra_release` exists).
- `RobotWithCameras` wraps the LeRobot follower, which is built **without** cameras, and
  merges the frames (BGR→RGB, resized to the configured shape) into `get_observation()`
  and `observation_features`. The dataset features are therefore identical to what
  `lerobot-record` produces with cameras inside the robot.

Two cameras are configured by default:

```python
CameraConfig(name="wrist", source=video(0))   # /dev/video0 on Linux, index 0 elsewhere
CameraConfig(name="eye",   source=video(2))
```

`python -m src.cameras find|test|preview` probes devices, measures the delivered fps per
backend, and shows live windows.

---

## 3. Raw session format

```
<sessions_root>/<YYYYmmdd_HHMMSS>/          one teleop launch = one run
    session_0/
        data.jsonl            one JSON object per frame, same order as the video frames
        meta.json             schema + fps + camera shapes (written on close)
        video_wrist.mp4       one mp4 per camera key, frame i == data.jsonl row i
        video_eye.mp4
    session_1/ ...
```

`data.jsonl` row. With the Nero the observation has 8 joints and the action the Koch
leader's 6; `source` says who was driving (`human` or `ai`):

```json
{
  "frame_index": 0, "timestamp": 0.0, "wall_time": 1758203471.512,
  "observation": {"shoulder_pan.pos": -12.3, "shoulder_lift.pos": 45.1, "forearm_roll.pos": 0.8,
                  "elbow_flex.pos": 30.2, "wrist_roll.pos": 0.4, "wrist_pitch.pos": -3.1,
                  "wrist_flex.pos": -5.0, "gripper.pos": 12.0},
  "action":      {"shoulder_pan.pos": -12.0, "shoulder_lift.pos": 45.5, "elbow_flex.pos": 30.0,
                  "wrist_roll.pos": 0.5, "wrist_flex.pos": -5.2, "gripper.pos": 15.0},
  "task": "Pick up the cube and place it in the bin.",
  "source": "human"
}
```

`meta.json` carries `robot_type`, `fps`, `state_names`, `action_names`, `cameras`
(`{name: {height, width, channels, video}}`), `num_frames`, the LeRobot feature dict and
which capture backend each camera used. `convert.py` reads everything from here, so no
joint names or shapes are hard-coded anywhere.

Units are whatever the follower reports: Nero and SO-101 in degrees, Koch in
`-100…100` unless `use_degrees=True`; grippers `0…100` from the leader.

---

## 4. Configuration (`src/config.py`)

No JSON. Edit the dataclasses, or add a preset:

| Section | What it holds |
| --- | --- |
| `robot` (dict) | `type` = a registered LeRobot robot, plus that dataclass's fields. Nero: `interface`, `channel` (`None` = auto), `firmware_version`, `speed_percent`, `teleop_action_joints` … Koch/SO-101: `port`, `id`, `use_degrees` … |
| `teleop` (dict) | `type` = a registered LeRobot teleoperator. Default `koch_leader_remote` (`host`, `port`, `max_action_age_s`). |
| `cameras` | list of `CameraConfig` (`name`, `source`, `width`, `height`, `fps`, `hw_decode`, `pixel_format`, `publish_frames`, `rotation`). |
| `leader_client` | defaults for the leader PC: `robot_ip`, `port`, `com`, `id`, `fps`, `gripper_open_pos`. |
| `record` | `codec`, `play_sounds`, `display_*`, `redo_starts_recording`, `frame_stale_s`, and the timed-record knobs. |
| `dataset` / `replay` / `policy` | `repo_id`, `root`, `push_to_hub`, `tags` · `ramp_s` · `path`, `device`, `duration_s`. |
| `ai` | `action_port` 5010, `state_port` 5011, `action_ttl_s`, `read_leader_while_ai`, `human_takeover_threshold`, `bus_timeout_s`. |

Select a preset with `--preset koch` on any script or `REVOBOTS_ARM_PRESET=koch`.

---

## 5. Workflows

Run from this folder in the Python environment that has LeRobot's dependencies.

Arguments use the lerobot CLI's names (`--robot.type`, `--teleop.port`, `--dataset.repo_id`,
`--display_data=true`, …) so commands migrate from `lerobot-teleoperate` / `-record` /
`-replay` unchanged; `--preset` fills in whatever is not given. Cameras, the task string
and the AI link are off unless given (`--cameras.wrist.source=0`, `--dataset.single_task=...`,
`--ai.enabled=true`). See HOW_TO_USE.md.

```bash
# leader PC
python -m src.leader_client --ip <robot pc ip>

# robot PC
python -m src.cameras find                      # which /dev/video* are the wrist and eye?
python -m src.cameras test --record_time_s=5    # backend + real fps + bus status per camera
python -m src.teleop --preset nero              # main tool, see keys below
python -m src.teleop --robot.type=so101_follower --robot.port=COM4 --robot.id=aadi_so101_follower \
                     --teleop.type=so101_leader --teleop.port=COM5 --teleop.id=aadi_so101_leader
python -m src.convert --sessions_dir <sessions_root>/<run> --dataset.repo_id=<user>/revobots_nero_v1
python -m src.replay  --preset nero --session_dir <sessions_root>/<run>/session_0
lerobot-train --dataset.repo_id=<user>/revobots_nero_v1 --policy.type=act --output_dir=outputs/train/act_nero ...

# policy: teleop stays running in another terminal
python -m src.inference --preset nero --policy.path=outputs/train/act_nero/checkpoints/last/pretrained_model \
                        --dataset_repo_id=<user>/revobots_nero_v1 --control_time_s=120
# then press `a` in the teleop terminal
```

### `teleop.py` keys

| Key | Idle | Recording |
| --- | --- | --- |
| **Space** | start `session_<i>` | stop and save |
| **Left** / **r** | delete the last session saved in this launch and record it again | discard the ongoing session and record it again |
| **a** | AI control on / off (inference.py must be sending) | same; the recorded rows carry `source: ai` |
| **Esc** / **q** | quit | save, then quit |

While AI is on, teleop keeps reading the leader; moving it by more than
`ai.human_takeover_threshold` hands control back at once. If no fresh AI action arrives
within `ai.action_ttl_s` the arm holds its pose. Recording works in both modes, so an
AI run recorded with Space is an eval session with the same format.

---

## 6. What is reused from LeRobot

| Script | Imported from LeRobot | Written here |
| --- | --- | --- |
| `teleop.py` | `make_default_processors`, `build_dataset_frame`, `hw_to_dataset_features`, `create_key_listener`, `precise_sleep`, `log_say`, visualisation | the loop (a copy of `record_loop`'s per-step flow plus recording/AI state), session sink |
| `inference.py` | `lerobot_inference_api.build_policy_pipeline` + `run_inference` (the whole loop) | `ArmProxyRobot` (a `Robot` backed by the frame bus and UDP) |
| `leader_client.py` | `koch_leader_remote_client.run_remote_client` | defaults from config |
| `record.py` | `lerobot_record.record_loop` (the real one), `init_keyboard_listener` | `SessionWriter` duck-typing the dataset |
| `convert.py` | `LeRobotDataset.create/resume`, `VideoEncodingManager`, `build_dataset_frame`, `make_default_processors` | session discovery, lock-step video/JSONL reader |
| `replay.py` | `LeRobotDataset`, `make_default_robot_action_processor` | raw-session source, start ramp |
| `cameras.py`, `frame_bus.py`, `ai_link.py` | nothing (they mirror the AGV repo) | all |

---

## 7. Assumptions and open points

- Not executed yet. Ports, CAN channel and camera indices in `config.py` are placeholders.
- `koch_leader_remote.get_action()` blocks until a fresh packet arrives, so teleop pauses
  while the leader client is not streaming (also in AI mode, unless
  `ai.read_leader_while_ai=False`, which then disables leader take-over).
- Raw session video is OpenCV `mp4v` (lossy, same as the AGV splitter); the LeRobot
  dataset re-encodes once more. Set `record.codec` if you need something else.
- The frame bus is BGR (AGV convention). Every LeRobot-facing path converts to RGB.
- Global hotkeys (`pynput`) fire whatever window has focus, like `lerobot-record`.
- The Nero needs `pyAgxArm` on the robot PC; the leader PC needs LeRobot's `dynamixel` extra.

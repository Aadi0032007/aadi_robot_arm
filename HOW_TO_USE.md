# How to use the Revobots arm tools

Everything runs from the `revobots_arm/` folder, in the Python environment that has
LeRobot's dependencies (the one you already use for `lerobot-teleoperate`):

```bash
cd revobots_arm
python -m src.<tool> [--preset nero|koch|so101|so101_remote|so101_keyboard] [lerobot-style args]
```

**Arguments use the same names as the lerobot CLI**, so a `lerobot-teleoperate` or
`lerobot-record` command line carries over as is: `--robot.type=... --robot.port=...
--robot.id=... --teleop.type=... --teleop.port=... --teleop.id=... --dataset.repo_id=...
--dataset.single_task=... --display_data=true --play_sounds=true --fps=30`.
Spoken prompts are off by default here (`--play_sounds=true` turns them on).
`--preset` is optional: it fills in whatever you do not spell out (default preset `nero`,
or `REVOBOTS_ARM_PRESET`). Flags that have no lerobot equivalent follow the same style
(`--run_dir`, `--sessions_dir`, `--session_dir`, `--cameras.wrist.source=0`, `--ai.action_port=5010`).

**Off unless you ask for them**: cameras (`--cameras.wrist.source=0 --cameras.eye.source=2`),
the task string (`--dataset.single_task="..."`), and the AI link to `inference.py`
(`--ai.enabled=true`). A bare teleop command therefore moves the arm and records joints only.

Contents

1. [One-time setup](#1-one-time-setup)
2. [Cameras](#2-cameras)
3. [Leader arm on another PC](#3-leader-arm-on-another-pc)
4. [teleop.py, the main tool](#4-teleoppy-the-main-tool)
5. [What a run looks like on disk](#5-what-a-run-looks-like-on-disk)
6. [record.py, timed recording](#6-recordpy-timed-recording)
7. [convert.py, sessions to a LeRobot dataset](#7-convertpy-sessions-to-a-lerobot-dataset)
8. [replay.py](#8-replaypy)
9. [Training](#9-training)
10. [inference.py](#10-inferencepy)
11. [SO-101 command sheet](#11-so-101-command-sheet)
12. [Troubleshooting](#12-troubleshooting)

---

## 1. One-time setup

1. **Pick a preset and fill in the hardware fields** in `src/config.py`, or pass them on
   the command line every time (section 11 shows both forms):

   | Preset | Follower | Leader | Fields to check |
   | --- | --- | --- | --- |
   | `nero` (default) | AgileX Nero over CAN | Koch leader on another PC | `interface`/`channel` (leave `None` for auto), `firmware_version`, `speed_percent` |
   | `koch` | Koch follower over serial | Koch leader on another PC | `robot["port"]` (`COM4`, `/dev/ttyACM0`) |
   | `so101` | SO-101 follower (`COM4`, id `aadi_so101_follower`) | SO-101 leader on the same PC (`COM5`, id `aadi_so101_leader`) | `robot["port"]`, `teleop["port"]` |
   | `so101_remote` | SO-101 follower | SO-101 leader on another PC (UDP 5002) | `robot["port"]`, `leader_client.com` |
   | `so101_keyboard` | SO-101 follower | keyboard joint jog, no leader, no IK | `robot["port"]`, `teleop["speed"]` |

   Also set `task` (`--dataset.single_task`), `dataset.repo_id`, and the two camera
   sources (section 2).

2. **Calibrate the arms with the normal LeRobot CLI** (Koch / SO-101 only, the Nero
   needs none). The `id` selects the calibration file and must match what you pass later:

   ```bash
   lerobot-calibrate --robot.type=so101_follower --robot.port=COM4 --robot.id=aadi_so101_follower
   lerobot-calibrate --teleop.type=so101_leader  --teleop.port=COM5 --teleop.id=aadi_so101_leader
   lerobot-calibrate --robot.type=koch_follower  --robot.port=COM4 --robot.id=revobots_koch_follower
   lerobot-calibrate --teleop.type=koch_leader   --teleop.port=COM3 --teleop.id=aadi              # on the leader PC
   ```

3. **Nero only**: install `pyAgxArm` on the robot PC. It is imported when the arm connects.

---

## 2. Cameras

Two cameras are configured, `wrist` and `eye`. Their names become
`observation.images.wrist` / `observation.images.eye` in the dataset, and the frame bus
regions `lab_wrist` / `lab_eye` for other processes.

### Find the devices

```bash
python -m src.cameras find
```

Prints every device OpenCV can open, its default mode, and whether it delivers a frame.
On Linux each physical camera usually shows two `/dev/video*` nodes; the one that says
`ok` is the capture node, the other is metadata. Put the capture node in the config, or
pass it on the command line:

```python
# src/config.py
def default_cameras() -> list[CameraConfig]:
    return [
        CameraConfig(name="wrist", source=video(0)),   # /dev/video0 on Linux, index 0 on Windows
        CameraConfig(name="eye",   source=video(2)),
    ]
```

```bash
--cameras.wrist.source=0 --cameras.eye.source=2        # per-camera fields: source width height fps
--cameras.eye.enabled=false                             #   pixel_format rotation publish_frames hw_decode
--cameras=none                                          # run without cameras
```

### Hardware decode and fallback

`hw_decode=True` opens a Jetson GStreamer pipeline (`v4l2src → nvv4l2decoder → nvvidconv`)
inside OpenCV. If that pipeline cannot open, capture falls back to plain OpenCV V4L2 on
its own and logs one warning. The default is on when running on a Jetson
(`/etc/nv_tegra_release` exists) and off elsewhere.

### Check they run

```bash
python -m src.cameras test --record_time_s=5 --cameras.wrist.source=0 --cameras.eye.source=2
python -m src.cameras preview --preset so101            # one window per camera, q to quit
```

`test` should report an fps close to the configured one. If it is far below, the USB
bus is saturated (try `pixel_format="MJPG"`, a lower resolution, or another port).

### Who reads the frames

- `teleop.py` and `record.py` read the in-process latest frame.
- `inference.py` reads the frame bus, which teleop publishes (`publish_frames=True`).
  Nothing else has to open the camera, so teleop and inference run at the same time.

---

## 3. Leader arm on another PC

With the `nero` and `koch` presets the Koch leader, and with `so101_remote` the SO-101
leader, is plugged into a second PC and streamed over UDP. On that PC:

```bash
python -m src.leader_client --ip <robot pc ip>                        # nero (default) or --preset koch
python -m src.leader_client --preset so101_remote --ip <robot pc ip>  # SO-101 leader
```

The script picks the Koch or the SO-101 client from the preset's `teleop["type"]`
(`koch_leader_remote` → port 5001, `so101_leader_remote` → port 5002). Its flags keep the
dash style of the repo's `koch_leader_remote_client`:

| Arg | Default (from `config.leader_client`) | Meaning |
| --- | --- | --- |
| `--ip` | `192.168.1.42` | robot PC address |
| `--port` | `5001` (Koch) / `5002` (SO-101) | must equal `teleop["port"]` on the robot PC |
| `--com` | `COM3` / `COM5` | serial port of the leader |
| `--id` | `aadi` / `aadi_so101_leader` | calibration file of the leader |
| `--fps` | `60` | read and stream rate |
| `--gripper-open-pos` | `45.0` | Koch only: spring-back position of the leader gripper |
| `--use-degrees` / `--no-use-degrees` | on | SO-101 only: units streamed, must match the follower's `use_degrees` |

The leader owns the joint mapping. With `--preset nero` every pose is translated into
Nero joint space on this PC (offsets, signs and gripper scale in
`src/lerobot/teleoperators/koch_leader_remote/koch_remote_nero_config.py`) before it is
sent, so the Nero follower applies no leader math. Any other preset sends raw Koch values.
Run the leader client with the same `--preset` as the robot PC.

The SO-101 client streams raw leader values (degrees by default); an SO leader drives an
SO follower one to one, so there is no mapping step.

It keeps streaming whether or not teleop is running; teleop picks it up mid-stream.
With `so101` (leader on the same PC) this step does not exist.

---

## 4. `teleop.py`, the main tool

Teleoperate, record sessions on demand, and host the link that `inference.py` uses.
Same arguments as `lerobot-teleoperate`, plus the recording ones from `lerobot-record`:

```bash
python -m src.teleop --robot.type=so101_follower --robot.port=COM4 --robot.id=aadi_so101_follower \
                     --teleop.type=so101_leader --teleop.port=COM5 --teleop.id=aadi_so101_leader
python -m src.teleop --preset so101 --display_data=true
python -m src.teleop --preset so101 --resume                       # append to the latest run
python -m src.teleop --preset so101 --run_dir ~/.cache/revobots/arm/sessions/20260918_200511
```

| Arg | Default | Meaning |
| --- | --- | --- |
| `--robot.type` `--robot.port` `--robot.id` | preset | follower, as in `lerobot-teleoperate` |
| `--teleop.type` `--teleop.port` `--teleop.id` | preset | leader (`so101_leader`, `koch_leader_remote`, `keyboard_joints`, …) |
| `--fps` | 30 | control-loop and video fps |
| `--teleop_time_s` | until Esc | stop after this many seconds |
| `--display_data=true` | false | stream observations/actions to the visualizer |
| `--display_mode` | `rerun` | `rerun` or `foxglove` |
| `--display_ip` `--display_port` | none | remote rerun viewer / foxglove bind |
| `--play_sounds=true` | false | speak the prompts (off by default; the text is always logged) |
| `--dataset.single_task` | preset `task` | task sentence stored in every frame |
| `--dataset.repo_id` | preset | only used for the "next: convert" hint |
| `--run_dir` | new `<sessions_root>/<YYYYmmdd_HHMMSS>/` | append sessions to an existing run, numbering continues |
| `--resume` | false | append to the most recent run under `sessions_root` |
| `--sessions_root` | `~/.cache/revobots/arm/sessions` | where runs are created |
| `--record.codec` | `mp4v` | OpenCV FOURCC for the raw session mp4s |
| `--record.redo_starts_recording=false` | true | after Left/`r`, stay idle instead of recording again |
| `--cameras.<name>.<field>` | none | camera sources and settings, e.g. `--cameras.wrist.source=0 --cameras.eye.source=2` |
| `--ai.enabled=true` | false | open the UDP link for `inference.py` and enable the `a` key |

### Keys

The keys are global (they work while the terminal is not focused), like `lerobot-record`.

| Key | When idle | When recording |
| --- | --- | --- |
| **Space** | start recording `session_<i>` | stop and save the session |
| **Left** or **r** | delete the last session saved in this launch and start recording it again | discard the ongoing session and start recording it again |
| **a** | AI control on (teleop started with `--ai.enabled=true`, inference.py running) | same; rows get `source: ai` |
| **Esc** or **q** | quit | save the ongoing session, then quit |

Typical episode: press Space, do the task, press Space. Reset the scene, repeat. If a
take went wrong while recording, press Left: the take is deleted and a new recording of
the same `session_<i>` starts at once. If you noticed after saving, press Left while
idle: the last saved session is deleted and re-recorded.

The status line shows the state:

```
REC session_3 00:12 360f              HUMAN     loop   33.1ms ( 30 Hz)   Space=record/stop  Left/r=redo  a=AI on/off  Esc/q=quit
```

Left never deletes sessions from an earlier launch, even inside an appended run.
A session stopped at zero frames is removed.

### Keyboard jog (`--teleop.type=keyboard_joints`)

Replaces the leader arm with hold-to-move keys, one joint at a time, no IK. The target
starts at the arm's measured pose, so nothing jumps.

| Joint | decrease / increase | | Joint | decrease / increase |
| --- | --- | --- | --- | --- |
| shoulder_pan | `j` / `l` | | wrist_flex | `h` / `y` |
| shoulder_lift | `k` / `i` | | wrist_roll | `g` / `t` |
| elbow_flex | `u` / `o` | | gripper | `[` close / `]` open |
| speed | `-` halve / `+` double | | | |

`--teleop.speed` is in the follower's units per second (degrees for SO-101, default 30),
`--teleop.gripper_speed` likewise. The teleop keys (Space, Left/`r`, `a`, Esc/`q`) keep
working, and sessions recorded this way have the same format as leader-recorded ones.
Keys are global, so do not type in other windows while it runs.

---

## 5. What a run looks like on disk

One teleop launch is one run. Each Space-to-Space take is one session, and one session
becomes one episode in the dataset. There is no split step, unlike the AGV, because the
operator already defines the episode boundaries with Space.

```
~/.cache/revobots/arm/sessions/            ← config.sessions_root
└── 20260918_200511/                       ← run (one teleop launch)
    ├── session_0/
    │   ├── data.jsonl                     ← one JSON object per frame
    │   ├── meta.json                      ← schema, fps, camera shapes, frame count
    │   ├── video_wrist.mp4                ← frame i of every mp4 == row i of data.jsonl
    │   └── video_eye.mp4
    ├── session_1/
    └── session_2/
```

`data.jsonl`, one row per frame (Nero: 8 joints observed, 6 commanded by the Koch leader):

```json
{"frame_index": 0, "timestamp": 0.0, "wall_time": 1758203471.512,
 "observation": {"shoulder_pan.pos": -12.3, "shoulder_lift.pos": 45.1, "forearm_roll.pos": 0.8,
                 "elbow_flex.pos": 30.2, "wrist_roll.pos": 0.4, "wrist_pitch.pos": -3.1,
                 "wrist_flex.pos": -5.0, "gripper.pos": 12.0},
 "action":      {"shoulder_pan.pos": -12.0, "shoulder_lift.pos": 45.5, "elbow_flex.pos": 30.0,
                 "wrist_roll.pos": 0.5, "wrist_flex.pos": -5.2, "gripper.pos": 15.0},
 "task": "Pick up the cube and place it in the bin.",
 "source": "human"}
```

`meta.json`:

```json
{
  "session_name": "session_0",
  "robot_type": "agx_nero_follower",
  "robot_id": "revobots_nero",
  "teleop_type": "koch_leader_remote",
  "preset": "nero",
  "task": "Pick up the cube and place it in the bin.",
  "fps": 30,
  "codec": "mp4v",
  "state_names":  ["shoulder_pan.pos", "shoulder_lift.pos", "forearm_roll.pos", "elbow_flex.pos",
                   "wrist_roll.pos", "wrist_pitch.pos", "wrist_flex.pos", "gripper.pos"],
  "action_names": ["shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos",
                   "wrist_roll.pos", "wrist_flex.pos", "gripper.pos"],
  "cameras": {
    "wrist": {"height": 480, "width": 640, "channels": 3, "video": "video_wrist.mp4"},
    "eye":   {"height": 480, "width": 640, "channels": 3, "video": "video_eye.mp4"}
  },
  "camera_backends": {"wrist": "gstreamer", "eye": "opencv"},
  "started_at": "2026-09-18T20:05:11",
  "num_frames": 900,
  "duration_s": 29.97,
  "lerobot_features": {"...": "the LeRobot feature dict, for reference"}
}
```

Values are exactly what the follower and leader report: degrees for Nero and SO-101,
`-100…100` for Koch body joints, `0…100` for grippers. Nothing is rescaled.

Delete a bad session by deleting its folder; `convert.py` skips gaps in the numbering.

---

## 6. `record.py`, timed recording

The `lerobot-record` style alternative: fixed episode length, fixed reset time, N
episodes, same session output as `teleop.py`. Uses LeRobot's own `record_loop` and
takes `lerobot-record`'s arguments:

```bash
python -m src.record --robot.type=so101_follower --robot.port=COM4 --robot.id=aadi_so101_follower \
                     --teleop.type=so101_leader --teleop.port=COM5 --teleop.id=aadi_so101_leader \
                     --dataset.repo_id=Aadi/revobots_so101_v1 --dataset.single_task="Pick the cube" \
                     --dataset.num_episodes=10 --dataset.episode_time_s=30 --dataset.reset_time_s=10
```

| Arg | Default (`config.record`) | Meaning |
| --- | --- | --- |
| `--dataset.num_episodes` | 50 | sessions to record |
| `--dataset.episode_time_s` | 30 | seconds per session |
| `--dataset.reset_time_s` | 10 | seconds of un-recorded teleop between sessions |
| `--dataset.single_task` | preset `task` | task sentence stored in every frame |
| `--display_data` `--display_mode` `--play_sounds` `--fps` `--run_dir` `--resume` | as in teleop | |

Keys: **→** or **n** end the episode early, **←** or **r** re-record it, **Esc** or **q** stop.

---

## 7. `convert.py`, sessions to a LeRobot dataset

Turns a run folder into a LeRobot dataset, one session per episode, with the same
helpers `lerobot-record` uses (`hw_to_dataset_features`, `build_dataset_frame`,
`save_episode`). Joint names, camera shapes, fps and robot type come from `meta.json`.
Dataset arguments are `lerobot-record`'s `--dataset.*`:

```bash
python -m src.convert --sessions_dir ~/.cache/revobots/arm/sessions/20260918_200511 \
                      --dataset.repo_id=Aadi/revobots_so101_v1
python -m src.convert --sessions_dir <run> --dataset.repo_id=Aadi/revobots_so101_v1 --resume            # newer sessions only
python -m src.convert --sessions_dir <run> --dataset.repo_id=Aadi/revobots_so101_v1 --dataset.push_to_hub=true
```

| Arg | Default | Meaning |
| --- | --- | --- |
| `--sessions_dir` | required | run folder with `session_0/`, `session_1/` … (or one session folder) |
| `--dataset.repo_id` | preset | `<user>/<name>` |
| `--dataset.root` | `$HF_LEROBOT_HOME/<repo_id>` | local dataset directory |
| `--dataset.single_task` | from the sessions | override the task string |
| `--dataset.fps`, `--robot_type` | from `meta.json` | overrides |
| `--resume` | false | append to an existing dataset; sessions below its episode count are assumed done |
| `--dataset.push_to_hub=true`, `--dataset.private=true`, `--dataset.tags='["a","b"]'` | preset | upload after writing |
| `--dataset.video=false` | true | store PNG images instead of video |
| `--dataset.num_image_writer_threads_per_camera` | 4 | |
| `--dataset.video_encoding_batch_size` | 1 | episodes per encoding batch |
| `--no_obs_processor` | false | skip the observation processor (keep it on, inference applies the same one) |
| `--limit` | all | convert at most N sessions |

All sessions in one run must share fps, joint names and cameras; the converter stops
otherwise. Merging several runs into one dataset: convert the first run, then the
others with `--resume` into the same `--dataset.repo_id`.

Check the result before training at https://huggingface.co/spaces/lerobot/visualize_dataset
(after pushing) or with `lerobot-dataset-viz`.

---

## 8. `replay.py`

Plays recorded actions back on the follower, from a raw session or from a dataset
episode. A sanity check that the data drives the arm the way you moved it. Same
arguments as `lerobot-replay`, plus `--session_dir`:

```bash
python -m src.replay --robot.type=so101_follower --robot.port=COM4 --robot.id=aadi_so101_follower \
                     --session_dir ~/.cache/revobots/arm/sessions/20260918_200511/session_0
python -m src.replay --robot.type=so101_follower --robot.port=COM4 --robot.id=aadi_so101_follower \
                     --dataset.repo_id=Aadi/revobots_so101_v1 --dataset.episode=3
```

| Arg | Default | Meaning |
| --- | --- | --- |
| `--robot.type` `--robot.port` `--robot.id` | preset | follower |
| `--session_dir` | | raw session folder |
| `--dataset.repo_id` + `--dataset.episode` | preset, episode 0 | dataset source (`--dataset.root` for a local copy) |
| `--dataset.fps` | from the source | playback rate |
| `--replay.ramp_s` | 1.5 (Nero 2.0) | seconds to glide from the current pose to the first recorded pose; `0` disables |
| `--play_sounds=true` | false | speak the prompts |

Keep a hand near the power switch the first time. The ramp avoids the initial jump,
but the recorded motion itself plays at full recorded speed. Cameras are not opened.

---

## 9. Training

Unchanged LeRobot CLI:

```bash
lerobot-train --dataset.repo_id=Aadi/revobots_so101_v1 --policy.type=act \
              --output_dir=outputs/train/act_so101 --job_name=act_so101 --batch_size=8
```

---

## 10. `inference.py`

Runs a trained policy while **teleop keeps running** in another terminal. Teleop owns
the arm and cameras; inference reads frames from the frame bus and joint state from
teleop, and sends actions back. The arm follows them only after you press `a` in the
teleop terminal. Arguments follow `lerobot_inference`:

```bash
# terminal 1: the AI link is off by default, and the cameras must match the training set
python -m src.teleop --preset so101 --ai.enabled=true --cameras.wrist.source=0 --cameras.eye.source=2
# terminal 2
python -m src.inference --robot.type=so101_follower \
                        --policy.path=outputs/train/act_so101/checkpoints/last/pretrained_model \
                        --dataset_repo_id=Aadi/revobots_so101_v1 --single_task="Pick the cube" \
                        --fps=30 --control_time_s=120
# then press `a` in terminal 1
```

| Arg | Default (`config.policy`) | Meaning |
| --- | --- | --- |
| `--policy.path` | `policy.path` | checkpoint dir or Hub id |
| `--policy.device` | `cuda` | `cuda` / `cpu` / `mps` |
| `--dataset_repo_id` | `policy.dataset_repo_id`, then `dataset.repo_id` | training dataset (feature spec + normalisation stats) |
| `--root` | `policy.dataset_root` | local dataset copy, skips the Hub lookup |
| `--single_task` | preset `task` | task prompt |
| `--fps` | `policy.fps`, then 30 | control rate |
| `--control_time_s` | 60 | seconds to run |
| `--display_data=true` | false | stream to rerun |
| `--robot.type` | preset | only the type is used (teleop owns the hardware) |

Hand-over rules, all enforced in teleop:

- `a` toggles AI control. While on, the status line shows `AI`, or `AI(wait)` when no
  fresh action has arrived within `ai.action_ttl_s` (the arm then holds its pose).
- Moving the leader by more than `ai.human_takeover_threshold` hands control back to
  you immediately. Pressing `a` again does the same.
- Space still records. A run recorded with AI on is an eval session in the same format,
  with `"source": "ai"` in every row.

Ports: state teleop → inference on UDP `5011`, actions inference → teleop on UDP `5010`
(`--ai.state_port`, `--ai.action_port`). Both default to `127.0.0.1`.

---

## 11. SO-101 command sheet

Fully spelled out (no preset). The default preset still supplies anything not given,
so `--dataset.repo_id` is included where it matters. Cameras are off by default: the
`--cameras.*` flags below are only needed when the cameras are plugged in (drop one, or
both, to run without them).

```bash
# teleop: leader + follower, without / with display
python -m src.teleop --robot.type=so101_follower --robot.port=COM4 --robot.id=aadi_so101_follower --teleop.type=so101_leader --teleop.port=COM5 --teleop.id=aadi_so101_leader --cameras.wrist.source=0 --cameras.eye.source=2
python -m src.teleop --robot.type=so101_follower --robot.port=COM4 --robot.id=aadi_so101_follower --teleop.type=so101_leader --teleop.port=COM5 --teleop.id=aadi_so101_leader --cameras.wrist.source=0 --cameras.eye.source=2 --display_data=true

# record: Space starts/stops sessions inside teleop (commands above), or timed episodes:
python -m src.record --robot.type=so101_follower --robot.port=COM4 --robot.id=aadi_so101_follower --teleop.type=so101_leader --teleop.port=COM5 --teleop.id=aadi_so101_leader --cameras.wrist.source=0 --cameras.eye.source=2 --dataset.repo_id=Aadi/revobots_so101_v1 --dataset.single_task="Pick the cube" --dataset.num_episodes=10 --dataset.episode_time_s=30 --dataset.reset_time_s=10
python -m src.record --robot.type=so101_follower --robot.port=COM4 --robot.id=aadi_so101_follower --teleop.type=so101_leader --teleop.port=COM5 --teleop.id=aadi_so101_leader --cameras.wrist.source=0 --cameras.eye.source=2 --dataset.repo_id=Aadi/revobots_so101_v1 --dataset.single_task="Pick the cube" --dataset.num_episodes=10 --dataset.episode_time_s=30 --dataset.reset_time_s=10 --display_data=true

# convert the run to a LeRobot dataset
python -m src.convert --sessions_dir %USERPROFILE%\.cache\revobots\arm\sessions\<run> --dataset.repo_id=Aadi/revobots_so101_v1

# replay: raw session / dataset episode
python -m src.replay --robot.type=so101_follower --robot.port=COM4 --robot.id=aadi_so101_follower --session_dir %USERPROFILE%\.cache\revobots\arm\sessions\<run>\session_0
python -m src.replay --robot.type=so101_follower --robot.port=COM4 --robot.id=aadi_so101_follower --dataset.repo_id=Aadi/revobots_so101_v1 --dataset.episode=0

# keyboard control of the follower, without / with display
python -m src.teleop --robot.type=so101_follower --robot.port=COM4 --robot.id=aadi_so101_follower --teleop.type=keyboard_joints --teleop.speed=30 --cameras.wrist.source=0 --cameras.eye.source=2
python -m src.teleop --robot.type=so101_follower --robot.port=COM4 --robot.id=aadi_so101_follower --teleop.type=keyboard_joints --teleop.speed=30 --cameras.wrist.source=0 --cameras.eye.source=2 --display_data=true

# the same with the preset
python -m src.teleop --preset so101
python -m src.teleop --preset so101 --display_data=true
python -m src.record --preset so101 --dataset.num_episodes=10 --dataset.episode_time_s=30 --dataset.reset_time_s=10
python -m src.replay --preset so101 --session_dir %USERPROFILE%\.cache\revobots\arm\sessions\<run>\session_0
python -m src.replay --preset so101 --dataset.episode=0
python -m src.teleop --preset so101_keyboard
```

---

## 11b. `test_mediapipe.py` (experimental): gripper from a thumb-index pinch

Webcam → MediaPipe Hands → pinch distance (normalised by hand size) → `gripper.pos`.
Only the gripper moves; the other joints are held where they are.

```bash
pip install mediapipe
python -m src.test_mediapipe --dry_run=true                 # tune first: read the on-screen "pinch" number
python -m src.test_mediapipe --robot.type=so101_follower --robot.port=COM4 --robot.id=aadi_so101_follower
```

| Arg | Default | Meaning |
| --- | --- | --- |
| `--camera_index` | 0 | webcam for the hand (independent of the robot cameras) |
| `--dry_run=true` | false | no robot, just the window and the value |
| `--pinch_closed` / `--pinch_open` | 0.25 / 1.0 | pinch ratio at fully closed / fully open; read them off the window with your hand pinched and open |
| `--invert=true` | false | flip if the gripper moves the wrong way |
| `--smooth` | 0.5 | EMA factor, lower = smoother and laggier |
| `--gripper_min` / `--gripper_max` | 0 / 100 | output clamp |
| `--show=false` | true | no window, print to the terminal instead |

Press `q` in the window or Ctrl-C to stop. Current MediaPipe builds have no legacy
`solutions.hands`; the script then uses the Tasks API and downloads `hand_landmarker.task`
(about 7 MB) into `revobots_arm/models/` on first run. Offline, download it yourself and
pass `--hand_model=<path>`.

---

## 11c. SO-101 IK and gamepad Cartesian control

`src/kinematics.py` is a numpy-only FK / IK solver that reads any serial-arm URDF (no
placo needed). `src/so101_ik.py` is the SO-101 profile, using the bundled
`urdf/so101/so101_new_calib.urdf` (joints only, no meshes). Coordinates are the URDF
`base_link` frame in metres, angles in degrees, tool point `gripper_frame_link`.

```bash
# 1. no robot: Jacobian check + FK/IK round trips, should print "self-test PASSED"
python -m src.so101_ik --self_test

# 2. where is the tool now? (move the arm by hand first to see which way +X / +Y / +Z point)
python -m src.so101_ik --robot.type=so101_follower --robot.port=COM4 --robot.id=aadi_so101_follower --print_pose

# 3. send a Cartesian target: any of --x --y --z (m) and --roll --pitch --yaw (deg); the rest keeps the current pose
python -m src.so101_ik --robot.type=so101_follower --robot.port=COM4 --robot.id=aadi_so101_follower --z=0.15 --dry_run
python -m src.so101_ik --robot.type=so101_follower --robot.port=COM4 --robot.id=aadi_so101_follower --z=0.15 --move_time_s=3
```

The move follows a straight Cartesian line, solved every tick. The end pose is solved
first, and nothing moves if it is out of reach (`--max_pos_error_mm`, default 3) or a
joint would jump (`--max_joint_step_deg`, default 10). The SO-101 has 5 arm joints:
position is always exact, orientation only as far as the remaining two DOF allow (an
arbitrary yaw usually is not reachable). The residual is printed. If `--print_pose`
disagrees with reality for one joint, fix the mapping with `--invert_joints=elbow_flex`
or `--offset_deg=elbow_flex:5`.

### Gamepad Cartesian jog (`--teleop.type=gamepad_ee`)

Needs `pip install pygame`. Check the pad first with `python -m src.gamepad`.

```bash
python -m src.teleop --robot.type=so101_follower --robot.port=COM4 --robot.id=aadi_so101_follower --teleop.type=gamepad_ee
```

| Gamepad input | Function |
| --- | --- |
| Left stick up / down | X forward / backward |
| Left stick left / right | Y left / right |
| Right stick up / down | Z up / down (height) |
| Right stick left / right | yaw: swing the whole arm left / right (shoulder_pan, see below) |
| RT / LT | pitch: tilt the gripper up / down, **wrist_flex only** |
| D-pad left / right | roll: rotate the gripper, **wrist_roll only** |
| D-pad up / down | speed up / down (x0.1 … x1.5, starts at x0.5 = 5 cm/s) |
| A / B | gripper open / close |
| X | precision mode on/off (x0.2) |
| Y | toggle base / tool reference frame |
| RB, LB, Start | reserved (RB becomes a hold-to-move deadman with `--teleop.require_deadman=true`) |

Recording keys (Space, Left/`r`, Esc/`q`) work as with the leader arm, and the recorded
action is still joint positions, so gamepad data trains the same policy.

**Why pitch / yaw behave the way they do (5-DOF arm).** The SO-101's shoulder, elbow and
wrist-flex axes are parallel, so the gripper always points inside the arm's vertical
plane. The default `orientation_mode=arm` uses only what the arm can do:

- *Pitch* (RT/LT) moves only the wrist-flex motor: the gripper swings up or down about
  the wrist and no other motor moves. It stops at the wrist-flex limit.
- *Roll* (D-pad) moves only the wrist-roll motor.
- *Yaw*: turning the gripper to face left or right **without moving it is physically
  impossible** on 5 joints, so yaw moves only the shoulder pan: the whole arm swings and
  the tool moves on an arc around the base.
- While translating (sticks), the IK keeps the gripper's tilt and roll, so the wrist
  moves along with shoulder and elbow; the gripper's heading follows the arm.
  `--teleop.wrist_direct=false` makes pitch/roll keep the tool point still instead (then
  shoulder and elbow move too).
  With the gripper pointing straight down, sideways moves keep its rotation via wrist roll.
- With `Y` (tool frame), the left stick moves along the gripper: up = push forward
  along the gripper axis, left / right = sideways, RT / LT = the gripper's own up / down.

`--teleop.orientation_mode=frame` restores plain base/tool-axis rotations (for 6+ DOF arms).

Useful `--teleop.*` settings (all in `GamepadEETeleopConfig`, `src/gamepad_teleop.py`):

| Setting | Default | Meaning |
| --- | --- | --- |
| `invert` | `""` | flip axes that move the wrong way, e.g. `--teleop.invert=x,pitch` |
| `max_linear_speed` / `max_angular_speed` | 0.10 m/s / 1.0 rad/s | at speed x1.0 |
| `initial_speed_level` | 2 | index into `speed_levels` `[0.1, 0.25, 0.5, 0.75, 1.0, 1.5]` |
| `precision_scale` | 0.2 | X button multiplier |
| `frame` | `base` | starting translation frame (`base` or `tool`) |
| `orientation_mode` | `arm` | `arm` = 5-DOF aware pitch/roll/yaw as above; `frame` = rotations about base/tool axes |
| `wrist_direct` | true | arm mode: pitch/roll move only their wrist motor; `false` = keep the tool point still via IK |
| `pitch_joint` / `roll_joint` / `yaw_joint` | `wrist_flex` / `wrist_roll` / `shoulder_pan` | joints that define the arm-mode axes |
| `z_min_m` | 0.02 | tool cannot be driven below this height; `none` disables |
| `orientation_weight` | 1.0 | 0 = position only, orientation left free |
| `max_pos_error_m` | 0.003 | steps the IK cannot reach within this are rejected (arm holds) |
| `max_joint_step_deg` | 5 | per tick |
| `deadzone` / `expo` | 0.12 / 1.5 | stick shaping |
| `require_deadman` | false | `true` = the arm only moves while RB is held |
| `debug` | false | log the active inputs and raw axes twice a second, to find a stick or trigger that is not at rest |
| `invert_joints` / `joint_offsets_deg` | `""` | URDF ↔ motor mapping fixes, same as for `so101_ik` |

**Other robots.** The mapping and the jog logic are robot-agnostic; only the kinematics
is per arm. Point `--teleop.urdf_path`, `--teleop.tip_link` and `--teleop.motors`
(JSON list, chain order) at that arm's URDF. The follower must report joints in
degrees (Koch: `--robot.use_degrees=true`). Nero now includes its arm/gripper URDF
and a native seven-joint interface. Use `nero_gamepad` or `nero_vr`; see
[NERO_IK.md](NERO_IK.md) for Cartesian moves and TCP configuration.

---

## 12. Troubleshooting

| Symptom | Cause / fix |
| --- | --- |
| `unknown override --foo` | a flag the script does not know; `--help` lists the script flags, the epilog lists the config sections |
| `'KochFollowerConfig' does not accept ['use_degrees_x']` | a `--robot.*` key that LeRobot's config dataclass does not have; the message lists the accepted keys |
| `cameras failed to open: ['eye']` at start | wrong `source`; run `python -m src.cameras find` and use the node marked `ok`, or drop the `--cameras.eye.*` flags |
| `target out of reach (… mm): workspace edge or joint limit; holding.` | gamepad jog hit the edge of the workspace or a joint limit; move back the other way |
| `[...] outside the URDF joint limits. Is the follower in degrees` | the follower is not in degrees, or a joint's sign/offset differs from the URDF; check with `so101_ik --print_pose` |
| `URDF not found` | copy the `urdf/` folder along with `src/` |
| `AI link is off. Restart teleop with --ai.enabled=true` | the `a` key was pressed without the AI link; add `--ai.enabled=true` when you intend to run `inference.py` |
| `open VIDEOIO/FFMPEG: Failed list devices for backend dshow` | harmless OpenCV warning on Windows while probing cameras |
| `cameras delivered no frame within 10s` | device opened but streams nothing; try `--cameras.eye.pixel_format=YUYV`, another USB port, or `hw_decode=False` |
| teleop status line frozen, `Waiting for actions from the remote leader arm` | `leader_client.py` is not streaming or `--ip` / port mismatch; the loop resumes by itself when packets arrive |
| `nothing received from teleop within 5s` from inference | teleop is not running, a camera has `publish_frames=False`, or the `--ai.*` ports differ between the two processes |
| `policy was trained with cameras [...] config has [...]` warning | camera names must match the dataset exactly; rename them in `config.py` |
| `Refusing to write into a non-empty session folder` | a leftover folder in the run; delete it or use a new run |
| `<session> has a different schema/fps/camera set` in convert | you changed cameras or fps between sessions; convert those runs into separate datasets |
| keys do nothing | the keyboard backend needs a display session (X11/Windows/macOS) or an interactive TTY; see the warning printed at start |
| Windows: `.venv` or env prompts | none of these tools create environments; run them with your existing `lerobot` interpreter |

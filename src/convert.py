"""Convert recorded sessions into a LeRobot (Hugging Face) dataset.

Dataset arguments follow lerobot-record's ``--dataset.*``:

    python -m src.convert --sessions_dir <run> --dataset.repo_id=Aadi/revobots_so101_v1
    python -m src.convert --sessions_dir <run> --dataset.repo_id=... --resume          # append newer sessions
    python -m src.convert --sessions_dir <run> --dataset.repo_id=... --dataset.push_to_hub=true --dataset.private=true
    python -m src.convert --sessions_dir <run> --dataset.repo_id=... --dataset.single_task="Pick the cube" --dataset.fps=30
    python -m src.convert --sessions_dir <run> --dataset.repo_id=... --dataset.video=false --dataset.video_encoding_batch_size=4

One ``session_<i>/`` folder becomes one episode. This mirrors
``src/lerobot/robots/revobots_agv_follower/data_convert_agv.py``: the same
``hw_to_dataset_features`` / ``build_dataset_frame`` / ``VideoEncodingManager`` /
``save_episode`` calls the live recorder uses, and the same observation-processor
symmetry with inference. Unlike the AGV version nothing about the robot is
hard-coded: joint names, camera shapes, fps and robot type come from ``meta.json``.
"""

from __future__ import annotations

if __package__ in (None, ""):  # allow `python src/convert.py`
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "src"

import logging
from pathlib import Path
from typing import Any

from ._bootstrap import ensure_lerobot_importable
from .recorder import SessionFrames, find_sessions, load_meta

logger = logging.getLogger(__name__)


def build_features(meta: dict[str, Any], use_videos: bool = True) -> dict:
    from lerobot.utils.constants import ACTION, OBS_STR
    from lerobot.utils.feature_utils import hw_to_dataset_features

    hw_obs: dict[str, Any] = {name: float for name in meta["state_names"]}
    for cam, info in meta.get("cameras", {}).items():
        if "height" not in info:
            raise ValueError(f"Camera '{cam}' in {meta.get('session_name')} has no recorded frames/shape.")
        hw_obs[cam] = (int(info["height"]), int(info["width"]), int(info.get("channels", 3)))
    hw_action: dict[str, Any] = {name: float for name in meta["action_names"]}

    return {
        **hw_to_dataset_features(hw_action, ACTION),
        **hw_to_dataset_features(hw_obs, OBS_STR, use_video=use_videos),
    }


def _schema_signature(meta: dict[str, Any]) -> tuple:
    cams = tuple(sorted((k, v.get("height"), v.get("width")) for k, v in meta.get("cameras", {}).items()))
    return (meta["fps"], tuple(meta["state_names"]), tuple(meta["action_names"]), cams)


def add_session_as_episode(session_dir: Path, dataset, task: str | None, robot_observation_processor) -> int:
    """Append one session as one episode. Mirrors the inner body of ``record_loop``."""
    from tqdm import tqdm

    from lerobot.utils.constants import ACTION, OBS_STR
    from lerobot.utils.feature_utils import build_dataset_frame

    n_written = 0
    with SessionFrames(session_dir) as frames:
        total = len(frames)
        if total == 0:
            logger.warning("%s has no frames, skipping", session_dir.name)
            return 0
        for row, images in tqdm(frames, total=total, desc=f"  {session_dir.name}", unit="f", leave=True, dynamic_ncols=True):
            raw_obs = {**row["observation"], **images}
            raw_action = dict(row["action"])

            obs_processed = robot_observation_processor(raw_obs)
            observation_frame = build_dataset_frame(dataset.features, obs_processed, prefix=OBS_STR)
            action_frame = build_dataset_frame(dataset.features, raw_action, prefix=ACTION)

            frame = {**observation_frame, **action_frame, "task": task or row.get("task") or ""}
            dataset.add_frame(frame)
            n_written += 1

    dataset.save_episode()
    return n_written


def run_convert(
    sessions_dir: str | Path,
    repo_id: str,
    task: str | None = None,
    fps: int | None = None,
    root: str | Path | None = None,
    robot_type: str | None = None,
    image_writer_threads_per_camera: int = 4,
    video_encoding_batch_size: int = 1,
    resume: bool = False,
    push_to_hub: bool = False,
    private: bool | None = None,
    tags: list[str] | None = None,
    no_videos: bool = False,
    no_obs_processor: bool = False,
    limit: int | None = None,
) -> Path:
    ensure_lerobot_importable()
    from lerobot.datasets import LeRobotDataset, VideoEncodingManager
    from lerobot.processor import make_default_processors

    sessions = find_sessions(Path(sessions_dir))
    if not sessions:
        raise SystemExit(f"No session_<i>/ folders (with data.jsonl + meta.json) under {sessions_dir}")

    metas = {s: load_meta(s) for s in sessions}
    first = metas[sessions[0]]
    signature = _schema_signature(first)
    for s, m in metas.items():
        if _schema_signature(m) != signature:
            raise SystemExit(
                f"{s.name} has a different schema/fps/camera set than {sessions[0].name}; "
                "convert them into separate datasets."
            )

    fps = fps or int(first["fps"])
    robot_type = robot_type or str(first.get("robot_type", "unknown"))
    features = build_features(first, use_videos=not no_videos)
    n_cams = len(first.get("cameras", {}))

    _, _, robot_observation_processor = make_default_processors()
    if no_obs_processor:
        robot_observation_processor = lambda obs: obs  # noqa: E731

    print(f"[*] sessions dir : {sessions_dir}")
    print(f"[*] sessions     : {len(sessions)}")
    print(f"[*] repo id      : {repo_id}")
    print(f"[*] fps          : {fps}")
    print(f"[*] robot type   : {robot_type}")
    print(f"[*] obs processor: {'DISABLED' if no_obs_processor else 'enabled (symmetric with inference)'}")
    print("\n[*] dataset features:")
    for k, v in features.items():
        print(f"      {k:35s} dtype={v['dtype']:8s} shape={tuple(v['shape'])} names={v.get('names', '')}")
    print()

    if resume:
        dataset = LeRobotDataset.resume(
            repo_id,
            root=root,
            batch_encoding_size=video_encoding_batch_size,
            image_writer_threads=image_writer_threads_per_camera * n_cams,
        )
        start_idx = dataset.num_episodes
        print(f"[*] resuming at episode {start_idx}: sessions before index {start_idx} are assumed converted")
    else:
        dataset = LeRobotDataset.create(
            repo_id,
            fps,
            features=features,
            root=root,
            robot_type=robot_type,
            use_videos=not no_videos,
            image_writer_threads=image_writer_threads_per_camera * n_cams,
            batch_encoding_size=video_encoding_batch_size,
        )
        start_idx = 0

    todo = sessions[start_idx:]
    if limit is not None:
        todo = todo[:limit]
    print(f"[*] converting {len(todo)} session(s)\n")

    total_frames = 0
    try:
        with VideoEncodingManager(dataset):
            for offset, session_dir in enumerate(todo):
                ep_idx = start_idx + offset
                print(f"Episode {ep_idx}  <-  {session_dir}")
                total_frames += add_session_as_episode(session_dir, dataset, task, robot_observation_processor)
    finally:
        print("\n[*] finalizing dataset ...")
        dataset.finalize()

    print(f"[✓] wrote {total_frames} frames across {len(todo)} episode(s)")
    print(f"[✓] dataset root: {dataset.root}")

    if push_to_hub:
        print("[*] pushing to the Hugging Face Hub ...")
        dataset.push_to_hub(tags=tags, private=private)
        print("[✓] pushed")

    return Path(dataset.root)


def main() -> None:
    from .devices import add_bool_flag, load_config, make_parser, overridden

    parser = make_parser(__doc__)
    parser.add_argument("--sessions_dir", required=True, help="Run folder containing session_0/, session_1/, ... (or one session)")
    add_bool_flag(parser, "--resume", "Append to an existing dataset (sessions below its episode count are skipped)")
    parser.add_argument("--robot_type", default=None, help="Override robot_type (default: from meta.json)")
    add_bool_flag(parser, "--no_obs_processor", "Skip the observation processor (keep it on; inference applies the same one)")
    parser.add_argument("--limit", type=int, default=None, help="Convert at most N sessions")
    args, overrides = parser.parse_known_args()

    cfg = load_config(args.preset, overrides)
    ds = cfg.dataset
    if not ds.repo_id:
        raise SystemExit("Pass --dataset.repo_id=<user>/<name> or set dataset.repo_id in src/config.py.")

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    run_convert(
        sessions_dir=args.sessions_dir,
        repo_id=ds.repo_id,
        # Only override what was given on the command line; otherwise keep the sessions' own values.
        task=cfg.task if overridden(overrides, "task", "dataset.single_task") else None,
        fps=cfg.fps if overridden(overrides, "fps", "dataset.fps") else None,
        root=ds.root,
        robot_type=args.robot_type,
        image_writer_threads_per_camera=ds.num_image_writer_threads_per_camera,
        video_encoding_batch_size=ds.video_encoding_batch_size,
        resume=args.resume,
        push_to_hub=ds.push_to_hub,
        private=ds.private,
        tags=ds.tags,
        no_videos=not ds.video,
        no_obs_processor=args.no_obs_processor,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()

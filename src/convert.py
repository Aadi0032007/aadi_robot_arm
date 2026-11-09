"""Convert recorded sessions into a LeRobot (Hugging Face) dataset.

Dataset arguments follow lerobot-record's ``--dataset.*``:

    python -m src.convert --sessions_dir <run> --dataset.repo_id=Aadi/revobots_so101_v1
    python -m src.convert --sessions_dir <run> --dataset.repo_id=... --resume          # append newer sessions
    python -m src.convert --sessions_dir <run> --dataset.repo_id=... --dataset.push_to_hub=true --dataset.private=true
    python -m src.convert --sessions_dir <run> --dataset.repo_id=... --dataset.single_task="Pick the cube" --dataset.fps=30
    python -m src.convert --sessions_dir <run> --dataset.repo_id=... --dataset.video=false --dataset.video_encoding_batch_size=4
    python -m src.convert --sessions_dir <run> --dataset.repo_id=... --dataset.depth_encoder.depth_max=1.0 --dataset.depth_encoder.use_log=false   # D405

One ``session_<i>/`` folder becomes one episode. This mirrors
``src/lerobot/robots/revobots_agv_follower/data_convert_agv.py``: the same
``hw_to_dataset_features`` / ``build_dataset_frame`` / ``VideoEncodingManager`` /
``save_episode`` calls the live recorder uses, and the same observation-processor
symmetry with inference. Unlike the AGV version nothing about the robot is
hard-coded: joint names, camera shapes, fps and robot type come from ``meta.json``.

Depth keys (``<cam>_depth``, one channel, uint16 mm) become
``observation.images.<cam>_depth`` with ``info.is_depth_map=True``, which makes
``LeRobotDataset`` store them with its depth encoder, as ``lerobot-record`` does for a
RealSense with ``use_depth=True``. ``--dataset.depth_encoder.*`` (depth_min, depth_max,
shift, use_log; metres) sets that encoder like ``lerobot-record`` does; LeRobot's
defaults (0.01-10 m, log) otherwise. ``--resume`` always keeps the dataset's own settings.
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
        # channels=1 → depth map (uint16 mm); hw_to_dataset_features flags it is_depth_map.
        hw_obs[cam] = (int(info["height"]), int(info["width"]), int(info.get("channels", 3)))
    hw_action: dict[str, Any] = {name: float for name in meta["action_names"]}

    return {
        **hw_to_dataset_features(hw_action, ACTION),
        **hw_to_dataset_features(hw_obs, OBS_STR, use_video=use_videos),
    }


def _schema_signature(meta: dict[str, Any]) -> tuple:
    cams = tuple(sorted((k, v.get("height"), v.get("width")) for k, v in meta.get("cameras", {}).items()))
    return (meta["fps"], tuple(meta["state_names"]), tuple(meta["action_names"]), cams)


def _depth_settings(enc: Any) -> dict[str, Any]:
    """The four depth quantizer settings of a ``DepthEncoderConfig`` or a plain dict, as floats / bool."""
    values = enc if isinstance(enc, dict) else vars(enc)
    return {
        "depth_min": float(values["depth_min"]),
        "depth_max": float(values["depth_max"]),
        "shift": float(values["shift"]),
        "use_log": bool(values["use_log"]),
    }


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
    depth_encoder: dict[str, Any] | None = None,
) -> Path:
    ensure_lerobot_importable()
    from lerobot.configs import DepthEncoderConfig, encoder_config_from_video_info
    from lerobot.datasets import LeRobotDataset, LeRobotDatasetMetadata, VideoEncodingManager
    from lerobot.processor import make_default_processors
    from lerobot.utils.constants import HF_LEROBOT_HOME

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

    if resume:
        # LeRobotDataset.resume() needs an explicit root; this is where create() put it.
        root = Path(root) if root else HF_LEROBOT_HOME / repo_id

    # Depth storage. On resume it must stay what the dataset already uses: info.json
    # carries one set of settings for every episode of a depth key.
    depth_encoder_cfg = None
    has_depth = any((ft.get("info") or {}).get("is_depth_map") for ft in features.values())
    if has_depth and not no_videos:
        if resume:
            meta = LeRobotDatasetMetadata(repo_id, root=root)
            stored = [k for k in meta.depth_keys if k in meta.video_keys]
            if stored:
                depth_encoder_cfg = encoder_config_from_video_info(meta.features[stored[0]].get("info"))
                if depth_encoder is not None and _depth_settings(depth_encoder_cfg) != _depth_settings(depth_encoder):
                    raise SystemExit(
                        f"{repo_id} stores depth with {_depth_settings(depth_encoder_cfg)}, the command asks for "
                        f"{_depth_settings(depth_encoder)}. New episodes must match: drop the "
                        "--dataset.depth_encoder.* flags, or convert into a new --dataset.repo_id."
                    )
        elif depth_encoder is not None:
            depth_encoder_cfg = DepthEncoderConfig(**_depth_settings(depth_encoder))

    print(f"[*] sessions dir : {sessions_dir}")
    print(f"[*] sessions     : {len(sessions)}")
    print(f"[*] repo id      : {repo_id}")
    print(f"[*] fps          : {fps}")
    print(f"[*] robot type   : {robot_type}")
    print(f"[*] obs processor: {'DISABLED' if no_obs_processor else 'enabled (symmetric with inference)'}")
    if depth_encoder_cfg is not None:
        d = _depth_settings(depth_encoder_cfg)
        print(f"[*] depth storage: {d['depth_min']:g}-{d['depth_max']:g} m, {'log' if d['use_log'] else 'linear'}"
              + (" (from the dataset)" if resume else ""))
    print("\n[*] dataset features:")
    for k, v in features.items():
        print(f"      {k:35s} dtype={v['dtype']:8s} shape={tuple(v['shape'])} names={v.get('names', '')}")
    print()

    if resume:
        dataset = LeRobotDataset.resume(
            repo_id,
            root=root,
            batch_encoding_size=video_encoding_batch_size,
            depth_encoder=depth_encoder_cfg,
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
            depth_encoder=depth_encoder_cfg,
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

    print(f"[ok] wrote {total_frames} frames across {len(todo)} episode(s)")
    print(f"[ok] dataset root: {dataset.root}")

    if push_to_hub:
        print("[*] pushing to the Hugging Face Hub ...")
        dataset.push_to_hub(tags=tags, private=private)
        print("[ok] pushed")

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

    depth_flags = ("dataset.depth_min", "dataset.depth_max", "dataset.depth_shift", "dataset.depth_use_log")
    depth_encoder = dict(depth_min=ds.depth_min, depth_max=ds.depth_max, shift=ds.depth_shift, use_log=ds.depth_use_log)

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
        # On resume the dataset's own settings apply; flags given there are only checked against them.
        depth_encoder=depth_encoder if not args.resume or overridden(overrides, *depth_flags) else None,
    )


if __name__ == "__main__":
    main()

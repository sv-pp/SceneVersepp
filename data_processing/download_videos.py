#!/usr/bin/env python3
"""Download YouTube videos for SceneVerse++ scenes from data_info.json."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
import yt_dlp 

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download YouTube videos for scenes that contain data_info.json."
    )
    parser.add_argument(
        "dataset_root",
        type=Path,
        help="Root directory containing scene folders.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Redownload videos even if the target file already exists.",
    )
    parser.add_argument(
        "--scene-name",
        default=None,
        help="Only process the specified scene folder name. Default: process all scenes.",
    )
    parser.add_argument(
        "--max-height",
        type=int,
        default=720,
        help="Maximum downloaded video height. Default: 720",
    )
    return parser.parse_args()


def get_progress_wrapper(iterable, description: str):
    try:
        from tqdm import tqdm  # type: ignore

        return tqdm(iterable, desc=description, unit="scene")
    except Exception:
        return iterable


def find_scene_dirs(dataset_root: Path) -> list[Path]:
    return sorted(
        scene_dir
        for scene_dir in dataset_root.iterdir()
        if scene_dir.is_dir() and (scene_dir / "data_info.json").is_file()
    )


def load_data_info(scene_dir: Path) -> dict:
    with (scene_dir / "data_info.json").open("r", encoding="utf-8") as f:
        return json.load(f)


def download_video(scene_dir: Path, overwrite: bool, max_height: int) -> tuple[str, str]:
    data_info = load_data_info(scene_dir)
    video_url = data_info.get("video_url")
    if not video_url:
        raise ValueError(f"Missing video_url in {scene_dir / 'data_info.json'}")

    output_path = scene_dir / "video.mp4"
    if output_path.exists() and not overwrite:
        return "skipped", str(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    outtmpl = str(output_path.with_suffix(".%(ext)s"))

    ydl_opts = {
        "outtmpl": outtmpl,
        "format": (
            f"bv*[height<={max_height}][ext=mp4]+ba[ext=m4a]/"
            f"b[height<={max_height}][ext=mp4]/"
            f"best[height<={max_height}]"
        ),
        "merge_output_format": "mp4",
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "overwrites": overwrite,
        "retries": 3,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([video_url])

    if not output_path.exists():
        alt_candidates = sorted(scene_dir.glob(f"{output_path.stem}.*"))
        if not alt_candidates:
            raise FileNotFoundError(f"Download finished but no output file found for {scene_dir.name}")
        if alt_candidates[0] != output_path:
            alt_candidates[0].replace(output_path)

    return "downloaded", str(output_path)


def main() -> int:
    args = parse_args()
    dataset_root = args.dataset_root.expanduser().resolve()
    if not dataset_root.is_dir():
        print(f"Dataset root does not exist: {dataset_root}", file=sys.stderr)
        return 1
    if args.max_height <= 0:
        print("--max-height must be a positive integer.", file=sys.stderr)
        return 1

    scene_dirs = find_scene_dirs(dataset_root)
    if args.scene_name is not None:
        scene_dirs = [
            scene_dir for scene_dir in scene_dirs
            if scene_dir.name == args.scene_name
        ]

    if not scene_dirs:
        if args.scene_name is not None:
            print(
                f"Scene {args.scene_name!r} not found under {dataset_root} "
                f"or missing data_info.json.",
                file=sys.stderr,
            )
        else:
            print(
                f"No scene folders with data_info.json found under {dataset_root}",
                file=sys.stderr,
            )
        return 1

    print(f"Found {len(scene_dirs)} scene(s) with data_info.json under {dataset_root}")

    downloaded = 0
    skipped = 0
    failed: list[tuple[str, str]] = []

    for scene_dir in get_progress_wrapper(scene_dirs, "Downloading videos"):
        try:
            status, output_path = download_video(
                scene_dir=scene_dir,
                overwrite=args.overwrite,
                max_height=args.max_height,
            )
            if status == "downloaded":
                downloaded += 1
            else:
                skipped += 1
        except Exception as exc:
            failed.append((scene_dir.name, str(exc)))
            print(f"[error] {scene_dir.name}: {exc}")

    print(
        f"Finished. downloaded={downloaded}, skipped={skipped}, failed={len(failed)}"
    )
    if failed:
        print("Failed scenes:")
        for scene_name, error in failed[:20]:
            print(f"  - {scene_name}: {error}")
        if len(failed) > 20:
            print(f"  ... and {len(failed) - 20} more")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

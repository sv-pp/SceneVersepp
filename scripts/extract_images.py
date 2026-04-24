#!/usr/bin/env python3
"""Extract SceneVerse++ frames from downloaded videos and create crop_images."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract frames listed in data_info.json from each scene video, save raw "
            "images into images/ and cropped images into crop_images/."
        )
    )
    parser.add_argument(
        "dataset_root",
        type=Path,
        help="Root directory containing scene folders.",
    )
    parser.add_argument(
        "--video-name",
        default="video.mp4",
        help="Video filename inside each scene folder. Default: video.mp4",
    )
    parser.add_argument(
        "--image-dir-name",
        default="images",
        help="Directory name for original extracted images. Default: images",
    )
    parser.add_argument(
        "--crop-dir-name",
        default="crop_images",
        help="Directory name for cropped images. Default: crop_images",
    )
    parser.add_argument(
        "--multiple",
        type=int,
        default=16,
        help="Crop output size to the nearest center crop divisible by this value. Default: 16",
    )
    parser.add_argument(
        "--max-size",
        type=int,
        default=512,
        help="Resize so the longer side is at most this value before center cropping. Default: 512",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-extract frames even if output files already exist.",
    )
    parser.add_argument(
        "--scene-name",
        default=None,
        help="Only process the specified scene folder name. Default: process all scenes.",
    )
    return parser.parse_args()

def find_scene_dirs(dataset_root: Path) -> list[Path]:
    return sorted(
        scene_dir
        for scene_dir in dataset_root.iterdir()
        if scene_dir.is_dir() and (scene_dir / "data_info.json").is_file()
    )

def load_data_info(scene_dir: Path) -> dict:
    with (scene_dir / "data_info.json").open("r", encoding="utf-8") as f:
        return json.load(f)

def format_frame_stem(frame_id: int) -> str:
    return f"{int(frame_id):06d}"


def crop_to_nearest_multiple_center(
    img: np.ndarray,
    multiple: int,
    max_size: int,
) -> np.ndarray:
    import cv2  # type: ignore

    height, width = img.shape[:2]
    if height <= 0 or width <= 0:
        raise ValueError("Input image has invalid shape")

    scale = min(1.0, float(max_size) / float(max(height, width)))
    if scale != 1.0:
        new_width = max(1, int(round(width * scale)))
        new_height = max(1, int(round(height * scale)))
        img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
        height, width = img.shape[:2]

    crop_width = max(multiple, (width // multiple) * multiple)
    crop_height = max(multiple, (height // multiple) * multiple)
    crop_width = min(crop_width, width)
    crop_height = min(crop_height, height)

    x0 = max(0, (width - crop_width) // 2)
    y0 = max(0, (height - crop_height) // 2)
    return img[y0 : y0 + crop_height, x0 : x0 + crop_width]


def extract_scene(
    scene_dir: Path,
    video_name: str,
    image_dir_name: str,
    crop_dir_name: str,
    multiple: int,
    max_size: int,
    overwrite: bool,
) -> tuple[int, int]:

    data_info = load_data_info(scene_dir)
    frame_ids = data_info.get("data_frames", [])
    if not isinstance(frame_ids, list):
        raise ValueError(f"data_frames must be a list in {scene_dir / 'data_info.json'}")

    video_path = scene_dir / video_name
    if not video_path.is_file():
        raise FileNotFoundError(f"Missing video file: {video_path}")

    image_dir = scene_dir / image_dir_name
    crop_dir = scene_dir / crop_dir_name
    image_dir.mkdir(parents=True, exist_ok=True)
    crop_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    written = 0
    skipped = 0
    try:
        for frame_id in frame_ids:
            frame_index = int(frame_id)
            stem = format_frame_stem(frame_index)
            raw_path = image_dir / f"{stem}.png"
            crop_path = crop_dir / f"{stem}.png"
            if raw_path.exists() and crop_path.exists() and not overwrite:
                skipped += 1
                continue

            ok = cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            if not ok:
                raise RuntimeError(f"Failed to seek to frame {frame_index} in {video_path}")

            ok, frame = cap.read()
            if not ok or frame is None:
                raise RuntimeError(f"Failed to read frame {frame_index} from {video_path}")

            if overwrite or not raw_path.exists():
                if not cv2.imwrite(str(raw_path), frame):
                    raise RuntimeError(f"Failed to save image: {raw_path}")

            cropped_img = crop_to_nearest_multiple_center(
                frame,
                multiple=multiple,
                max_size=max_size,
            )
            if overwrite or not crop_path.exists():
                if not cv2.imwrite(str(crop_path), cropped_img):
                    raise RuntimeError(f"Failed to save cropped image: {crop_path}")

            written += 1
    finally:
        cap.release()

    return written, skipped


def main() -> int:
    args = parse_args()
    dataset_root = args.dataset_root.expanduser().resolve()
    if not dataset_root.is_dir():
        print(f"Dataset root does not exist: {dataset_root}", file=sys.stderr)
        return 1
    if args.multiple <= 0:
        print("--multiple must be a positive integer.", file=sys.stderr)
        return 1
    if args.max_size <= 0:
        print("--max-size must be a positive integer.", file=sys.stderr)
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

    written_total = 0
    skipped_total = 0
    failed: list[tuple[str, str]] = []
    for scene_dir in tqdm(scene_dirs, desc="Extracting frames", unit="scene"):
        try:
            written, skipped = extract_scene(
                scene_dir=scene_dir,
                video_name=args.video_name,
                image_dir_name=args.image_dir_name,
                crop_dir_name=args.crop_dir_name,
                multiple=args.multiple,
                max_size=args.max_size,
                overwrite=args.overwrite,
            )
            written_total += written
            skipped_total += skipped
        except Exception as exc:
            failed.append((scene_dir.name, str(exc)))
            print(f"[error] {scene_dir.name}: {exc}")

    print(
        f"Finished. written={written_total}, skipped={skipped_total}, failed={len(failed)}"
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

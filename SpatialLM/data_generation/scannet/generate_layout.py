# -*- coding: utf-8 -*-
import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import open3d as o3d
import pandas as pd
from tqdm import tqdm

VOXEL_SIZE = 0.02
NUM_WORKERS = 16

SCANNET20_IDS = {
    3: 'cabinet',
    4: 'bed',
    5: 'chair',
    6: 'sofa',
    7: 'table',
    10: 'bookshelf',
    11: 'picture',
    12: 'counter',
    14: 'desk',
    16: 'curtain',
    24: 'refrigerator',
    28: 'shower curtain',
    33: 'toilet',
    34: 'sink',
    36: 'bathtub'
}

def load_label_map(label_map_file: Path) -> pd.DataFrame:
    if not label_map_file.exists():
        raise FileNotFoundError(f"Label map file not found: {label_map_file}")
    return pd.read_csv(label_map_file, sep="\t")


def map_label_to_scannet20(raw_label: str, label_df: pd.DataFrame):
    rows = label_df[label_df["raw_category"].str.lower() == raw_label.lower()]
    if rows.empty:
        return None
    row = rows.iloc[0]
    nyu40id = int(row["nyu40id"])
    if nyu40id not in SCANNET20_IDS:
        return None
    return nyu40id, SCANNET20_IDS[nyu40id]


def load_splits(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def load_segs(segs_file: Path) -> np.ndarray:
    with segs_file.open("r", encoding="utf-8") as f:
        seg_data = json.load(f)
    return np.array(seg_data["segIndices"], dtype=np.int32)


def load_aggregation(agg_file: Path, label_df: pd.DataFrame):
    seg_to_instance = {}
    instance_info = {}
    with agg_file.open("r", encoding="utf-8") as f:
        agg = json.load(f)

    for obj in agg["segGroups"]:
        raw_label = obj["label"]
        mapped = map_label_to_scannet20(raw_label, label_df)
        if mapped is None:
            continue
        _, mapped_label = mapped
        instance_id = obj["id"]
        instance_info[instance_id] = mapped_label
        for seg_id in obj["segments"]:
            seg_to_instance[seg_id] = instance_id
    return seg_to_instance, instance_info


def load_axis_alignment(txt_file: Path) -> np.ndarray:
    if not txt_file.exists():
        return np.eye(4)

    with txt_file.open("r", encoding="utf-8") as f:
        lines = f.readlines()

    for line in lines:
        if line.startswith("axisAlignment"):
            parts = line.replace("=", " ").split()
            nums = [float(x) for x in parts[1:]]
            if len(nums) == 16:
                return np.array(nums, dtype=np.float64).reshape(4, 4)
    return np.eye(4)


def extract_bboxes(points: np.ndarray, seg_indices: np.ndarray, seg_to_instance: dict, instance_info: dict) -> list[str]:
    bboxes = []
    instance_points = {}
    for idx, seg_id in enumerate(seg_indices):
        if seg_id not in seg_to_instance:
            continue
        inst_id = seg_to_instance[seg_id]
        if inst_id not in instance_info:
            continue
        instance_points.setdefault(inst_id, []).append(points[idx])

    for bbox_idx, (inst_id, pts) in enumerate(instance_points.items()):
        pts = np.asarray(pts)
        if pts.shape[0] < 50:
            continue
        min_pt, max_pt = pts.min(0), pts.max(0)
        center = (min_pt + max_pt) / 2
        size = max_pt - min_pt
        anglez = 0.0
        label = instance_info[inst_id]
        bbox_str = (
            f"bbox_{bbox_idx}=Bbox({label},"
            f"{center[0]:.3f},{center[1]:.3f},{center[2]:.3f},{anglez:.3f},"
            f"{size[0]:.3f},{size[1]:.3f},{size[2]:.3f})"
        )
        bboxes.append(bbox_str)
    return bboxes


def save_bboxes_to_txt(bbox_list: list[str], filename: Path) -> None:
    with filename.open("w", encoding="utf-8") as f:
        f.write("\n".join(bbox_list))


def resolve_scannet_paths(data_root: Path) -> dict[str, Path]:
    return {
        "scene_root": Path(data_root / "scans"),
        "label_map_file": Path("scannetv2-labels.combined.tsv"),
        "train_splits": Path("data_generation/scannet/scannetv2_train.txt"),
        "val_splits": Path("data_generation/scannet/scannetv2_val.txt"),
    }


def process_scene(
    task: tuple[str, str],
    scene_root: Path,
    output_root: Path,
    label_df: pd.DataFrame,
    voxel_size: float,
) -> str:
    scene_name, split_name = task
    try:
        data_root = scene_root / scene_name
        if not data_root.is_dir():
            return f"{scene_name}: skipped (not a dir)"

        mesh_file = data_root / f"{scene_name}_vh_clean_2.ply"
        segs_file = data_root / f"{scene_name}_vh_clean_2.0.010000.segs.json"
        agg_file = data_root / f"{scene_name}.aggregation.json"
        txt_file = data_root / f"{scene_name}.txt"

        if not mesh_file.exists() or not segs_file.exists() or not agg_file.exists():
            return f"{scene_name}: skipped (missing mesh/seg/agg file)"

        full_pcd = o3d.io.read_point_cloud(str(mesh_file))
        if full_pcd.is_empty():
            return f"{scene_name}: skipped (empty point cloud)"

        transform = load_axis_alignment(txt_file)
        full_pcd.transform(transform)
        full_points = np.asarray(full_pcd.points)

        seg_indices = load_segs(segs_file)
        if seg_indices.shape[0] != full_points.shape[0]:
            return f"{scene_name}: skipped (segIndices and point count mismatch)"

        seg_to_instance, instance_info = load_aggregation(agg_file, label_df)
        bboxes = extract_bboxes(full_points, seg_indices, seg_to_instance, instance_info)

        save_pcd = full_pcd.voxel_down_sample(voxel_size)

        pcd_dir = output_root / "pcd" / split_name
        layout_dir = output_root / "layout" / split_name
        pcd_dir.mkdir(parents=True, exist_ok=True)
        layout_dir.mkdir(parents=True, exist_ok=True)

        o3d.io.write_point_cloud(str(pcd_dir / f"{scene_name}.ply"), save_pcd)
        save_bboxes_to_txt(bboxes, layout_dir / f"{scene_name}.txt")

        return f"{scene_name}: done"
    except Exception as exc:
        return f"{scene_name}: error ({exc})"


def main(args):
    data_root = Path(args.data_root)
    save_path = Path(args.save_path)
    paths = resolve_scannet_paths(data_root)
    scene_root = paths["scene_root"]
    label_df = load_label_map(paths["label_map_file"])

    train_scenes = load_splits(paths["train_splits"])
    val_scenes = load_splits(paths["val_splits"])
    all_tasks = [(scene_name, "train") for scene_name in train_scenes] + [
        (scene_name, "val") for scene_name in val_scenes
    ]
    if args.process_number > 0:
        all_tasks = all_tasks[: args.process_number]

    if not scene_root.exists():
        print(f"⚠️ {scene_root} not found, skipped")
        return

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(
                process_scene,
                task,
                scene_root,
                save_path,
                label_df,
                args.voxel_size,
            )
            for task in all_tasks
        ]
        for future in tqdm(as_completed(futures), total=len(futures), desc=f"Processing {scene_root.name}"):
            print(future.result())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True, help="ScanNet dataset root")
    parser.add_argument("--save_path", type=str, required=True, help="SpatialLM output root")
    parser.add_argument("--voxel_size", type=float, default=VOXEL_SIZE, help="Point cloud voxel size")
    parser.add_argument("--workers", type=int, default=NUM_WORKERS, help="Number of parallel workers")
    parser.add_argument("--process_number", type=int, default=-1, help="Number of scenes to process")
    args = parser.parse_args()

    main(args)

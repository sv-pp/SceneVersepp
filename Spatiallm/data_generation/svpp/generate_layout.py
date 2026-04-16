# -*- coding: utf-8 -*-
import os
import argparse
import numpy as np
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import json
import open3d as o3d
from tqdm import tqdm

CLASS_LABELS_20 = [
    'cabinet', 'bed', 'chair', 'sofa', 'table', 'bookshelf',
    'picture', 'counter', 'desk', 'curtain', 'refrigerator',
    'shower curtain', 'toilet', 'sink', 'bathtub'
]
CLASS_LABELS_20_SET = set(CLASS_LABELS_20)

MAX_POINTS = 300_000

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

LABEL_MAP_PATH = Path(
    "./scannetv2-labels.combined.tsv"
)

def build_scannet20_label_map(tsv_path: Path) -> dict[str, str]:
    """Build a mapping from raw ScanNet200 categories to ScanNet20 labels."""
    if not tsv_path.exists():
        raise FileNotFoundError(f"ScanNet label map not found: {tsv_path}")
    mapper: dict[str, str] = {}
    with tsv_path.open("r", encoding="utf-8") as f:
        header = f.readline().strip().split("\t")
        try:
            raw_idx = header.index("raw_category")
            nyu_idx = header.index("nyu40id")
        except ValueError as exc:
            raise ValueError("ScanNet label map missing expected columns") from exc
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) <= max(raw_idx, nyu_idx):
                continue
            raw_label = parts[raw_idx].strip().lower()
            if not raw_label:
                continue
            try:
                nyu40id = int(parts[nyu_idx])
            except ValueError:
                continue
            mapped = SCANNET20_IDS.get(nyu40id)
            if mapped:
                mapper[raw_label] = mapped
    return mapper


def rotation_yup_to_zup() -> np.ndarray:
    return np.array([
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, -1.0, 0.0]
    ], dtype=np.float64)


def rotate_pcd_yup_to_zup(mesh: o3d.geometry.TriangleMesh) -> None:
    R = rotation_yup_to_zup()
    mesh.rotate(R, center=(0.0, 0.0, 0.0))


def load_bbox(metadata: str, scene_points: np.ndarray, label_map: dict[str, str]) -> list:
    """Load npz file to bbox list"""
    bbox = []
    with open(metadata, "r", encoding="utf-8") as f:
        data = json.loads(f.read())
    for instance_id, instance_info in data.items():
        pred_mask = np.array(instance_info['point_ids'])
        raw_label = str(instance_info['pred_class_name']).strip().lower()
        mapped_class = label_map.get(raw_label)
        if mapped_class is None:
            if raw_label in CLASS_LABELS_20_SET:
                mapped_class = raw_label
            else:
                continue
        if mapped_class not in CLASS_LABELS_20_SET:
            continue
        pred_class = mapped_class
        instance_points = scene_points[pred_mask]
        if instance_points.shape[0] < 100:
            continue
        mins = instance_points.min(axis=0)
        maxs = instance_points.max(axis=0)
        center = (mins + maxs) / 2
        size = (maxs - mins)
        anglez = 0.0
        bbox_str = (
            f"bbox_{instance_id}=Bbox({pred_class},"
            f"{center[0]},{center[1]},{center[2]},{anglez},"
            f"{size[0]},{size[1]},{size[2]})"
        )
        bbox.append(bbox_str)
    return bbox


def save_bboxes_to_txt(bbox_list, filename: Path):
    with open(filename, "w", encoding="utf-8") as f:
        f.write("\n".join(bbox_list))


def process_scene(
    seq_name: str,
    root: Path,
    args,
    layout_path: Path,
    pcd_path: Path,
    label_map: dict[str, str]
):
    data_root = root / seq_name
    if not os.path.isdir(data_root):
        return f"{seq_name}: skipped (not a dir)"

    scene_ply = data_root / 'mesh.ply'
    metadata = data_root / 'metadata.json'
    if not metadata.exists():
        return f"{seq_name}: skipped (no metadata.json)"

    layout_file = layout_path / f"{seq_name}.txt"
    pcd_file = pcd_path / f"{seq_name}.ply"
    

    mesh = o3d.io.read_triangle_mesh(str(scene_ply))
    scene_points = np.asarray(mesh.vertices, dtype=np.float32)   # 👈 强制 float32
    if scene_points.shape[0] == 0:
        return f"{seq_name}: skipped (empty mesh)"
    mins = scene_points.min(axis=0)
    maxs = scene_points.max(axis=0)
    z_shift = -float(mins[2])
    center_xy = (mins[:2] + maxs[:2]) / 2.0
    xy_shift = -center_xy
    shift = np.array([xy_shift[0], xy_shift[1], z_shift], dtype=np.float32)
    scene_points += shift
    scene_colors = np.asarray(mesh.vertex_colors, dtype=np.float32)
    bbox = load_bbox(metadata, scene_points, label_map)

    save_bboxes_to_txt(bbox, layout_file)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(scene_points)
    pcd.colors = o3d.utility.Vector3dVector(scene_colors)
    if args.voxel_size > 0:
        pcd = pcd.voxel_down_sample(voxel_size=args.voxel_size)

    points = np.asarray(pcd.points, dtype=np.float32)
    colors = np.asarray(pcd.colors, dtype=np.float32)
    if points.shape[0] > MAX_POINTS:
        idx = np.random.choice(points.shape[0], MAX_POINTS, replace=False)
        points, colors = points[idx], colors[idx]
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)

    o3d.io.write_point_cloud(str(pcd_file), pcd)

    return f"{seq_name}: done"


def main(args):
    save_path = Path(args.save_path)
    data_root = Path(args.data_root)
    label_map = build_scannet20_label_map(Path(args.label_map))
    layout_path = save_path / 'layout'
    pcd_path = save_path / 'pcd'
    layout_path.mkdir(parents=True, exist_ok=True)
    pcd_path.mkdir(parents=True, exist_ok=True)
    results = []
    if not data_root.exists():
        print(f"⚠️ {data_root} not found, skipped")
        return

    seq_names = os.listdir(data_root)[:args.process_number] if args.process_number > 0 else os.listdir(data_root)
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(process_scene, seq, data_root, args, layout_path, pcd_path, label_map): seq
            for seq in seq_names
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc=f"Processing {data_root.name}"):
            results.append(future.result())

    print("\n=== Summary ===")
    for r in results:
        print(r)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, required=True, help="svpp dataset root")
    parser.add_argument('--save_path', type=str, required=True, help="spatial layout output directory")
    parser.add_argument('--voxel_size', type=float, default=0.02, help="point cloud voxel size")
    parser.add_argument('--workers', type=int, default=16, help="number of parallel workers")
    parser.add_argument('--process_number', type=int, default=-1, help="number of process to run")
    parser.add_argument(
        '--label-map',
        type=str,
        default=str(LABEL_MAP_PATH),
        help="ScanNet label map TSV for ScanNet200->ScanNet20 conversion"
    )
    args = parser.parse_args()

    main(args)

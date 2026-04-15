# -*- coding: utf-8 -*-
import open3d as o3d
import numpy as np
import json
import random
import argparse
from collections import Counter
from pathlib import Path
import yaml
import torch
import rerun as rr
from dataclasses import dataclass
import time 

@dataclass
class Scan:
    dataset_name: str
    scene_name: str
    mesh_file: Path
    segments_file: Path
    pcd_file: Path
    aux_file: Path

def load_scan(pcd_path: Path, inst2label_path: Path):
    pcd_data = torch.load(pcd_path)
    inst_to_label = torch.load(inst2label_path)
    points, colors, instance_labels = pcd_data[0], np.asarray(pcd_data[1]), pcd_data[-1]
    if colors.size > 0 and np.max(colors) <= 1:
        colors = colors * 255.0
    colors = colors.astype(np.uint8)
    pcds = np.concatenate([points, colors], 1)
    obj_pcds = []
    for i in inst_to_label.keys():
        mask = instance_labels == i     # time consuming
        if np.sum(mask) == 0:
            continue
        color = np.random.randint(0, 256, size=3, dtype=np.uint8)
        obj_pcds.append((pcds[mask], inst_to_label[i], color))
        colors[mask] = color
    return obj_pcds, points, colors


def make_safe_entity_name(name: str) -> str:
    return name.replace(" ", "_").replace("/", "_")


def resolve_mesh_file(data_root: Path, scene_name: str, dataset_name: str) -> Path:
    scene_root = data_root / scene_name
    candidates = []
    if dataset_name.lower() == "scannet":
        candidates.append(scene_root / f"{scene_name}_vh_clean_2.rot.ply")
        # candidates.append(scene_root / f"{scene_name}_vh_clean_2.ply")
    else:
        candidates.append(scene_root / "mesh.ply")
        # candidates.append(scene_root / f"{scene_name}_vh_clean_2.rot.ply")

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def visualize_segments(scan: Scan):

    # just for vis segments.s.k.json
    mesh_path = Path(scan.mesh_file)
    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    if mesh.is_empty():
        raise ValueError(f"❌ can not read mesh file: {mesh_path}")
    mesh.compute_vertex_normals()

    with open(scan.segments_file, 'r') as f:
        data = json.load(f)
    seg_indices = np.array(data["segIndices"], dtype=int)

    id_counts = Counter(seg_indices)
    print(f"segments numbers: {len(id_counts)}")

    id2color = {sid: [random.random(), random.random(), random.random()] for sid in id_counts.keys()}
    colors = np.array([id2color[sid] for sid in seg_indices])
    mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([mesh])

def visualize_training_data(scan: Scan):
    # for vis training data
    obj_pcds, points, colors = load_scan(scan.pcd_file, scan.aux_file)
    seg_indices = np.load(scan.aux_file)
    unique_ids = np.unique(seg_indices)
    print(f"Segment IDs in scene {scan.scene_name}: {len(unique_ids)}")
    safe_scene_name = make_safe_entity_name(scan.scene_name)

    rr.log(
        f"{safe_scene_name}/pointcloud",
        rr.Points3D(
            positions=points.astype(np.float32),
            colors=colors.astype(np.uint8),
            radii=0.005,
        ),
    )
    # draw bbox
    for idx, (obj_points, class_name, color) in enumerate(obj_pcds):
        # if class_name in ['wall', 'floor', 'ceiling']:
        #     continue
        min_xyz = obj_points[:, :3].min(axis=0)
        max_xyz = obj_points[:, :3].max(axis=0)
        center = (min_xyz + max_xyz) / 2
        half_size = (max_xyz - min_xyz) / 2
        entity_name = make_safe_entity_name(class_name)
        rr.log(
            f"{safe_scene_name}/bboxes/{idx:03d}_{entity_name}",
        rr.Boxes3D(
                centers=center.reshape(1, 3).astype(np.float32),
                half_sizes=half_size.reshape(1, 3).astype(np.float32),
                colors=color.reshape(1, 3).astype(np.uint8),
                labels=[f"{class_name}"],
            ),
        )
    rr.log("origin", rr.ViewCoordinates.RIGHT_HAND_Z_UP)
    print(f"✅ Visualized {len(obj_pcds)} objects with Boxes3D in Rerun.")

def is_valid_scan(scan: Scan) -> bool:
    # Check if the scan has valid data
    return all([Path(scan.mesh_file).exists(), Path(scan.segments_file).exists(), Path(scan.pcd_file).exists(), Path(scan.aux_file).exists()])

def visualize_scene(args: argparse.Namespace, scan: Scan):
    if not is_valid_scan(scan):
        return
    if args.vis_segments:
        visualize_segments(scan)
    if args.vis_training_data:
        visualize_training_data(scan)


if __name__ == "__main__":
    data_root = Path('/mnt/fillipo/yaowei/svpp_release_data/svpp_data')
    parser = argparse.ArgumentParser(description="Visualize segmentation using simple Open3D draw_geometries")
    parser.add_argument("--config", type=str, default="data_process/config.yaml", help="Path to the config file")
    parser.add_argument("--dataset_name", type=str, default="svpp", help="Name of the dataset to visualize")
    parser.add_argument("--scene_name", type=str, default=None, help="Name of the scene to visualize (default: all scenes)")
    parser.add_argument("--vis_segments", action="store_true", help="Whether to visualize segments using Open3D")
    parser.add_argument("--vis_training_data", action="store_true", help="Whether to visualize training data using Rerun")
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)    
    data_root = Path(config['datasets'][args.dataset_name]['data_root'])
    save_root = Path(config['save_root'])
    
    segments_path = save_root / "segments" / args.dataset_name
    inst2label_path = save_root / config['base_dir'] / args.dataset_name / "scan_data" / "instance_id_to_label"
    pcd_path = save_root / config['base_dir'] / args.dataset_name / "scan_data" / "pcd_with_global_alignment"
    
    s, k = config['datasets'][args.dataset_name]['segmentator_s'], config['datasets'][args.dataset_name]['segmentator_k']
    s_str = f"{s:.3f}"
    if args.vis_training_data:
        rr.init(
            application_id='svpp pq3d training data visualization',
            recording_id=time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime()),
            spawn=True,
        )
    if args.scene_name:
        scene_names = [args.scene_name]
    else:
        scene_names = [d.name for d in data_root.iterdir() if d.is_dir()]
    for scene_name in scene_names:
        mesh_file = resolve_mesh_file(data_root, scene_name, args.dataset_name)
        segments_file = Path(segments_path / f"{scene_name}.{s_str}_{k}.segs.json")
        pcd_file = Path(pcd_path / f"{scene_name}.pth")
        aux_file = Path(inst2label_path / f"{scene_name}.pth")
        scan = Scan(dataset_name=args.dataset_name, scene_name=scene_name, mesh_file=mesh_file, segments_file=segments_file, pcd_file=pcd_file, aux_file=aux_file)
        visualize_scene(args, scan)

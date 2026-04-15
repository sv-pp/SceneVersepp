# -*- coding: utf-8 -*-
import argparse
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import numpy as np
import open3d as o3d
import torch
import yaml
from tqdm import tqdm


DEFAULT_CONFIG = Path("data_process/config.yaml")
DEFAULT_MAX_WORKERS = 16
DEFAULT_SEGMENTATOR_BIN = Path("data_process/segmentator")


def load_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Invalid config format: {config_path}")
    return cfg

def resplit_segments(seg_json_path: Path, instance_labels: np.ndarray, resplit_aug: bool) -> Optional[np.ndarray]:
    if not seg_json_path.exists():
        return None
    with seg_json_path.open("r", encoding="utf-8") as f:
        seg_data = json.load(f)
    if "segIndices" not in seg_data:
        return None
    segments = np.asarray(seg_data["segIndices"], dtype=np.int64)
    if resplit_aug is False:
        return segments
    if segments.size == 0:
        return None

    if len(segments) != len(instance_labels):
        raise ValueError(
            f"Data length mismatch: seg({len(segments)}) vs instance({len(instance_labels)})"
        )

    new_segments = segments.copy()
    next_seg_id = int(segments.max() + 1)
    unique_segs = np.unique(segments)

    # Keep pure background segments unchanged.
    for seg_id in unique_segs:
        seg_mask = segments == seg_id
        seg_indices = np.where(seg_mask)[0]
        if len(seg_indices) == 0:
            continue

        inst_ids = np.unique(instance_labels[seg_indices])
        inst_ids = inst_ids[inst_ids > 0]
        if len(inst_ids) == 0:
            continue

        # Split the intersection part into new segment ids.
        for inst_id in inst_ids:
            inter_mask = seg_mask & (instance_labels == inst_id)
            if np.sum(inter_mask) == 0:
                continue
            new_segments[inter_mask] = next_seg_id
            next_seg_id += 1

    return new_segments


def load_scannet_segments(seg_json_path: Path) -> np.ndarray:
    with seg_json_path.open("r", encoding="utf-8") as f:
        seg_data = json.load(f)
    if "segIndices" not in seg_data:
        raise ValueError(f"Missing segIndices in: {seg_json_path}")
    return np.asarray(seg_data["segIndices"], dtype=np.int64)


def load_scannet_aggregation(agg_path: Path):
    seg_to_instance = {}
    inst_to_label = {}
    with agg_path.open("r", encoding="utf-8") as f:
        agg = json.load(f)

    for obj in agg.get("segGroups", []):
        mapped_label = str(obj.get("label", "")).strip()
        if not mapped_label:
            continue
        inst_id = int(obj["id"]) + 1
        inst_to_label[inst_id] = mapped_label
        for seg_id in obj.get("segments", []):
            seg_to_instance[int(seg_id)] = inst_id
    return seg_to_instance, inst_to_label


def build_vertex_instance_from_segments(segments: np.ndarray, seg_to_instance: dict[int, int]) -> np.ndarray:
    vertex_instance = np.zeros(len(segments), dtype=np.int32)
    for idx, seg_id in enumerate(segments):
        inst_id = seg_to_instance.get(int(seg_id), 0)
        vertex_instance[idx] = inst_id
    return vertex_instance


class SVPPProcessor:
    def __init__(self, dataset_name: str, dataset_cfg: dict, cfg: dict):
        self.dataset_name = dataset_name
        self.dataset_cfg = dataset_cfg
        self.resplit_aug = dataset_cfg['resplit_aug']

        self.save_root = Path(cfg.get("save_root", "./training_datas"))
        self.base_dir = str(cfg.get("base_dir", "base")).strip()
        self.aux_dir = str(cfg.get("aux_dir", "aux")).strip()

        self.process_number = int(dataset_cfg.get("process_number", -1))
        self.num_workers = int(dataset_cfg.get("workers", cfg.get("workers", DEFAULT_MAX_WORKERS)))
        self.output = cfg.get("output", {"pcd": True, "segment_id": True})
        self.segmentator_s = float(dataset_cfg.get("segmentator_s", 0.1))
        self.segmentator_k = int(dataset_cfg.get("segmentator_k", 20))
        self.auto_segmentator = bool(dataset_cfg.get("resplit_aut", True))
        self.segmentator_bin = Path(cfg.get("segmentator_bin", DEFAULT_SEGMENTATOR_BIN)).expanduser().resolve()

        self.data_root = Path(dataset_cfg["data_root"])
        self.segment_root = self.save_root / "segments" / self.dataset_name
        self.segment_root.mkdir(parents=True, exist_ok=True)

        self.base_root = self.save_root / self.base_dir / self.dataset_name
        self.aux_root = self.save_root / self.aux_dir / self.dataset_name / "segment_id"
        
        self.inst2label_path = self.base_root / "scan_data" / "instance_id_to_label"
        self.pcd_path = self.base_root / "scan_data" / "pcd_with_global_alignment"
        self.split_dir = self.base_root / "split"
 
        self.inst2label_path.mkdir(parents=True, exist_ok=True)
        self.pcd_path.mkdir(parents=True, exist_ok=True)
        self.split_dir.mkdir(parents=True, exist_ok=True)
        self.aux_root.mkdir(parents=True, exist_ok=True)

    def segment_json_path(self, scene_name: str) -> Path:
        s_str = f"{self.segmentator_s:.3f}"
        return self.segment_root / f"{scene_name}.{s_str}_{self.segmentator_k}.segs.json"

    def ensure_segments_json(self, scene_name: str, scene_ply_path: Path) -> Optional[Path]:
        """
        Make sure the segmentator output exists, and return its path.
        This mirrors the behavior from segmentator_part.py.
        """
        seg_json_path = self.segment_json_path(scene_name)
        if seg_json_path.exists():
            return seg_json_path
        if not scene_ply_path.exists():
            raise FileNotFoundError(f"scene ply not found for segmentator: {scene_ply_path}")
        if not self.segmentator_bin.exists():
            raise FileNotFoundError(f"segmentator binary not found: {self.segmentator_bin}")

        cmd = [str(self.segmentator_bin), str(scene_ply_path), f"{self.segmentator_s}", str(self.segmentator_k), str(seg_json_path)]
        seg_json_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            return seg_json_path
        except subprocess.CalledProcessError as e:
            stderr = e.stderr.decode("utf-8", errors="ignore").strip() if e.stderr else ""
            raise RuntimeError(f"segmentator failed for {scene_ply_path}: {stderr or e}") from e

    def scene_proc(self, scene_name: str) -> str:
        scene_root = self.data_root / scene_name
        scene_ply_path = scene_root / "mesh.ply"
        if not scene_ply_path.exists():
            return f"[MISS] {scene_name} missing mesh.ply"
        # process segments part
        segments_json_path = self.ensure_segments_json(scene_name, scene_ply_path)
        
        # load GT metadata
        metadata_path = scene_root / "metadata.json"
        if not metadata_path.exists():
            return f"[MISS] {scene_name} missing metadata.json"
        metadata = json.load(open(metadata_path, "r"))

        # process scan_data(instance_id_to_label, pad_with_global_alignment)
        mesh = o3d.io.read_triangle_mesh(str(scene_ply_path))
        vertices = np.asarray(mesh.vertices)
        vertex_colors = np.asarray(mesh.vertex_colors)
        if vertex_colors.size == 0:
            vertex_colors = np.zeros_like(vertices)
        elif np.max(vertex_colors) <= 1:
            vertex_colors = vertex_colors * 255.0

        vertex_instance = np.zeros(vertices.shape[0], dtype=np.int32)
        inst_to_label = {}

        for inst_id, inst_info in metadata.items():
            inst_id = int(inst_id) + 1
            point_ids = inst_info.get("point_ids", [])
            pred_class_name = inst_info.get("pred_class_name")
            pred_class_id = inst_info.get("pred_class_id", -1)
            if pred_class_id == -1:
                continue
            vertex_instance[point_ids] = inst_id
            inst_to_label[inst_id] = pred_class_name

        center_points = np.mean(vertices, axis=0)
        center_points[2] = np.min(vertices[:, 2])
        vertices = vertices - center_points
        torch.save(inst_to_label, self.inst2label_path / f"{scene_name}.pth")
        torch.save((vertices, vertex_colors, vertex_instance), self.pcd_path / f"{scene_name}.pth")
        
        # process resplit_segments and generate aux data
        new_segments = resplit_segments(segments_json_path, vertex_instance, self.resplit_aug)  
        np.save(self.aux_root / f"{scene_name}.npy", new_segments.astype(np.int32))
        return f"[DONE] {scene_name}"

    def process_scans(self) -> None:
        scene_names = sorted([d.name for d in self.data_root.iterdir() if d.is_dir()])
        if self.process_number > 0:
            scene_names = scene_names[:self.process_number]

        with open(self.split_dir / "train_split.txt", "w", encoding="utf-8") as fp:
            fp.write("\n".join(scene_names))

        if not scene_names:
            print(f"[WARN] No scenes found for dataset {self.dataset_name}: {self.data_root}")
            return

        with tqdm(total=len(scene_names), desc=self.dataset_name, unit="scene") as pbar:
            with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                futures = {executor.submit(self.scene_proc, scene_name): scene_name for scene_name in scene_names}
                for future in as_completed(futures):
                    print(future.result())
                    pbar.update(1)


class ScanNetProcessor:
    def __init__(self, dataset_name: str, dataset_cfg: dict, cfg: dict):
        self.dataset_name = dataset_name
        self.dataset_cfg = dataset_cfg
        self.resplit_aug = bool(dataset_cfg.get("resplit_aug", False))

        self.save_root = Path(cfg.get("save_root", "./training_datas"))
        self.base_dir = str(cfg.get("base_dir", "base")).strip()
        self.aux_dir = str(cfg.get("aux_dir", "aux")).strip()
        self.process_number = int(dataset_cfg.get("process_number", -1))
        self.num_workers = int(dataset_cfg.get("workers", cfg.get("workers", DEFAULT_MAX_WORKERS)))
        self.output = cfg.get("output", {"pcd": True, "segment_id": True})
        self.segmentator_s = float(dataset_cfg.get("segmentator_s", 0.1))
        self.segmentator_k = int(dataset_cfg.get("segmentator_k", 10))
        self.segmentator_bin = Path(cfg.get("segmentator_bin", DEFAULT_SEGMENTATOR_BIN)).expanduser().resolve()

        self.data_root = Path(dataset_cfg["data_root"])
        self.segment_root = self.save_root / "segments" / self.dataset_name
        self.segment_root.mkdir(parents=True, exist_ok=True)
        self.base_root = self.save_root / self.base_dir / self.dataset_name
        self.aux_root = self.save_root / self.aux_dir / self.dataset_name / "segment_id"
        self.inst2label_path = self.base_root / "scan_data" / "instance_id_to_label"
        self.pcd_path = self.base_root / "scan_data" / "pcd_with_global_alignment"
        self.split_dir = self.base_root / "split"

        self.inst2label_path.mkdir(parents=True, exist_ok=True)
        self.pcd_path.mkdir(parents=True, exist_ok=True)
        self.split_dir.mkdir(parents=True, exist_ok=True)
        self.aux_root.mkdir(parents=True, exist_ok=True)

    def scene_mesh_path(self, scene_name: str) -> Path:
        scene_root = self.data_root / scene_name
        return scene_root / f"{scene_name}_vh_clean_2.rot.ply"

    def scene_raw_segments_path(self, scene_name: str) -> Path:
        scene_root = self.data_root / scene_name
        return scene_root / f"{scene_name}_vh_clean_2.0.010000.segs.json"

    def scene_aggregation_path(self, scene_name: str) -> Path:
        scene_root = self.data_root / scene_name
        return scene_root / f"{scene_name}.aggregation.json"

    def segment_json_path(self, scene_name: str) -> Path:
        s_str = f"{self.segmentator_s:.3f}"
        return self.segment_root / f"{scene_name}.{s_str}_{self.segmentator_k}.segs.json"

    def ensure_segments_json(self, scene_name: str, scene_ply_path: Path) -> Optional[Path]:
        seg_json_path = self.segment_json_path(scene_name)
        if seg_json_path.exists():
            return seg_json_path
        if not scene_ply_path.exists():
            raise FileNotFoundError(f"scene ply not found for segmentator: {scene_ply_path}")
        if not self.segmentator_bin.exists():
            raise FileNotFoundError(f"segmentator binary not found: {self.segmentator_bin}")

        cmd = [
            str(self.segmentator_bin),
            str(scene_ply_path),
            f"{self.segmentator_s}",
            str(self.segmentator_k),
            str(seg_json_path),
        ]
        seg_json_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            return seg_json_path
        except subprocess.CalledProcessError as e:
            stderr = e.stderr.decode("utf-8", errors="ignore").strip() if e.stderr else ""
            raise RuntimeError(f"segmentator failed for {scene_ply_path}: {stderr or e}") from e

    def scene_proc(self, scene_name: str) -> str:
        scene_root = self.data_root / scene_name
        mesh_path = self.scene_mesh_path(scene_name)
        raw_seg_path = self.scene_raw_segments_path(scene_name)
        agg_path = self.scene_aggregation_path(scene_name)

        if not scene_root.is_dir():
            return f"[MISS] {scene_name} missing scene dir"
        if not mesh_path.exists():
            return f"[MISS] {scene_name} missing mesh file"
        if not raw_seg_path.exists():
            return f"[MISS] {scene_name} missing raw seg json"
        if not agg_path.exists():
            return f"[MISS] {scene_name} missing aggregation json"
        seg_json_path = self.ensure_segments_json(scene_name, mesh_path)
        
        if seg_json_path is None:
            return f"[MISS] {scene_name} failed to build seg json"

        mesh = o3d.io.read_triangle_mesh(str(mesh_path))
        vertices = np.asarray(mesh.vertices)
        vertex_colors = np.asarray(mesh.vertex_colors)
        if vertex_colors.size == 0:
            vertex_colors = np.zeros_like(vertices)
        elif np.max(vertex_colors) <= 1:
            vertex_colors = vertex_colors * 255.0

        raw_segments = load_scannet_segments(raw_seg_path)
        if len(raw_segments) != len(vertices):
            return f"[MISS] {scene_name} seg/mesh length mismatch"

        seg_to_instance, inst_to_label = load_scannet_aggregation(agg_path)
        vertex_instance = build_vertex_instance_from_segments(raw_segments, seg_to_instance)

        center_points = np.mean(vertices, axis=0)
        center_points[2] = np.min(vertices[:, 2])
        vertices = vertices - center_points

        torch.save(inst_to_label, self.inst2label_path / f"{scene_name}.pth")
        torch.save((vertices, vertex_colors, vertex_instance), self.pcd_path / f"{scene_name}.pth")

        new_segments = resplit_segments(seg_json_path, vertex_instance, self.resplit_aug)
        if new_segments is None:
            return f"[MISS] {scene_name} failed to build segments"
        np.save(self.aux_root / f"{scene_name}.npy", new_segments.astype(np.int32))
        return f"[DONE] {scene_name}"

    def process_scans(self) -> None:
        scene_names = sorted([d.name for d in self.data_root.iterdir() if d.is_dir()])
        if self.process_number > 0:
            scene_names = scene_names[:self.process_number]

        with open(self.split_dir / "train_split.txt", "w", encoding="utf-8") as fp:
            fp.write("\n".join(scene_names))

        if not scene_names:
            print(f"[WARN] No scenes found for dataset {self.dataset_name}: {self.data_root}")
            return

        with tqdm(total=len(scene_names), desc=self.dataset_name, unit="scene") as pbar:
            with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                futures = {executor.submit(self.scene_proc, scene_name): scene_name for scene_name in scene_names}
                for future in as_completed(futures):
                    print(future.result())
                    pbar.update(1)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG), help="Path to the root config.yaml")
    args = parser.parse_args()

    config_path = Path(args.config).expanduser().resolve()
    cfg = load_config(config_path)

    process_datasets = cfg.get("process_dataset", [])
    if not process_datasets:
        print("[WARN] process_dataset is empty, nothing to do.")
        return

    datasets_cfg = cfg.get("datasets", {})
    processor_registry = {
        "svpp": SVPPProcessor,
        "ScanNet": ScanNetProcessor,
    }
    for dataset_name in process_datasets:
        if dataset_name not in datasets_cfg:
            print(f"[WARN] dataset config missing for: {dataset_name}")
            continue

        processor_cls = processor_registry.get(dataset_name)
        if processor_cls is None:
            print(f"[WARN] no processor registered for dataset: {dataset_name}")
            continue

        print(f"[INFO] Processing dataset: {dataset_name}")
        processor = processor_cls(
            dataset_name=dataset_name,
            dataset_cfg=datasets_cfg[dataset_name],
            cfg=cfg,
        )
        processor.process_scans()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Visualize a SceneVerse++ mesh and camera poses with Open3D.

Example:
    python view_sceneverse_camera_poses.py \
        /mnt/fillipo/yaowei/svpp_release_data/svpp_data \
        --scene-name bedroom_100_3o5KSzfdOSE
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Load mesh.ply and camera_info.json for a scene and visualize them "
            "with Open3D."
        )
    )
    parser.add_argument(
        "data_root",
        type=Path,
        help="Root directory containing SceneVerse++ scene folders.",
    )
    parser.add_argument(
        "--scene-name",
        required=True,
        help="Scene folder name to visualize.",
    )
    parser.add_argument(
        "--mesh-name",
        default="mesh.ply",
        help="Mesh filename inside the scene folder. Default: mesh.ply",
    )
    parser.add_argument(
        "--camera-info-name",
        default="camera_info.json",
        help="Camera info filename inside the scene folder. Default: camera_info.json",
    )
    parser.add_argument(
        "--extrinsic-type",
        choices=("world_to_camera", "camera_to_world"),
        default="world_to_camera",
        help=(
            "Interpretation of the stored extrinsic matrices. "
            "Default: world_to_camera"
        ),
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Visualize every N-th camera pose. Default: 1",
    )
    parser.add_argument(
        "--max-cameras",
        type=int,
        default=None,
        help="Limit the number of visualized camera poses after stride. Default: all",
    )
    parser.add_argument(
        "--frustum-scale",
        type=float,
        default=0.18,
        help="Camera frustum scale in world units. Default: 0.18",
    )
    parser.add_argument(
        "--origin-frame-size",
        type=float,
        default=0.4,
        help="Origin coordinate frame size. Default: 0.4",
    )
    parser.add_argument(
        "--camera-axis-size",
        type=float,
        default=0.08,
        help="Per-camera axis frame size. Set 0 to disable. Default: 0.08",
    )
    return parser.parse_args()


def load_camera_info(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def sorted_extrinsics(extrinsic_dict: dict[str, list[list[float]]]) -> list[tuple[str, np.ndarray]]:
    def sort_key(item: tuple[str, list[list[float]]]):
        frame_id = item[0]
        return int(frame_id) if frame_id.isdigit() else frame_id

    return [
        (frame_id, np.asarray(matrix, dtype=np.float64))
        for frame_id, matrix in sorted(extrinsic_dict.items(), key=sort_key)
    ]


def to_camera_to_world(extrinsic: np.ndarray, extrinsic_type: str) -> np.ndarray:
    if extrinsic.shape != (4, 4):
        raise ValueError(f"Expected 4x4 extrinsic matrix, got shape {extrinsic.shape}")
    if extrinsic_type == "camera_to_world":
        return extrinsic
    return np.linalg.inv(extrinsic)


def transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    rotation = transform[:3, :3]
    translation = transform[:3, 3]
    return (rotation @ points.T).T + translation


def create_camera_frustum(
    intrinsic: dict,
    camera_to_world: np.ndarray,
    scale: float,
    color: tuple[float, float, float] = (0.95, 0.35, 0.1),
):
    import open3d as o3d

    w = float(intrinsic["w"])
    h = float(intrinsic["h"])
    fx = float(intrinsic["fx"])
    fy = float(intrinsic["fy"])
    cx = float(intrinsic["cx"])
    cy = float(intrinsic["cy"])

    camera_center = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)
    image_corners = np.array(
        [
            [(0.0 - cx) / fx * scale, (0.0 - cy) / fy * scale, scale],
            [(w - cx) / fx * scale, (0.0 - cy) / fy * scale, scale],
            [(w - cx) / fx * scale, (h - cy) / fy * scale, scale],
            [(0.0 - cx) / fx * scale, (h - cy) / fy * scale, scale],
        ],
        dtype=np.float64,
    )
    points_camera = np.concatenate([camera_center, image_corners], axis=0)
    points_world = transform_points(points_camera, camera_to_world)

    lines = np.array(
        [
            [0, 1],
            [0, 2],
            [0, 3],
            [0, 4],
            [1, 2],
            [2, 3],
            [3, 4],
            [4, 1],
        ],
        dtype=np.int32,
    )
    colors = np.tile(np.asarray(color, dtype=np.float64), (len(lines), 1))

    frustum = o3d.geometry.LineSet()
    frustum.points = o3d.utility.Vector3dVector(points_world)
    frustum.lines = o3d.utility.Vector2iVector(lines)
    frustum.colors = o3d.utility.Vector3dVector(colors)
    return frustum


def build_geometries(
    mesh_path: Path,
    camera_info_path: Path,
    extrinsic_type: str,
    stride: int,
    max_cameras: int | None,
    frustum_scale: float,
    origin_frame_size: float,
    camera_axis_size: float,
):
    import open3d as o3d

    camera_info = load_camera_info(camera_info_path)
    intrinsic = camera_info["intrinsic"]
    extrinsics = sorted_extrinsics(camera_info["extrinsic"])

    if stride <= 0:
        raise ValueError("--stride must be a positive integer")
    extrinsics = extrinsics[::stride]
    if max_cameras is not None:
        if max_cameras <= 0:
            raise ValueError("--max-cameras must be a positive integer")
        extrinsics = extrinsics[:max_cameras]

    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    if mesh.is_empty():
        raise ValueError(f"Failed to load mesh or mesh is empty: {mesh_path}")
    if not mesh.has_vertex_normals():
        mesh.compute_vertex_normals()
    if not mesh.has_vertex_colors():
        mesh.paint_uniform_color([0.78, 0.78, 0.8])

    geometries: list = [mesh]
    geometries.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=origin_frame_size))

    for frame_id, extrinsic in extrinsics:
        camera_to_world = to_camera_to_world(extrinsic, extrinsic_type)
        frustum = create_camera_frustum(
            intrinsic=intrinsic,
            camera_to_world=camera_to_world,
            scale=frustum_scale,
        )
        geometries.append(frustum)

        if camera_axis_size > 0:
            axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=camera_axis_size)
            axis.transform(camera_to_world)
            geometries.append(axis)

    return geometries, len(extrinsics)


def main() -> int:
    args = parse_args()
    scene_dir = args.data_root.expanduser().resolve() / args.scene_name
    if not scene_dir.is_dir():
        print(f"Scene directory does not exist: {scene_dir}", file=sys.stderr)
        return 1

    mesh_path = scene_dir / args.mesh_name
    camera_info_path = scene_dir / args.camera_info_name
    if not mesh_path.is_file():
        print(f"Mesh file not found: {mesh_path}", file=sys.stderr)
        return 1
    if not camera_info_path.is_file():
        print(f"Camera info file not found: {camera_info_path}", file=sys.stderr)
        return 1

    try:
        import open3d as o3d
    except Exception as exc:
        print(
            "Failed to import open3d. Install a working Open3D environment first. "
            f"Import error: {exc}",
            file=sys.stderr,
        )
        return 1

    try:
        geometries, num_cameras = build_geometries(
            mesh_path=mesh_path,
            camera_info_path=camera_info_path,
            extrinsic_type=args.extrinsic_type,
            stride=args.stride,
            max_cameras=args.max_cameras,
            frustum_scale=args.frustum_scale,
            origin_frame_size=args.origin_frame_size,
            camera_axis_size=args.camera_axis_size,
        )
    except Exception as exc:
        print(f"Failed to build visualization: {exc}", file=sys.stderr)
        return 1

    print(f"Scene: {args.scene_name}")
    print(f"Mesh: {mesh_path}")
    print(f"Camera info: {camera_info_path}")
    print(f"Visualizing {num_cameras} camera pose(s)")

    o3d.visualization.draw_geometries(
        geometries,
        window_name=f"SceneVerse++ Cameras - {args.scene_name}",
        width=1600,
        height=1000,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

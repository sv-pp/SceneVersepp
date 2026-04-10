import os
from copy import deepcopy
from typing import TYPE_CHECKING, Dict, List, Union, Sequence, Optional, Tuple

import torch
import numpy as np
from scipy.spatial.transform import Rotation as R

from spatiallm.layout.layout import Layout
from spatiallm.layout.entity import NORMALIZATION_PRESET
from spatiallm.pcd import load_o3d_pcd, get_points_and_colors
from spatiallm.pcd.transform import Compose
import numpy as np
import open3d as o3d
import re
import random
import torch.distributed as dist
if TYPE_CHECKING:
    from numpy.typing import NDArray
    from transformers import PreTrainedTokenizer

    PointCloudInput = Union[str, dict, NDArray]

LAYOUT_S_PLACEHOLDER = os.environ.get("LAYOUT_S_PLACEHOLDER", "<|layout_s|>")
LAYOUT_E_PLACEHOLDER = os.environ.get("LAYOUT_E_PLACEHOLDER", "<|layout_e|>")
POINT_S_TOKEN = os.environ.get("POINT_S_TOKEN", "<|point_start|>")
POINT_E_TOKEN = os.environ.get("POINT_E_TOKEN", "<|point_end|>")
POINT_CLOUD_PLACEHOLDER = os.environ.get("POINT_CLOUD_PLACEHOLDER", "<point_cloud>")

def parse_layout_str(layout_str: str):
    """把 layout string 解析成 list(dict)"""
    objs = []
    # 提取 bbox_x=Bbox(...) 里的内容
    matches = re.findall(r"bbox_(\d+)=Bbox\((.*?)\)", layout_str)
    for idx, content in matches[:15]:
        parts = content.split(",")
        name = parts[0]
        nums = [float(x) for x in parts[1:]]
        cx, cy, cz = nums[0:3]
        anglez = nums[3]
        sx, sy, sz = nums[4:7]
        center = np.array([cx, cy, cz])
        size = np.array([sx, sy, sz])
        bbox_min = center - size / 2
        bbox_max = center + size / 2
        objs.append({
            "index": idx,
            "name": name,
            "center": center,
            "anglez": anglez,
            "size": size,
            "bbox": (bbox_min, bbox_max)
        })
    return objs

def format_layout_str(objs):
    """
    把裁剪后的 list(dict) 转回和输入一致的 layout string，
    并且重新编号 bbox，从 0 开始。
    """
    body_lines = []
    for new_idx, obj in enumerate(objs):
        c = obj["center"]
        sz = obj["size"]
        anglez = obj["anglez"]
        line = (
            f"bbox_{new_idx}=Bbox({obj['name']},"
            f"{c[0]:.6f},{c[1]:.6f},{c[2]:.6f},"
            f"{anglez:.6f},"
            f"{sz[0]:.6f},{sz[1]:.6f},{sz[2]:.6f})"
        )
        body_lines.append(line)

    return "<|layout_s|>" + "\n".join(body_lines) + "<|layout_e|>"

def crop_layout_and_pcd(
    layout_str: str,
    pcd: o3d.geometry.PointCloud,
    radius: float = 3,
    height: float = 5.0,
    max_trials: int = 3,
    max_points: int = 150000,
    voxel_size: float = 0.025,
):
    """
    随机选择一个 layout 中的物体，裁剪点云 + layout
    - 若 layout 为空，则以点云几何中心作为中心裁剪
    - 保证返回点数 <= max_points
    - 返回时不再进行平移居中
    """
    objs = parse_layout_str(layout_str) or []
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors) if pcd.has_colors() else np.zeros_like(points)

    def limit_points(pcd_in):
        out = pcd_in
        if voxel_size and voxel_size > 0:
            out = out.voxel_down_sample(voxel_size)
        n = len(out.points)
        if n > max_points:
            ratio = max_points / n
            out = out.random_down_sample(ratio)
        return out

    # ---------- 没有 layout objs ----------
    if len(objs) == 0:
        if len(points) == 0:
            return "<|layout_s|><|layout_e|>", pcd

        center = points.mean(0)  # 用几何中心
        d2 = ((points[:, :2] - center[:2]) ** 2).sum(axis=1)
        mask_radius = d2 < radius ** 2
        z_min = points[:, 2].min()
        z_max = z_min + height
        mask_height = (points[:, 2] >= z_min) & (points[:, 2] <= z_max)
        mask = mask_radius & mask_height

        cropped_points = points[mask]
        cropped_colors = colors[mask]

        cropped_pcd = o3d.geometry.PointCloud()
        cropped_pcd.points = o3d.utility.Vector3dVector(cropped_points)
        cropped_pcd.colors = o3d.utility.Vector3dVector(cropped_colors)

        cropped_pcd = limit_points(cropped_pcd)
        return "<|layout_s|><|layout_e|>", cropped_pcd

    # ---------- 有 layout objs ----------
    for _ in range(max_trials):
        chosen = random.choice(objs)
        center = chosen["center"]

        d2 = ((points[:, :2] - center[:2]) ** 2).sum(axis=1)
        mask_radius = d2 < radius ** 2
        z_min = points[:, 2].min()
        z_max = z_min + height
        mask_height = (points[:, 2] >= z_min) & (points[:, 2] <= z_max)
        mask = mask_radius & mask_height

        if not np.any(mask):
            continue

        cropped_points = points[mask]
        cropped_colors = colors[mask]

        cropped_pcd = o3d.geometry.PointCloud()
        cropped_pcd.points = o3d.utility.Vector3dVector(cropped_points)
        cropped_pcd.colors = o3d.utility.Vector3dVector(cropped_colors)

        cropped_pcd = limit_points(cropped_pcd)

        cropped_layout = [
            obj.copy()
            for obj in objs
            if ((obj["center"][:2] - center[:2]) @ (obj["center"][:2] - center[:2]) < radius ** 2)
               and (z_min <= obj["center"][2] <= z_max)
        ]

        return format_layout_str(cropped_layout), cropped_pcd

    # ---------- fallback ----------
    safe_pcd = limit_points(pcd)
    return "<|layout_s|><|layout_e|>", safe_pcd

def print_pcd_size(pcd: o3d.geometry.PointCloud, name="pcd"):
    if not pcd.has_points():
        print(f"[DEBUG] {name} has no points.")
        return

    pts = np.asarray(pcd.points)  # (N, 3)
    min_xyz = pts.min(axis=0)
    max_xyz = pts.max(axis=0)
    size = max_xyz - min_xyz

    print(f"[DEBUG] {name} - min: {min_xyz}, max: {max_xyz}, size: {size}, total points: {pts.shape[0]}")
    return min_xyz, max_xyz, size


def debug_grid_stats(points: torch.Tensor, grid_size: float, tag: str = ""):
    """
    Debug 输出点云 voxel 化后的 grid_coord 统计信息
    Args:
        points: (N,3) torch.Tensor
        grid_size: float
        tag: str, 调试标记字符串
    """
    # 确定当前卡号
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        rank = torch.distributed.get_rank()
    elif torch.cuda.is_available():
        rank = torch.cuda.current_device()
    else:
        rank = 0  # 单机 CPU 情况
    # voxel 化
    grid_coord = torch.div(
        points - points.min(0).values,
        grid_size,
        rounding_mode="trunc"
    ).int()

    min_xyz = grid_coord.min(dim=0).values.cpu().numpy()
    max_xyz = grid_coord.max(dim=0).values.cpu().numpy()
    mean_xyz = grid_coord.float().mean(dim=0).cpu().numpy()

    print(f"[DEBUG][Rank {rank}] {tag} - "
          f"grid_coord stats: min={min_xyz}, max={max_xyz}, mean={mean_xyz}")

def debug_grid_coord(grid_coord, tag=""):
    rank = dist.get_rank() if dist.is_initialized() else 0

    if isinstance(grid_coord, torch.Tensor):
        min_vals = grid_coord.min(0).values.cpu().numpy()
        max_vals = grid_coord.max(0).values.cpu().numpy()
        mean_vals = grid_coord.float().mean(0).cpu().numpy()
    elif isinstance(grid_coord, np.ndarray):
        min_vals = grid_coord.min(axis=0)
        max_vals = grid_coord.max(axis=0)
        mean_vals = grid_coord.mean(axis=0)
    else:
        raise TypeError(f"Unsupported type for grid_coord: {type(grid_coord)}")

    print(f"[DEBUG][Rank {rank}] {tag} grid_coord min: {min_vals}, "
          f"max: {max_vals}, mean: {mean_vals}")
class SpatialLMPlugin:
    def __init__(
        self,
        point_token: str = "<|point_pad|>",
        num_bins: int = 1280,
        do_augmentation: bool = False,
        random_rotation: bool = False,
    ):
        self.point_token = point_token

        global_extent = NORMALIZATION_PRESET["world"]
        self.num_bins = num_bins
        self.grid_size = (global_extent[1] - global_extent[0]) / self.num_bins
        self.do_augmentation = do_augmentation
        self.random_rotation = random_rotation
        self.augmentation = Compose(
            [
                dict(type="RandomColorGrayScale", p=0.05),
                dict(type="ChromaticAutoContrast", p=0.2, blend_factor=None),
                dict(type="ChromaticTranslation", p=0.75, ratio=0.1),
                dict(type="ChromaticJitter", p=0.8, std=0.05),
                dict(type="HueSaturationTranslation", hue_max=0.2, saturation_max=0.2),
                dict(type="RandomColorDrop", p=0.1, color_augment=0.0),
                dict(type="RandomJitter", sigma=0.025, clip=0.05, ratio=0.8, p=0.9),
                dict(type="RandomJitter", sigma=0.2, clip=0.2, ratio=0.05, p=0.85),
                dict(type="RandomJitter", sigma=0.4, clip=1.0, ratio=0.001, p=0.75),
                dict(type="RandomJitter", sigma=0.5, clip=4.0, ratio=0.0005, p=0.7),
                dict(
                    type="ElasticDistortion",
                    distortion_params=[[0.2, 0.4], [0.8, 1.6]],
                    p=[0.85, 0.5],
                ),
            ]
        )

        self.transform = Compose(
            [
                dict(type="PositiveShift"),
                dict(type="NormalizeColor"),
                dict(
                    type="GridSample",
                    grid_size=self.grid_size,
                    hash_type="fnv",
                    mode="train",
                    keys=("coord", "color"),
                    return_grid_coord=True,
                    max_grid_coord=self.num_bins,
                ),
            ]
        )

    def _preprocess_point_cloud(self, point_cloud: dict) -> np.ndarray:
        r"""
        Pre-processes a single point cloud.
        """
        point_cloud = self.transform(point_cloud)
        coord = point_cloud["grid_coord"]
        xyz = point_cloud["coord"]
        color = point_cloud["color"]
        assert len(coord) == len(xyz) == len(color)
        return np.concatenate([coord, xyz, color], axis=1)

    def _regularize_point_clouds(
        self, point_clouds: Sequence["PointCloudInput"], **kwargs
    ) -> torch.Tensor:
        points_list = []
        max_len = 0
        for point_cloud in point_clouds:
            if not isinstance(point_cloud, dict):
                raise ValueError(
                    "Point cloud input must be a dictionary with 'name' and 'coord' keys."
                )
            point_feats = self._preprocess_point_cloud(point_cloud, **kwargs)
            max_len = max(max_len, len(point_feats))
            points_list.append(point_feats)

        for i in range(len(points_list)):
            points_list[i] = np.pad(
                points_list[i],
                ((0, max_len - len(points_list[i])), (0, 0)),
                mode="constant",
                constant_values=np.nan,
            )

        # convert list of point clouds to batch with shape (batch_size, max_len, 3)
        return torch.as_tensor(np.stack(points_list, axis=0))

    def _get_mm_inputs(
        self,
        batched_messages: Sequence[Dict[str, str]],
        point_clouds: Sequence["PointCloudInput"],
    ) -> dict:
        input_dict = {"point_clouds": None}  # default key

        point_clouds_data = []
        transformations = []
        for index, pcd_path in enumerate(point_clouds):
            pcd_name = os.path.basename(pcd_path)
            pcd = load_o3d_pcd(pcd_path)
            cropped_layout_str, cropped_pcd = crop_layout_and_pcd(batched_messages[index][1]['content'], pcd, radius=3)
            points, colors = get_points_and_colors(cropped_pcd)
            # debug_grid_stats(torch.as_tensor(points), self.grid_size, tag=f"Before Aug {pcd_name}")
            # print_pcd_size(pcd, name=pcd_path)
            # print_pcd_size(cropped_pcd, name=pcd_name)
            if self.do_augmentation:
                data_aug = {"name": "pcd", "coord": points, "color": colors}
                data_aug = self.augmentation(data_aug)
                points = data_aug["coord"]
                colors = data_aug["color"]
            batched_messages[index][1]['content'] = cropped_layout_str
            # randomly apply scale and rotation transformation to the point cloud
            if self.random_rotation:
                angle_z = np.random.random() * 2 * np.pi
            else:
                angle_z = np.random.choice(np.array([0, 0.5, 1.0, 1.5]) * np.pi)

            scaling = np.random.uniform(0.75, 1.25)
            rotmat = R.from_rotvec(np.array([0, 0, angle_z])).as_matrix()
            min_bound = points.min(axis=0)
            max_bound = points.max(axis=0)
            center_pt = (min_bound + max_bound) / 2
            scaled_points = (points - center_pt) * scaling
            transformed_points = (rotmat @ scaled_points.T).T + center_pt
            # store transformation parameters for sync the augmentation to the layout
            transformations.append(
                {
                    "angle_z": angle_z,
                    "center_pt": center_pt,
                    "scaling": scaling,
                    "min_bound": np.min(transformed_points, axis=0),
                    "transformed_points": transformed_points,
                }
            )

            point_cloud = {"name": "pcd", "coord": transformed_points, "color": colors}
            point_clouds_data.append(point_cloud)

        # Here we assume each conversation has exactly one point cloud
        assert len(batched_messages) == len(point_clouds_data)
        processed_messages = []
        for mi, messages in enumerate(batched_messages):
            processed_messages.append(
                self.process_messages(messages, [transformations[mi]])
            )

        if len(processed_messages) != 0:
            input_dict["messages"] = processed_messages
        if len(point_clouds_data) != 0:
            # convert point clouds to batched tensors with shape (batch_size, max_len, 9)
            input_dict["point_clouds"] = self._regularize_point_clouds(
                point_clouds_data
            )
        return input_dict

    def _validate_input(
        self,
        point_clouds: Sequence["PointCloudInput"],
    ) -> None:
        r"""
        Validates if this model accepts the input modalities.
        """
        if len(point_clouds) != 0 and self.point_token is None:
            raise ValueError(
                "This model does not support point cloud input. Please check whether the correct `template` is used."
            )

    def process_token_ids(
        self,
        input_ids: List[int],
        labels: Optional[List[int]],
        point_clouds: Sequence["PointCloudInput"],
        tokenizer: "PreTrainedTokenizer",
    ) -> Tuple[List[int], Optional[List[int]]]:
        self._validate_input(point_clouds)
        return input_ids, labels

    def process_messages(
        self,
        messages: Sequence[Dict[str, str]],
        transformations: Sequence[dict],
    ) -> List[Dict[str, str]]:
        r"""
        Pre-processes input messages to sync the transformation between point cloud and layout.
        """
        self._validate_input(transformations)
        messages = deepcopy(messages)
        num_point_tokens = 0

        for message in messages:
            content = message["content"]
            if LAYOUT_S_PLACEHOLDER in content and LAYOUT_E_PLACEHOLDER in content:
                transformation = transformations[num_point_tokens - 1]
                min_bound = transformation["min_bound"]
                center_pt = transformation["center_pt"]
                scaling = transformation["scaling"]
                transformed_points = transformation["transformed_points"]
                layout_start_pos = content.index(LAYOUT_S_PLACEHOLDER)
                layout_end_pos = content.index(LAYOUT_E_PLACEHOLDER)
                layout_content = content[
                    layout_start_pos + len(LAYOUT_S_PLACEHOLDER) : layout_end_pos
                ]
                # parse layout_content
                layout = Layout(layout_content)
                # transformation augmentation
                layout.translate(-center_pt)
                layout.scale(scaling)
                layout.rotate(transformation["angle_z"])
                layout.translate(center_pt)
                layout.filter_empty_bboxes(transformed_points, num_points=100)
                layout.reorder_entities()
                layout.translate(-min_bound)
                layout.normalize_and_discretize(self.num_bins)
                new_layout_content = layout.to_language_string()
                content = content.replace(
                    f"{LAYOUT_S_PLACEHOLDER}{layout_content}{LAYOUT_E_PLACEHOLDER}",
                    new_layout_content,
                )
                message["content"] = content

            if POINT_CLOUD_PLACEHOLDER in content:
                content = content.replace(
                    POINT_CLOUD_PLACEHOLDER,
                    f"{POINT_S_TOKEN}{self.point_token}{POINT_E_TOKEN}",
                    1,
                )
                num_point_tokens += 1
                message["content"] = content

        if len(transformations) != num_point_tokens:
            raise ValueError(
                f"The number of point clouds does not match the number of {POINT_CLOUD_PLACEHOLDER} tokens."
            )
        return messages

    def get_mm_inputs(
        self,
        point_clouds: Sequence["PointCloudInput"],
        batch_prompts: Sequence[List[int]],
    ) -> Dict[str, Union[List[dict]]]:
        r"""
        Builds batched multimodal inputs for VLMs.

        Arguments:
            point_clouds: a list of point cloud inputs, shape (num_point_clouds,)
            pointlens: number of point clouds in each sample, shape (batch_size,)
            batch_ids: token ids of input samples, shape (batch_size, seq_len)
            processor: a processor for pre-processing images and videos
        """
        self._validate_input(point_clouds)
        return self._get_mm_inputs(batch_prompts, point_clouds)


def get_mm_plugin(
    point_token: str = "<|point_pad|>",
    **kwargs,
) -> "SpatialLMPlugin":
    return SpatialLMPlugin(point_token, **kwargs)

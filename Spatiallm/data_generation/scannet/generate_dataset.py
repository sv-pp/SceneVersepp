# -*- coding: utf-8 -*-

import os
import argparse
import json
from glob import glob

import pandas as pd
from tqdm import tqdm

from spatiallm.tuner.data import (
    LAYOUT_S_PLACEHOLDER,
    LAYOUT_E_PLACEHOLDER,
    POINT_CLOUD_PLACEHOLDER,
)
from spatiallm.layout.layout import Layout


def collect_scenes(split, pcd_dir, layout_dir):
    """收集某个 split (train/val) 下的场景id"""
    pcd_files = glob(os.path.join(pcd_dir, split, "*.ply"))
    pcd_scene_ids = [os.path.basename(pcd_file).split(".")[0] for pcd_file in pcd_files]
    layout_files = glob(os.path.join(layout_dir, split, "*.txt"))
    layout_scene_ids = [
        os.path.basename(layout_file).split(".")[0] for layout_file in layout_files
    ]
    scene_ids = set(pcd_scene_ids) & set(layout_scene_ids)
    return list(scene_ids)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--dataset_dir",
        type=str,
        required=True,
        help="Path to the preprocessed dataset (with pcd/train,val and layout/train,val)",
    )
    parser.add_argument(
        "--code_template_file",
        type=str,
        default="code_template.txt",
    )
    parser.add_argument(
        "-n",
        "--dataset_name",
        type=str,
        required=True,
        help="Name of the dataset",
    )
    args = parser.parse_args()

    pcd_dir = os.path.join(args.dataset_dir, "pcd")
    layout_dir = os.path.join(args.dataset_dir, "layout")

    # 分别收集 train / val
    train_scene_ids = collect_scenes("train", pcd_dir, layout_dir)
    val_scene_ids = collect_scenes("val", pcd_dir, layout_dir)

    print(f"Creating dataset with {len(train_scene_ids)} train scenes and {len(val_scene_ids)} val scenes...")

    with open(args.code_template_file, "r") as f:
        code_template = f.read()

    dataset = {
        "train": [],
        "val": [],
    }

    # === 处理 train split ===
    for scene_id in tqdm(train_scene_ids, desc="Processing train"):
        try:
            with open(os.path.join(layout_dir, "train", f"{scene_id}.txt"), "r") as f:
                layout_content = f.read()

            layout = Layout(layout_content)
            language_string = layout.to_language_string()

            conversation_data = {
                "conversations": [
                    {
                        "from": "human",
                        "value": f"{POINT_CLOUD_PLACEHOLDER}Detect boxes. The reference code is as followed: {code_template}",
                    },
                    {
                        "from": "gpt",
                        "value": f"{LAYOUT_S_PLACEHOLDER}{language_string}{LAYOUT_E_PLACEHOLDER}",
                    },
                ],
                "point_clouds": [
                    os.path.join("pcd", "train", f"{scene_id}.ply"),
                ],
            }
            dataset["train"].append(conversation_data)

        except Exception as e:
            print(f"Error processing train scene {scene_id}: {e}")
            continue

    # === 处理 val split ===
    for scene_id in tqdm(val_scene_ids, desc="Processing val"):
        try:
            with open(os.path.join(layout_dir, "val", f"{scene_id}.txt"), "r") as f:
                layout_content = f.read()

            layout = Layout(layout_content)
            language_string = layout.to_language_string()

            conversation_data = {
                "conversations": [
                    {
                        "from": "human",
                        "value": f"{POINT_CLOUD_PLACEHOLDER}Detect boxes. The reference code is as followed: {code_template}",
                    },
                    {
                        "from": "gpt",
                        "value": f"{LAYOUT_S_PLACEHOLDER}{language_string}{LAYOUT_E_PLACEHOLDER}",
                    },
                ],
                "point_clouds": [
                    os.path.join("pcd", "val", f"{scene_id}.ply"),
                ],
            }
            dataset["val"].append(conversation_data)

        except Exception as e:
            print(f"Error processing val scene {scene_id}: {e}")
            continue

    # 保存 json
    print(f"Saving train set with {len(dataset['train'])} samples...")
    with open(
        os.path.join(args.dataset_dir, f"{args.dataset_name}_train.json"), "w"
    ) as f:
        json.dump(dataset["train"], f, indent=2)

    print(f"Saving val set with {len(dataset['val'])} samples...")
    with open(
        os.path.join(args.dataset_dir, f"{args.dataset_name}_val.json"), "w"
    ) as f:
        json.dump(dataset["val"], f, indent=2)

    # 更新 dataset_info.json
    dataset_info = {
        f"{args.dataset_name}_train": {
            "file_name": f"{args.dataset_name}_train.json",
            "formatting": "sharegpt",
            "columns": {
                "messages": "conversations",
                "point_clouds": "point_clouds",
            },
        },
        f"{args.dataset_name}_val": {
            "file_name": f"{args.dataset_name}_val.json",
            "formatting": "sharegpt",
            "columns": {
                "messages": "conversations",
                "point_clouds": "point_clouds",
            },
        },
    }

    info_file = os.path.join(args.dataset_dir, "dataset_info.json")
    if not os.path.exists(info_file):
        with open(info_file, "w") as f:
            json.dump(dataset_info, f, indent=2)
    else:
        with open(info_file, "r") as f:
            original_dataset_info = json.load(f)
        original_dataset_info.update(dataset_info)
        with open(info_file, "w") as f:
            json.dump(original_dataset_info, f, indent=2)

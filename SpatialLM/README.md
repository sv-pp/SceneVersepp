# SpatialLM for SceneVerse++

This folder adapts [manycore-research/SpatialLM](https://github.com/manycore-research/SpatialLM) for training on SceneVerse++ data.

The end-to-end workflow is:

1. Install the SpatialLM environment.
2. Generate training data for SVPP and ScanNet.
3. (Optional) Visualize point clouds and layouts.
4. Pretrain on SVPP, fine-tune on ScanNet, then run inference and evaluation.

All commands below are run with `SpatialLM/` as the working directory.

## 1. Environment

Follow the upstream installation style:

```bash
# create a conda environment
conda create -n spatiallm python=3.11 -y
conda activate spatiallm

# install dependencies (Poetry, managed locally inside the conda env)
pip install poetry
poetry config virtualenvs.create false --local
poetry install

# install the model-specific extras (pick the one matching your target model)
# poe install-torchsparse
poe install-sonata
```

## 2. Data generation

### 2.1 SVPP

The SVPP pipeline reads each scene folder, uses `metadata.json` to build instance-level boxes, and writes the layout and point cloud outputs.

Generate layouts:

```bash
python data_generation/svpp/generate_layout.py \
  --data_root /path/to/svpp_data \
  --save_path ./data/svpp \
  --voxel_size 0.02 \
  --workers 16 \
  --process_number -1 \
  --label-map ./scannetv2-labels.combined.tsv
```

| Flag | Meaning |
| --- | --- |
| `--data_root` | SVPP raw dataset root. Each scene is a subdirectory containing `mesh.ply` and `metadata.json`. |
| `--save_path` | Output root for the generated `layout/` and `pcd/`. |
| `--voxel_size` | Downsample size for the saved point cloud. |
| `--workers` | Number of parallel workers. |
| `--process_number` | Scene cap; use `-1` to process all scenes. |
| `--label-map` | TSV used for label mapping. Pass an absolute path if the file is not in the repo root. |

Then build the training dataset:

```bash
python data_generation/svpp/generate_dataset.py \
  --dataset_dir ./data/svpp \
  --dataset_name svpp \
  --code_template_file ./code_template.txt
```

| Flag | Meaning |
| --- | --- |
| `--dataset_dir` | Directory produced by `generate_layout.py`. Must contain `layout/` and `pcd/`. |
| `--dataset_name` | Used to create `svpp_train.json`, `svpp_val.json`, and `dataset_info.json`. |
| `--code_template_file` | Prompt template used to build the instruction text. |

Resulting structure under `./data/svpp`:

```text
./data/svpp/
  layout/
  pcd/
  svpp_train.json
  svpp_val.json
  dataset_info.json
```

### 2.2 ScanNet

The ScanNet pipeline reads raw ScanNet scenes, loads `*_vh_clean_2.ply`, `*_vh_clean_2.0.010000.segs.json`, and `*.aggregation.json`, then writes outputs split by train/val.

Generate layouts:

```bash
python data_generation/scannet/generate_layout.py \
  --data_root /path/to/scannet \
  --save_path ./data/scannet \
  --voxel_size 0.02 \
  --workers 16 \
  --process_number -1
```

| Flag | Meaning |
| --- | --- |
| `--data_root` | ScanNet dataset root. Raw scenes are expected under `data_root/scans/`. |
| `--save_path` | Output root for the generated `layout/{train,val}` and `pcd/{train,val}`. |
| `--voxel_size` | Downsample size for the saved point cloud. |
| `--workers` | Number of parallel workers. |
| `--process_number` | Scene cap; use `-1` to process all scenes. |

Notes:

- The script expects the ScanNet train/val split files under `data_generation/scannet/`.
- The script also looks for `scannetv2-labels.combined.tsv` in the `SpatialLM/` root.
- If your file locations differ, update the paths in `data_generation/scannet/generate_layout.py` accordingly.

Then build the training dataset:

```bash
python data_generation/scannet/generate_dataset.py \
  --dataset_dir ./data/scannet \
  --dataset_name scannet \
  --code_template_file ./code_template.txt
```

| Flag | Meaning |
| --- | --- |
| `--dataset_dir` | Directory produced by `generate_layout.py`. Must contain `layout/{train,val}` and `pcd/{train,val}`. |
| `--dataset_name` | Used to create `scannet_train.json`, `scannet_val.json`, and `dataset_info.json`. |
| `--code_template_file` | Prompt template used to build the instruction text. |

Resulting structure under `./data/scannet`:

```text
./data/scannet/
  layout/
    train/
    val/
  pcd/
    train/
    val/
  scannet_train.json
  scannet_val.json
  dataset_info.json
```

## 3. Visualization

Inspect a point cloud together with its layout file using Rerun:

```bash
python visualize.py \
  --point_cloud ./data/scannet/pcd/train/scene0007_00.ply \
  --layout ./data/scannet/layout/train/scene0007_00.txt
```

| Flag | Meaning |
| --- | --- |
| `--point_cloud` | Path to the `.ply` point cloud. |
| `--layout` | Path to the corresponding `.txt` layout file. |
| `--radius` | Radius of rendered points in Rerun. |
| `--max_points` | Maximum number of points to visualize. |

For SVPP, use the matching paths under `./data/svpp/pcd/` and `./data/svpp/layout/`.

## 4. Training and evaluation

### 4.1 Download the base model

```bash
huggingface-cli download manycore-research/SpatialLM1.1-Qwen-0.5B \
  --local-dir ./models/basemodel
```

Hugging Face page: <https://huggingface.co/manycore-research/SpatialLM1.1-Qwen-0.5B>

### 4.2 Pretrain on SceneVerse++ / SVPP

```bash
python train.py configs/pretrain.yaml
```

Recommended [`configs/pretrain.yaml`](configs/pretrain.yaml) settings:

```yaml
model_name_or_path: ./models/basemodel
dataset: svpp_train
dataset_dir: ./data/svpp
template: spatiallm_qwen
save_dir: saves/pretrain
output_dir: saves/pretrain
```

| Field | Meaning |
| --- | --- |
| `model_name_or_path` | Path to the downloaded base model. |
| `dataset` | Must match the dataset name from `generate_dataset.py` plus the split suffix (e.g. `svpp_train`). |
| `dataset_dir` | Directory containing `dataset_info.json` and the generated `*.json` dataset files. |
| `template` | Model template used by the trainer. |
| `save_dir` / `output_dir` | Where checkpoints and logs are written. |

### 4.3 Fine-tune on ScanNet

```bash
python train.py configs/finetune.yaml
```

Recommended [`configs/finetune.yaml`](configs/finetune.yaml) settings:

```yaml
model_name_or_path: ./saves/svpp_pretrain/checkpoint-xxxx
dataset: scannet_train
dataset_dir: ./data/scannet
template: spatiallm_qwen
save_dir: saves/finetune
output_dir: saves/finetune
```

| Field | Meaning |
| --- | --- |
| `model_name_or_path` | Path to the pretrained checkpoint from stage 4.2. |
| `dataset` | Generated dataset name (e.g. `scannet_train`). |
| `dataset_dir` | Directory containing `dataset_info.json` and the generated ScanNet JSON files. |
| `save_dir` / `output_dir` | Where the fine-tuned checkpoints and logs are written. |

### 4.4 Inference

```bash
python inference.py \
  --data_file ./data/scannet/scannet_val.json \
  --output ./results/finetune \
  --model_path ./saves/finetune/checkpoint-*** \
  --code_template_file ./code_template.txt \
  --detect_type object
```

| Flag | Meaning |
| --- | --- |
| `--data_file` | Input dataset JSON produced by `generate_dataset.py`. |
| `--output` | Output directory (or file) for the predicted layout `.txt` files. |
| `--model_path` | Fine-tuned checkpoint path. |
| `--code_template_file` | Prompt template used during inference. |
| `--detect_type` | Detection mode, one of `all`, `arch`, or `object`. |
| `--category` | Optional list of object categories, used when `--detect_type object`. |

### 4.5 Evaluation

Evaluate predicted layouts against ground-truth layouts:

```bash
python eval.py \
  --gt_dir ./data/scannet/layout/train \
  --pred_dir ./results/finetune
```

| Flag | Meaning |
| --- | --- |
| `--gt_dir` | Directory of ground-truth `.txt` layout files. |
| `--pred_dir` | Directory of predicted `.txt` layout files. |

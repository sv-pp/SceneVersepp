# SpatialLM for SceneVerse++

This code is adapted from [manycore-research/SpatialLM](https://github.com/manycore-research/SpatialLM) and is used to train on SceneVerse++ data.

The workflow in this folder is:

1. Install the SpatialLM environment.
2. Generate training data for SVPP and ScanNet.
3. Visualize point clouds and layouts.
4. Train, fine-tune, run inference, and evaluate.

## 1. Environment Setup

Please follow the official SpatialLM installation guide first. The commands below are the same style as the upstream project and are sufficient for the training and data-processing scripts in this folder.

```bash
# clone the repository
git clone https://github.com/manycore-research/SpatialLM.git
cd SpatialLM

# create a conda environment
conda create -n spatiallm python=3.11
conda activate spatiallm

# install dependencies
pip install poetry
poetry config virtualenvs.create false --local
poetry install

# install the model-specific extra dependencies from the official guide
# choose the one that matches your target model
poe install-torchsparse
# poe install-sonata
```

Notes:

- The codebase is tested with Python 3.11.
- Follow the upstream SpatialLM README if you need the exact CUDA / PyTorch combination.
- Keep the repository root as the working directory when running the commands below.

## 2. Data Generation

### 2.1 Generate SVPP Training Data

The SVPP pipeline reads each scene folder, uses `metadata.json` to build instance-level boxes, then writes the layout and point cloud outputs.

Run layout generation first:

```bash
python data_generation/svpp/generate_layout.py \
  --data_root /path/to/svpp_data \
  --save_path ./data/svpp \
  --voxel_size 0.02 \
  --workers 16 \
  --process_number -1 \
  --label-map ./scannetv2-labels.combined.tsv
```

Parameter explanation:

- `--data_root`: SVPP raw dataset root. Each scene should be a subdirectory containing `mesh.ply` and `metadata.json`.
- `--save_path`: Output root for generated `layout/` and `pcd/`.
- `--voxel_size`: Downsample size for the saved point cloud.
- `--workers`: Number of parallel workers.
- `--process_number`: Limit the number of scenes to process. Use `-1` to process all scenes.
- `--label-map`: TSV file used for label mapping. If the file is not in the repository root, pass an absolute path.

After layout generation, build the training dataset:

```bash
python data_generation/svpp/generate_dataset.py \
  --dataset_dir ./data/svpp \
  --dataset_name svpp \
  --code_template_file ./code_template.txt
```

Parameter explanation:

- `--dataset_dir`: The directory produced by `generate_layout.py`. It must contain `layout/` and `pcd/`.
- `--dataset_name`: Dataset name used to create `svpp_train.json`, `svpp_val.json`, and `dataset_info.json`.
- `--code_template_file`: Prompt template file used to construct the instruction text.

The generated files will be placed under `./data/svpp`, for example:

```text
./data/svpp/
  layout/
  pcd/
  svpp_train.json
  svpp_val.json
  dataset_info.json
```

### 2.2 Generate ScanNet Training Data

The ScanNet pipeline reads the raw ScanNet scene folders, loads `*_vh_clean_2.ply`, `*_vh_clean_2.0.010000.segs.json`, and `*.aggregation.json`, then writes layout and point cloud outputs split by train / val.

Run layout generation first:

```bash
python data_generation/scannet/generate_layout.py \
  --data_root /path/to/scannet \
  --save_path ./data/scannet \
  --voxel_size 0.02 \
  --workers 16 \
  --process_number -1
```

Parameter explanation:

- `--data_root`: ScanNet dataset root. The script expects the raw scenes under `data_root/scans/`.
- `--save_path`: Output root for generated `layout/train`, `layout/val`, `pcd/train`, and `pcd/val`.
- `--voxel_size`: Downsample size for the saved point cloud.
- `--workers`: Number of parallel workers.
- `--process_number`: Limit the number of scenes to process. Use `-1` to process all scenes.

Notes:

- The current script expects the ScanNet train / val split files under `data_generation/scannet/`.
- The current script also looks for `scannetv2-labels.combined.tsv` in the repository root.
- If your file locations differ, update the paths in `data_generation/scannet/generate_layout.py` accordingly.

Then generate the training dataset:

```bash
python data_generation/scannet/generate_dataset.py \
  --dataset_dir ./data/scannet \
  --dataset_name scannet \
  --code_template_file ./code_template.txt
```

Parameter explanation:

- `--dataset_dir`: The directory produced by `generate_layout.py`. It must contain `layout/train`, `layout/val`, `pcd/train`, and `pcd/val`.
- `--dataset_name`: Dataset name used to create `scannet_train.json`, `scannet_val.json`, and `dataset_info.json`.
- `--code_template_file`: Prompt template file used to construct the instruction text.

The generated files will be placed under `./data/scannet`, for example:

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

Use the visualization script to inspect a point cloud together with its layout file:

```bash
python visualize.py \
  --point_cloud ./data/scannet/pcd/train/scene0007_00.ply \
  --layout ./data/scannet/layout/train/scene0007_00.txt
```

Parameter explanation:

- `--point_cloud`: Path to the `.ply` point cloud file.
- `--layout`: Path to the corresponding layout `.txt` file.
- `--radius`: Radius of the rendered points in Rerun.
- `--max_points`: Maximum number of points to visualize.

For SVPP, use the matching paths under `./data/svpp/pcd/` and `./data/svpp/layout/`.

## 4. Training and Evaluation

### 4.1 Download the Base Model

Download the base model to `./models/basemodel`:

```bash
huggingface-cli download manycore-research/SpatialLM1.1-Qwen-0.5B \
  --local-dir ./models/basemodel
```

You can also download it from the Hugging Face web page:

<https://huggingface.co/manycore-research/SpatialLM1.1-Qwen-0.5B>

### 4.2 Pretraining on SceneVerse++ / SVPP

Run pretraining:

```bash
python train.py configs/pretrain.yaml
```

Recommended `configs/pretrain.yaml` settings:

```yaml
model_name_or_path: ./models/basemodel
dataset: svpp_train
dataset_dir: ./data/svpp
template: spatiallm_qwen
save_dir: saves/pretrain
output_dir: saves/pretrain
```

Meaning of the key fields:

- `model_name_or_path`: Path to the downloaded base model.
- `dataset`: Must match the dataset name generated by `generate_dataset.py` plus the split suffix, for example `svpp_train`.
- `dataset_dir`: Directory that contains `dataset_info.json` and the generated `*.json` dataset files.
- `template`: Model template used by the trainer.
- `save_dir` / `output_dir`: Where checkpoints and logs will be written.

### 4.3 Fine-tuning on ScanNet

Run fine-tuning:

```bash
python train.py configs/finetune.yaml
```

Recommended `configs/finetune.yaml` settings:

```yaml
model_name_or_path: ./saves/svpp_pretrain/checkpoint-xxxx
dataset: scannet_train
dataset_dir: ./data/scannet
template: spatiallm_qwen
save_dir: saves/finetune
output_dir: saves/finetune
```

Meaning of the key fields:

- `model_name_or_path`: Path to the pretrained checkpoint.
- `dataset`: Must match the generated dataset name, for example `scannet_train`.
- `dataset_dir`: Directory that contains `dataset_info.json` and the generated ScanNet JSON files.
- `save_dir` / `output_dir`: Where the fine-tuned checkpoints and logs will be written.

### 4.4 Inference

Run inference on a generated dataset JSON file:

```bash
python inference.py \
  --data_file ./data/scannet/scannet_val.json \
  --output ./results/finetune \
  --model_path ./saves/finetune/checkpoint-37500 \
  --code_template_file ./code_template.txt \
  --detect_type object
```

Parameter explanation:

- `--data_file`: Input dataset JSON generated by `generate_dataset.py`.
- `--output`: Output directory or file for the predicted layout text files.
- `--model_path`: Fine-tuned checkpoint path.
- `--code_template_file`: Prompt template file used during inference.
- `--detect_type`: Detection mode, one of `all`, `arch`, or `object`.
- `--category`: Optional list of object categories when `--detect_type object` is used.

### 4.5 Evaluation

Evaluate the predicted layouts against the ground-truth layouts:

```bash
python eval.py \
  --gt_dir ./data/scannet/layout/train \
  --pred_dir ./results/finetune
```

Parameter explanation:

- `--gt_dir`: Directory that contains the ground-truth `.txt` layout files.
- `--pred_dir`: Directory that contains the predicted `.txt` layout files.


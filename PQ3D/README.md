# PQ3D for SceneVerse++

This folder adapts [PQ3D](https://github.com/PQ3D/PQ3D) for SceneVerse++ data generation and 3D instance-segmentation training.

For the underlying environment setup (MinkowskiEngine, flash-attn, torch-scatter, spconv, etc.), follow the installation instructions in the upstream PQ3D repository.

All commands below are run with `PQ3D/` as the working directory.

## 1. Data generation

The generation stage is driven by [`data_process/config.yaml`](data_process/config.yaml).

### Selecting datasets

`process_dataset` controls which datasets are processed in a given run. For example,

```yaml
process_dataset: [svpp, ScanNet]
```

will process both `svpp` and `ScanNet`.

### Per-dataset fields

Each entry under `datasets` accepts:

| Field | Meaning |
| --- | --- |
| `data_root` | Root directory of the raw dataset. Each scene is read from here. |
| `process_number` | Maximum number of scenes to process. Use `-1` to process all. |
| `segmentator_s` | `s` parameter of the segmentator. Affects the segment filename and the segmentation result. |
| `segmentator_k` | `k` parameter of the segmentator. Affects the segment filename and the segmentation result. |

### Run

```bash
python data_process/generate_dataset.py
```

The script reads `data_process/config.yaml` and generates the intermediate files for every dataset listed in `process_dataset`.

## 2. Visualization

```bash
python data_process/visulization.py --vis_training_data --dataset_name ScanNet
```

Arguments:

| Flag | Description |
| --- | --- |
| `--vis_training_data` | Visualize training data (point clouds and bounding boxes). |
| `--vis_segments` | Visualize segment results. |
| `--dataset_name` | Dataset to visualize. Must match a key under `datasets` in `config.yaml` (e.g. `svpp`, `ScanNet`). |
| `--scene_name` | Visualize a single scene. If omitted, iterates over all scenes. |
| `--config` | Path to the config file. Defaults to `data_process/config.yaml`. |

Example:

```bash
python data_process/visulization.py \
  --vis_training_data \
  --dataset_name ScanNet \
  --scene_name scene0007_00
```

## 3. Output layout

Generated data is written under the `save_root` specified in `config.yaml`. A typical run produces:

- `segments/` — segment files produced by the segmentator.
- `base/<dataset_name>/scan_data/` — point clouds and instance labels.
- `aux/<dataset_name>/segment_id/` — re-split segment ids.

Changing `config.yaml` will change these paths accordingly.

## 4. Training

Training proceeds in two stages: first pretrain on SVPP, then fine-tune on ScanNet using the SVPP checkpoint.

### Stage 1 — pretrain on SVPP

```bash
python run.py --config-path configs --config-name svpp_gt.yaml
```

### Stage 2 — fine-tune on ScanNet

```bash
python run.py --config-path configs --config-name svpp_gt_scannet_fps.yaml
```

Before stage 2, update `pretrain_ckpt_path` in [`configs/svpp_gt_scannet_fps.yaml`](configs/svpp_gt_scannet_fps.yaml) to point to the checkpoint saved by stage 1 (default: `exp_results/svpp_gt/ckpt/best.pth`). If your stage-1 output directory differs, set the path manually.

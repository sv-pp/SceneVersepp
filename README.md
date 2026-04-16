# SceneVerse++

SceneVerse++ is a project for lifting unlabeled internet-scale data into structured 3D scene understanding. It focuses on generating and using large-scale 3D supervision for tasks such as 3D object detection, instance segmentation, spatial VQA, and vision-language navigation.

Project page: <https://sv-pp.github.io/>

Paper: [**Lifting Unlabeled Internet-level Data for 3D Scene Understanding** (CVPR 2026)](https://arxiv.org/abs/2506.07491)

Hugging Face dataset: [bigai/SceneVersepp](https://huggingface.co/datasets/bigai/SceneVersepp)

## TODO

- [x] 3D detection training code: `Spatiallm`
- [x] 3D segmentation training code: `PQ3d`
- [ ] Data generation from web video

## Repository Structure

```text
.
├── PQ3d/        # 3D segmentation training code with PQ3D
├── Spatiallm/   # 3D detection training code with SpatialLM
└── README.md
```

For detailed instructions, see the component-level documentation:

- [`PQ3d/README.md`](PQ3d/README.md)
- [`Spatiallm/README.md`](Spatiallm/README.md)

## Quick Start

1. Download the Hugging Face dataset:

   ```bash
   huggingface-cli download bigai/SceneVersepp --repo-type dataset --local-dir ./svpp
   ```

2. Generate the SpatialLM training data:

   ```bash
   cd Spatiallm
   ```

   Follow the detailed instructions in [`Spatiallm/README.md`](Spatiallm/README.md).

3. Generate the PQ3D training data:

   ```bash
   cd PQ3d
   ```

   Follow the detailed instructions in [`PQ3d/README.md`](PQ3d/README.md).


## Citation

If you use this project, please cite the SceneVerse++ paper:

```bibtex
@inproceedings{chen2026lifting,
  title     = {Lifting Unlabeled Internet-level Data for 3D Scene Understanding},
  author    = {Chen, Yixin and Zhang, Yaowei and Yu, Huangyue and He, Junchao and Wang, Yan and Huang, Jiangyong and Shen, Hongyu and Ni, Junfeng and Wang, Shaofei and Jia, Baoxiong and Zhu, Song-Chun and Huang, Siyuan},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2026}
}
```

## Acknowledgements

This repository is built on top of:

- [PQ3D](https://github.com/PQ3D/PQ3D/tree/main)
- [SpatialLM](https://github.com/manycore-research/SpatialLM)

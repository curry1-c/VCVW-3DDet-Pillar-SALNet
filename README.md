# VCVW-3DDet-Pillar-SALNet

Official implementation for the manuscript:

**Depth-Reconstructed Point Cloud Understanding and Lightweight 3D Detection for Scene-Level Engineering Vehicles**

This repository provides the code, configuration files, data-processing scripts, and reproducibility materials for **VCVW-3DDet** and **Pillar-SALNet**, a lightweight 3D detection framework for scene-level engineering vehicles from depth-reconstructed point clouds.

This repository is directly associated with the manuscript submitted to **The Visual Computer**. It is intended to support transparent and reproducible research.

If you use this code, configuration files, data-processing scripts, or benchmark protocol in your research, please cite the related manuscript and the archived Zenodo release.

**Zenodo DOI:** https://doi.org/10.5281/zenodo.19886473
---

## 1. Important Note

This repository provides modified modules, configurations, and experimental settings based on **OpenPCDet**.

It is **not a standalone 3D detection framework**.

Please install and use it together with OpenPCDet:

```text
https://github.com/open-mmlab/OpenPCDet
```

To prepare the OpenPCDet environment, please first install OpenPCDet following its official instructions.

A typical installation procedure is:

```bash
git clone https://github.com/open-mmlab/OpenPCDet.git
cd OpenPCDet
python setup.py develop
```

Then merge the modified modules, configuration files, and scripts from this repository into the corresponding OpenPCDet directories.

---

## 2. Overview

Three-dimensional object detection in construction scenes differs from standard autonomous-driving benchmarks because engineering vehicles usually exhibit:

- large cross-category scale variation;
- frequent occlusion and clutter in construction scenes;
- sparse and anisotropic geometry in depth-reconstructed point clouds;
- limited available scene-level point-cloud benchmarks for engineering vehicles.

To address these issues, this work constructs **VCVW-3DDet**, an OpenPCDet-compatible 3D detection dataset derived from depth-reconstructed point clouds, and proposes **Pillar-SALNet**, a lightweight PointPillars-based detection framework.

Pillar-SALNet integrates:

- category-specific size priors;
- Size-Aware Label Assignment, referred to as **SALA**;
- Efficient Channel Attention, referred to as **ECA**.

---

## 3. Main Contributions

The main contributions of this repository and the corresponding manuscript are summarized as follows:

1. **Depth-reconstructed point-cloud benchmark**  
   We construct **VCVW-3DDet**, a scene-level 3D detection dataset for engineering vehicles, by back-projecting depth maps from the VCVW-3D virtual construction-scene dataset and organizing the results into an OpenPCDet-compatible format.

2. **Category-specific size priors**  
   We introduce category-specific geometric size priors to improve anchor initialization for engineering vehicles with large inter-class scale variation.

3. **Size-Aware Label Assignment**  
   We propose **SALA**, a size-aware adaptive label assignment strategy that adjusts supervision according to class-level geometric priors and target-anchor size deviation.

4. **Lightweight BEV feature enhancement**  
   We introduce **ECA** into BEV multi-scale feature fusion to strengthen channel-wise feature representation with negligible computational overhead.

5. **Reproducible experimental setting**  
   We provide configuration files, modified modules, data-processing scripts, split metadata, training/evaluation commands, and experimental records to support reproducibility.

---

## 4. Framework

### 4.1 Overall Pipeline

<p align="center">
  <img src="docs/pipeline.png" width="900"/>
</p>

The proposed framework is built upon **PointPillars** and improves the baseline from three aspects:

- geometry-aware anchor initialization using category-specific size priors;
- adaptive supervision through **SALA**;
- lightweight BEV feature enhancement with **ECA**.

---

## 5. Key Modules

### 5.1 Efficient Channel Attention

<p align="center">
  <img src="docs/eca_module.png" width="700"/>
</p>

The ECA module is inserted after multi-scale BEV feature fusion. It enhances channel-wise feature representation while maintaining a lightweight structure.

Main characteristics:

- captures local cross-channel interaction;
- avoids dimensionality reduction;
- introduces negligible computational overhead;
- improves BEV feature discrimination for multi-scale engineering vehicles.

---

### 5.2 Size-Aware Label Assignment

<p align="center">
  <img src="docs/sala_strategy.png" width="900"/>
</p>

SALA dynamically adjusts label assignment according to class-specific geometric priors and target-anchor size deviation.

Main effects:

- improves supervision quality for multi-scale targets;
- alleviates mismatch caused by fixed IoU thresholds;
- reduces false negatives for hard and scale-sensitive examples;
- improves detection stability for engineering vehicles with large geometric differences.

---

## 6. Experimental Results

### 6.1 Main Detection Results

The following table reports the main 3D detection performance on the 5000-frame VCVW-3DDet setting.

| Method | 3D AP_R40 (%) | Description |
|---|---:|---|
| SECOND | 39.34 | Voxel-based reference method |
| PointPillars Baseline | 66.70 | Main baseline |
| EMA | 67.33 | Lightweight attention reference |
| ECA-final | 67.88 | BEV feature enhancement |
| SALA(dimweight) | 67.34 | Size-aware supervision |
| **Pillar-SALNet (ECA + SALA(dimweight))** | **69.29** | **Final setting** |
| CenterPoint | 81.98 | Center-based cross-framework reference |

Compared with the PointPillars baseline, the final **ECA + SALA(dimweight)** setting improves 3D AP_R40 from **66.70%** to **69.29%**, yielding an improvement of **+2.59 percentage points**.

CenterPoint is reported as a supplementary cross-framework reference. Since it follows a center-based detection paradigm, it should not be interpreted as a direct replacement for the proposed PointPillars-based Pillar-SALNet framework.

---

### 6.2 Complexity and Efficiency

| Method | Params (M) | FPS | Latency (ms) | Memory (GB) |
|---|---:|---:|---:|---:|
| PointPillars Baseline | 4.932 | 26.67 | 37.501 | 0.575 |
| ECA-final | 4.932 | 28.76 | 34.767 | 0.502 |
| SALA(dimweight) | 4.932 | 26.67 | 37.501 | 0.575 |
| **ECA + SALA(dimweight)** | **4.932** | **28.76** | **34.767** | **0.502** |
| CenterPoint | 5.227 | 2.99 | 334.600 | 3.202 |

The final Pillar-SALNet setting improves detection performance without increasing the parameter size. Runtime statistics were measured under the same local experimental setting and may vary with hardware, software environment, CUDA version, runtime warm-up, and implementation details.

---

### 6.3 BEV Detection Comparison

Representative BEV detection results are shown below.

<p align="center">
  <img src="docs/bev_comparison.png" width="900"/>
</p>

Visualization legend:

- black boxes: ground truth;
- red dashed boxes: predictions;
- blue points: point cloud.

---

### 6.4 Point-Cloud Reconstruction Examples

The point-cloud inputs are reconstructed from depth images in the VCVW-3D virtual construction-scene dataset.

<p align="center">
  <img src="docs/pc_scene_1.png" width="48%"/>
  <img src="docs/pc_scene_2.png" width="48%"/>
</p>

These examples show that the reconstructed point clouds preserve the geometric structures of engineering vehicles and surrounding construction environments, providing effective input for downstream 3D object detection.

---

## 7. Dataset

### 7.1 Overview

This work is based on the **VCVW-3D** virtual construction-scene dataset.

Unlike standard LiDAR-based 3D detection benchmarks, VCVW-3D provides depth images, camera parameters, and JSON-format 3D annotations instead of ready-to-train point clouds. In this project, depth maps are back-projected into point clouds and then converted into an OpenPCDet-compatible 3D detection format.

---

### 7.2 Data Availability

The original VCVW-3D raw dataset is **not redistributed** in this repository due to dataset size and licensing restrictions.

Users should obtain the original VCVW-3D dataset from its official source and follow the corresponding license terms.

To facilitate reproducibility without redistributing the licensed raw dataset, this repository provides:

- dataset configuration files;
- class-name definitions and mapping;
- training/validation split files;
- processed annotation format example;
- data conversion script;
- OpenPCDet-style model configurations;
- modified model modules;
- training and evaluation commands;
- visualization examples and experimental materials.

The following lightweight reproducibility materials are provided under `data/`:

```text
data/
├── README.md
├── annotation_example.json
├── class_mapping.yaml
└── splits/
    ├── train.txt
    └── val.txt
```

The full raw VCVW-3D dataset, full processed point clouds, and full metadata files are not included in this repository.

---

### 7.3 Data Processing Pipeline

The dataset is constructed through the following steps:

1. depth-map back-projection to 3D point clouds;
2. invalid-depth filtering and point normalization;
3. JSON-format annotation parsing;
4. coordinate alignment and 3D box correction;
5. conversion to OpenPCDet-compatible data format;
6. generation of training and validation metadata files.

A data conversion script is provided at:

```text
tools/data_converter/export_fix_aligned_to_pcdet_infos.py
```

Key preprocessing settings:

```python
POINT_CLOUD_RANGE = [0, -70, -10, 70.4, 70, 10]
VOXEL_SIZE = [0.2, 0.2, 20]
MAX_POINTS_PER_FRAME = 200000
```

The vertical voxel size is set to compress the height dimension into pillars, which is consistent with the pillar-based representation used by PointPillars.

---

### 7.4 Expected Processed Data Structure

After preprocessing, the dataset should be organized in OpenPCDet-compatible format as follows:

```text
data/vcvw_extend5000_like1000_9cls/
├── ImageSets/
│   ├── train.txt
│   └── val.txt
├── infos_train.pkl
├── infos_val.pkl
└── training/
    └── velodyne/
        ├── 000001.bin
        ├── 000002.bin
        └── ...
```

Each processed sample contains:

- point-cloud file: `.bin`;
- point feature format: `(x, y, z, intensity)`;
- 3D annotations: `(x, y, z, l, w, h, yaw)`;
- metadata files: `infos_train.pkl` and `infos_val.pkl`.

The train/validation split is generated using a fixed random seed with an 8:2 ratio to support reproducibility.

---

### 7.5 Dataset Characteristics

VCVW-3DDet contains 9 engineering-vehicle categories with clear multi-scale and long-tail characteristics:

- Forklift
- DumpTruck
- Bulldozer
- ConcreteMixer
- Loader
- Excavator
- RoadRoller
- Crane
- Grader

The significant geometric differences among categories motivate the use of category-specific size priors and the proposed SALA strategy.

---

## 8. Installation

### 8.1 Install OpenPCDet

Please first install OpenPCDet:

```bash
git clone https://github.com/open-mmlab/OpenPCDet.git
cd OpenPCDet
python setup.py develop
```

Please prepare PyTorch, CUDA, spconv, and other dependencies according to the OpenPCDet environment requirements.

---

### 8.2 Prepare This Repository

Clone this repository:

```bash
git clone https://github.com/curry1-c/VCVW-3DDet-Pillar-SALNet.git
```

This repository contains modified modules and configuration files. Please merge them into the corresponding OpenPCDet directories.

A typical mapping is:

```text
VCVW-3DDet-Pillar-SALNet/cfgs/vcvw_models/
    -> OpenPCDet/tools/cfgs/vcvw_models/

VCVW-3DDet-Pillar-SALNet/tools/cfgs/vcvw_models/
    -> OpenPCDet/tools/cfgs/vcvw_models/

VCVW-3DDet-Pillar-SALNet/tools/cfgs/dataset_configs/
    -> OpenPCDet/tools/cfgs/dataset_configs/

VCVW-3DDet-Pillar-SALNet/models/backbones_2d/
    -> OpenPCDet/pcdet/models/backbones_2d/

VCVW-3DDet-Pillar-SALNet/models/backbones_3d/vfe/
    -> OpenPCDet/pcdet/models/backbones_3d/vfe/

VCVW-3DDet-Pillar-SALNet/models/dense_heads/
    -> OpenPCDet/pcdet/models/dense_heads/

VCVW-3DDet-Pillar-SALNet/tools/train.py
    -> OpenPCDet/tools/train.py

VCVW-3DDet-Pillar-SALNet/tools/test.py
    -> OpenPCDet/tools/test.py

VCVW-3DDet-Pillar-SALNet/tools/train_utils/
    -> OpenPCDet/tools/train_utils/

VCVW-3DDet-Pillar-SALNet/tools/data_converter/
    -> OpenPCDet/tools/data_converter/
```

Please check the file paths carefully before overwriting existing OpenPCDet files.

---

### 8.3 Python Environment

A typical conda environment can be created as follows:

```bash
conda create -n vcvw3ddet python=3.10 -y
conda activate vcvw3ddet
pip install -r requirements.txt
```

The experiments reported in the manuscript were conducted with Python 3.10.19, PyTorch 2.9.1+cu128, and CUDA 12.8. Other OpenPCDet-compatible environments may also work, but the reported runtime results may vary with hardware and software versions.

---

## 9. Training

### 9.1 PointPillars Baseline

```bash
python tools/train.py \
    --cfg_file tools/cfgs/vcvw_models/pointpillar_vcvw_5000.yaml \
    --batch_size 1 \
    --epochs 80 \
    --workers 0 \
    --fix_random_seed
```

---

### 9.2 Pillar-SALNet

```bash
python tools/train.py \
    --cfg_file tools/cfgs/vcvw_models/pointpillar_vcvw_5000_eca_sala_dimweight.yaml \
    --batch_size 1 \
    --epochs 80 \
    --workers 0 \
    --fix_random_seed
```

If your final local configuration uses another filename, please modify the `--cfg_file` path accordingly.

---

### 9.3 CenterPoint Reference

```bash
python tools/train.py \
    --cfg_file tools/cfgs/vcvw_models/centerpoint_vcvw_5000.yaml \
    --batch_size 1 \
    --epochs 80 \
    --workers 0 \
    --fix_random_seed
```

---

## 10. Evaluation

### 10.1 Evaluate PointPillars Baseline

```bash
python tools/test.py \
    --cfg_file tools/cfgs/vcvw_models/pointpillar_vcvw_5000.yaml \
    --ckpt path/to/checkpoint_epoch_80.pth
```

---

### 10.2 Evaluate Pillar-SALNet

```bash
python tools/test.py \
    --cfg_file tools/cfgs/vcvw_models/pointpillar_vcvw_5000_eca_sala_dimweight.yaml \
    --ckpt path/to/checkpoint_epoch_80.pth
```

---

### 10.3 Evaluate CenterPoint Reference

```bash
python tools/test.py \
    --cfg_file tools/cfgs/vcvw_models/centerpoint_vcvw_5000.yaml \
    --ckpt path/to/centerpoint_checkpoint_epoch_80.pth
```

Please replace the checkpoint path with your actual trained model checkpoint.

---

## 11. Repository Structure

```text
VCVW-3DDet-Pillar-SALNet
├── cfgs/
│   └── vcvw_models/                 # VCVW-3DDet model configurations
├── models/
│   ├── backbones_2d/                # ECA-enhanced BEV backbone modules
│   ├── backbones_3d/vfe/            # pillar feature encoding modules
│   └── dense_heads/
│       ├── anchor_head_single.py
│       ├── anchor_head_template.py
│       └── target_assigner/         # SALA target assignment modules
├── tools/
│   ├── _init_path.py
│   ├── train.py                     # training entry
│   ├── test.py                      # evaluation entry
│   ├── train_utils/                 # training utilities
│   ├── data_converter/              # VCVW-to-OpenPCDet conversion script
│   └── cfgs/
│       ├── vcvw_models/             # OpenPCDet-style model configs
│       └── dataset_configs/         # dataset configuration files
├── data/
│   ├── README.md
│   ├── annotation_example.json
│   ├── class_mapping.yaml
│   └── splits/                      # train / val split metadata
├── docs/                            # figures and visual materials
├── requirements.txt
├── LICENSE
└── README.md
```

---

## 12. Supplementary Center-Based Reference

This repository mainly focuses on the PointPillars/Pillar-SALNet framework, including category-specific size priors, SALA, and ECA.

The CenterPoint result is included as a supplementary cross-framework reference. Since CenterPoint follows a center-based detection paradigm, it is not used as the main method in this work and should not be interpreted as a direct replacement for Pillar-SALNet.

The main claims and reproducible experiments of this repository focus on the lightweight PointPillars/Pillar-SALNet line.

---

## 13. Reproducibility Notes

To improve reproducibility, this repository provides:

- model configuration files;
- dataset configuration files;
- class mapping and split metadata;
- annotation format example;
- data conversion script;
- modified model modules;
- training and evaluation commands;
- fixed random seed setting;
- preprocessing settings;
- quantitative results under the 5000-frame setting;
- visualization examples.

Please note that FPS, latency, and memory usage may vary across hardware platforms, CUDA versions, PyTorch versions, and runtime settings.

---

## 14. License and Data Usage

This repository is released under the Apache-2.0 License.

Please note:

- the original VCVW-3D dataset belongs to its original authors;
- this repository does not redistribute the original raw dataset;
- this repository does not redistribute the full processed point clouds or full metadata files;
- this repository provides configurations, data-processing scripts, split metadata, class mapping, annotation examples, modified modules, and visualization examples;
- users should obtain the original dataset from its official source and follow the corresponding license terms.

---

## 15. Code and Data Availability

The code, configurations, data-processing scripts, and documentation are publicly available at:

```text
https://github.com/curry1-c/VCVW-3DDet-Pillar-SALNet
```

An archived version of this repository with DOI is available at:

```text
Zenodo DOI: https://doi.org/10.5281/zenodo.19886473
```

Please note that the original VCVW-3D raw dataset is not redistributed in this repository due to size and licensing restrictions. Users should obtain the original dataset from its official source and follow the corresponding license terms.

To facilitate reproducibility without redistributing the licensed raw dataset, this repository releases processed-data preparation materials, including split files, class mapping, annotation format examples, dataset configuration files, and data conversion scripts.

---

## 16. Citation

If this repository is useful for your research, please cite the related manuscript and the archived Zenodo release:

```bibtex
@misc{zhang2026vcvw3ddet,
  title        = {Depth-Reconstructed Point Cloud Understanding and Lightweight 3D Detection for Scene-Level Engineering Vehicles},
  author       = {Ye Xu and Xinzhi Zhang},
  year         = {2026},
  note         = {Manuscript submitted to The Visual Computer. Code archived at Zenodo},
  doi          = {10.5281/zenodo.19886473}
}
```

Please replace the citation information with the final published metadata after acceptance.

---

## 17. Contact

For questions about this repository, please contact the authors through GitHub Issues or the corresponding author email provided in the manuscript.

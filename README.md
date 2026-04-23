# VCVW-3DDet-Pillar-SALNet

**3D Detection of Construction Vehicles from Depth-Reconstructed Point Clouds via Pillar-SALNet**

This repository contains the implementation and experimental materials for **VCVW-3DDet-Pillar-SALNet**, a 3D detection framework for construction vehicles based on depth-reconstructed point clouds.

---
## ⚠️ Important Note

This repository provides modified modules, configurations, and experimental settings based on OpenPCDet.

It is **not a standalone 3D detection framework**.

Please install and use it together with OpenPCDet:
https://github.com/open-mmlab/OpenPCDet

---

## 1. Overview

Construction vehicle detection differs substantially from standard autonomous driving benchmarks due to:

- large cross-category scale variation
- frequent occlusion and clutter in construction scenes
- sparse and anisotropic geometry in depth-reconstructed point clouds

To address these challenges, we construct the **VCVW-3DDet** dataset and propose **Pillar-SALNet**, an enhanced PointPillars-based detector that integrates:

- category-aware size priors
- Size-Aware Label Assignment (SALA)
- Efficient Channel Attention (ECA)

### Contributions

The main contributions of this work are summarized as follows:

- We construct an OpenPCDet-compatible 3D detection dataset for construction vehicles from depth-reconstructed point clouds based on the VCVW-3D virtual construction environment.
- We propose **SALA**, a Size-Aware Label Assignment strategy that dynamically adjusts IoU thresholds according to geometric priors and target-anchor size deviation.
- We introduce **ECA** into BEV feature fusion to improve feature representation with negligible computational overhead.
- We show that the proposed **Pillar-SALNet** achieves consistent performance gains while maintaining real-time efficiency.

---

## 2. Framework

### Pipeline

<p align="center">
  <img src="docs/pipeline.png" width="900"/>
</p>

The proposed framework is built upon **PointPillars** and improves the baseline from three aspects:

- geometry-aware anchor initialization using category size priors
- adaptive supervision through SALA
- lightweight BEV feature enhancement with ECA

---

## 3. Key Modules

### 3.1 Efficient Channel Attention (ECA)

<p align="center">
  <img src="docs/eca_module.png" width="700"/>
</p>

The ECA module is inserted at the multi-scale BEV fusion output stage to enhance channel-wise feature representation while maintaining efficiency.

**Main characteristics:**

- captures local cross-channel interaction
- avoids dimensionality reduction
- introduces negligible computational overhead

### 3.2 Size-Aware Label Assignment (SALA)

<p align="center">
  <img src="docs/sala_strategy.png" width="900"/>
</p>

SALA dynamically adjusts the assignment thresholds according to class-specific geometric priors and target size deviation.

**Main effects:**

- improves supervision quality for multi-scale targets
- alleviates mismatch caused by fixed IoU thresholds
- reduces false negatives for hard examples

---

## 4. Experimental Results

### 4.1 Main Quantitative Results

| Method | AP_R40 (%) | Params (M) | FPS |
|--------|------------|------------|-----|
| SECOND | 39.34 | - | - |
| Baseline | 66.70 | 4.932 | 26.67 |
| EMA | 67.33 | - | - |
| ECA-final | 67.88 | 4.932 | 28.76 |
| SALA (dimweight) | 67.34 | 4.932 | 26.67 |
| **ECA + SALA (dimweight)** | **69.29** | **4.932** | **28.76** |

**Observation.**  
The proposed joint model achieves the best overall performance in our experiments. Compared with the baseline, the final model improves AP_R40 by **+2.59** without increasing parameter size. The reported FPS values were measured under the same local experimental setting and may vary with hardware, software environment, runtime warm-up, and implementation details.

### 4.2 BEV Detection Comparison

Representative BEV comparison results between the baseline and the proposed method are shown below.

<p align="center">
  <img src="docs/bev_comparison.png" width="900"/>
</p>

**Visualization legend**

- Black boxes: Ground Truth
- Red dashed boxes: Predictions
- Blue points: Point Cloud

### 4.3 Point Cloud Reconstruction Examples

The point cloud input is reconstructed from depth images in the VCVW-3D virtual construction scene dataset.

<p align="center">
  <img src="docs/pc_scene_1.png" width="48%"/>
  <img src="docs/pc_scene_2.png" width="48%"/>
</p>

*Figure: Representative reconstructed point cloud samples in construction scenarios*

These examples show that the reconstructed point clouds preserve the geometric structure of construction vehicles and surrounding environments, providing reliable input for downstream 3D object detection.

---

## 5. Cross-Framework Extension

To evaluate whether the proposed design is specific to PointPillars or can generalize to other 3D detection paradigms, this project is being further extended to a **CenterPoint-based framework** for cross-framework comparison.

The current study focuses on the PointPillars / Pillar-SALNet line, where the proposed improvements are implemented and validated in a complete anchor-based BEV detection pipeline. In parallel, we are conducting additional experiments on a center-based detector to examine the transferability of the key ideas under a different detection formulation.

### Current Status

| Framework | Detector | AP_R40 (%) | Status |
|----------|----------|------------|--------|
| Anchor-based | PointPillars | 66.70 | Done |
| Anchor-based | Pillar-SALNet | 69.29 | Done |
| Center-based | CenterPoint | TBD | Running |

### Description

The goal of this extension is not only to compare detector performance, but also to investigate whether the design principles behind **size-aware supervision** and **lightweight BEV feature enhancement** remain effective when transferred from an anchor-based pipeline to a center-based one.

At the current stage, the CenterPoint branch should be regarded as an **ongoing extension** rather than a finalized benchmark result. Final quantitative results will be released after the corresponding experiments are fully completed and verified.

### Important Note

For **PointPillars**, the vertical voxel size is intentionally set to compress the height dimension into pillars, which is consistent with the pillar-based representation.  
For **CenterPoint**, experiments are conducted with an **independent configuration**, and its voxelization and detection settings should be interpreted separately rather than as a direct reuse of the PointPillars configuration.

### Planned Update

The CenterPoint extension will be updated with:

- finalized AP_R40 results
- corresponding configuration files
- reproducible training and evaluation commands
- cross-framework comparison analysis

This section is included to present the broader research direction of the project and will be updated as the CenterPoint branch becomes fully available.
---

## 6. Dataset

### Overview

This work is based on the **VCVW-3D** virtual construction scene dataset. Unlike standard LiDAR-based 3D detection benchmarks, VCVW-3D provides **depth images, camera parameters, and JSON-format 3D annotations** instead of ready-to-train point clouds. In this project, we reconstruct depth-based point clouds and build an OpenPCDet-ready dataset for construction vehicle 3D detection.

### What This Repository Provides

The dataset is not included in this repository due to size and licensing restrictions. Please prepare the dataset in OpenPCDet format according to the provided pipeline.

This repository provides:

- depth-to-point-cloud reconstruction pipeline
- annotation parsing and alignment
- OpenPCDet-style data organization
- training and evaluation configurations
- visualization examples and a small number of sample outputs

The original **VCVW-3D raw dataset is not redistributed** in this repository.

### Data Processing Pipeline

The dataset is constructed through the following steps:

1. Depth back-projection to 3D point clouds  
2. Invalid-depth filtering and normalization  
3. Annotation parsing from JSON files  
4. Coordinate alignment and box correction  
5. Export to OpenPCDet format  

**Key preprocessing settings**

```python
POINT_CLOUD_RANGE = [0, -70, -10, 70.4, 70, 10]
VOXEL_SIZE = [0.2, 0.2, 20]
max_points = 200000
```

### Data Format

Each processed sample includes:

- point cloud file: `.bin`
- 3D annotations: `(x, y, z, l, w, h, yaw)`
- metadata files: `infos_train.pkl`, `infos_val.pkl`

The train/val split is generated with a fixed random seed using an **8:2 ratio** for reproducibility.

### Dataset Characteristics

The final dataset contains **9 construction vehicle categories** with clear **multi-scale** and **long-tail** characteristics. These class-wise geometric differences motivate the use of **size priors** and **SALA** in our detection framework.

### Note

This repository focuses on the **data construction pipeline**, **processed format**, and **reproducible experiments**. It does not claim ownership of the original VCVW-3D data.

---
## 7. Dependency

This project is built upon OpenPCDet.

Please install OpenPCDet first:

```bash
git clone https://github.com/open-mmlab/OpenPCDet.git
cd OpenPCDet
python setup.py develop

```
Then use this repository for configs and modified modules.

PyTorch, CUDA, and spconv should be installed according to the OpenPCDet environment requirements.

---

## 8. Installation

### 8.1 Environment

```bash
conda create -n vcvw3ddet python=3.8 -y
conda activate vcvw3ddet
pip install -r requirements.txt
```

If you are using an OpenPCDet-based environment, please additionally prepare the corresponding CUDA, PyTorch, and spconv versions required by your local setup.


---

## 9. Training

### 9.1 PointPillars / Pillar-SALNet

    python tools/train.py \
        --cfg_file tools/cfgs/vcvw_models/pointpillar_vcvw_5000.yaml \
        --batch_size 1 \
        --epochs 80 \
        --workers 0 \
        --fix_random_seed

### 9.2 CenterPoint

    python tools/train.py \
        --cfg_file tools/cfgs/vcvw_models/centerpoint_vcvw_5000.yaml \
        --batch_size 1 \
        --epochs 80 \
        --workers 0 \
        --fix_random_seed

Replace the CenterPoint config path with your actual local filename if needed.

---

## 10. Evaluation

### 10.1 PointPillars / Pillar-SALNet

    python tools/test.py \
        --cfg_file tools/cfgs/vcvw_models/pointpillar_vcvw_5000.yaml \
        --ckpt path/to/your_checkpoint.pth

### 10.2 CenterPoint

    python tools/test.py \
        --cfg_file tools/cfgs/vcvw_models/centerpoint_vcvw_5000.yaml \
        --ckpt path/to/your_centerpoint_checkpoint.pth

---

## 11. Repository Structure

    VCVW-3DDet-Pillar-SALNet
    ├── cfgs/
    │   └── vcvw_models/                 # baseline / ECA / SALA / final configs
    ├── models/
    │   ├── backbones_2d/                # ECA-enhanced BEV backbone
    │   └── dense_heads/
    │       └── target_assigner/         # SALA target assignment
    ├── tools/                           # training and testing scripts
    ├── data/                            # dataset description and examples
    ├── docs/                            # figures and visual materials
    └── README.md

---

## 12. Notes

- Built upon PointPillars and OpenPCDet
- Designed for construction vehicle 3D detection
- Supports adaptive supervision through SALA
- Extended toward CenterPoint-based cross-framework experiments
---

## 13. License & Data Usage

- The original dataset belongs to the VCVW-3D authors
- This repository does **not** redistribute the original raw dataset
- Only configurations, processing pipeline, and visualization materials are provided

Please obtain the original dataset from its official source and follow the corresponding license terms.

---

## 14. Citation

If you find this work useful, please consider citing it as:

```bibtex
@misc{vcvw_sala,
  title={VCVW-3DDet: 3D Detection of Construction Vehicles from Depth-Reconstructed Point Clouds via Pillar-SALNet},
  author={Your Name and Coauthors},
  note={Under review}
}

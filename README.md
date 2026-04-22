# VCVW-3DDet-Pillar-SALNet

> **3D Detection of Construction Vehicles from Depth-Reconstructed Point Clouds via Pillar-SALNet**

This repository contains the implementation and experimental materials for **VCVW-3DDet-Pillar-SALNet**, a 3D detection framework for construction vehicles based on **depth-reconstructed point clouds**.

---

## 1. Overview

Construction vehicle detection differs substantially from standard autonomous driving benchmarks due to:

- large cross-category scale variation
- frequent occlusion and clutter in construction scenes
- sparse and anisotropic geometry in depth-reconstructed point clouds

To address these challenges, we construct the **VCVW-3DDet** dataset and propose **Pillar-SALNet**, an enhanced PointPillars-based detector that integrates:

- **category-aware size priors**
- **Size-Aware Label Assignment (SALA)**
- **Efficient Channel Attention (ECA)**

---

## 2. Framework

![Pipeline](docs/pipeline.png)

The proposed framework is built upon **PointPillars** and improves the baseline from three aspects:

- **Geometry-aware anchor initialization** using category size priors
- **Adaptive supervision** through SALA
- **Lightweight BEV feature enhancement** with ECA

---

## 3. Key Modules

### 3.1 Efficient Channel Attention (ECA)

![ECA](docs/eca_module.png)

The ECA module is inserted at the multi-scale BEV fusion output stage to enhance channel-wise feature representation while maintaining efficiency.

Main characteristics:

- captures local cross-channel interaction
- avoids dimensionality reduction
- introduces negligible computational overhead

---

### 3.2 Size-Aware Label Assignment (SALA)

![SALA](docs/sala_strategy.png)

SALA dynamically adjusts the assignment threshold according to class-specific geometric priors and target size deviation.

Main effects:

- improves supervision quality for multi-scale targets
- alleviates mismatch caused by fixed IoU thresholds
- reduces false negatives for hard examples

---

## 4. Experimental Results

### 4.1 Main Quantitative Results

| Method               | AP_R40 (%) | Params (M) | FPS   |
|----------------------|------------|------------|-------|
| SECOND               | 39.34      | -          | -     |
| Baseline             | 66.70      | 4.932      | 26.67 |
| EMA                  | 67.33      | -          | -     |
| ECA-final            | 67.88      | 4.932      | 28.76 |
| SALA(dimweight)      | 67.34      | 4.932      | 26.67 |
| ECA + SALA(dimweight)| **69.29**  | 4.932      | 28.76 |

**Observation.**  
The proposed joint model achieves the best overall performance while preserving real-time efficiency.

---

### 4.2 BEV Detection Comparison

Representative BEV comparison results between the baseline and the proposed method are shown below.

![BEV Comparison](docs/bev_comparison.png)

**Visualization legend**

- **Black boxes**: Ground Truth
- **Red dashed boxes**: Predictions
- **Blue points**: Point Cloud

> **Note:** Replace the CenterPoint config path with your local file path if necessary.

---

### 4.3 Point Cloud Reconstruction Examples

The point cloud input is reconstructed from depth images in the VCVW-3D virtual construction scene dataset.

<p align="center">
  <img src="docs/pc_scene_2.png" width="45%">
  <img src="docs/pc_scene_3.png" width="45%">
</p>

<p align="center">
  <em>Figure: Representative reconstructed point cloud samples in construction scenarios</em>
</p>
These examples show that the reconstructed point clouds preserve the geometric structure of construction vehicles and surrounding environments, providing reliable input for downstream 3D object detection.

---

### 5. Cross-Framework Extension

In addition to the PointPillars/Pillar-SALNet framework, this project is being extended toward CenterPoint-based experiments for cross-framework comparison.

#### Current Status

| Framework     | Detector      | AP_R40 (%) | Status   |
|--------------|--------------|------------|----------|
| Anchor-based | PointPillars | 66.70      | Done     |
| Anchor-based | Pillar-SALNet| 69.29      | Done     |
| Center-based | CenterPoint  | TBD        | Running  |

> **Note:** CenterPoint experiments are currently in progress. Final results will be updated upon completion.
---

## 6. Dataset

The dataset is constructed based on the **VCVW-3D virtual construction scene dataset**.

This repository provides:

- processed data format description
- visualization examples
- experimental configurations
- training and evaluation pipeline

This repository does **not** redistribute the original raw dataset.

---

## 7. Installation

### 7.1 Environment

```bash
conda create -n vcvw3ddet python=3.8 -y
conda activate vcvw3ddet
pip install -r requirements.txt
```
If you are using an OpenPCDet-based environment, please additionally prepare the corresponding CUDA, PyTorch, and spconv versions required by your local setup.

---

## 8. Training

### 8.1 PointPillars / Pillar-SALNet

```bash
python tools/train.py \
    --cfg_file tools/cfgs/vcvw_models/pointpillar_vcvw_5000.yaml \
    --batch_size 1 \
    --epochs 80 \
    --workers 0 \
    --fix_random_seed
```
### 8.2  CenterPoint 

```bash
python tools/train.py \
    --cfg_file tools/cfgs/vcvw_models/centerpoint_vcvw_5000.yaml \
    --batch_size 1 \
    --epochs 80 \
    --workers 0 \
    --fix_random_seed
```
Replace the CenterPoint config path with your actual local filename if needed.
---

 ## 9. Evaluation
 
### 9.1  PointPillars / Pillar-SALNet

```bash
python tools/test.py \
    --cfg_file tools/cfgs/vcvw_models/pointpillar_vcvw_5000.yaml \
    --ckpt path/to/your_checkpoint.pth
```
### 9.2 CenterPoint

```bash
python tools/test.py \
    --cfg_file tools/cfgs/vcvw_models/centerpoint_vcvw_5000.yaml \
    --ckpt path/to/your_centerpoint_checkpoint.pth
```
---
## 10. Repository Structure

```text
VCVW-3DDet-Pillar-SALNet
├── cfgs        # configuration files
├── docs        # figures and visual materials
├── tools       # training and testing scripts
├── data        # dataset description and examples
├── README.md
```
---

## 11. Notes

- Built upon the PointPillars framework  
- Designed for construction vehicle 3D detection  
- Supports multi-scale feature fusion  
- Introduces adaptive supervision strategy  
- Extended toward CenterPoint-based cross-framework experiments  

---

## 12. License & Data Usage

- Dataset belongs to the original VCVW-3D authors
- This repository does NOT redistribute the original raw dataset
- Only configurations, processing pipeline, and visualization materials are provided

Please obtain the original dataset from its official source and follow the corresponding license terms.
---

## 13. Citation

```bash
@article{vcvw_sala2026,
  title={VCVW-3DDet: 3D Detection of Construction Vehicles from Depth-Reconstructed Point Clouds via Pillar-SALNet},
  author={curry1-c},
  journal={The Visual Computer},
  year={2026}
}
```
If this work is helpful for your research, please consider citing it.

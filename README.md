# VCVW-3DDet-Pillar-SALNet

🚧 **3D Detection of Construction Vehicles from Depth-Reconstructed Point Clouds via Pillar-SALNet**

This repository contains the implementation and experimental materials of our work on **3D detection of construction vehicles from depth-reconstructed point clouds**.  
The project is built around the **VCVW-3DDet** dataset and the proposed **Pillar-SALNet** framework, which improves PointPillars by combining:

- **Category-aware size priors**
- **Size-Aware Label Assignment (SALA)**
- **Efficient Channel Attention (ECA)**

---

## 📖 Introduction

3D object detection has achieved significant success in autonomous driving, but research on **construction vehicle detection** remains limited. Compared with road traffic objects, construction vehicles exhibit:

- larger cross-category scale variation
- more complex occlusion and viewpoint changes
- different point cloud characteristics when reconstructed from depth maps rather than LiDAR

To address these issues, we construct **VCVW-3DDet**, a 3D detection dataset for construction vehicles generated from **depth-reconstructed point clouds**, and propose **Pillar-SALNet**, an enhanced PointPillars-based detector.

According to the uploaded paper, the method improves detection from three aspects:  
(1) category size prior for anchor initialization,  
(2) adaptive supervision with SALA, and  
(3) lightweight BEV feature enhancement with ECA. :contentReference[oaicite:1]{index=1}

---

## ✨ Highlights

- **VCVW-3DDet dataset** for engineering vehicle 3D detection
- **Depth-to-point-cloud reconstruction pipeline**
- **Pillar-SALNet** based on PointPillars
- **SALA** for size-aware adaptive label assignment
- **ECA** for lightweight multi-scale BEV feature enhancement
- **Cross-framework extension** toward CenterPoint-based experiments

---

## 🔥 Method Overview

![Pipeline](docs/pipeline.png)

The proposed framework is built upon **PointPillars** and introduces three key improvements:

- 📐 **Geometry-aware anchor initialization** with category size priors  
- 🎯 **Adaptive label assignment** through SALA  
- ⚡ **Lightweight feature enhancement** using ECA at the BEV fusion output  

According to the paper, the full framework keeps the efficient pillar-based detection paradigm while improving geometric matching, supervision quality, and final feature discriminability. :contentReference[oaicite:2]{index=2}

---

## ⚙️ ECA Module

![ECA](docs/eca_module.png)

The Efficient Channel Attention (ECA) module is inserted at the **multi-scale BEV fusion output stage**.  
It enhances feature representation by:

- capturing local cross-channel interaction
- avoiding dimensionality reduction
- maintaining computational efficiency

The paper reports that among several lightweight attention insertion strategies, **ECA-final** achieves the best result on the 5000-frame setting. :contentReference[oaicite:3]{index=3}

---

## 📐 SALA Strategy

![SALA](docs/sala_strategy.png)

The proposed **Size-Aware Label Assignment (SALA)** dynamically adjusts positive/negative assignment thresholds according to the deviation between an object and its class-specific size prior.

Main effects:

- improves supervision quality for multi-scale targets
- handles large intra-class and inter-class size variation
- reduces biased matching caused by fixed IoU thresholds

The paper shows that **SALA(dimweight)** performs best among several supervision variants. :contentReference[oaicite:4]{index=4}

---

## 📊 Dataset: VCVW-3DDet

The dataset is constructed from the **VCVW-3D virtual construction scene dataset** by:

1. depth map back-projection
2. invalid depth filtering
3. point cloud standardization
4. box alignment correction
5. OpenPCDet-format export

The uploaded paper states that the final dataset contains **9 categories** of construction vehicles, including:

- Forklift
- DumpTruck
- Bulldozer
- ConcreteMixer
- Loader
- Excavator
- RoadRoller
- Crane
- Grader

It also reports strong cross-category scale differences, which directly motivate the design of size priors and SALA. :contentReference[oaicite:5]{index=5}

---

## 🌐 Point Cloud Reconstruction Examples

The point cloud data is reconstructed from depth images in the VCVW-3D virtual construction scene dataset.

### Example 1
![Point Cloud Example 1](docs/pc_scene_1.png)

### Example 2
![Point Cloud Example 2](docs/pc_scene_2.png)

These examples illustrate that the reconstructed point clouds preserve the geometric structure of vehicles and surrounding construction environments, providing usable input for downstream 3D detection. This is consistent with the dataset construction and visualization discussion in the uploaded paper. :contentReference[oaicite:6]{index=6}

---

## 📊 Experimental Results

### Main results on VCVW-3DDet (5000-frame setting)

| Method               | AP_R40 (%) | Params (M) | FPS   |
|----------------------|------------|------------|-------|
| SECOND               | 39.34      | -          | -     |
| Baseline             | 66.70      | 4.932      | 26.67 |
| EMA                  | 67.33      | -          | -     |
| ECA-final            | 67.88      | 4.932      | 28.76 |
| SALA(dimweight)      | 67.34      | 4.932      | 26.67 |
| ECA + SALA(dimweight)| **69.29**  | 4.932      | 28.76 |

> **Note:** The proposed joint model achieves the best overall performance while maintaining real-time efficiency.

These results are directly reported in the uploaded paper, where the joint strategy improves over the baseline by **2.59 percentage points** on the 5000-frame setting. :contentReference[oaicite:7]{index=7}

---

## 🚗 BEV Detection Comparison

Comparison between the baseline detector and our proposed method:

![BEV Comparison](docs/bev_comparison.png)

**Visualization legend**

- **Black boxes**: Ground Truth  
- **Red dashed boxes**: Predictions  
- **Blue points**: Point Cloud  

The uploaded paper reports that compared with the baseline, the joint model improves:

- box completeness
- localization stability
- recall in difficult multi-object scenes

while distant, heavily occluded, and extremely sparse targets remain challenging. :contentReference[oaicite:8]{index=8}

---

## 🧪 Ablation Studies

### Feature enhancement path

| Method      | AP_R40 (%) |
|-------------|------------|
| EMA         | 67.33      |
| ECA-final   | **67.88**  |
| ECA-middle  | 66.41      |
| ECA-multi   | 65.49      |
| ECA-k5      | 67.20      |

### Supervision optimization path

| Method                 | AP_R40 (%) |
|------------------------|------------|
| Original SALA          | 63.46      |
| Threshold-corrected SALA | 64.40    |
| SALA(dimweight)        | **67.34**  |
| SALA(dimweight-bi)     | 65.68      |

The paper concludes that:
- **ECA-final** is the best attention insertion strategy
- **SALA(dimweight)** is the most effective supervision optimization variant :contentReference[oaicite:9]{index=9}

---

## 🔁 Multi-seed Stability

The uploaded paper further reports stable performance across different random seeds:

| Method                 | Seed   | AP_R40 (%) |
|------------------------|--------|------------|
| Baseline               | seed42 | 66.20      |
| Baseline               | seed66 | 66.70      |
| ECA-final              | seed42 | 67.20      |
| ECA-final              | seed66 | 66.98      |
| SALA(dimweight)        | seed42 | 66.87      |
| SALA(dimweight)        | seed66 | 67.25      |
| ECA+SALA(dimweight)    | seed42 | 68.74      |
| ECA+SALA(dimweight)    | seed66 | **69.29**  |

This shows that the joint model not only performs best, but also remains stable under multiple seeds. :contentReference[oaicite:10]{index=10}

---

## 🏗️ Cross-Framework Extension: CenterPoint

Besides the PointPillars/Pillar-SALNet line, this repository is also being extended toward **CenterPoint-based experiments**.

CenterPoint is mentioned in the uploaded paper as a representative center-based 3D detection framework with strong localization capability and good efficiency. :contentReference[oaicite:11]{index=11}

### Ongoing CenterPoint experiments

The current plan is to evaluate whether the improvements validated on Pillar-SALNet can also provide benefits under a **center-based detection paradigm**, including:

- dataset compatibility on VCVW-3DDet
- cross-framework comparison between anchor-based and center-based detectors
- analysis of whether size-aware supervision remains beneficial for CenterPoint
- comparison of detection behavior on depth-reconstructed point clouds

### Planned comparison table

| Framework     | Detector      | AP_R40 (%) | Status |
|---------------|---------------|------------|--------|
| Anchor-based  | PointPillars  | 66.70      | Done   |
| Anchor-based  | Pillar-SALNet | 69.29      | Done   |
| Center-based  | CenterPoint   | TBD        | Running |

> **Note:** CenterPoint results are being tested and will be updated after the current experiments finish.  
> This avoids reporting unsupported numbers before verification.

---

## 📦 Installation

### Environment

```bash
conda create -n vcvw3ddet python=3.8 -y
conda activate vcvw3ddet
pip install -r requirements.txt

If you are using OpenPCDet-based code, please additionally prepare the corresponding CUDA, PyTorch, and spconv versions required by your local environment.

## 🚀 Training

### Train
PointPillars / Pillar-SALNet
```bash
python tools/train.py \
    --cfg_file tools/cfgs/vcvw_models/pointpillar_vcvw_5000.yaml \
    --batch_size 1 \
    --epochs 80 \
    --workers 0 \
    --fix_random_seed
```
CenterPoint (ongoing)
```bash
python tools/train.py \
    --cfg_file tools/cfgs/vcvw_models/centerpoint_vcvw_5000.yaml \
    --batch_size 1 \
    --epochs 80 \
    --workers 0 \
    --fix_random_seed
```
Replace the cfg path above with your actual CenterPoint config file if the filename differs in your local project.

 ## 🚀 Evaluation
 PointPillars / Pillar-SALNet
```bash
python tools/test.py \
    --cfg_file tools/cfgs/vcvw_models/pointpillar_vcvw_5000.yaml \
    --ckpt path/to/your_checkpoint.pth
```
CenterPoint
```bash
python tools/test.py \
    --cfg_file tools/cfgs/vcvw_models/centerpoint_vcvw_5000.yaml \
    --ckpt path/to/your_centerpoint_checkpoint.pth
```

## 📂 Repository Structure

```text
VCVW-3DDet-Pillar-SALNet
├── cfgs        # configuration files
├── docs        # figures and visual materials
├── tools       # training and testing scripts
├── data        # dataset description and examples
├── README.md
```

## 📌 Notes

·Built upon the PointPillars framework
·Designed for construction vehicle 3D detection
·Supports multi-scale feature fusion
·Introduces adaptive supervision strategy
·Extended toward CenterPoint-based cross-framework experiments

## ⚠️ License & Data Usage

·Dataset belongs to the original VCVW-3D authors
·This repository does NOT redistribute the original raw dataset
·Only configurations, processing pipeline, and visualization materials are provided

According to the uploaded paper, the repository provides processed-format support and experimental materials rather than redistributing the original source dataset.

## 📎 Citation
```bash
@article{vcvw_sala2026,
  title={VCVW-3DDet: 3D Detection of Construction Vehicles from Depth-Reconstructed Point Clouds via Pillar-SALNet},
  author={curry1-c},
  journal={The Visual Computer},
  year={2026}
}
```
If this work is helpful for your research, please consider citing it.

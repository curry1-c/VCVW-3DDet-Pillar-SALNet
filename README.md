# VCVW-3DDet-Pillar-SALNet

This repository provides the official implementation of the paper:

**"VCVW-3DDet: 3D Detection of Construction Vehicles from Depth-Reconstructed Point Clouds via Pillar-SALNet"**

It includes the model design, experimental configurations, and visualization results.

---

## 🔥 Method Overview

This work proposes a Pillar-SALNet framework built upon the PointPillars architecture.  
It integrates category size priors, size-aware label assignment, and lightweight channel attention to improve geometric alignment and feature representation for construction vehicle detection.

![Pipeline](docs/pipeline.png)

---

## ⚙️ ECA Module

The Efficient Channel Attention (ECA) module is introduced at the multi-scale BEV feature fusion stage.  
It captures local cross-channel interactions via a lightweight 1D convolution without dimensionality reduction, improving feature discriminability with minimal computational cost.

![ECA](docs/eca_module.png)

---

## 📐 SALA Strategy

The Size-Aware Label Assignment (SALA) strategy dynamically adjusts the matching threshold according to category-specific geometric priors.  
This mechanism enhances supervision quality, especially for objects with large intra-class scale variations.

![SALA](docs/sala_strategy.png)

---

## 📊 Dataset

The dataset used in this work is constructed based on the **VCVW-3D virtual construction scene dataset**.

- We do **NOT** distribute the original dataset in this repository.
- This repository only provides:
  - data processing description
  - dataset format
  - example usage

Users should obtain the dataset from its official source and follow its license.

---

## ⚠️ License & Data Usage

The dataset used in this work is derived from the VCVW-3D dataset.  
All rights belong to the original authors.

This repository only provides processed data format and experimental configuration.  
**It does NOT redistribute the original dataset.**

---

## 🚀 Training

Example training command:

```bash
python tools/train.py \
    --cfg_file tools/cfgs/vcvw_models/pointpillar_vcvw_5000.yaml \
    --batch_size 1 \
    --epochs 80 \
    --workers 0 \
    --fix_random_seed

Example evaluation command:

```bash
python tools/test.py \
    --cfg_file tools/cfgs/vcvw_models/pointpillar_vcvw_5000.yaml \
    --ckpt path/to/your_checkpoint.pth

# VCVW-3DDet-Pillar-SALNet

This repository provides the official implementation of the paper:

**"VCVW-3DDet: 3D Detection of Construction Vehicles from Depth-Reconstructed Point Clouds via Pillar-SALNet"**

It includes the model design, experimental configurations, and visualization results.

---

## 🔥 Method Overview

This work proposes a Pillar-SALNet framework built upon the PointPillars architecture. It integrates category size priors, size-aware label assignment, and lightweight channel attention to improve geometric alignment and feature representation for construction vehicle detection.

![Pipeline](docs/pipeline.png)

---

## ECA Module

The Efficient Channel Attention (ECA) module is introduced at the multi-scale BEV feature fusion stage. It captures local cross-channel interactions via a lightweight 1D convolution without dimensionality reduction, improving feature discriminability with minimal computational cost.

![ECA](docs/eca_module.png)

---

## SALA Strategy

The Size-Aware Label Assignment (SALA) strategy dynamically adjusts the matching threshold according to category-specific geometric priors. This mechanism enhances supervision quality, especially for objects with large intra-class scale variations.

![SALA](docs/sala_strategy.png)

---

## Overview

This project focuses on 3D object detection of construction vehicles using depth-reconstructed point clouds.

The method is built upon the PointPillars framework and introduces three main improvements:

- Category size prior for anchor initialization
- Size-aware adaptive label assignment (SALA)
- Lightweight ECA-based feature enhancement

---

## Dataset

The dataset used in this work is constructed based on the VCVW-3D virtual construction scene dataset.

We do **not** distribute the original VCVW-3D data in this repository.

This repository only provides:

- data processing description
- dataset format
- example usage

Users should obtain the original dataset from its official source and follow its license.

The dataset used in this work is derived from the VCVW-3D dataset. All rights belong to the original authors. This repository only provides processed data format and does not redistribute the original data.

---

## Repository Structure

```text
VCVW-3DDet-Pillar-SALNet
├── cfgs        # configuration files
├── docs        # figures and documentation
├── tools       # training and testing tools
├── data        # dataset description and examples

================================================================================
FILE 1: .gitignore
================================================================================
# Byte-compiled / optimized / DLL files
__pycache__/
*.py[cod]
*$py.class

# C extensions
*.so

# Distribution / packaging
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual environments
venv/
env/
ENV/
.env

# Jupyter Notebook checkpoints
.ipynb_checkpoints

# Datasets (Strictly exclude large CSV files to avoid exceeding repository limits)
Dataset/
Dataset/*
Dataset/Edge-IIoTset dataset/
Dataset/Multi/
Dataset/Client/
Dataset/ValidationSet/
Dataset/TestSet/
*.csv

# Model weights and serialized tensors (Exclude files > 25MB)
*.h5
*.weights.h5
*.sparse.npz
*.npz
*.joblib

# Operational synchronization signals and temporary logs
edge_server_signals/
edge_server_signals_multi/
*.done
DL_Attack_Logs/
DL_Local_Performance_Logs_Multi/

# IDE and OS metadata
.vscode/
.idea/
.DS_Store
Thumbs.db


================================================================================
FILE 2: requirements.txt
================================================================================
tensorflow>=2.15.0
keras-tcn>=3.5.0
lightgbm>=4.0.0
scikit-learn>=1.3.0
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0
joblib>=1.3.0
tqdm>=4.65.0
psutil>=5.9.0


================================================================================
FILE 3: README.md
================================================================================
# Securing IIoT Networks: A Dual-Layer Defence Approach in Hierarchical Federated Learning

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://tensorflow.org)
[![LightGBM](https://img.shields.io/badge/LightGBM-Ensemble-green.svg)](https://lightgbm.readthedocs.io/)

This repository contains the official implementation of the Final Year Project (FYP2) at **Universiti Tunku Abdul Rahman (UTAR)**:
> **"Federated Learning for IIoT Security: Decentralized Anomaly Detection"**  
> *Author:* Hong Yee Hang (22ACB03200)  
> *Supervisor:* Dr. Nadeem Muhammad Waqas  
> *Faculty:* Faculty of Information and Communication Technology (FICT, Kampar Campus)  
> *Programme:* Bachelor of Information Technology (Honours) Communications and Networking  

---

## 📌 Project Overview
Industrial Internet of Things (IIoT) environments produce large volumes of sensitive network telemetry. Transmitting this raw data to a central cloud server violates privacy regulations and creates network bottlenecks. Federated Learning (FL) provides a solution by training models locally on edge devices, sharing only the model parameters. 

However, conventional federated learning frameworks rely on the assumption that all nodes are honest. Under adversarial settings, standard aggregation algorithms such as Federated Averaging (FedAvg) collapse completely when exposed to:
1. **Data Poisoning (Targeted Label Flipping):** Malicious clients invert or alter training labels to confuse decision boundaries.
2. **Model Poisoning (Gradient Scaling):** Malicious clients amplify gradient updates to overpower honest contributions during aggregation.

To mitigate these threats, this project implements a **three-tier Hierarchical Federated Learning (HFL)** framework featuring a **Dual-Layer Defence** mechanism combined with communication optimisation.

---

## 🛠️ System Architecture

The architecture is divided into three distinct operational tiers:

```text
[ Tier 1: Client Layer ]
    │  - Local training on Non-IID "Enriched Specialist" partitions
    │  - 1D-TCN + GRU (Sequential) / Tabular Deep Neural Network (T-DNN)
    │  - Top-k Sparsification (transmitting only the top 45% of updates)
    ▼
[ Tier 2: Edge Gatekeeper Layer ]
    │  - Stage 1: Hard-Threshold Filtering (Accuracy < 0.50 & L2 Norm > 100.0)
    │  - Stage 2: FLTrust Directional Verification (ReLU-clipped Cosine Similarity)
    │  - Magnitude normalisation and weighted zone aggregation
    ▼
[ Tier 3: Global Aggregation Layer ]
    │  - Replaces traditional parameter averaging (FedAvg)
    │  - Base model prediction extraction on a dedicated meta-dataset
    │  - Stacking Ensemble meta-learner (LightGBM with balanced class weights)
    ▼
[ Unified Global Intrusion Detection Model ]

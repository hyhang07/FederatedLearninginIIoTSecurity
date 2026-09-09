# Securing IIoT Networks: A Dual-Layer Defence Approach in Hierarchical Federated Learning

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python: 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![Framework: TensorFlow](https://img.shields.io/badge/Framework-TensorFlow-orange.svg)](https://tensorflow.org)
[![Ensemble: LightGBM](https://img.shields.io/badge/Ensemble-LightGBM-green.svg)](https://lightgbm.readthedocs.io/)

Official repository for the Final Year Project (FYP2) at **Universiti Tunku Abdul Rahman (UTAR)**:
* **Project Title:** Federated Learning for IIoT Security: Decentralized Anomaly Detection
* **Author:** Hong Yee Hang
* **Supervisor:** Dr. Nadeem Muhammad Waqas
* **Faculty:** Faculty of Information and Communication Technology (FICT, Kampar Campus)
* **Programme:** Bachelor of Information Technology (Honours) Communications and Networking

---

## 1. Project Overview

Industrial Internet of Things (IIoT) systems generate continuous, mission-critical network telemetry. Centralising this data on cloud servers creates severe privacy risks and bandwidth bottlenecks. Federated Learning (FL) addresses this constraint by training models locally on edge nodes.

However, standard FL assumes honest participation. When exposed to adversarial threats, conventional Federated Averaging (FedAvg) collapses:
* **Data Poisoning (Label Flipping):** Inverting training labels to mislead classification boundaries.
* **Model Poisoning (Gradient Scaling):** Artificially magnifying update vectors to hijack global aggregation.

This project implements a **three-tier Hierarchical Federated Learning (HFL)** architecture featuring a **Dual-Layer Defence** strategy combined with communication compression.

---

## 2. System Architecture

The architecture consists of three operational tiers:

1. **Tier 1 — Client Layer:**
   * Local models: **1D-TCN + GRU** (for sequential flow telemetry) and **Tabular Deep Neural Network (T-DNN)** (for static flow statistics).
   * Data distribution follows an **"Enriched Specialist"** Non-IID strategy (65% Normal, 30% Target Attack, 5% Background Mix).
   * **Top-k Sparsification:** Transmits only the top 45% largest absolute gradient updates to conserve bandwidth.

2. **Tier 2 — Edge Gatekeeper Layer:**
   * **Hard Thresholds:** Instantly drops updates with validation accuracy < 0.50 (stopping data poisoners) or L2 norm > 100.0 (stopping model scaling).
   * **FLTrust Directional Check:** Computes Cosine Similarity against a clean Root-of-Trust server gradient. Negative scores are zeroed via a ReLU filter.
   * **Magnitude Normalisation:** Rescales accepted updates to match the server baseline before weighted aggregation.

3. **Tier 3 — Global Aggregation Layer:**
   * Replaces standard FedAvg with a **Stacking Ensemble**.
   * Reconstructed base models generate prediction probabilities over an independent meta-dataset.
   * A **LightGBM meta-learner** (`class_weight='balanced'`) learns which regional model to trust for specific attack categories, preventing feature dilution under Non-IID conditions.

---

## 3. Experimental Results

The framework was evaluated on two public cybersecurity benchmarks under mixed adversarial attacks (80% label flipping ratio and scaling factor $\lambda = 100.0$):

| Evaluation Metric | Baseline FedAvg (No Defence) | Proposed Framework (Dual-Layer HFL) |
| :--- | :---: | :---: |
| **Edge-IIoTset Accuracy** | 4.34% *(Collapsed)* | **90.80%** |
| **TON_IoT Accuracy** | 5.56% *(Collapsed)* | **82.87%** |
| **Normal Traffic Precision** | 0.00 (Edge) / 0.12 (TON) | **1.00 (Edge) / 0.96 (TON)** |
| **Malicious Update Rejection** | 0% *(Accepted All)* | **100% (Zero False Positives)** |
| **Communication Reduction** | 0% (Dense Update) | **77.5% (Edge) / 50.2% (TON)** |

---

## 4. Repository Structure

```text
├── Edge_IIoTset_Implementation/       # 1D-TCN + GRU pipeline for Edge-IIoTset
│   ├── Multi_CreateInitialModel.py    # Global baseline weight initialisation
│   ├── Multi_Client.py                # Honest client with Top-k sparsification
│   ├── Multi_Client_DataPoison.py     # Malicious client (Label Flipping)
│   ├── Multi_Client_ModelPoison.py    # Malicious client (Gradient Scaling)
│   ├── EdgeServer_Multi.py            # Gatekeeper (Hard thresholds + FLTrust)
│   ├── EdgeServer_NoDefense.py        # Baseline FedAvg edge aggregator
│   ├── GlobalServer_Distiller_Multi.py# Global server with LightGBM Stacking
│   ├── GlobalServer_NoDefense.py      # Baseline FedAvg global aggregator
│   ├── Multi_GlobalServer.py          # Comparative DBSCAN clustering aggregation
│   ├── Multi_DataPreparation.py       # Enriched Specialist Non-IID partitioner
│   ├── Multi_DataPreparation_Test.py  # Global test set generator
│   ├── Multi_CreateValidationSet_PerZone.py # Root-of-Trust validation builder
│   ├── Multi_DataUtils.py             # Label mapping and conversion utility
│   ├── Multi_DataUtils_Client.py      # 10s rolling-window feature engineering
│   ├── Multi_TestModel.py             # Standalone offline model evaluation
│   └── Multi_GenerateGraph.py         # Figure and chart generator
│
├── TON_IoT_Implementation/            # T-DNN pipeline for TON_IoT
│   ├── Create_Initial_Model.py        # 47-feature T-DNN model builder
│   ├── Client.py                      # Honest client with smoothed class weighting
│   ├── Client_DataPoison.py           # Malicious client (Cyclic shift label attack)
│   ├── Client_ModelPoison.py          # Malicious client (Gradient scaling attack)
│   ├── EdgeServer.py                  # Gatekeeper with zone quarantine logic
│   ├── EdgeServer_NoDefense.py        # Baseline FedAvg edge aggregator
│   ├── GlobalServer.py                # Global server with memory-optimised Stacking
│   ├── GlobalServer_NoDefense.py      # Baseline FedAvg global aggregator
│   ├── DatasetPreparation.py          # TON_IoT 2.5:1 data distribution script
│   ├── DataUtils.py                   # 9-class label mapping utility
│   ├── DataUtils_Client.py            # High-cardinality filtering and scaling
│   ├── Generate_Schema.py             # Master JSON schema generator
│   ├── GenerateGraph.py               # Evaluation chart generator
│   └── checkDataset.py                # Class distribution verification utility
│
├── .gitignore                         # Excludes raw datasets and large models
├── requirements.txt                   # Dependency manifest
└── README.md                          # Repository documentation
```

---

## 5. Getting Started

### Prerequisites
* Python 3.12+
* Linux / WSL2 on Windows 11
* Dedicated NVIDIA GPU (recommended for client training)

### Setup
1. Clone this repository:
```bash
git clone https://github.com/hyhang07/FederatedLearningInIIoTSecurity.git
cd FederatedLearningInIIoTSecurity
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

---

## 6. Execution Workflow

### Step 1: Data Preparation
Download the datasets from their official sources:
* **Edge-IIoTset:** [Kaggle Dataset](https://www.kaggle.com/datasets/mohamedamineferrag/edgeiiotset-cyber-security-dataset-of-iot-iiot)
* **TON_IoT:** [UNSW Canberra Research Repository](https://research.unsw.edu.au/projects/toniot-datasets)

**To prepare Edge-IIoTset:**
```bash
cd Edge_IIoTset_Implementation
python Multi_DataPreparation.py
python Multi_DataPreparation_Test.py
python Multi_CreateValidationSet_PerZone.py
python Multi_CreateInitialModel.py
```

**To prepare TON_IoT:**
```bash
cd TON_IoT_Implementation
python DatasetPreparation.py
python Generate_Schema.py
python Create_Initial_Model.py
```

### Step 2: Simulation Pipeline
Run the client training scripts to generate sparse `.npz` updates. Launch the corresponding edge server gatekeepers to validate updates and compute zone models, followed by the global server to execute the LightGBM stacking ensemble.

---

## 7. Citation

If you reference this work or codebase, please cite:

```bibtex
@thesis{Hong2026Federated,
  author       = {Hong Yee Hang},
  title        = {Federated Learning for IIoT Security: Decentralized Anomaly Detection},
  school       = {Universiti Tunku Abdul Rahman (UTAR)},
  year         = {2026},
  month        = {February},
  type         = {Bachelor's Final Year Project Report}
}
```

---

## 8. License

Distributed under the MIT License. See [LICENSE](LICENSE) for details.

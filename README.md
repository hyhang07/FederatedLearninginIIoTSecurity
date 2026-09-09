# Securing IIoT Networks: A Dual-Layer Defence Approach in Hierarchical Federated Learning

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://tensorflow.org)
[![LightGBM](https://img.shields.io/badge/LightGBM-Ensemble-green.svg)](https://lightgbm.readthedocs.io/)

This repository contains the official implementation of the Final Year Project (FYP2) submitted to **Universiti Tunku Abdul Rahman (UTAR)**:
> **"Federated Learning for IIoT Security: Decentralized Anomaly Detection"**  
> *Author:* Hong Yee Hang  
> *Supervisor:* Dr. Nadeem Muhammad Waqas  
> *Degree:* Bachelor of Information Technology (Honours) Communications and Networking  

---

## 📌 Project Overview
Industrial Internet of Things (IIoT) infrastructures generate continuous, sensitive network telemetry that cannot be transferred to a centralised cloud due to strict privacy regulations. While Federated Learning (FL) enables distributed on-device model training, conventional aggregation protocols (such as standard Federated Averaging / FedAvg) are vulnerable to:
1. **Data Poisoning (Targeted Label Flipping):** Malicious nodes flip training labels to alter decision boundaries.
2. **Model Poisoning (Gradient Scaling):** Malicious nodes amplify weight updates to dominate aggregation.

This project implements a **three-tier Hierarchical Federated Learning (HFL)** architecture equipped with a **Dual-Layer Defence** mechanism:
* **Tier 1 (Client Layer):** Deploys local deep feature extractors (**1D-TCN + GRU** for sequential traffic; **Tabular Deep Neural Networks / T-DNN** for static flow telemetry) combined with **Top-k Sparsification** to conserve communication bandwidth.
* **Tier 2 (Edge Gatekeeper Layer):** Applies a two-stage verification strategy:
  1. *Hard-threshold filters:* Accuracy validation (< 0.50 threshold) and L2 norm checks (> 100.0 threshold).
  2. *FLTrust directional verification:* ReLU-clipped Cosine Similarity against a Root-of-Trust reference update, followed by magnitude normalisation.
* **Tier 3 (Global Aggregation Layer):** Discards parameter averaging in favour of a **Stacking Ensemble** meta-learner (**LightGBM**), which systematically fuses regional base models on dedicated meta-features to manage extreme Non-IID data skew without feature dilution.

---

## 📊 Benchmark Results (Under Mixed Poisoning Attacks)

Comparative simulations against an unprotected baseline demonstrate the resilience of the proposed architecture:

| Evaluation Metric | Baseline FedAvg (No Defence) | Proposed Framework (Dual-Layer HFL) |
| :--- | :---: | :---: |
| **Edge-IIoTset Global Accuracy** | 4.34% *(Model Collapsed)* | **90.80%** |
| **TON_IoT Global Accuracy** | 5.56% *(Model Collapsed)* | **82.87%** |
| **Normal Traffic Precision** | 0.00 (Edge) / 0.12 (TON) | **1.00 (Edge) / 0.96 (TON)** |
| **Malicious Update Rejection Rate** | 0% *(Accepted All)* | **100% (Zero False Positives)** |
| **Communication Payload Reduction** | 0% (Full Updates) | **77.5% (Edge) / 50.2% (TON)** |

---

## 📂 Repository Structure

The codebase is organized into two primary implementations based on data topology:

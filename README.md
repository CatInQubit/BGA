# BGA: Encrypted Traffic Detection via BiLSTM and Gated Multi-Head Attention

This repository provides the implementation of the **BGA Framework**, a deep learning system designed for robust encrypted traffic detection in Industrial IoT (IIoT) environments. 

## 🚀 Overview
The BGA framework integrates a **Bidirectional Long Short-Term Memory (BiLSTM)** network with an **Adaptive Gated Multi-Head Attention** mechanism. To address the extreme class imbalance, we employ **WGAN-GP** (Wasserstein GAN with Gradient Penalty) for high-fidelity data augmentation.

## 📂 File Structure
- `model.py`: Architecture of the BGA model (BiLSTM + Gated Attention).
- `wgan_gp_engine.py`: Implementation of the WGAN-GP generative module for data augmentation.
- `data_loader.py`: Script for loading and pre-processing the traffic data (Normalization & ANOVA).
- `main.py`: The entry point for training and evaluating the model.
- `gas_final.arff.csv`: A sample subset of the processed Edge-IIoT dataset (Gas Pipeline).
- `README.md`: Project documentation.

## 🛠 Installation
Ensure you have Python 3.8+ and the following dependencies installed:
```bash
pip install torch pandas numpy scikit-learn matplotlib
```

## 💻 Usage
### 1. Data Preparation
The `data_loader.py` handles the Min-Max normalization and feature selection based on ANOVA.
### 2. Data Augmentation
To balance the minority attack classes using WGAN-GP:
```bash
# This logic is integrated within the training pipeline
```
### 3. Training & Evaluation
To train the BGA model and generate the classification report:
```bash
python main.py
```

## 📊 Performance
The model achieves a performance ceiling of over **95%** across all major metrics (Precision, Recall, F1-Score) on the Edge-IIoT and CIC-IDS-2018 datasets.

## 📝 Data Availability
The raw datasets can be found here:
- **Edge-IIoTset**: [Official Link](https://ieee-dataport.org/documents/edge-iiotset-dataset)
- **CIC-IDS-2018**: [Official Link](https://www.unb.ca/cic/datasets/ids-2018.html)

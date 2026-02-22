# IIoT Intrusion Detection System based on BGA-BiLSTM and WGAN-GP

This repository implements a robust Intrusion Detection System (IDS) specifically designed for Industrial Internet of Things (IIoT) environments. The framework integrates **WGAN-GP data augmentation** to address class imbalance and a **BiLSTM with Residual Gated Attention (BGA)** to enhance feature extraction and model interpretability.

## 🚀 Key Features

1.  **WGAN-GP Data Augmentation**: Utilizes Wasserstein GAN with Gradient Penalty to synthesize minority class samples (e.g., MSCI, MPCI attacks). This ensures a balanced training set, significantly improving the recall of rare attacks.
2.  **Residual Gated Attention (BGA)**: Implements a novel attention mechanism combining Multi-Head Attention (MHA) with a Sigmoid-gated residual connection. This allows the model to dynamically focus on critical traffic features while filtering out industrial background noise.
3.  **Four Experimental Modes**:
    *   **Main**: Full training and evaluation with visualization.
    *   **Ablation**: Comparative study between BaseLSTM, BiLSTM, BiLSTM+MHA, and the proposed BGA model.
    *   **Sensitivity**: Analysis of performance trends across different hidden dimensions and attention heads.
    *   **Robustness**: Stress testing against Gaussian noise to simulate encrypted or unstable network environments (Table 4.4 in the thesis).
4.  **Explainable AI (XAI)**: Automatically generates Gated Heatmaps to visualize the attention weights, demonstrating how the model distinguishes between normal and malicious traffic.

---

## 📂 Project Structure

*   `main.py`: **The Central Engine**. Controls the execution flow and switches between the four experimental modes.
*   `model.py`: **Architecture Definition**. Contains the `BiLSTMWithResidualGatedAttention` and `GatedMultiheadAttention` modules.
*   `data_loader.py`: **Data Pipeline**. Handles CSV loading, label cleaning, ANOVA feature selection, WGAN-GP augmentation, and normalization.
*   `gas_final.arff.csv`: **Dataset**. Industrial gas sensor subset derived from the Edge-IIoT dataset.
*   `README.md`: Project documentation.

---

## 🛠️ Installation

Ensure you have Python 3.8+ installed. Install the required dependencies via pip:

```bash
pip install torch numpy pandas matplotlib seaborn scikit-learn
```

---

## ⚙️ Usage

You can switch the execution logic by modifying the `MODE` variable in `main.py`:

```python
# main.py - Line 26
MODE = 'Main'  # Options: 'Main', 'Ablation', 'Sensitivity', 'Robustness'
```

### Execution Modes:
*   **'Main'**: Standard run. Outputs Training Loss curves, Confusion Matrix, Classification Report, and **Attention Heatmaps**.
*   **'Ablation'**: Generates a performance comparison table for all model variants to verify the effectiveness of the Gated Attention module.
*   **'Sensitivity'**: Evaluates how the Weighted F1-score changes with different hyper-parameters (Hidden Dimensions: 32, 64, 128).
*   **'Robustness'**: Executes the noise-injection test. It outputs a comparison table showing F1-scores under different noise levels (None, Low, Mid, High).

---

## 📊 Model Architecture

1.  **Input Layer**: 1D features selected via ANOVA F-Test from IIoT traffic.
2.  **Data Balancing**: WGAN-GP generates synthetic samples for minority attack classes during the training phase.
3.  **Encoding Layer**: Bi-directional LSTM captures long-term temporal dependencies in sensor data.
4.  **Attention Layer**:
    *   **MHA**: Captures global feature correlations.
    *   **Gated Unit**: A Sigmoid gate controls the flow of information.
    *   **Residual Connection**: Prevents gradient vanishing and retains original temporal context.
5.  **Output Layer**: Softmax classifier for 5 categories (`Normal`, `CMRI`, `MSCI`, `MPCI`, `DoS`).

---

## 📈 Expected Outputs

The system generates several artifacts for analysis:
*   `attention_heatmap.png`: Visual evidence of the model's focus on malicious vs. normal traffic.
*   **Detailed Metrics**: Precision, Recall, F1-score, and Accuracy for every attack type.
*   **Latex-Ready Tables**: The console outputs formatted tables for direct use in academic papers (e.g., Ablation results and Robustness data).

---

## 🏷️ Citation
*This project was developed for research into Industrial Sensor Network Security. If you find this code useful for your research, please cite the corresponding thesis/paper.*

--- 
**Author**: CatInQubit  
**Last Updated**: 2024  
**License**: MIT License

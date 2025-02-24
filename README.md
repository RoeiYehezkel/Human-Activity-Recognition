# Human-Activity-Recognition

---

## 1. Importing & Extracting the Data
- Data extracted using **Kaggle API** and organized in a **custom Python environment**.
- Required libraries for **data processing, machine learning, and deep learning** were set up.
- Dataset contains labeled **training data and metadata** for robust evaluation of **human activity recognition (HAR) models**.

---

## 2. Exploratory Data Analysis (EDA)
### Key Findings
1. Dataset consists of **18 human activities** performed by **8 unique users** in training.
2. Activities like **stairs_down** and **stairs_up** are underrepresented.
3. **Sequence lengths vary** significantly across sensor measurements.

### Data Insights
- **Test users are distinct from training users**, ensuring a **challenging generalization task**.
- Unique patterns exist in **time-series data** for different activities.

---

## 3. Preprocessing and Feature Engineering
### Feature Extraction
Extracted features from raw sensor data:
1. **Statistical Features**: Mean, standard deviation, skewness, kurtosis (for x, y, and z axes).
2. **Signal Magnitude Area (SMA)**: Measures overall activity intensity.
3. **Zero-Crossing Rate (ZCR)**: Indicates frequency of signal sign changes.

### Data Preparation
- **Train-validation split by user ID** to prevent data leakage.
- **Padded/truncated sequences** for uniform length.
- Integrated metadata like **sensor type and body part**.

---

## 4. Naïve Baseline Model
- Implemented based on **class distribution**.
- Achieved **validation log-loss: 3.0516** (reference for advanced models).

---

## 5. Classical Machine Learning Approach
### **Random Forest Classifier**
- Trained **100 estimators** using extracted features.
- Results:
  - **Validation Accuracy**: ~95%
  - **Validation Log-Loss**: ~0.33
- **Logged results with MLflow** and generated **classification report & confusion matrix**.

---

## 6. Deep Learning Approaches
### 6.1 Convolutional Neural Network (CNN)
#### **Architecture**:
1. **Three convolutional layers** with increasing filters.
2. **Max-pooling** for temporal feature extraction.
3. **Fully connected layers** with **dropout regularization**.
4. **Softmax output layer** for classification.

- **Validation accuracy improved across epochs**.
- Confusion matrix highlighted **areas for misclassification**.

### 6.2 Long Short-Term Memory (LSTM)
#### **Architecture**:
1. **Multi-layer LSTM** for temporal dependencies.
2. **Batch normalization** for convergence stability.
3. **Fully connected output layer**.
4. **Early stopping & model checkpointing**.

- **Validation Accuracy**: **81%**
- **Validation Loss**: **0.52**
- Model generalizes well but struggles with **overlapping sensor patterns**.

---

## 7. Comparative Analysis: LSTM vs. 1D CNN
| Feature  | LSTM | 1D CNN |
|----------|------|--------|
| **Temporal Dependencies** | Captures long-term dependencies | Struggles with long sequences, captures local patterns |
| **Validation Accuracy** | 81% | 50.4% |
| **Validation Loss** | 0.52 | 1.29 |
| **Generalization** | Some overfitting | Less accurate but fewer overfitting issues |
| **Architectural Complexity** | Higher (3 LSTM layers) | Lower (convolutional layers) |

---

## 8. Suggestions for Improvement
### 1. Distinguishing Similar Activities
- **Custom Weighted Loss Function**: Penalizes misclassifications between **similar activities**.
- **Feature Augmentation**: Introduces **new features** emphasizing differences (e.g., acceleration deltas, frequency-domain features).

### 2. Improving Generalization & Avoiding Overfitting
- **Dropout Layers** after LSTM for regularization.
- **Bidirectional LSTMs** for context from past & future time steps.
- **Simplified LSTM Architecture** (fewer layers, reduced neurons).

### 3. Addressing Class Imbalance
- **Oversampling or Data Augmentation** for underrepresented activities.
- **Weighted Sampling** in DataLoader to balance training distribution.

### 4. Hybrid Model Approach
- Combine **1D CNN + LSTM**: CNN extracts **local features**, LSTM models **long-term dependencies**.
- **Attention Mechanisms** for better **feature selection**.

---

## 9. Self-Supervised Pretraining: Masking for Pretraining
### **Approach**:
1. **Randomly mask** 20% of sequence data.
2. Train model to **predict masked values**.
3. **Use Mean Squared Error (MSE) loss** for reconstruction.

### **Model Architecture**:
- **LSTM backbone** extracts temporal dependencies.
- **Fully connected layers** reconstruct masked values.
- Pretrained on **unlabeled sequences**, then fine-tuned on **classification task**.

### **Results**:
- **Validation Accuracy**: **48.25%**
- **Validation Loss**: **1.4**
- **Struggles with subtle activity variations**.

---

## 10. Final Comparative Analysis
| Model | Validation Accuracy | Validation Loss |
|--------|------------------|----------------|
| **Baseline LSTM** | **81%** | **0.52** |
| **Self-Supervised Masking LSTM** | **48.25%** | **1.4** |

### **Key Takeaways**:
- **Regular LSTM significantly outperforms Masking LSTM** in recognizing activities.
- **Self-Supervised Masking needs improvements** in **pretraining strategies** and **regularization**.

### **Future Improvements**:
1. **Optimize Learning Rate Scheduler**.
2. **Introduce Dropout for Regularization**.
3. **Experiment with Bidirectional LSTMs**.
4. **Improve Pretraining Strategy (adjust masking probabilities)**.
5. **Augment training data with noise & transformation techniques**.

---

## Conclusion
- **Pretrained models outperform naive baselines**.
- **LSTM performs better** than CNN for HAR.
- **Hybrid CNN-LSTM models & attention mechanisms** are promising future approaches.

---



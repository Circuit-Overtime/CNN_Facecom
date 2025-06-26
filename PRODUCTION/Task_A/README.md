# 🧠 Task A: Gender Classification (VGG19)

A robust deep learning pipeline for **gender classification** under challenging, real-world conditions, using a fine-tuned VGG19 model with strong augmentation and fairness principles.

---

## 🚀 Overview

- **Goal:** Predict gender (male/female) from face images, even in poor lighting or weather.
- **Approach:** Fine-tuned VGG19 with focal loss, heavy augmentation, and class balancing.
- **Key Features:**
    - Handles class imbalance (more male images than female).
    - Auto-threshold tuning for optimal F1 score.
    - Grad-CAM for model explainability.

---

## 📁 Folder Structure

```
PRODUCTION/Task_A/
├── training/         # Training scripts, logs, plots
├── inference/        # Inference & Grad-CAM scripts
├── test/             # Test images for inference
├── README.md         # (This file)
```

---

## 🏗️ How to Use

### 1. **Training**

> **Note:** Training is optional; pretrained weights are provided.

- Script: `training/train_vgg19_updated.py`
- Requirements: Data in `Data/Task_A/` (see `Task.md` for structure)

### 2. **Inference**

- Script: `inference/inference_vgg19_updated.py`
- Place your test image in `test/` or provide a path.
- Run the script to get gender prediction and Grad-CAM visualization.

**Example (from deprecated script):**
```python
python inference/inference_vgg19_updated.py
```
Output:
```
🧠 Predicted Gender: Male
🔢 Confidence: 98.12%
📊 Probabilities: Male: 98.12%, Female: 1.88%
```

### 3. **Model Weights**

- Download `vgg19_final_epoch.h5` from [releases](https://github.com/Circuit-Overtime/CNN_vedic/releases/tag/publish102)
- Place in `PRODUCTION/models/`

---

## 📊 Results

**Validation Performance:**  
(See `training/vgg19_training_logs.txt`)

- **Accuracy:** 96.85%
- **F1 Score:** 0.9684
- **Precision:** 0.9714
- **Recall:** 0.9653
- **Male Accuracy:** 0.9716
- **Female Accuracy:** 0.9653

|        | Precision | Recall | F1-score | Support |
|--------|-----------|--------|----------|---------|
| Male   |   0.97    |  0.97  |   0.97   |   317   |
| Female |   0.97    |  0.97  |   0.97   |   317   |
| **Overall** | **0.97** | **0.97** | **0.97** | **634** |

Confusion Matrix:
```
[[308   9]
 [ 11 306]]
```

---

## 📝 Key Implementation Details

- **Model:** VGG19 backbone, custom dense head, softmax output.
- **Loss:** Focal loss for class imbalance.
- **Augmentation:** Rotation, brightness, flips, etc.
- **Threshold:** Auto-tuned for best F1 (default: 0.40).
- **Explainability:** Grad-CAM visualization available.

---

## 📚 References

- See `Task.md` for dataset structure and challenge details.
- Model weights and releases:  
    [GitHub Releases](https://github.com/Circuit-Overtime/CNN_vedic/releases/)

---

## 🤝 Acknowledgements

Developed by [Circuit-Overtime](https://github.com/Circuit-Overtime) and contributors.  
For academic/educational use only.

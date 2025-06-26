# 🧠 CNN COMSYS Hackathon

A robust deep learning pipeline for **gender classification** and **face verification** under challenging, real-world conditions, inspired by Vedic principles of fairness and accuracy.

![Elixpo_Generated (2)](https://github.com/user-attachments/assets/3e38c081-8576-419b-9503-37adcd2bb9b4)

---

## 📂 Repository Structure

```
CNN_vedic/
├── PRODUCTION/
│   ├── Task_A/               # Gender classification (VGG19-based)
│   │   ├── training/         # Training scripts, logs, plots
│   │   ├── inference/        # Inference & Grad-CAM scripts
│   │   └── test/             # Test images for inference
│   ├── Task_B/               # Face verification (Triplet Network)
│   │   ├── training/         # Triplet model training, embedding extraction
│   │   ├── inference/        # Face matching demo
│   │   └── test/             # Test/reference images for verification
├── [DEPRECATED MODELS]/      # Old/experimental models & scripts
├── Data/                     # (Git-ignored) Training/validation/test data
├── TASK.md                   # Challenge description & dataset structure
└── README.md                 # (You are here)
├── app/                      # Contains the server.py for the flask server inference
├── index.html                # frontend entry to check inference using a GUI
```
---

## 🚀 Project Overview

This project addresses two core computer vision tasks:

### **Task A: Gender Classification**
- **Goal:** Predict gender (male/female) from face images, even in poor lighting or weather.
- **Approach:** Fine-tuned VGG19 with focal loss, heavy augmentation, and class balancing.
- **Key Features:**
    - Handles class imbalance (more male images than female).
    - Auto-threshold tuning for optimal F1 score.
    - Grad-CAM for model explainability.
    - **Visual Explainability:** Uses Grad-CAM to highlight which regions of the face the model focuses on to determine gender, displaying a heatmap overlay. See [PRODUCTION/Task_A/inference/inference_vgg19_updated.py](#file:inference_vgg19_updated.py-context) for implementation details.
- **Relevant Files:**
    - `PRODUCTION/Task_A/training/train_vgg19_updated.py` — Full training pipeline.
    - `PRODUCTION/Task_A/inference/inference_vgg19_updated.py` — Inference & Grad-CAM.
    > Put the model in this folder from the link:  
    [vgg19_final_epoch](https://github.com/Circuit-Overtime/CNN_vedic/releases/tag/publish102)
    - `PRODUCTION/models/vgg19_final_epoch.h5` — Final model weights.
    - `PRODUCTION/Task_A/training/vgg19_training_logs.txt` — Training/evaluation logs.

### **Task B: Face Matching (Verification)**
- **Goal:** Match distorted face images to correct identity using embeddings (not classification).
- **Approach:** Triplet Network with ResNet50 backbone, trained to minimize intra-class and maximize inter-class distances.
- **Key Features:**
    - Embedding extraction for similarity-based matching.
    - Automatic threshold tuning for best accuracy.
- **Relevant Files:**
    - `PRODUCTION/Task_B/training/tripletNetwork_updated.py` — Triplet training pipeline.
    - `PRODUCTION/Task_B/training/embedding_model_extract.py` — Extracts embedding submodel.
    - `PRODUCTION/Task_B/inference/verify_face.py` — Face verification demo.
    > Put the model in this folder from the link 

    > **Note: You only need the embedding_sequel.h5 model** from here: 
    [embedding_sequel.h5](https://github.com/Circuit-Overtime/CNN_vedic/releases/tag/publish101)
    - `PRODUCTION/models/tripletNetwork.h5` — Full triplet model.
    - `PRODUCTION/models/embedding_sequel.h5` — Embedding-only model.
    - `PRODUCTION/Task_B/training/tripletTrainingLogs.txt` — Training/evaluation logs.

---

## 📁 Folder Details

### `PRODUCTION/`
- **Task_A/**: All scripts, logs, and models for gender classification.
    - `training/`: Training code, logs, and plots.
    - `inference/`: Inference script with Grad-CAM visualization.
- **Task_B/**: All scripts, logs, and models for face verification.
    - `training/`: Triplet network training, embedding extraction.
    - `inference/`: Face verification demo using embeddings.
- **models/**: Final production models (`.h5`), ready for inference.

### `[DEPRECATED MODELS]/`
- Contains experimental, ablation, and legacy models/scripts (Keras, PyTorch, EfficientNet, etc.).
- **Not for production use**; see `README.md` inside for details.

### `Data/`
- **Not included in repo** (see `.gitignore`).
- Structure for Task A and Task B as described in `Task.md`.

### `Task.md`
- Full challenge description, dataset structure, and objectives for both tasks.

---

## 🏗️ How to Use

### **1. Gender Classification (Task A)**
- **Train:**  
    Run `PRODUCTION/Task_A/training/train_vgg19_updated.py` (requires data in `Data/Task_A/`).
- **Inference:**  
    Place test image in `PRODUCTION/Task_A/test/`, run `PRODUCTION/Task_B/inference/inference_vgg19_updated.py`.
- **Model:**  
    Download `vgg19_final_epoch.h5` from [releases](https://github.com/Circuit-Overtime/CNN_vedic/releases/tag/publish102).

### **2. Face Verification (Task B)**
- **Train:**  
    Run `PRODUCTION/Task_B/training/tripletNetwork_updated.py` (requires data in `Data/Task_B/`).
- **Extract Embedding Model:**  
    Run `embedding_model_extract.py` after training.
- **Inference:**  
    Place reference/test images, run `PRODUCTION/Task_B/inference/verify_face.py`.
- **Model:**  
    Download `embedding_sequel.h5` from [releases](https://github.com/Circuit-Overtime/CNN_vedic/releases/tag/publish101).

---

## 📝 Key Features

- **Class Imbalance Handling:**  
    Automatic class weights and focal loss for fair gender classification.
- **Data Augmentation:**  
    Robust to real-world distortions (rotation, brightness, etc.).
- **Auto Threshold Tuning:**  
    Finds best probability/distance threshold for optimal F1/accuracy.
- **Explainability:**  
    Grad-CAM visualizations for gender model, showing which parts of the face are being scanned to determine gender using a heatmap.
- **Production-Ready:**  
    All production models and scripts are in `PRODUCTION/`.

---

## 📈 GRAD-CAM Result Showcase (Male and Female)



## 📊 Results

- **Gender Classification:**  
    - Accuracy: ~97% (see `vgg19_training_logs.txt`)
    - Balanced precision/recall for both genders.
- **Face Verification:**  
    - Accuracy: ~96% (see `tripletTrainingLogs.txt`)
    - High precision/recall for identity matching.

---

## 🔧 Development Environment

| Component     | Version          |
|---------------|------------------|
| Python        | 3.10.0           |
| TensorFlow    | 2.9.0            |
| CUDA Toolkit  | 11.8             |
| cuDNN         | 8.6.0            |
| GPU           | NVIDIA RTX (6GB or higher) recommended

---

## 📦 Requirements

### CPU Environment

- **Python**: 3.11 or higher
- **TensorFlow**: Latest (CPU version)
- **Keras**
- **scikit-learn**
- **OpenCV**
- **matplotlib**
- **tqdm**

Install with:
```bash
pip install tensorflow keras scikit-learn opencv-python matplotlib tqdm
```

---

### GPU Environment

- **Python**: 3.10 (required)
- **TensorFlow**: 2.9.0 (GPU version)
- **Keras**
- **scikit-learn**
- **OpenCV**
- **matplotlib**
- **tqdm**
- **CUDA Toolkit**: 11.8
- **cuDNN**: 8.6.0
- **NVIDIA GPU**: RTX series (6GB+ VRAM recommended)

Install with:
```bash
pip install tensorflow==2.9.0 keras scikit-learn opencv-python matplotlib tqdm
```
> Ensure CUDA 11.8 and cuDNN 8.6.0 are installed and properly configured for GPU support.

---

## 📚 References

- See `Task.md` for dataset structure and challenge details.
- Model weights and releases:  
    [GitHub Releases](https://github.com/Circuit-Overtime/CNN_vedic/releases/)

---

## 🤝 Acknowledgements

Developed by [Circuit-Overtime](https://github.com/Circuit-Overtime) and contributors.  
For academic/educational use only.

---

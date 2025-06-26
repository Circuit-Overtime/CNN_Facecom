# 🧑‍🤝‍🧑 Task B: Face Verification (Triplet Network)

A robust deep learning pipeline for **face verification** under challenging, real-world conditions, using a Triplet Network with a ResNet50 backbone. This system matches distorted face images to the correct identity using learned embeddings and a distance-based threshold.

---

## 🚩 Overview

- **Goal:** Match distorted face images to the correct identity folder using similarity in embedding space (not classification).
- **Approach:**  
    - Triplet Network with ResNet50 backbone, trained to minimize intra-class and maximize inter-class distances.
    - Embedding extraction for similarity-based matching.
    - Automatic threshold tuning for best accuracy.

---

## 📂 Key Files

- `training/tripletNetwork_updated.py` — Triplet network training pipeline.
- `training/embedding_model_extract.py` — Extracts embedding submodel for inference.
- `inference/verify_face.py` — Face verification demo script (see usage below).
- `training/tripletTrainingLogs.txt` — Training and evaluation logs.
- `../models/embedding_sequel.h5` — Embedding-only model for inference (download from [releases](https://github.com/Circuit-Overtime/CNN_vedic/releases/tag/publish101)).

---

## 🏗️ How It Works

1. **Training:**  
     - The triplet network is trained on anchor, positive, and negative face images to learn an embedding space where similar faces are close and dissimilar faces are far apart.
2. **Embedding Extraction:**  
     - After training, the embedding submodel is extracted for fast inference.
3. **Verification:**  
     - For two images, embeddings are computed and L2-normalized.
     - The Euclidean distance between embeddings is compared to a threshold to decide if the faces match.

---

## 📝 Usage

1. **Download the Model:**  
     - Place `embedding_sequel.h5` in `PRODUCTION/models/` (see [releases](https://github.com/Circuit-Overtime/CNN_vedic/releases/tag/publish101)).

2. **Run Verification Demo:**
     ```bash
     python PRODUCTION/Task_B/inference/verify_face.py
     ```
     - Edit the `reference_img` and `test_img` paths in the script to your images.
     - Output will indicate if the faces are a match or not.



## ⚙️ Configuration

- **Model Path:** `PRODUCTION/models/embedding_sequel.h5`
- **Threshold:** 0.945 (auto-tuned, see logs)
- **Input Size:** (224, 224)

---

## 📊 Results

- **Best Threshold:** 1.0652
- **Accuracy:** 96.00%
- **Precision:** 0.9729
- **Recall:** 0.9841
- **F1 Score:** 0.9785

(See `training/tripletTrainingLogs.txt` for full logs.)

---


## 📚 References

- See main [README.md](../../README.md) and `Task.md` for full challenge details and dataset structure.
- Model weights: [GitHub Releases](https://github.com/Circuit-Overtime/CNN_vedic/releases/)

---
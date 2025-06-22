
# 🧠 CNN Vedic Principles Facial and Gender Detection

---
##  Challenge Overview

###  Task A: Gender Classification
- **Objective:** Predict the gender (male/female) of a person from face images captured in difficult lighting/weather conditions.
- **Dataset Structure:**
```
dataset/
├── train/
│ ├── male/ # 1532 images
│ └── female/ # 394 images
└── val/
├── male/ # 317 images
└── female/ # 105 images
```
- **Model Goal:** Train a binary classifier that is accurate, fair, and generalizes well on distorted or real-world images.

---

###  Task B: Face Matching (Face Verification)
- **Objective:** Match distorted face images to the correct identity folder using embeddings, **not** classification.
- **Dataset Structure:**
```
dataset/
├── identities/
│ ├── person_001/
│ │ ├── clean.jpg
│ │ └── distorted/
│ │ ├── img1.jpg
│ │ ├── img2.jpg
│ │ └── ... (total 7 distorted images)
│ ├── person_002/
│ │ └── ...
│ └── ... (total: 877 identity folders)
```

- **Model Goal:** Learn a similarity-based system that embeds faces such that:
- Similar identities are **close in embedding space**
- Dissimilar faces are **far apart**

---

##  Folder Structure (Repo)
```
├── dataset/ # Not included in repo; user-supplied
├── models/ # Pretrained weights (.pt or .h5)
├── utils/
│ └── vedic_ops.py # Vedic arithmetic utilities (optional)
├── train_model.py # Trains both Task A and Task B
├── inference.py # Runs on test data and returns metrics
├── requirements.txt # Dependencies
└── README.md # Project explanation and usage guide
```


---

##  Submission Deadline
- **30 June 2025**

All code must be working and error-free on final test data.

---

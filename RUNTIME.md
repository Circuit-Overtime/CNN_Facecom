# 🧩 Project Runtime Instructions

This guide explains how to run the project on both **CPU** and **GPU** environments. The project includes pretrained models for both Task A (Gender Classification) and Task B (Face Verification), so training is not required.

---

## 🧠 Default: CPU Runtime (Recommended for Most Users)

You can run the inference and evaluation scripts using your local CPU. No GPU is required.

### ✅ Requirements

- Python **3.10**
- `requirements.txt` (already provided)

### ⚙️ Setup Instructions

1. **Create a virtual environment**:
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

2. **Install dependencies**:
    ```bash
    pip install --upgrade pip
    pip install -r requirements.txt
    ```

3. **Run Inference / Evaluation**:
    ```bash
    python PRODUCTION\Task_A\inference\inference_vgg19_updated.py
    python PRODUCTION\Task_B\inference\verify_face.py
    ```

> 📝 This will use CPU by default and works on most systems without CUDA or GPU drivers.

---

## 🚀 Advanced: GPU Runtime via Docker (CUDA 11.8 + cuDNN 8.6)

To run the model on GPU (TensorFlow 2.9.0 + Python 3.10), you can use our GPU Docker environment.

### ✅ GPU Runtime Requirements

- A system with an NVIDIA GPU (Ampere/Volta/Turing preferred)
- Installed:

  | Tool                     | Version        |
  |--------------------------|----------------|
  | Docker                   | Latest         |
  | NVIDIA GPU Driver        | ≥ 515.x        |
  | NVIDIA Container Toolkit | [Install Guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) |
  
> ⚠️ No need to install CUDA or cuDNN manually — they are included in the Docker image.

---

### 🧱 GPU Environment Details

| Component      | Version     |
|----------------|-------------|
| Python         | 3.10        |
| TensorFlow     | 2.9.0       |
| CUDA           | 11.8        |
| cuDNN          | 8.6         |
| Base Image     | `nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu20.04` |

---

### 📦 Build & Run with Docker

1. **Test your NVIDIA GPU with Docker**:
    ```bash
    docker run --rm --gpus all nvidia/cuda:11.8.0-base nvidia-smi
    ```

2. **Build the container**:
    ```bash
    docker compose -f docker-compose.yml up --build
    ```

3. **Run inference inside the container**:
    ```bash
    docker compose run facecom python inference_taskA.py
    docker compose run facecom python inference_taskB.py
    ```

---

### 🛠 Alternative: Manual GPU Setup (Not Recommended)

If you wish to run on GPU *without Docker*, ensure:

- Python == **3.10**
- TensorFlow == **2.9.0**
- CUDA 11.8 and cuDNN 8.6 are installed **and correctly configured**.

> See [TensorFlow GPU Setup Guide](https://www.tensorflow.org/install/pip#windows-native) for details.

---

## ✅ Files Provided

| File                     | Description                                       |
|--------------------------|---------------------------------------------------|
| `requirements.txt`       | CPU runtime dependencies                         |
| `requirements_GPU.txt`   | GPU runtime dependencies (TensorFlow 2.9.0, etc.)|
| `Dockerfile.gpu`         | Custom Dockerfile with GPU setup                 |
| `docker-compose.yml`     | Compose config to run in GPU container           |
| `inference_taskA.py`     | Gender classification inference script (Task A)  |
| `inference_taskB.py`     | Face verification inference script (Task B)      |
| `RUNTIME.md`             | You are here                                     |

---

### 📬 Need Help?

Please open an issue on the GitHub repo or contact the maintainer.

---

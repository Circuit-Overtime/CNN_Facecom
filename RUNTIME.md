# 🧩 Project Runtime Instructions

This guide explains how to run the project on both **CPU** and **GPU** environments. The project includes pretrained models for both Task A (Gender Classification) and Task B (Face Verification), so **training is not required**.

---


## 🧬 Conda Environment (Recommended)

For guaranteed reproducibility across systems, we provide a ready-to-use Conda environment file directly in this repository.

### 🔧 Setup Instructions (CPU/GPU Compatible)

1. **Install [Miniconda](https://docs.conda.io/en/latest/miniconda.html)** if you haven't already.

   - ✅ During installation, **check the box to add Conda to your PATH**.
   - ✅ After installing, open a new terminal and run:
     ```bash
     conda init powershell  # or conda init bash/zsh depending on your shell
     ```
   - ✅ Restart the terminal to apply changes.

2. **Create the Conda environment from the included YAML file**:
   ```bash
   conda env create -f facecom_env.yml
   ```

3. **Activate the Environment**
    ```bash
    conda activate facecom
    ```

4. **Run inference For Gender Prediction**:
    ```bash
    python PRODUCTION/Task_A/inference/inference_vgg19_updated.py
    ```
5. **Run inference For Face Classification**:
    ```bash
    python PRODUCTION/Task_B/inference/verify_face.py
    ```
6. **Deactivate the Environment**
    ```bash
    conda deactivate
    ```

---


## 🧠 Default: CPU Runtime (Easy but may not be optimal in terms of results)

You can run inference and evaluation scripts using your local **CPU**. No GPU is required.

### ✅ Requirements

- Python **3.10+**
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
    python PRODUCTION/Task_A/inference/inference_vgg19_updated.py
    python PRODUCTION/Task_B/inference/verify_face.py
    ```

> ✅ CPU mode works by default on all systems. You don’t need to install CUDA or cuDNN.

---

## 🖥️ Running the Web API & GUI

You can use the provided Flask server and web interface for an interactive experience (supports both CPU and GPU, depending on your environment):

### 1. **Start the Flask API server**

With your Conda environment activated (see above), run:

```bash
python app/server.py
```

- This will start the API at `http://127.0.0.1:5000/`.
- The server will automatically use GPU if available and supported by your environment.

### 2. **Use the Web GUI**

Open `index.html` in your browser (double-click or use a local web server).

- The GUI allows you to upload images for gender classification and face verification.
- Make sure the Flask server is running before using the GUI.

> **Note:** If accessing from another device or over a network, adjust the API endpoint URLs in `index.html` accordingly.

---

## 🚀 Advanced: GPU Runtime via Docker (CUDA 11.8 + cuDNN 8.6) {Optional}

To run on GPU using **Docker**, a preconfigured runtime is provided.

### ✅ GPU Runtime Requirements

- An NVIDIA GPU with driver version **≥ 515**
- Installed:

  | Tool                     | Version        |
  |--------------------------|----------------|
  | Docker                   | Latest         |
  | NVIDIA Container Toolkit | [Install Guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) |

> ⚠️ No need to install CUDA/cuDNN locally — they're included in the container image.

---

### 🧱 GPU Environment Specs

| Component      | Version     |
|----------------|-------------|
| Python         | 3.10        |
| TensorFlow     | 2.9.0       |
| NumPy          | 1.23.5      |
| CUDA           | 11.8        |
| cuDNN          | 8.6         |
| Base Image     | `nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu20.04` |

---

### 🐳 Run with Docker Compose

1. **Verify GPU support**:
    ```bash
    docker run --rm --gpus all nvidia/cuda:11.8.0-base nvidia-smi
    ```

2. **Build and start the container**:
    ```bash
    docker compose -f docker-compose.yml up --build
    ```

3. **Run a specific script in the container**:
    ```bash
    docker compose run facecom python PRODUCTION/Task_A/inference/inference_vgg19_updated.py
    docker compose run facecom python PRODUCTION/Task_B/inference/verify_face.py
    ```

---

## 🧾 Files Provided

| File                     | Description                                        |
|--------------------------|----------------------------------------------------|
| `requirements.txt`       | CPU-only dependencies                             |
| `requirements_GPU.txt`   | GPU-specific dependencies (TF 2.9.0, NumPy 1.23.5) |
| `env/facecom_gpu.yml`    | Conda YAML for reproducible GPU runtime           |
| `Dockerfile.gpu`         | Docker setup for GPU                              |
| `docker-compose.yml`     | Docker Compose to run GPU containers              |
| `app/server.py`          | Flask API server for GUI/web usage                |
| `index.html`             | Web GUI for gender and face verification          |
| `inference_vgg19_updated.py` | Gender classification (Task A)              |
| `verify_face.py`         | Face matching (Task B)                             |
| `RUNTIME.md`             | This file                                          |

---

## 🆘 Need Help?

If you're stuck or unsure how to proceed:

- Open an issue in the GitHub repository.
- Contact the project maintainer with logs and system info.

---

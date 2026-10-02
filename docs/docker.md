# Running BEVFormer with Docker

This guide details how to build, run, and develop **BEVFormer** inside a Docker container. Using Docker ensures a completely reproducible environment across both local Windows testing machines (via Docker Desktop & WSL2) and multi-GPU Linux production clusters, eliminating manual compilation issues for `mmcv-full`, `mmdetection3d`, and `detectron2`.

---

## 1. Host Prerequisites

### For Windows Users (Development & Testing)
1. **NVIDIA GPU Driver**: Ensure you have an updated NVIDIA driver installed (Game Ready or Studio driver). Modern drivers support GPU acceleration in WSL2/Docker automatically.
2. **Docker Desktop**:
   - Download and install [Docker Desktop for Windows](https://www.docker.com/products/docker-desktop/).
   - During setup, ensure **"Use the WSL 2 based engine"** is checked.
   - Verify GPU support from PowerShell:
     ```powershell
     docker run --rm --gpus all nvidia/cuda:11.3.1-base-ubuntu20.04 nvidia-smi
     ```

### For Linux Users (Training & Benchmarking)
1. Install [Docker Engine](https://docs.docker.com/engine/install/) and the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
2. Verify GPU access:
   ```bash
   docker run --rm --gpus all nvidia/cuda:11.3.1-base-ubuntu20.04 nvidia-smi
   ```

---

## 2. Docker Architecture & Files

The project includes pre-configured Docker files in the repository:
- `docker/Dockerfile`: Base image `pytorch/pytorch:1.10.0-cuda11.3-cudnn8-devel`, precompiled `mmcv-full==1.4.0`, source-built `mmdetection3d==v0.17.1`, `detectron2`, `timm`, `nuscenes-devkit`, and system dependencies (`dos2unix`, `libgl1`, etc.).
- `docker-compose.yml`: Mounts the repository to `/workspace/BEVFormer`, enables `ipc: host` (essential for PyTorch multi-process dataloaders), and requests NVIDIA GPU access.
- `.dockerignore`: Prevents large dataset directories (`data/`) and model checkpoints (`ckpts/`) from copying into the build context.

---

## 3. Quickstart Guide

### Step 1: Build the Docker Image
From your host terminal (PowerShell on Windows or Bash on Linux) in the repository root:

```bash
docker compose build
```
*(This may take ~10–15 minutes on the first build while CUDA extensions are compiled).*

### Step 2: Start an Interactive Container Session
```bash
docker compose run --rm bevformer
```
You will enter a shell inside the container at `/workspace/BEVFormer`.

> **Note on Windows Line Endings**: If the repository was cloned on Windows with CRLF line endings, run this command once inside the container to prevent script syntax errors:
> ```bash
> dos2unix tools/*.sh tools/**/*.sh
> ```

---

## 4. Dataset Setup & Storage on a Separate Disk

nuScenes is very large (the full dataset is over 100 GB). **You should never bake the dataset inside the Docker container image.** 

In our setup, the dataset is kept on your host storage via Docker volume bind mounts:

### Option A: Store in the Project Directory (Default)
By default, the container mounts `./data` on your host machine to `/workspace/BEVFormer/data`. If your repository is on drive `D:\`, the dataset stays physically on your `D:\` drive and does not consume space in Docker's internal virtual disk.

### Option B: Store on a Completely Separate Physical Disk (Recommended for Large Datasets)
If you have an external SSD, a dedicated NVMe drive, or a separate server mount (`/mnt/datasets`):
1. Copy [`.env.example`](../.env.example) to `.env`:
   ```bash
   cp .env.example .env
   ```
2. Set the path to your external directory in `.env`:
   - **On Windows** (e.g. secondary drive `E:`):
     ```env
     DATA_DIR=E:/datasets/bevformer_data
     ```
   - **On Linux** (e.g. dedicated NVMe or network storage):
     ```env
     DATA_DIR=/mnt/nvme/datasets/nuscenes
     ```
3. Docker will automatically mount that external disk to `/workspace/BEVFormer/data` inside the container without modifying any code.

### Dataset Directory Layout
Whichever option you choose, organize the dataset so it matches:
```text
<DATA_DIR>/
├── can_bus/
└── nuscenes/
    ├── maps/
    ├── samples/
    ├── sweeps/
    ├── v1.0-mini/       # or v1.0-trainval/
```

### Generate Temporal Annotations (Inside Container)
- **For nuScenes v1.0-mini:**
  ```bash
  python tools/create_data.py nuscenes --root-path ./data/nuscenes --out-dir ./data/nuscenes --extra-tag nuscenes --version v1.0-mini --canbus ./data
  ```
- **For nuScenes full dataset (v1.0):**
  ```bash
  python tools/create_data.py nuscenes --root-path ./data/nuscenes --out-dir ./data/nuscenes --extra-tag nuscenes --version v1.0 --canbus ./data
  ```

---

## 5. Download Pretrained Checkpoints

Create a `ckpts/` directory and download the checkpoint you wish to run:

```bash
mkdir -p ckpts

# BEVFormer-tiny checkpoint (recommended for 8GB VRAM)
wget -O ckpts/bevformer_tiny_epoch_24.pth https://github.com/zhiqi-li/storage/releases/download/v1.0/bevformer_tiny_epoch_24.pth

# Optional: R101-DCN backbone pretrain (for BEVFormer-base / small)
wget -O ckpts/r101_dcn_fcos3d_pretrain.pth https://github.com/zhiqi-li/storage/releases/download/v1.0/r101_dcn_fcos3d_pretrain.pth
```

---

## 6. Running Evaluation, Training & Visualization

Run all commands inside the active Docker container:

### A. Evaluation (1 GPU)
BEVFormer requires distributed evaluation mode even for a single GPU. Use [`tools/dist_test.sh`](../tools/dist_test.sh):

```bash
./tools/dist_test.sh ./projects/configs/bevformer/bevformer_tiny.py ./ckpts/bevformer_tiny_epoch_24.pth 1 --eval bbox
```

### B. Training (1 GPU)
```bash
./tools/dist_train.sh ./projects/configs/bevformer/bevformer_tiny.py 1
```

### C. Multi-GPU Training (Linux Server / 8 GPUs)
When deploying to a multi-GPU Linux node:
```bash
./tools/dist_train.sh ./projects/configs/bevformer/bevformer_base.py 8
```

### D. Visualization
To visualize predictions and ground-truth 3D bounding boxes projected onto multi-camera images:
```bash
python tools/analysis_tools/visual.py
```

---

## 7. Troubleshooting & Important Notes

- **GPU Memory Management**:
  - `BEVFormer-tiny` requires ~6.5GB VRAM (suitable for single 8GB GPUs like RTX 3060/4060 Laptop).
  - `BEVFormer-base` requires ~28.5GB VRAM across multiple GPUs (e.g. A100 / V100 / RTX 3090/4090 clusters).
- **Shared Memory Bus Errors**:
  - If PyTorch raises `RuntimeError: DataLoader worker ... is killed by signal: Bus error`, ensure `ipc: host` is set in `docker-compose.yml` (or pass `--ipc=host` if using `docker run`).
- **File Persistence**:
  - Everything saved under `/workspace/BEVFormer` inside the container persists directly on your host machine filesystem, including training logs in `work_dirs/`, checkpoints in `ckpts/`, and data in `data/`.

---

## 8. Key Optimizations for Speed & Stability

### 1. Persistent PyTorch Model Cache
By default, PyTorch downloads vision backbones (like ResNet-50) into `~/.cache/torch`. We mapped this directory to `${TORCH_CACHE:-./.cache/torch}` in `docker-compose.yml` so weights are retained on your host disk instead of re-downloading every time the container is started.

### 2. CUDA Memory Defragmentation
To avoid sudden out-of-memory errors on 8GB GPUs:
- `docker-compose.yml` sets `PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128`.
- This prevents PyTorch's caching allocator from fragmenting large contiguous memory blocks during the spatial cross-attention phase.

### 3. Mixed Precision Training (FP16)
For 2x faster training and ~40% lower VRAM usage on modern NVIDIA GPUs (like RTX 40-series Tensor Cores), use the FP16 config and launcher:
```bash
./tools/fp16/dist_train.sh ./projects/configs/bevformer_fp16/bevformer_tiny_fp16.py 1
```

### 4. Windows WSL2 Host Tuning (`.wslconfig`)
On Windows, WSL2 can consume up to 80% of system RAM by default. You can limit and stabilize memory usage by creating a file named `.wslconfig` in your Windows user profile folder (`C:\Users\<YourUsername>\.wslconfig`):
```ini
[wsl2]
memory=16GB       # Adjust according to your total RAM (e.g. 16GB or 24GB)
processors=8      # Limit vCPUs allocated to WSL2
swap=8GB
```
Restart WSL after editing: `wsl --shutdown` in PowerShell.

### 5. Line Ending Safety (`.gitattributes`)
Bash scripts checked out on Windows with CRLF can cause syntax errors (`\r: command not found`). We added a `.gitattributes` file enforcing `eol=lf` for all `.sh` scripts, and `dos2unix` is pre-installed in the Docker image.


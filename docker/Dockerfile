FROM pytorch/pytorch:1.10.0-cuda11.3-cudnn8-devel

# Avoid interactive prompts during apt install
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# Optimization: Ensure real-time terminal output and prevent CUDA memory fragmentation
ENV PYTHONUNBUFFERED=1
ENV PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128"

# Fix NVIDIA GPG key expiration issue in legacy Ubuntu images, then install system utilities
RUN rm -f /etc/apt/sources.list.d/cuda.list /etc/apt/sources.list.d/nvidia-ml.list \
    && apt-get update && apt-get install -y --no-install-recommends \
    git \
    wget \
    curl \
    ca-certificates \
    build-essential \
    ninja-build \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxrender-dev \
    libxext6 \
    ffmpeg \
    unzip \
    dos2unix \
    && rm -rf /var/lib/apt/lists/*

# Ensure CUDA compiler architectures are configured for both legacy and modern GPUs
ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0 7.5 8.0 8.6+PTX"
ENV FORCE_CUDA="1"

# Pin setuptools to avoid distutils deprecation issues in older OpenMMLab versions
RUN pip install --no-cache-dir --upgrade pip "setuptools<60.0.0" wheel

# Install MMCV-full with precompiled CUDA 11.3 + PyTorch 1.10 wheels
RUN pip install --no-cache-dir mmcv-full==1.4.0 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.10.0/index.html

# Install mmdet and mmsegmentation
RUN pip install --no-cache-dir mmdet==2.14.0 mmsegmentation==0.14.1

# Install mmdetection3d v0.17.1 from source (limit MAX_JOBS to prevent OOM during compilation)
ENV MAX_JOBS=4
RUN git clone -b v0.17.1 https://github.com/open-mmlab/mmdetection3d.git /opt/mmdetection3d \
    && cd /opt/mmdetection3d \
    && pip install --no-cache-dir -e .

# Install additional project dependencies (pin yapf==0.40.1 to avoid FormatCode verify argument bug)
RUN pip install --no-cache-dir \
    einops \
    fvcore \
    seaborn \
    iopath==0.1.9 \
    timm==0.6.13 \
    typing-extensions \
    pylint \
    ipython \
    numpy==1.19.5 \
    matplotlib==3.5.2 \
    pandas \
    scikit-image==0.19.3 \
    nuscenes-devkit \
    pyquaternion \
    shapely \
    yapf

# Install Detectron2 using official pre-compiled wheel for PyTorch 1.10 + CUDA 11.3
RUN python -m pip install --no-cache-dir detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu113/torch1.10/index.html

# Pin setuptools and yapf to avoid distutils/verify runtime bugs
RUN pip install --no-cache-dir "setuptools==59.5.0" "yapf==0.40.1"

WORKDIR /workspace/BEVFormer
CMD ["/bin/bash"]

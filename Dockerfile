# ============================================================================
# SAVi – Slot Attention for Video (Google Research)
# Docker image: CUDA 12.8 + cuDNN + Python 3.10 + JAX (GPU)
# ============================================================================
FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_ROOT_USER_ACTION=ignore

# Install Python 3.10 (ships with Ubuntu 22.04) and system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-venv \
    python3-pip \
    git \
    wget \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1 \
    && ln -sf /usr/bin/python3 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/slot-attention-video

# Copy requirements first for Docker layer caching
COPY requirements.txt .

# Install JAX 0.4.13 with CUDA 12 support (bundles its own CUDA/cuDNN via pip)
# Then install remaining dependencies from requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir "jax[cuda12_pip]==0.4.13" \
      -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html && \
    pip install --no-cache-dir -r requirements.txt

# Copy the full codebase (respects .dockerignore)
COPY . .

CMD ["/bin/bash"]
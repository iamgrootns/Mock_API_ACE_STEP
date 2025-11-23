FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

WORKDIR /app
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# ------------------------------------------------------------
# 1. System Dependencies
# ------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget git ffmpeg libsndfile1 \
    python3.10 python3.10-venv python3.10-distutils \
    build-essential libssl-dev zlib1g-dev \
    libbz2-dev liblzma-dev libreadline-dev libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# Use Python 3.10 as default python
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

RUN python -m ensurepip --upgrade
RUN python -m pip install --upgrade pip setuptools wheel

# ------------------------------------------------------------
# 2. Install PyTorch CUDA 12.4
# ------------------------------------------------------------
RUN python -m pip install --no-cache-dir \
    torch==2.6.0+cu124 \
    torchvision==0.21.0+cu124 \
    torchaudio==2.6.0+cu124 \
    --index-url https://download.pytorch.org/whl/cu124

# ------------------------------------------------------------
# 3. Install ACE-Step compatible HF stack
# ------------------------------------------------------------
RUN python -m pip install --no-cache-dir \
    safetensors==0.7.0 \
    huggingface-hub==0.36.0 \
    accelerate==1.6.0 \
    pillow==11.0.0 \
    tqdm==4.67.1 \
    regex==2025.11.3 \
    packaging==25.0 \
    loguru==0.7.3 \
    numpy==2.0.2

# transformers / diffusers / peft pinned exactly for ACE-Step 0.2.0
RUN python -m pip install --no-cache-dir --no-deps \
    transformers==4.31.0 \
    diffusers==0.21.4 \
    peft==0.3.0 \
    tokenizers==0.13.3

# ------------------------------------------------------------
# 4. Install ACE-Step
# ------------------------------------------------------------
RUN python -m pip install --no-cache-dir --no-deps \
    git+https://github.com/ace-step/ACE-Step.git

# ------------------------------------------------------------
# 5. Install Application Requirements
# ------------------------------------------------------------
COPY requirements.txt /app/requirements.txt
RUN python -m pip install --no-cache-dir -r /app/requirements.txt

# ------------------------------------------------------------
# 6. Copy Code
# ------------------------------------------------------------
COPY mockhandler.py /app/mockhandler.py

CMD ["python", "mockhandler.py"]

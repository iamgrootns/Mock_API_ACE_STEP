FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

WORKDIR /app
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# ------------------------------------------------------------
# 1. System Dependencies
# ------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget git ffmpeg \
    libsndfile1 libsndfile1-dev \
    python3.10 python3.10-venv python3.10-distutils python3-pip \
    build-essential libssl-dev zlib1g-dev \
    libbz2-dev liblzma-dev libreadline-dev libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# Force Python 3.10
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

RUN python3.10 -m pip install --upgrade pip setuptools wheel

# ------------------------------------------------------------
# 2. Install soundfile
# ------------------------------------------------------------
RUN python3.10 -m pip install soundfile

# ------------------------------------------------------------
# 3. PyTorch CUDA 12.4
# ------------------------------------------------------------
RUN python3.10 -m pip install --no-cache-dir \
    torch==2.6.0+cu124 \
    torchvision==0.21.0+cu124 \
    torchaudio==2.6.0+cu124 \
    --index-url https://download.pytorch.org/whl/cu124

# ------------------------------------------------------------
# 4. HuggingFace core libs (compatible with SD3 + ACE-Step)
# ------------------------------------------------------------
RUN python3.10 -m pip install --no-cache-dir \
    safetensors==0.7.0 \
    huggingface-hub==0.22.2 \
    accelerate==1.6.0 \
    pillow==11.0.0 \
    tqdm==4.67.1 \
    regex==2025.11.3 \
    packaging==25.0 \
    numpy==2.0.2 \
    loguru==0.7.3

# ------------------------------------------------------------
# 5. Transformers / Diffusers / PEFT (SD3-compatible)
# ------------------------------------------------------------
RUN python3.10 -m pip install --no-cache-dir --no-deps \
    diffusers==0.30.2 \
    transformers==4.42.4 \
    peft==0.6.2 \
    tokenizers==0.19.1

# ------------------------------------------------------------
# 6. Install ACE-Step WITHOUT dependencies
# ------------------------------------------------------------
RUN python3.10 -m pip install --no-cache-dir --no-deps \
    git+https://github.com/ace-step/ACE-Step.git

# ------------------------------------------------------------
# 7. App Requirements
# ------------------------------------------------------------
COPY requirements.txt /app/requirements.txt
RUN python3.10 -m pip install --no-cache-dir -r /app/requirements.txt

# ------------------------------------------------------------
# 8. Application Entrypoint
# ------------------------------------------------------------
COPY mockhandler.py /app/mockhandler.py

CMD ["python3.10", "mockhandler.py"]

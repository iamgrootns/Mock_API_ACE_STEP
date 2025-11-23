FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

WORKDIR /app
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# ----------------------------
# 1. System Dependencies
# ----------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget git ffmpeg libsndfile1 \
    build-essential libssl-dev zlib1g-dev \
    libncurses5-dev libncursesw5-dev libreadline-dev \
    libsqlite3-dev libgdbm-dev libdb5.3-dev libbz2-dev \
    libexpat1-dev liblzma-dev tk-dev libffi-dev \
    python3-venv ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# ----------------------------
# 2. Install Python 3.12.2
# ----------------------------
RUN wget https://www.python.org/ftp/python/3.12.2/Python-3.12.2.tgz && \
    tar -xf Python-3.12.2.tgz && \
    cd Python-3.12.2 && \
    ./configure --enable-optimizations && \
    make -j$(nproc) && make altinstall && \
    cd .. && rm -rf Python-3.12.2 Python-3.12.2.tgz

RUN update-alternatives --install /usr/bin/python3 python3 /usr/local/bin/python3.12 1 && \
    update-alternatives --install /usr/bin/python python /usr/local/bin/python3.12 1

RUN python3.12 -m pip install --upgrade pip setuptools wheel

# ----------------------------
# 3. Install PyTorch CUDA 12.4
# ----------------------------
RUN python3.12 -m pip install --no-cache-dir \
    torch==2.6.0+cu124 \
    torchvision==0.21.0+cu124 \
    torchaudio==2.6.0+cu124 \
    --index-url https://download.pytorch.org/whl/cu124

# ----------------------------
# 4. Install the ONLY working HF stack for ACE-Step
# ----------------------------
RUN python3.12 -m pip install --no-cache-dir \
    safetensors==0.7.0 \
    huggingface-hub==0.36.0 \
    accelerate==1.6.0 \
    requests==2.32.5 \
    tqdm==4.67.1 \
    regex==2025.11.3 \
    packaging==25.0 \
    pillow==11.0.0 \
    numpy==2.0.2 \
    loguru==0.7.3

# Transformers/Diffusers/PEFT MUST be pinned exactly:
RUN python3.12 -m pip install --no-cache-dir --no-deps \
    transformers==4.31.0 \
    diffusers==0.21.4 \
    peft==0.3.0 \
    tokenizers==0.13.3

# ----------------------------
# 5. Install ACE-Step (NO DEPENDENCIES)
# ----------------------------
RUN python3.12 -m pip install --no-cache-dir --no-deps \
    git+https://github.com/ace-step/ACE-Step.git

# ----------------------------
# 6. Install Your App Dependencies
# ----------------------------
COPY requirements.txt /app/requirements.txt
RUN python3.12 -m pip install --no-cache-dir -r /app/requirements.txt

# ----------------------------
# 7. Copy Application
# ----------------------------
COPY mockhandler.py /app/mockhandler.py

CMD ["python3.12", "mockhandler.py"]

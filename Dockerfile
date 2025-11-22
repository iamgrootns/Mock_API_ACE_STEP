# CUDA 12.4.1 + cuDNN + Ubuntu 22.04
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

WORKDIR /app

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3.12 \
    python3.12-dev \
    python3.12-venv \
    git \
    wget \
    curl \
    ffmpeg \
    libsndfile1 \
    libsndfile1-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install pip for Python 3.12
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.12

# Set Python 3.12 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1

# Upgrade pip
RUN python3.12 -m pip install --upgrade pip setuptools wheel

# Install PyTorch with CUDA 12.4 support
RUN pip install --no-cache-dir \
    torch==2.6.0+cu124 \
    torchvision==0.21.0+cu124 \
    torchaudio==2.6.0+cu124 \
    --index-url https://download.pytorch.org/whl/cu124

# Copy requirements and handler script
COPY requirements.txt /app/requirements.txt
COPY handler.py /app/mockhandler.py    

# Install Python dependencies (including ace_step)
RUN python3.12 -m pip install --no-cache-dir -r /app/requirements.txt

CMD ["python", "/app/mockhandler.py"]   

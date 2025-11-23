FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

# Set work directory
WORKDIR /app
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# System dependencies needed for Python build and ML/audio
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    git \
    ffmpeg \
    libsndfile1 \
    build-essential \
    libssl-dev \
    zlib1g-dev \
    libncurses5-dev \
    libncursesw5-dev \
    libreadline-dev \
    libsqlite3-dev \
    libgdbm-dev \
    libdb5.3-dev \
    libbz2-dev \
    libexpat1-dev \
    liblzma-dev \
    tk-dev \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python 3.12 from source
RUN wget https://www.python.org/ftp/python/3.12.2/Python-3.12.2.tgz && \
    tar -xf Python-3.12.2.tgz && \
    cd Python-3.12.2 && \
    ./configure --enable-optimizations && \
    make -j$(nproc) && \
    make altinstall && \
    cd .. && \
    rm -rf Python-3.12.2*

# Set python alternatives explicitly
RUN update-alternatives --install /usr/bin/python3 python3 /usr/local/bin/python3.12 1 && \
    update-alternatives --install /usr/bin/python python /usr/local/bin/python3.12 1

# Upgrade pip and basic build tools
RUN python3.12 -m pip install --upgrade pip setuptools wheel

# Install PyTorch CUDA libraries (ONLY FROM OFFICIAL CUDA WHEEL INDEX)
RUN python3.12 -m pip install --no-cache-dir \
    torch==2.6.0+cu124 \
    torchvision==0.21.0+cu124 \
    torchaudio==2.6.0+cu124 \
    --index-url https://download.pytorch.org/whl/cu124

# Clone ACE-Step source code, install in editable mode for reliable import (fixes ModuleNotFoundError)
RUN git clone https://github.com/ace-step/ACE-Step.git
RUN python3.12 -m pip install -e /app/ACE-Step

# Other direct dependencies (NO ace_step in requirements.txt)
COPY requirements.txt /app/requirements.txt
RUN python3.12 -m pip install --no-cache-dir -r /app/requirements.txt

# Copy application entrypoint last, after all installs and builds
COPY mockhandler.py /app/mockhandler.py

CMD ["python3.12", "mockhandler.py"]

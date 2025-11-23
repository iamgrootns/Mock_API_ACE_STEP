FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

WORKDIR /app
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies
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

RUN update-alternatives --install /usr/bin/python3 python3 /usr/local/bin/python3.12 1 && \
    update-alternatives --install /usr/bin/python python /usr/local/bin/python3.12 1

RUN python3.12 -m pip install --upgrade pip setuptools wheel

# Install PyTorch CUDA libraries
RUN python3.12 -m pip install --no-cache-dir \
    torch==2.6.0+cu124 \
    torchvision==0.21.0+cu124 \
    torchaudio==2.6.0+cu124 \
    --index-url https://download.pytorch.org/whl/cu124

# Clone ACE-Step and install in editable mode
RUN git clone https://github.com/ace-step/ACE-Step.git
RUN python3.12 -m pip install -e /app/ACE-Step

# Install app-specific dependencies
COPY requirements.txt /app/requirements.txt
RUN python3.12 -m pip install --no-cache-dir -r /app/requirements.txt

# Copy application entrypoint
COPY mockhandler.py /app/mockhandler.py

CMD ["python3.12", "mockhandler.py"]

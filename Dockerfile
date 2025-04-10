FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

WORKDIR /app

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    HF_HOME=/app/models \
    HY3DGEN_MODELS=/app/models

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    #test
    wget \
    python3-dev \
    build-essential \
    ninja-build \
    pkg-config \
    cmake \
    libegl1-mesa-dev \
    libgl1-mesa-dev \
    libgles2-mesa-dev \
    libosmesa6-dev \
    ffmpeg \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip3 install --no-cache-dir --upgrade pip

# Install Python dependencies
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Install runpod
RUN pip3 install --no-cache-dir runpod

# Pre-download models (do this before copying application code)
COPY download_models.py .
RUN pip3 install --no-cache-dir huggingface_hub && \
    python3 download_models.py && \
    # Create .u2net directory in the image
    mkdir -p /root/.u2net

# Verify model paths are correct
RUN find /app/models -type f | grep safetensors

# Copy application code
COPY rp_handler.py .
COPY test_input.json .

# Start the handler
CMD ["python3", "-u", "rp_handler.py"] 
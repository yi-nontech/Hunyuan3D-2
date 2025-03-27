FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

WORKDIR /app

# Install Python and dependencies
RUN apt-get update && apt-get install -y python3 python3-pip git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements and install them
COPY requirements.txt .
RUN pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu118
RUN pip3 install -r requirements.txt

# Copy the code
COPY . .

# Install the package
RUN pip3 install -e .

# Install custom rasterizer and renderer
RUN cd hy3dgen/texgen/custom_rasterizer && python3 setup.py install && cd ../../..
RUN cd hy3dgen/texgen/differentiable_renderer && python3 setup.py install && cd ../../..

# Create a directory for cache
RUN mkdir -p /app/gradio_cache

# RunPod handler file
COPY handler.py .

# Set the entrypoint
ENTRYPOINT ["python3", "-u", "handler.py"] 
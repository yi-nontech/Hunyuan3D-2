import os
from huggingface_hub import snapshot_download
import urllib.request
import shutil

print("Downloading Hunyuan 3D models...")

# Set model directory
model_dir = os.environ.get('HY3DGEN_MODELS', '/app/models')
os.makedirs(model_dir, exist_ok=True)

# Download shape generation model (main model)
shape_model_path = os.path.join(model_dir, 'tencent/Hunyuan3D-2mini')
os.makedirs(shape_model_path, exist_ok=True)

print(f"Downloading shape model to {shape_model_path}...")
snapshot_download(
    repo_id='tencent/Hunyuan3D-2mini',
    allow_patterns=["hunyuan3d-dit-v2-mini-turbo/*", "hunyuan3d-vae-v2-mini-turbo/*"],
    local_dir=shape_model_path
)

# Download background removal model
u2net_dir = "/root/.u2net"
os.makedirs(u2net_dir, exist_ok=True)
u2net_path = os.path.join(u2net_dir, "u2net.onnx")

if not os.path.exists(u2net_path):
    print(f"Downloading background removal model to {u2net_path}...")
    urllib.request.urlretrieve(
        "https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2net.onnx",
        u2net_path
    )

print("All models downloaded successfully!") 
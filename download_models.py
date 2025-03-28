import os
from huggingface_hub import snapshot_download

print("Downloading Hunyuan 3D models...")

# Set model directory
model_dir = os.environ.get('HY3DGEN_MODELS', '/app/models')
os.makedirs(model_dir, exist_ok=True)

# Download shape generation model
shape_model_path = os.path.join(model_dir, 'tencent/Hunyuan3D-2mini')
os.makedirs(shape_model_path, exist_ok=True)

print(f"Downloading shape model to {shape_model_path}...")
snapshot_download(
    repo_id='tencent/Hunyuan3D-2mini',
    allow_patterns=["hunyuan3d-dit-v2-mini-turbo/*"],
    local_dir=shape_model_path
)

print("Download complete!") 
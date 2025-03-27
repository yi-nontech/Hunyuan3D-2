import base64
import os
import json
import sys
import torch

# Set environment variables directly in the script
os.environ["MODEL_PATH"] = "tencent/Hunyuan3D-2mini"
os.environ["TEX_MODEL_PATH"] = "tencent/Hunyuan3D-2"
os.environ["SUBFOLDER"] = "hunyuan3d-dit-v2-mini-turbo"
os.environ["ENABLE_TEX"] = "true"

# Import just the handler function, not the whole module
sys.path.append('.')  # Make sure current directory is in path
# Import the ModelLoader and other required functions directly
from handler import ModelLoader, load_image_from_base64

# Initialize the model loader
model_loader = ModelLoader()

# Path to your test image
IMAGE_PATH = "assets/demo.png"

# Read and encode the image
with open(IMAGE_PATH, "rb") as image_file:
    encoded_image = base64.b64encode(image_file.read()).decode("utf-8")

# Create input params similar to what handler would receive
params = {
    "image": encoded_image,
    "texture": True,
    "seed": 42,
    "num_inference_steps": 5,
    "guidance_scale": 5.0,
    "octree_resolution": 128,
    "type": "glb"
}

try:
    # Handle image input
    if 'image' in params:
        image = params["image"]
        image = load_image_from_base64(image)
        image = model_loader.rembg(image)
        params['image'] = image
    
    # Generate mesh from image
    seed = params.get("seed", 1234)
    params['generator'] = torch.Generator('cuda').manual_seed(seed)
    params['octree_resolution'] = params.get("octree_resolution", 128)
    params['num_inference_steps'] = params.get("num_inference_steps", 5)
    params['guidance_scale'] = params.get('guidance_scale', 5.0)
    params['mc_algo'] = 'dmc'
    
    mesh = model_loader.pipeline(**params)[0]
    
    # Apply texture if requested
    if params.get('texture', False) and model_loader.enable_tex:
        from hy3dgen.shapegen import FloaterRemover, DegenerateFaceRemover, FaceReducer
        mesh = FloaterRemover()(mesh)
        mesh = DegenerateFaceRemover()(mesh)
        mesh = FaceReducer()(mesh, max_facenum=params.get('face_count', 40000))
        mesh = model_loader.pipeline_tex(mesh, image)
    
    # Save mesh
    output_type = params.get('type', 'glb')
    output_file = f"test_output.{output_type}"
    mesh.export(output_file)
    
    print(f"Successfully generated 3D model: {output_file}")
    
except Exception as e:
    print(f"Error: {str(e)}") 
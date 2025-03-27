import runpod
import os
import base64
import torch
import tempfile
import trimesh
from io import BytesIO
from PIL import Image

from hy3dgen.rembg import BackgroundRemover
from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline, FloaterRemover, DegenerateFaceRemover, FaceReducer
from hy3dgen.texgen import Hunyuan3DPaintPipeline

# Initialize models (load them once at startup)
class ModelLoader:
    def __init__(self):
        # Choose appropriate model path and subfolder based on your needs
        model_path = os.environ.get('MODEL_PATH', 'tencent/Hunyuan3D-2mini')
        tex_model_path = os.environ.get('TEX_MODEL_PATH', 'tencent/Hunyuan3D-2')
        subfolder = os.environ.get('SUBFOLDER', 'hunyuan3d-dit-v2-mini-turbo')
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.rembg = BackgroundRemover()
        self.pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
            model_path,
            subfolder=subfolder,
            use_safetensors=True,
            device=device,
        )
        self.pipeline.enable_flashvdm(mc_algo='mc')
        
        # Load texture model if enabled
        if os.environ.get('ENABLE_TEX', 'false').lower() == 'true':
            self.pipeline_tex = Hunyuan3DPaintPipeline.from_pretrained(tex_model_path)
            self.enable_tex = True
        else:
            self.enable_tex = False

model_loader = ModelLoader()

def load_image_from_base64(image):
    return Image.open(BytesIO(base64.b64decode(image)))

def handler(event):
    try:
        # Extract parameters
        params = event["input"]
        
        # Handle image input
        if 'image' in params:
            image = params["image"]
            image = load_image_from_base64(image)
            image = model_loader.rembg(image)
            params['image'] = image

        # Handle mesh
        if 'mesh' in params:
            mesh = trimesh.load(BytesIO(base64.b64decode(params["mesh"])), file_type='glb')
        else:
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
            mesh = FloaterRemover()(mesh)
            mesh = DegenerateFaceRemover()(mesh)
            mesh = FaceReducer()(mesh, max_facenum=params.get('face_count', 40000))
            mesh = model_loader.pipeline_tex(mesh, image)

        # Save mesh and return as base64
        output_type = params.get('type', 'glb')
        with tempfile.NamedTemporaryFile(suffix=f'.{output_type}', delete=False) as temp_file:
            mesh.export(temp_file.name)
            with open(temp_file.name, 'rb') as f:
                mesh_data = base64.b64encode(f.read()).decode('utf-8')
            
            # Clean up
            os.unlink(temp_file.name)
        
        # Clear CUDA cache
        torch.cuda.empty_cache()
        
        return {
            "model_base64": mesh_data,
            "output_type": output_type
        }
        
    except Exception as e:
        return {"error": str(e)}

# Start the RunPod serverless handler
runpod.serverless.start({"handler": handler})
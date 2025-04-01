import runpod
import base64
import json
import os
import tempfile
import torch
import trimesh
import time
from io import BytesIO
from PIL import Image

from hy3dgen.rembg import BackgroundRemover
from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline, FloaterRemover, DegenerateFaceRemover, FaceReducer

# Global variables to store loaded models
pipeline = None
rembg = None
device = "cuda"

print("Starting handler initialization...")

print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA device count: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"CUDA device name: {torch.cuda.get_device_name(0)}")

# At initialization
# Explicitly set paths to pre-downloaded models
os.environ['HY3DGEN_MODELS'] = '/app/models'
os.environ['U2NET_HOME'] = '/root/.u2net'

def load_models():
    global pipeline, rembg
    
    print("Loading models from pre-downloaded files...")
    # Load background remover with explicit path
    rembg = BackgroundRemover()
    
    # Initialize pipeline with local models
    pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
        'tencent/Hunyuan3D-2mini',
        subfolder='hunyuan3d-dit-v2-mini-turbo',
        use_safetensors=True,
        device=device,
        local_files_only=True  # Force local files only
    )
    pipeline.enable_flashvdm(mc_algo='mc', topk_mode='merge')
    
    # Pre-compile common CUDA operations
    print("Pre-warming CUDA operations...")
    dummy_input = torch.randn(1, 3, 256, 256, device=device)
    _ = dummy_input + dummy_input  # Force CUDA initialization
    torch.cuda.synchronize()
    
    print("Models loaded successfully!")
    print(f"Models loaded. Pipeline type: {type(pipeline)}")

def load_image_from_base64(image_data):
    try:
        # Add padding if needed
        padding = len(image_data) % 4
        if padding:
            image_data += '=' * (4 - padding)
        
        return Image.open(BytesIO(base64.b64decode(image_data)))
    except Exception as e:
        print(f"Error decoding base64 image: {str(e)}")
        raise

def handler(event):
    global pipeline, rembg
    
    # Initialize models if not already done
    if pipeline is None or rembg is None:
        load_models()
    
    # Get input from the event
    job_input = event["input"]
    
    print(f"Received job input: {job_input.keys()}")
    
    # Extract parameters from the job input
    params = {}
    temp_file_path = None
    
    try:
        # Handle image input
        if 'image' in job_input:
            if job_input["image"].startswith("data:image"):
                print(f"Handling data URI image")
                # Handle data URI format
                image_data = job_input["image"].split(",")[1]
                image = load_image_from_base64(image_data)
            else:
                print(f"Handling base64 image")
                # Regular base64
                image = load_image_from_base64(job_input["image"])
            
            # Simply apply background removal without checking mode
            print(f"Applying background removal to image")
            image = rembg(image)
            params['image'] = image
        elif 'image_path' in job_input and os.path.exists(job_input['image_path']):
            print(f"Loading image from path: {job_input['image_path']}")
            # For local testing, support direct file path
            image = Image.open(job_input['image_path'])
            
            # Simply apply background removal without checking mode or conversion
            print(f"Applying background removal to image from path")
            image = rembg(image)
            params['image'] = image
        else:
            return {"error": "No valid input image provided"}
        
        # Process optional parameters
        seed = job_input.get("seed", 1234)
        params['generator'] = torch.manual_seed(seed)
        params['octree_resolution'] = job_input.get("octree_resolution", 256)
        params['num_inference_steps'] = job_input.get("num_inference_steps", 30)
        params['guidance_scale'] = job_input.get('guidance_scale', 7.5)
        params['num_chunks'] = job_input.get('num_chunks', 20000)
        params['output_type'] = 'trimesh'
        
        # Generate 3D mesh
        print("Generating 3D mesh...")
        mesh = pipeline(**params)[0]
        print("Mesh generation complete")
        
        # Apply post-processing
        if job_input.get('post_process', True):
            print("Applying post-processing...")
            mesh = FloaterRemover()(mesh)
            mesh = DegenerateFaceRemover()(mesh)
            # Optional face reduction
            if job_input.get('reduce_faces', False):
                mesh = FaceReducer()(mesh, max_facenum=job_input.get('face_count', 40000))
        
        # Export the mesh to a temporary file
        output_format = job_input.get('type', 'glb')
        print(f"Exporting mesh as {output_format}...")
        
        # Create a unique filename for the temp file
        temp_dir = tempfile.gettempdir()
        temp_file_path = os.path.join(temp_dir, f"mesh_{os.getpid()}_{int(time.time())}.{output_format}")
        
        # Export to the temporary file
        mesh.export(temp_file_path)
        
        # Read the file and convert to base64
        with open(temp_file_path, 'rb') as f:
            mesh_data = base64.b64encode(f.read()).decode('utf-8')
        
        # Return the results
        return {
            "model_base64": mesh_data,
            "format": output_format,
            "seed": seed,
            "vertices": len(mesh.vertices),
            "faces": len(mesh.faces)
        }
    
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Error generating mesh: {str(e)}\n{error_details}")
        return {"error": str(e), "traceback": error_details}
    
    finally:
        # Clean up temporary files in the finally block
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                # Give a small delay to ensure file handles are closed (Windows specific)
                time.sleep(0.5)
                os.remove(temp_file_path)
                print(f"Temporary file {temp_file_path} deleted successfully")
            except Exception as e:
                print(f"Warning: Could not delete temporary file {temp_file_path}: {str(e)}")
                # Non-critical error, proceed anyway

# Start the RunPod serverless handler
if __name__ == "__main__":
    runpod.serverless.start({"handler": handler}) 
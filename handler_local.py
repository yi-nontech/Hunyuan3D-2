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
    # Same implementation as before
    # ...

model_loader = ModelLoader()

def load_image_from_base64(image):
    return Image.open(BytesIO(base64.b64decode(image)))

def handler(event):
    # Same implementation as before
    # ...

# No runpod.serverless.start call here 
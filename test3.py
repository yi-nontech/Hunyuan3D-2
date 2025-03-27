from hy3dgen.texgen import Hunyuan3DPaintPipeline
from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
from trimesh import Trimesh

# Text-to-3D generation

# pipeline:Hunyuan3DDiTFlowMatchingPipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained('tencent/Hunyuan3D-2mini', subfolder='hunyuan3d-dit-v2-mini-turbo')
text_pipeline: Hunyuan3DPaintPipeline = Hunyuan3DPaintPipeline.from_pretrained('tencent/Hunyuan3D-2')

# Replace this text with your description
text_prompt = "a cute cat sitting on a couch"

# Generate the 3D model from text
text_mesh: Trimesh = text_pipeline(prompt=text_prompt)[0]
text_mesh.export('export/text_to_3d.glb')

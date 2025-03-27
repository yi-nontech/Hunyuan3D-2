from hy3dgen.texgen import Hunyuan3DPaintPipeline
from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
from trimesh import Trimesh

# let's generate a mesh first
pipeline:Hunyuan3DDiTFlowMatchingPipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained('tencent/Hunyuan3D-2mini', subfolder='hunyuan3d-dit-v2-mini-turbo')
mesh: Trimesh = pipeline(image='assets/demo.png')[0]
mesh.export('export/test2.glb')

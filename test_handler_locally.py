import base64
import os
import json
from handler import handler

# Path to your test image
IMAGE_PATH = "assets/demo.png"

# Read and encode the image
with open(IMAGE_PATH, "rb") as image_file:
    encoded_image = base64.b64encode(image_file.read()).decode("utf-8")

# Create a test event similar to what RunPod would send
test_event = {
    "input": {
        "image": encoded_image,
        "texture": True,  # Set to True to test texture generation
        "seed": 42,
        "num_inference_steps": 5,
        "guidance_scale": 5.0,
        "octree_resolution": 128,
        "type": "glb"
    }
}

# Call the handler function
result = handler(test_event)

# Check for errors
if "error" in result:
    print(f"Error: {result['error']}")
else:
    # Save the output to a file
    output_file = "test_output.glb"
    with open(output_file, "wb") as f:
        f.write(base64.b64decode(result["model_base64"]))
    
    print(f"Successfully generated 3D model: {output_file}")
    print(f"Output type: {result['output_type']}") 
import os
import json
import sys
import base64  # Add base64 import
from datetime import datetime
print("Python version:", sys.version)
print("Current directory:", os.getcwd())
print("Files in directory:", os.listdir("."))

# Check if test_input.json exists
if os.path.exists("test_input.json"):
    print("test_input.json found")
    with open("test_input.json", "r") as f:
        test_input = json.load(f)
        print("Loaded test input:", test_input.keys())
else:
    print("test_input.json NOT found")
    test_input = {
        "input": {
            "image_path": "https://media.sketchfab.com/models/644b113218b94c94b5f07be0b6f455a3/thumbnails/c21a2bf1dfee4819a7ababa451f10447/c74f778e43d546778162aa139d763689.jpeg",
            "seed": 1234,
            "octree_resolution": 64,  # Lower for quicker testing
            "num_inference_steps": 3,  # Lower for quicker testing
        }
    }

print("\n--- Starting direct handler test ---\n")

# Import handler directly
try:
    from rp_handler import handler
    print("Handler imported successfully")
    
    # Test image path existence
    image_path = test_input["input"].get("image_path")
    if image_path and os.path.exists(image_path):
        print(f"Image path exists: {image_path}")
    else:
        print(f"WARNING: Image path does not exist: {image_path}")
        
    # Call the handler directly
    print("Calling handler...")
    result = handler({"input": test_input["input"]})
    print("\nTest complete!")
    print("Result keys:", result.keys() if isinstance(result, dict) else "Not a dictionary")
    
    # Save the result mesh to a GLB file
    if isinstance(result, dict) and "model_base64" in result:
        try:
            # Decode the base64 string
            mesh_data = base64.b64decode(result["model_base64"])
            
            # Save to file
            output_file = "export/rp_handler_local_test_"+datetime.now().strftime("%Y%m%d_%H%M%S")+".glb"
            with open(output_file, "wb") as f:
                f.write(mesh_data)
            
            print(f"Mesh saved to: {os.path.abspath(output_file)}")
            print(f"Mesh info: {result.get('vertices', 'N/A')} vertices, {result.get('faces', 'N/A')} faces")
        except Exception as e:
            print(f"Error saving mesh: {str(e)}")
    else:
        print("No mesh data found in result")
    
except Exception as e:
    import traceback
    print(f"Error during test: {str(e)}")
    traceback.print_exc() 
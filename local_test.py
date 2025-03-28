import os
import json
import sys
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
            "image_path": "assets/demo.png",
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
    
except Exception as e:
    import traceback
    print(f"Error during test: {str(e)}")
    traceback.print_exc() 
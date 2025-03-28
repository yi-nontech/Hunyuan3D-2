import requests
import json
import time
import base64
from datetime import datetime
import os 

# Your RunPod API key and endpoint ID
# API_KEY = "rpa_PM2YUPP7X0TPKGFSCT8ZHGDTKTVXJIA31FO092L8vzo7db" // Read
API_KEY = "rpa_R4J3MFU9A6OBEU9A7C22J7I1ZHYROWQ5R4PYBBMFbmbe18"
ENDPOINT_ID = "gu4fk0x5f7m0iv"

# Path to your test image
IMAGE_PATH = "assets/demo3.jpeg"

# Read and encode the image
with open(IMAGE_PATH, "rb") as image_file:
    encoded_image = base64.b64encode(image_file.read()).decode("utf-8")

# Simple test request 
test_payload = {
    "input": {
        "image": encoded_image
        # "image_path":"https://media.sketchfab.com/models/644b113218b94c94b5f07be0b6f455a3/thumbnails/c21a2bf1dfee4819a7ababa451f10447/c74f778e43d546778162aa139d763689.jpeg"
    }
}

# Start timer
start_time = time.time()

# Send the request
print("Sending test request...")
response = requests.post(
    f"https://api.runpod.ai/v2/{ENDPOINT_ID}/run",
    headers={
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    },
    json=test_payload
)

print(f"Response status: {response.status_code}")
# print(f"Response content: {response.json()}")

# If successful, get job ID and poll for result
if response.status_code == 200:
    job_id = response.json().get("id")
    print(f"Job ID: {job_id}")
    
    # Poll for the result
    while True:
        status_response = requests.get(
            f"https://api.runpod.ai/v2/{ENDPOINT_ID}/status/{job_id}",
            headers={"Authorization": f"Bearer {API_KEY}"}
        )
        
        status = status_response.json()
        print(f"Current status: {status.get('status')}")
        
        if status.get("status") == "COMPLETED":
            print("Job completed!")
            output = status.get('output')
            print(f"Result: {output.get('vertices')}")
            
            # Save the result mesh to a GLB file
            if isinstance(output, dict) and "model_base64" in output:
                try:
                    # Decode the base64 string
                    mesh_data = base64.b64decode(output["model_base64"])
                    
                    # Save to file
                    output_file = "export/runpod_api_test_" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".glb"
                    with open(output_file, "wb") as f:
                        f.write(mesh_data)
                    
                    print(f"Mesh saved to: {os.path.abspath(output_file)}")
                    print(f"Mesh info: {output.get('vertices', 'N/A')} vertices, {output.get('faces', 'N/A')} faces")
                except Exception as e:
                    print(f"Error saving mesh: {str(e)}")
            else:
                print("No mesh data found in result")
            
            elapsed_time = time.time() - start_time
            print(f"Total time elapsed: {elapsed_time:.2f} seconds")
            break
        
        elif status.get("status") == "FAILED":
            print("Job failed!")
            print(f"Error: {status.get('error')}")
            elapsed_time = time.time() - start_time
            print(f"Total time elapsed: {elapsed_time:.2f} seconds")
            break
            
        time.sleep(5)  # Check every 5 seconds
else:
    print("Failed to send the initial request.")
    elapsed_time = time.time() - start_time
    print(f"Total time elapsed: {elapsed_time:.2f} seconds")

import requests
import json
import time
import base64
# Your RunPod API key and endpoint ID
API_KEY = "YOUR_RUNPOD_API_KEY"
ENDPOINT_ID = "YOUR_ENDPOINT_ID"


# Path to your test image
IMAGE_PATH = "assets/demo.png"

# Read and encode the image
with open(IMAGE_PATH, "rb") as image_file:
    encoded_image = base64.b64encode(image_file.read()).decode("utf-8")

# Simple test request 
test_payload = {
    "input": {
        "image": encoded_image
    }
}



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
print(f"Response content: {response.json()}")

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
            print(f"Result: {status.get('output')}")
            break
        elif status.get("status") == "FAILED":
            print("Job failed!")
            print(f"Error: {status.get('error')}")
            break
            
        time.sleep(5)  # Check every 5 seconds
import runpod

def handler(event):
    return {"status": "success", "message": "Hello from handler!"}

if __name__ == "__main__":
    runpod.serverless.start({"handler": handler}) 
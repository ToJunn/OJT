import numpy as np
import httpx
import cv2
from fastapi import FastAPI, File, UploadFile
from typing import Dict
from PIL import Image
import io

# Initialize FastAPI
app = FastAPI()

# Triton Server details
TRITON_SERVER_URL = "http://localhost:8000/v2/models/resnet50/infer"

# Preprocess function for ResNet-50
def preprocess_image(image: bytes) -> np.ndarray:
    image = Image.open(io.BytesIO(image)).convert("RGB")
    image = image.resize((224, 224))  # Resize to match ResNet-50 input
    image = np.array(image).astype(np.float32) / 255.0  # Normalize
    image = np.transpose(image, (2, 0, 1))  # Change shape to (3, 224, 224)
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> Dict:
    # Read file and preprocess
    image_bytes = await file.read()
    input_data = preprocess_image(image_bytes)

    # Triton inference request
    payload = {
        "inputs": [{
            "name": "data",
            "shape": input_data.shape,
            "datatype": "FP32",
            "data": input_data.tolist()
        }]
    }

    async with httpx.AsyncClient() as client:
        response = await client.post(TRITON_SERVER_URL, json=payload)

    if response.status_code != 200:
        return {"error": "Inference failed", "details": response.text}

    # Parse response
    result = response.json()
    predictions = result["outputs"][0]["data"]
    predicted_class = int(np.argmax(predictions))

    return {"predicted_class": predicted_class}

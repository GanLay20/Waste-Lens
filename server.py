#!/usr/bin/env python3
"""
WasteLens FastAPI Server - Serves a PyTorch waste classification model.
"""
import io
from pathlib import Path

import torch
import torch.nn as nn
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
from starlette.staticfiles import StaticFiles
from torchvision import models, transforms

# --- Configuration ---
MODEL_PATH = Path("model/best_model.pth")
LABELS = ["Organic", "Recyclable"]
IMAGE_SIZE = 224
CONFIDENCE_THRESHOLD = 0.65

# --- App Setup ---
app = FastAPI(title="WasteLens API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# --- Model Loading ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, len(LABELS))
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

infer_transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.CenterCrop(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
print(f"Model loaded successfully on device: {device}")


# --- API Routes ---
@app.post("/predict")
@torch.no_grad()
async def predict(file: UploadFile = File(...)):
    """Receives an image, classifies it, and returns the prediction."""
    try:
        # Read and validate the image from the uploaded file
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file.")

    # Transform the image and prepare it for the model
    tensor = infer_transform(image).unsqueeze(0).to(device)

    # Get model predictions
    logits = model(tensor)
    probabilities = torch.nn.functional.softmax(logits, dim=1).squeeze().cpu().numpy()
    
    # Format the prediction
    top_prob = float(probabilities.max())
    top_idx = int(probabilities.argmax())
    pred_label = LABELS[top_idx]
    is_unsure = top_prob < CONFIDENCE_THRESHOLD
    
    # Create a list of probabilities for each label
    all_probs = {LABELS[i]: float(prob) for i, prob in enumerate(probabilities)}

    return JSONResponse({
        "prediction": pred_label,
        "confidence": top_prob,
        "is_unsure": is_unsure,
        "probabilities": all_probs,
    })

# Serve the static front-end
app.mount("/", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

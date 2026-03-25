from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import os
import gdown

MODEL_FILE = "final_mental_health_model/model.safetensors"

if not os.path.exists(MODEL_FILE):
    print("Downloading model...")
    url = "https://drive.google.com/uc?id=YOUR_FILE_ID"
    gdown.download(url, MODEL_FILE, quiet=False)

app = FastAPI()

# 📦 Load model
model_path = "final_mental_health_model"

tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

labels = [
    "anxiety",
    "stress",
    "depression",
    "normal",
    "suicidal",
    "bipolar",
    "personality disorder"
]

# 📥 Request format
class InputText(BaseModel):
    text: str

@app.post("/predict")
def predict(data: InputText):
    inputs = tokenizer(data.text, return_tensors="pt", truncation=True, padding=True)

    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.softmax(outputs.logits, dim=1)
    pred = torch.argmax(probs, dim=1).item()

    return {
        "label": labels[pred],
        "confidence": float(probs[0][pred])
    }
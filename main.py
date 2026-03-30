import torch
import numpy as np
import shap
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load your trained model
model_path = "mental_health_model"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# Labels (must match training)
label_cols = ["Anxiety", "Stress", "Depression", "Suicidal", "Bipolar"]


# OPTIONAL: Long text chunking (for prediction)
def split_text(text, tokenizer, max_length=512, stride=50):
    inputs = tokenizer(
        text,
        return_overflowing_tokens=True,
        max_length=max_length,
        stride=stride,
        truncation=True
    )
    
    chunks = []
    for ids in inputs["input_ids"]:
        chunks.append(tokenizer.decode(ids, skip_special_tokens=True))
    
    return chunks


def predict_long_text(text):
    chunks = split_text(text, tokenizer)
    
    all_probs = []
    
    for chunk in chunks:
        inputs = tokenizer(
            chunk,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512
        ).to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        probs = torch.sigmoid(outputs.logits)
        all_probs.append(probs.cpu().numpy())
    
    # Use MAX to capture strongest signal
    return np.max(all_probs, axis=0)[0]


# Main prediction function
def predict(text, threshold=0.5, min_conf=0.3):
    
    probs = predict_long_text(text)
    
    predicted_labels = [
        label_cols[i] for i, p in enumerate(probs) if p > threshold
    ]
    
    if len(predicted_labels) == 0 or max(probs) < min_conf:
        return {
            "labels": ["Normal"],
            "probabilities": dict(zip(label_cols, probs))
        }
    
    return {
        "labels": predicted_labels,
        "probabilities": dict(zip(label_cols, probs))
    }


# =========================
# SHAP SECTION
# =========================

# SHAP prediction function (NO chunking here)
def predict_fn_shap(texts):
    results = []
    
    for text in texts:
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512
        ).to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        probs = torch.sigmoid(outputs.logits).cpu().numpy()[0]
        results.append(probs)
    
    return np.array(results)


# Initialize SHAP (once)
explainer = shap.Explainer(predict_fn_shap, tokenizer)


# Explain function
def explain(text):
    print("\n SHAP Explanation:")
    
    # SHAP works best on shorter text
    text = text[:512]
    
    shap_values = explainer([text])
    html = shap.plots.text(shap_values[0], display=False)

    with open("shap_output.html", "w") as f:
        f.write(html)

    print("SHAP saved to shap_output.html")


# Combined function (prediction + explanation)
def predict_with_explanation(text):
    
    result = predict(text)
    
    print("\n==============================")
    print(f"Text: {text}")
    print(f"Predicted Labels: {result['labels']}")
    
    print("Probabilities:")
    for k, v in result["probabilities"].items():
        print(f"  {k}: {v:.3f}")
    
    # Add explanation
    explain(text)


# =========================
# TEST EXAMPLES
# =========================

texts = [
    "I feel happy and satisfied",
    "I feel empty and anxious",
    "I want to die and end my life"
]

for text in texts:
    predict_with_explanation(text)

import torch
import torch.nn.functional as F
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from pathlib import Path

# --------------------------------------------------
# Load model + tokenizer ONCE
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[2]
MODEL_PATH = BASE_DIR / "model" / "final_model"

tokenizer = DistilBertTokenizer.from_pretrained(
    str(MODEL_PATH),
    local_files_only=True
)

model = DistilBertForSequenceClassification.from_pretrained(
    str(MODEL_PATH),
    local_files_only=True
)

model.eval()

# --------------------------------------------------
# Classification with confidence
# --------------------------------------------------

def classify_news(text: str):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=256
    )

    with torch.no_grad():
        outputs = model(**inputs)

    probs = F.softmax(outputs.logits, dim=1)[0]
    confidence = float(torch.max(probs).item()) * 100
    pred = torch.argmax(probs).item()

    label = "Real" if pred == 1 else "Fake"
    return label, round(confidence, 2)

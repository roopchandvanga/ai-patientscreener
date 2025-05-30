
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import torch


tokenizer = DistilBertTokenizerFast.from_pretrained("roopchand/bert-smoking-classifier")
model = DistilBertForSequenceClassification.from_pretrained("roopchand/bert-smoking-classifier")



def classify_smoking_status(text: str) -> str:
    text_lower = text.lower()

    # Fallback rule-based overrides for high-confidence non-smoker
    if any(phrase in text_lower for phrase in [
        "non-smoker", "never smoked", "no tobacco", "denies smoking", "no history of smoking"
    ]):
        return "non-smoker"

    # Use BERT model
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    predicted_class = torch.argmax(probs, dim=1).item()

    return "smoker" if predicted_class == 1 else "non-smoker"

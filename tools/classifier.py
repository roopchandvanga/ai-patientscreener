import joblib
import os

# Lazy load the model once
_model = None

def classify_smoking_status(text):
    global _model
    if _model is None:
        model_path = r'C:\Users\roopc\Desktop\Projects\ai-patienteligibility\tools\models\lifestyle_classifier.pkl'
        if not os.path.exists(model_path):
            raise FileNotFoundError("Model not found. Run train_classifier.py first.")
        _model = joblib.load(model_path)

    return _model.predict([text])[0]

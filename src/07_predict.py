import joblib
import pandas as pd
from utils import MODELS_DIR
from readability import get_readability_metrics
from completeness import detect_elements

def predict_quality(text: str):
    model = joblib.load(MODELS_DIR / "quality_model.joblib")
    encoder = joblib.load(MODELS_DIR / "label_encoder.joblib")

    X = pd.DataFrame({"instruction_text": [text]})
    pred = model.predict(X)[0]
    label = encoder.inverse_transform([pred])[0]

    return {
        "quality_prediction": label,
        "readability": get_readability_metrics(text),
        "completeness": detect_elements(text)
    }

if __name__ == "__main__":
    sample = "Take your antibiotics as prescribed. Drink fluids. Follow up with your doctor in one week. Seek urgent care if breathing gets worse."
    print(predict_quality(sample))
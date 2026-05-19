import joblib
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from utils import SYNTH_DIR, MODELS_DIR

def main():
    df = pd.read_csv(SYNTH_DIR / "synthetic_discharge_instructions.csv")
    model = joblib.load(MODELS_DIR / "quality_model.joblib")
    encoder = joblib.load(MODELS_DIR / "label_encoder.joblib")

    y_true = encoder.transform(df["quality_label"])
    preds = model.predict(df[["instruction_text"]])

    print(classification_report(y_true, preds, target_names=encoder.classes_))
    print(confusion_matrix(y_true, preds))

if __name__ == "__main__":
    main()
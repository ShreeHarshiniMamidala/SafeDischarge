import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from utils import SYNTH_DIR, MODELS_DIR

def main():
    df = pd.read_csv(SYNTH_DIR / "synthetic_discharge_instructions.csv")

    X = df[["instruction_text"]]
    y = df["quality_label"]

    encoder = LabelEncoder()
    y_enc = encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_enc,
        test_size=0.3,
        random_state=42
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("text", TfidfVectorizer(ngram_range=(1, 2)), "instruction_text")
        ]
    )

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000))
    ])

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print(classification_report(y_test, y_pred, zero_division=0))

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODELS_DIR / "quality_model.joblib")
    joblib.dump(encoder, MODELS_DIR / "label_encoder.joblib")

    print("Saved model files in:", MODELS_DIR)

if __name__ == "__main__":
    main()
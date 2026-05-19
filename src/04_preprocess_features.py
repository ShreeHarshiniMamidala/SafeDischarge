import pandas as pd
from utils import SYNTH_DIR, PROCESSED_DIR
from readability import get_readability_metrics
from completeness import detect_elements

def main():
    df = pd.read_csv(SYNTH_DIR / "synthetic_discharge_instructions.csv")
    read_df = df["instruction_text"].apply(lambda x: pd.Series(get_readability_metrics(x)))
    comp_df = df["instruction_text"].apply(lambda x: pd.Series(detect_elements(x)))
    out_df = pd.concat([df, read_df, comp_df], axis=1)
    out_path = PROCESSED_DIR / "discharge_features.csv"
    out_df.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")

if __name__ == "__main__":
    main()
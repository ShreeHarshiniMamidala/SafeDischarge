import pandas as pd
from utils import RAW_DIR, PROCESSED_DIR

KEYWORDS = {
    "medications": ["medication", "medicine", "drug", "prescribed", "take"],
    "followup": ["follow up", "follow-up", "appointment", "doctor", "clinic"],
    "warning_signs": ["call 911", "emergency", "seek care", "worsens", "trouble breathing", "chest pain"],
    "test_results": ["test result", "lab result", "x-ray", "scan", "blood test"],
    "home_care": ["rest", "drink fluids", "at home", "avoid", "monitor"]
}

def detect_topics(text):
    lowered = text.lower()
    return {k: int(any(term in lowered for term in terms)) for k, terms in KEYWORDS.items()}

def main():
    records = []
    for txt_file in RAW_DIR.glob("*.txt"):
        text = txt_file.read_text(encoding="utf-8", errors="ignore")
        record = {
            "file_name": txt_file.name,
            "text_length": len(text),
        }
        record.update(detect_topics(text))
        records.append(record)

    df = pd.DataFrame(records)
    out_path = PROCESSED_DIR / "public_corpus_index.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")

if __name__ == "__main__":
    main()
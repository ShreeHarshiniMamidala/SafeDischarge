import textstat

def get_readability_metrics(text: str) -> dict:
    text = text or ""
    return {
        "flesch_reading_ease": textstat.flesch_reading_ease(text),
        "flesch_kincaid_grade": textstat.flesch_kincaid_grade(text),
        "gunning_fog": textstat.gunning_fog(text),
        "smog_index": textstat.smog_index(text),
        "sentence_count": textstat.sentence_count(text),
        "word_count": textstat.lexicon_count(text, removepunct=True),
    }
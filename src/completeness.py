KEYWORDS = {
    "pred_has_medication_review": ["take", "medicine", "medication", "pill", "antibiotic", "insulin", "prescribed"],
    "pred_has_followup": ["follow up", "follow-up", "doctor", "clinic", "appointment", "one week", "within 7 days"],
    "pred_has_warning_signs": ["emergency room", "urgent care", "call 911", "trouble breathing", "chest pain", "worsening"],
    "pred_has_home_care": ["rest", "drink fluids", "limit salt", "weigh yourself", "at home", "avoid"],
    "pred_has_test_results": ["test result", "lab result", "blood test", "x-ray", "scan"]
}

def detect_elements(text: str) -> dict:
    lowered = (text or "").lower()
    return {label: int(any(term in lowered for term in terms)) for label, terms in KEYWORDS.items()}
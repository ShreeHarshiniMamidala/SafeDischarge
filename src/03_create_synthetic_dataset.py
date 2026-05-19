import pandas as pd
from utils import SYNTH_DIR

conditions = {
    "heart_failure": {
        "high": [
            "You were treated for heart failure. Take your medicines exactly as prescribed. Weigh yourself every morning before breakfast. Limit salt. Follow up with cardiology within 7 days. Go to the emergency room for chest pain, fainting, or severe shortness of breath.",
            "Your heart failure symptoms improved. Keep taking your water pill and heart medicines. Check your weight every day and call your doctor if you gain more than 2 pounds in a day. Follow up in one week. Get urgent help for trouble breathing or chest pain.",
            "Take your heart failure medicines every day. Limit salt and fluids as instructed. Weigh yourself each morning and write it down. Follow up with your heart doctor next week. Seek emergency care for severe breathing trouble or chest pain."
        ],
        "medium": [
            "Take your heart medicines as prescribed. Limit salt and follow up with your doctor next week. Call if symptoms get worse.",
            "Continue your medicines and watch for swelling or breathing problems. Make a follow-up appointment soon.",
            "Take your medications every day and try to limit salt. Contact your doctor if swelling or shortness of breath increases."
        ],
        "low": [
            "Stable for discharge. Continue meds. Follow up as needed.",
            "Discharge home today. Resume treatment plan.",
            "Heart failure improved. Continue usual care."
        ]
    },
    "pneumonia": {
        "high": [
            "You were treated for pneumonia. Finish all antibiotics. Rest and drink plenty of fluids. Follow up with your doctor in one week. Seek urgent care for worsening fever, chest pain, or trouble breathing.",
            "Take the full antibiotic course even if you feel better. Use your inhaler if prescribed. Drink fluids and rest. Call your doctor if your cough worsens or you have difficulty breathing.",
            "You are going home after pneumonia treatment. Take your medicines exactly as directed. Rest, drink water, and follow up with your doctor within 7 days. Get emergency help for severe breathing problems."
        ],
        "medium": [
            "Take antibiotics as directed. Rest and drink fluids. Follow up with your doctor.",
            "Complete your medicine and contact your doctor if symptoms worsen.",
            "Continue treatment at home, get rest, and call if breathing gets worse."
        ],
        "low": [
            "Pneumonia improved. Discharge today.",
            "Go home and continue treatment.",
            "Patient stable for home discharge."
        ]
    },
    "diabetes": {
        "high": [
            "Check your blood sugar before meals and at bedtime. Take insulin exactly as prescribed. Eat regular meals. Follow up with your doctor within one week. Call if your blood sugar is repeatedly low or above your target range.",
            "You were treated for diabetes-related problems. Continue insulin and diabetes pills as directed. Watch for sweating, shaking, confusion, or very high sugar readings. Schedule follow-up in 7 days.",
            "Monitor your blood sugar closely at home. Take insulin and medicines exactly as instructed. Follow up with clinic next week. Get help right away for severe low sugar, confusion, or vomiting."
        ],
        "medium": [
            "Take insulin as prescribed and monitor your blood sugar. Follow up with clinic next week.",
            "Check sugars regularly and call your doctor for low blood sugar.",
            "Continue diabetes medications and watch your blood sugar at home."
        ],
        "low": [
            "Diabetes instructions given. Continue medications.",
            "Resume diabetes care at home.",
            "Stable for discharge with diabetes treatment."
        ]
    },
    "post_surgery": {
        "high": [
            "Keep your incision clean and dry. Take pain medicine only as directed. Do not lift heavy objects for 2 weeks. Follow up with surgery clinic in 7 days. Go to the emergency room for fever, heavy bleeding, or worsening pain.",
            "You had surgery and are ready to go home. Walk a little each day. Watch for redness, swelling, drainage, or fever. Follow up with your surgeon next week.",
            "Take your pain medicine only if needed. Keep the wound dry and avoid heavy lifting. Call your surgeon for drainage, redness, fever, or worsening pain. Follow up in one week."
        ],
        "medium": [
            "Keep the wound clean and take pain medicine if needed. Follow up with surgery clinic.",
            "Avoid heavy lifting and call if pain worsens.",
            "Take care of the incision and arrange follow-up soon."
        ],
        "low": [
            "Post-op instructions provided. Activity as tolerated.",
            "Discharge home after surgery.",
            "Surgery recovery stable. Resume routine as able."
        ]
    },
    "copd": {
        "high": [
            "Use your inhalers exactly as prescribed. Finish your steroid or antibiotic if one was ordered. Avoid smoke exposure. Follow up with your doctor in one week. Go to urgent care for worsening shortness of breath or blue lips.",
            "You were treated for a COPD flare. Rest, use inhalers correctly, and avoid smoking. Seek help right away for severe breathing trouble. Make a follow-up appointment within 7 days.",
            "Continue your inhalers and breathing treatments at home. Do not smoke. Call your doctor if your cough or breathing worsens. Get emergency help for severe shortness of breath."
        ],
        "medium": [
            "Use inhalers as directed and follow up with your doctor. Call if breathing gets worse.",
            "Continue breathing treatments and avoid smoking.",
            "Take your medicines and monitor your breathing at home."
        ],
        "low": [
            "COPD stable for discharge.",
            "Return home and continue care.",
            "Patient improved and ready for discharge."
        ]
    }
}

def infer_flags(text):
    t = text.lower()
    return {
        "has_medication_review": int(any(x in t for x in [
            "medicine", "medicines", "medication", "medications", "antibiotic",
            "insulin", "inhaler", "pain medicine", "pill", "water pill", "steroid"
        ])),
        "has_followup": int(any(x in t for x in [
            "follow up", "follow-up", "appointment", "clinic", "doctor", "surgeon",
            "within 7 days", "next week", "one week"
        ])),
        "has_warning_signs": int(any(x in t for x in [
            "emergency", "urgent", "call", "trouble breathing", "chest pain",
            "fever", "bleeding", "worsening", "confusion", "blue lips"
        ])),
        "has_home_care": int(any(x in t for x in [
            "rest", "drink", "limit salt", "walk", "avoid", "clean", "dry",
            "weigh yourself", "monitor", "write it down"
        ])),
        "has_test_results": int(any(x in t for x in [
            "test result", "lab result", "x-ray", "scan", "blood test"
        ]))
    }

def main():
    rows = []
    idx = 1

    label_counts = {"high": 34, "medium": 33, "low": 33}
    generated = {"high": 0, "medium": 0, "low": 0}

    ordered_conditions = list(conditions.keys())

    while sum(generated.values()) < 100:
        for condition in ordered_conditions:
            for label in ["high", "medium", "low"]:
                if generated[label] >= label_counts[label]:
                    continue

                text_list = conditions[condition][label]
                text = text_list[generated[label] % len(text_list)]

                row = {
                    "instruction_id": f"ROW_{idx:03d}",
                    "condition": condition,
                    "source_type": f"synthetic_{label}",
                    "instruction_text": text,
                    "quality_label": label
                }
                row.update(infer_flags(text))
                rows.append(row)

                idx += 1
                generated[label] += 1

                if sum(generated.values()) >= 100:
                    break
            if sum(generated.values()) >= 100:
                break

    df = pd.DataFrame(rows)
    out_path = SYNTH_DIR / "synthetic_discharge_instructions.csv"
    df.to_csv(out_path, index=False)

    print(f"Saved {len(df)} rows to {out_path}")
    print("\nLabel distribution:")
    print(df["quality_label"].value_counts())
    print("\nCondition distribution:")
    print(df["condition"].value_counts())

if __name__ == "__main__":
    main()
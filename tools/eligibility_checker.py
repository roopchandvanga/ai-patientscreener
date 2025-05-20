import json
from tools.ner import extract_conditions
from tools.classifier import classify_smoking_status

def check_eligibility(note, criteria):
    reasons = []
    eligible = True

    conditions = extract_conditions(note)
    smoker_status = classify_smoking_status(note)
    print(smoker_status)

    if "hypertension" not in conditions:
        eligible = False
        reasons.append("Does not have hypertension")

    if smoker_status != "non-smoker":
        eligible = False
        reasons.append("Is/Was a smoker")

    if "heart disease" in conditions:
        eligible = False
        reasons.append("Has heart disease")

    if "pregnant" in conditions:
        eligible = False
        reasons.append("Is pregnant")

    return {
        "eligible": eligible,
        "reasons": reasons if reasons else ["Meets all criteria"]
    }

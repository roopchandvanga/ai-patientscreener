import json

def get_patient_note(patient_id: str, filepath="data/patients_notes.json"):
    patient_id = patient_id.strip().strip('"').strip("'")
    with open(filepath, "r") as f:
        data = json.load(f)
    return data.get(patient_id, "Patient ID not found.")

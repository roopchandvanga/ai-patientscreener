import json

def create_new_patient_note(note: str, filepath="data/patients_notes.json"):
    # Load existing notes
    with open(filepath, "r") as f:
        data = json.load(f)

    # Generate new ID
    existing_ids = [int(k) for k in data.keys()]
    new_id = str(max(existing_ids) + 1 if existing_ids else 1)

    # Add new note
    data[new_id] = note

    # Save back to file
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)

    return f"Patient note saved with ID {new_id}"

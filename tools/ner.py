import spacy

# Load scispaCy model (specific to biomedical NER)
nlp = spacy.load("en_core_sci_sm")

# Normalize and map detected entities to common condition labels
def extract_conditions(text):
    doc = nlp(text)
    conditions = set()
    print(text)
    if "pregnant" in text.lower() and "not pregnant" not in text.lower():
        conditions.add("pregnant")


    for ent in doc.ents:
        ent_text = ent.text.lower()
        #print(f"Text: {ent.text}, Label: {ent.label_}")
        if "hypertension" in ent_text:
            conditions.add("hypertension")
        if "heart" in ent_text and "disease" in ent_text:
            conditions.add("heart disease")
    print(list(conditions))
    return list(conditions)

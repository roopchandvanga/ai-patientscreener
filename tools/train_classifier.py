# train_classifier.py
import joblib
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Synthetic labeled data
train_texts = [
    "The patient smokes one pack a day.",
    "Smoker for 10 years.",
    "Currently smokes.",
    "Smoker.",
    "Active tobacco user.",
    "Smokes daily.",
    "Smoker, 1 pack per day.",
    "She quit smoking 5 years ago.",
    "Former smoker, now healthy.",
    "Stopped smoking 3 years ago.",
    "Previously smoked but quit.",
    "History of smoking.",
    "Patient has never smoked.",
    "Non-smoker.",
    "Does not smoke at all.",
    "Never smoked.",
    "Not a smoker.",
    "Denies smoking.",
    "No tobacco use.",
    "He is currently a heavy smoker."
]
train_labels = [
    "smoker", "smoker", "smoker", "smoker", "smoker", "smoker", "smoker",
    "former-smoker", "former-smoker", "former-smoker", "former-smoker", "former-smoker",
    "non-smoker", "non-smoker", "non-smoker", "non-smoker", "non-smoker", "non-smoker", "non-smoker",
    "smoker"
]


# Train + save model
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', LogisticRegression(max_iter=1000))
])
pipeline.fit(train_texts, train_labels)

joblib.dump(pipeline, r'C:\Users\roopc\Desktop\Projects\ai-patienteligibility\tools\models\lifestyle_classifier.pkl')
print("✅ Model trained and saved to models/lifestyle_classifier.pkl")

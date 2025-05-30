from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import torch

texts = [
    # Smoker
    "Smokes one pack a day.",
    "Chronic smoker.",
    "Still smoking daily.",
    "20 year pack history.",
    "Currently smokes cigarettes.",
    "Heavy tobacco user.",
    "Admits to smoking regularly.",
    "Daily smoker.",
    "Cigarette use ongoing.",
    "Continues to smoke despite advice.",

    # Former-smoker (treated as smoker)
    "Quit smoking 5 years ago.",
    "Former smoker, stopped after surgery.",
    "Used to smoke for 10 years.",
    "History of smoking, now abstinent.",
    "Stopped using tobacco last year.",
    "Previously smoked regularly.",
    "Quit tobacco recently.",
    "No longer smokes.",
    "Stopped smoking recently.",
    "Smoked in past but now quit.",

    # Non-smoker
    "Never smoked.",
    "Non-smoker.",
    "Denies any tobacco use.",
    "No history of smoking.",
    "Does not use tobacco.",
    "Lifelong non-smoker.",
    "No exposure to tobacco.",
    "Denies cigarette use.",
    "Has never smoked.",
    "Not a smoker."
]

labels = [1]*20 + [0]*10  # 1 = smoker/former-smoker, 0 = non-smoker

# 20 realistic non-smoker notes
additional_non_smoker_texts = [
    "The patient is a lifelong non-smoker with no exposure to tobacco products.",
    "Denies ever smoking cigarettes, cigars, or using any form of tobacco.",
    "Reports no history of nicotine use, smoking, or tobacco consumption.",
    "No evidence of past or present tobacco use in patient history.",
    "Never used tobacco. Non-smoker status confirmed by family.",
    "Patient is a non-smoker with no secondhand smoke exposure.",
    "Does not smoke or use any tobacco-related substances.",
    "Explicitly denies smoking or vaping. No history of tobacco use.",
    "No smoking history documented in medical records.",
    "Patient has consistently reported as non-smoker in all prior visits.",
    "No mention of smoking or tobacco habits in patient intake form.",
    "Patient reports being a non-smoker and avoids tobacco environments.",
    "Non-smoker for life. No known nicotine dependency.",
    "Medical history negative for tobacco use.",
    "Denies any current or past smoking habits.",
    "Never initiated tobacco use of any kind.",
    "No history of tobacco dependence or smoking.",
    "Reports zero exposure to active or passive smoking.",
    "Patient has maintained non-smoker status throughout adulthood.",
    "There is no record of smoking or tobacco use in any prior admission."
]

texts += additional_non_smoker_texts
labels += [0] * len(additional_non_smoker_texts)  # all are non-smoker

additional_smoker_texts = [
    "Patient smokes despite multiple attempts to quit.",
    "Continues smoking even after COPD diagnosis.",
    "Current smoker, advised to reduce intake.",
    "Admits to smoking during social events regularly.",
    "Smokes 10 cigarettes a day for over 15 years.",
    "Still uses cigarettes daily despite counseling.",
    "Active smoker, mostly rolls his own cigarettes.",
    "Patient states he smokes more under stress.",
    "Smoking continues; not interested in quitting.",
    "Current tobacco use confirmed by toxicology screen."
]

additional_former_smoker_texts = [
    "Patient quit smoking two years ago after developing high blood pressure.",
    "Used to smoke daily but quit after hospitalization.",
    "Stopped smoking at age 50 after 30 years of tobacco use.",
    "Former smoker, used nicotine patch for cessation.",
    "Patient quit cigarettes in 2021; reports no relapse.",
    "History of smoking, currently abstinent.",
    "Used to smoke a pack a day, quit gradually.",
    "No smoking for the last year after 20-year history.",
    "Past smoker, successfully completed cessation program.",
    "Ex-smoker, abstinent for 18 months now."
]

texts += additional_smoker_texts + additional_former_smoker_texts
labels += [1] * (len(additional_smoker_texts) + len(additional_former_smoker_texts))

#Tokenize
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
encodings = tokenizer(texts, truncation=True, padding=True)

dataset = Dataset.from_dict({
    'text': texts,
    'label': labels,
    'input_ids': encodings['input_ids'],
    'attention_mask': encodings['attention_mask']
})

train_test = dataset.train_test_split(test_size=0.2)
train_ds = train_test['train'].remove_columns(['text'])
val_ds = train_test['test'].remove_columns(['text'])

model = DistilBertForSequenceClassification.from_pretrained(
    'distilbert-base-uncased',
    num_labels=2
)

training_args = TrainingArguments(
    output_dir="./bert-smoking-classifier",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=5,
    logging_dir="./logs"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds
)

trainer.train()

model.save_pretrained("models/bert_smoking_classifier")
tokenizer.save_pretrained("models/bert_smoking_classifier")

print("DistilBERT smoking classifier trained and saved.")

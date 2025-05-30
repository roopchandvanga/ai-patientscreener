# AI Patient Eligibility Screener

# 🧠 AI-Powered Patient Eligibility Screener for Clinical Trials

This is an AI agent-based app that helps clinical trial coordinators **screen patients for eligibility** based on medical notes. The system uses natural language processing (NLP) to extract conditions, predict smoking status using a BERT classifier, and apply inclusion/exclusion criteria defined by the trial.

[🌐 View the Live App on hosted on Streamlit](https://ai-patientscreener-nvsdmey7c7w7xmxn3uutlf.streamlit.app/)

---

## 💡 Features

- 🧾 **Patient Screening**  
  Extracts key medical conditions from unstructured text using spaCy/scispaCy + BERT.

- 🔍 **Smoking Status Classification**  
  Uses a fine-tuned DistilBERT model(uploaded to huggingface) to classify if a patient is a smoker or non-smoker based on realistic, free-text medical notes.

- 🧠 **LLM-Powered Agent**  
  GPT-4 agent decides whether to check patient eligivility for trials, fetch existing patient data or create new entries based on new patient data.

- 📋 **Interactive UI**  
  Built with Streamlit for easy interaction: view, add, or test patient records. **Full data can be in the sidebar**.

---

## ⚙️ Technologies Used

- 🧠 **LangChain** – For tool-enabled LLM agents  
- 💬 **OpenAI GPT-3.5 / GPT-4** – For natural language understanding and decision-making  
- 🧪 **Transformers + Hugging Face** – Fine-tuned DistilBERT classifier for smoking detection  
- 📚 **scispaCy** – Biomedical Named Entity Recognition  
- 🎛️ **Streamlit** – Web app interface  
- 🗃️ **JSON** – Lightweight criteria and patient record store

---


## 🔧 How It Works

1. **User inputs** patient note or selects a record.
2. **Agent determines** whether to check eligibility, fetch or create a record.
3. **NER extracts** conditions like hypertension, pregnancy, heart disease.
4. **BERT model classifies** the smoking status.
6. **LLM returns** eligibility result with reasons.

---

## 📎 Example Prompts

- `"Check eligibility of patient 3"` - to check patient eligibility
- `"Add: 65-year-old male with hypertension. Former smoker. No heart problems."` - to add a new patient record (note - mention the word 'add' or similar)
- `"show patient 2"` - to show patient's information

---

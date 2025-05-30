from langchain.agents import initialize_agent, Tool
from langchain.agents.agent_types import AgentType
from langchain_openai import ChatOpenAI
import streamlit as st
import json

from tools.eligibility_checker import check_eligibility
from tools.create_note import create_new_patient_note
from tools.get_note import get_patient_note

OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]


# Load data
with open("data/patients_notes.json") as f:
    notes = json.load(f)

with open("data/trial_criteria.json") as f:
    trial_criteria = json.load(f)

# Tool 1: Check eligibility for a patient ID
def eligibility_tool_func(input_str: str):
    try:
        # Clean quotes, whitespace, and potential misparsing
        patient_id = input_str.strip().strip('"').strip("'")

        if patient_id not in notes:
            return f"❌ Patient ID '{patient_id}' not found."

        result = check_eligibility(notes[patient_id], trial_criteria)
        out = f"Patient {patient_id}\nEligibility: {result['eligible']}\nReasons:\n"
        for r in result['reasons']:
            out += f"- {r}\n"
        return out
    
    except Exception as e:
        return f"❌ Error: {str(e)}"


# Tool 2: Add a new patient note
def create_note_func(note: str):
    return create_new_patient_note(note)

# Tool 3: View a note
def get_note_func(patient_id: str):
    return get_patient_note(patient_id)

tools = [
    Tool(
        name="CheckEligibility",
        func=eligibility_tool_func,
        description=(
            "Check if a patient is eligible for the clinical trial. "
            "Provide a patient ID as string input (e.g., '3'). "
            "Use this when the user wants to verify eligibility of an existing patient."
        )
    ),
    Tool(
        name="CreateNewPatientNote",
        func=create_note_func,
        description=(
            "Add a new patient record using natural language input. "
            "Input should be a full patient note text like "
            "'65-year-old female with hypertension. Non-smoker. No heart disease.' "
            "Use this when the user provides medical details of a new patient, "
            "even if they don't explicitly say 'add' or 'create'."
        )
    ),
    Tool(
        name="GetPatientNote",
        func=get_note_func,
        description=(
            "Fetch and display an existing patient's note based on their ID. "
            "Input should be a patient ID as a string (e.g., '3'). "
            "Use this when the user wants to view the medical note for a patient."
        )
    )
]


llm = ChatOpenAI(model= "gpt-4", api_key=OPENAI_API_KEY, temperature=0)

agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

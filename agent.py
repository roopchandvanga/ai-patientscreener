from langchain.agents import initialize_agent, Tool
from langchain.agents.agent_types import AgentType
from langchain_openai import ChatOpenAI
from config import OPENAI_API_KEY
import json
from tools.eligibility_checker import check_eligibility

# Load data
with open("data/patients_notes.json") as f:
    notes = json.load(f)

with open("data/trial_criteria.json") as f:
    trial_criteria = json.load(f)

# Define a tool
def eligibility_tool_func(patient_id: str):
    note = notes[patient_id]
    result = check_eligibility(note, trial_criteria)
    out = f"Patient {patient_id}\nEligibility: {result['eligible']}\nReasons:\n"
    for r in result['reasons']:
        out += f"- {r}\n"
    return out

tools = [
    Tool(
        name="CheckEligibility",
        func=eligibility_tool_func,
        description="Use to check eligibility of a patient based on their ID (as a string)"
    )
]

llm = ChatOpenAI(api_key=OPENAI_API_KEY, temperature=0)

agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)


import streamlit as st
from agent import agent
import json


st.set_page_config(page_title="Patient Eligibility Agent", page_icon="🧬", layout="centered")

st.title("Clinical Trial Eligibility Agent")

st.markdown("Ask a question about a patient note or trial eligibility. You can reference existing patient IDs or describe new ones to save the record.")

st.sidebar.title("📁 Patient Records")

import json
import streamlit as st

# Load trial criteria from JSON
with open("data/trial_criteria.json") as f:
    trial_criteria = json.load(f)

# Display criteria in sidebar inside an expander
with st.sidebar.expander("📋 Trial Criteria", expanded=True):
    st.markdown(f"### 🧪 {trial_criteria.get('trial_name', 'Clinical Trial')}")

    st.markdown("**✅ Inclusion Criteria**")
    for item in trial_criteria.get("inclusion_criteria", []):
        st.markdown(f"- {item}")

    st.markdown("**❌ Exclusion Criteria**")
    for item in trial_criteria.get("exclusion_criteria", []):
        st.markdown(f"- {item}")


if st.sidebar.button("View All Notes"):
    try:
        with open("data/patients_notes.json", "r") as f:
            notes_data = json.load(f)
        
        st.sidebar.subheader("Patient Notes")
        for pid, note in notes_data.items():
            st.sidebar.markdown(f"**Patient {pid}**: {note}")
    except Exception as e:
        st.sidebar.error(f"Error reading patient notes: {e}")

if st.button("🧹 Clear Chat History"):
    st.session_state.chat_history = []
    st.success("Chat history cleared!")

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_input = st.text_input("💬 Your message", placeholder="E.g. Is patient 3 eligible?")

if st.button("Submit") and user_input:
    with st.spinner("Thinking..."):
        try:
            response = agent.run(user_input)
        except Exception as e:
            response = f"❌ Error: {str(e)}"

        st.session_state.chat_history.append(("You", user_input))
        st.session_state.chat_history.append(("Agent", response))

i=0
for sender, msg in st.session_state.chat_history:
    st.markdown(f"**{sender}:** {msg}")
    i+=1
    if i%2 == 0:
        st.markdown("---")

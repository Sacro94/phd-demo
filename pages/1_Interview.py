import streamlit as st
import pandas as pd
from datetime import datetime
import os

st.set_page_config(page_title="Agentic AI Interview")

st.title("AI in je bedrijf – Agentic SME Interview (Demo Mode)")
st.caption("This is a simulated version of the AI interviewer. It follows the exact same structure as the full AI-based interview but does not require an API key.")

# ---------------- Question script (same as the AI's structure) ----------------
QUESTIONS = [
    "Hello! Thank you for joining. This interview is part of a research demonstration for the AI in je bedrijf initiative. It is anonymous and takes about 10 minutes. Do you consent to continue? (yes/no)",
    "Great. What sector is your company in?",
    "How many employees work at your company?",
    "How many years has your company been in business?",
    "What digital tools or AI-based software does your company currently use? (e.g., inventory management, chatbots, predictive maintenance)",
    "Have these tools improved your productivity? Can you estimate the percentage improvement (e.g., 15%) or time/cost savings?",
    "What is the biggest barrier that prevents you from adopting more AI? (e.g., cost, skills, data quality, trust, integration, regulation, vendor lock-in)",
    "Who owns your business data? Can you easily switch to another vendor if needed, or do you feel locked in?",
    "Would you say you feel in control of the AI tools you use, or do they feel like a 'black box'?",
    "Thank you for your answers. Here is a summary: based on your responses, your company appears to be at a **medium** level of AI adoption, with moderate productivity gains and some concerns about digital autonomy. Your completion code is: AI-SME-2026."
]

# ---------------- Session state -----------------------------------------------
if "q_index" not in st.session_state:
    st.session_state.q_index = 0
if "answers" not in st.session_state:
    st.session_state.answers = []
if "interview_complete" not in st.session_state:
    st.session_state.interview_complete = False

append_demo = st.sidebar.checkbox(
    "Append my response to the demo dataset after completion",
    value=False,
    help="Your anonymised answers will be added to the synthetic dataset for dashboard analysis."
)

# ---------------- Display chat history ---------------------------------------
for i, answer in enumerate(st.session_state.answers):
    # Interviewer question
    with st.chat_message("assistant"):
        st.markdown(QUESTIONS[i])
    # User answer
    with st.chat_message("user"):
        st.markdown(answer)

# ---------------- Show current question --------------------------------------
if not st.session_state.interview_complete:
    current_q = QUESTIONS[st.session_state.q_index]
    with st.chat_message("assistant"):
        st.markdown(current_q)

    if prompt := st.chat_input("Type your answer here..."):
        # Save user answer
        st.session_state.answers.append(prompt)
        # Move to next question
        st.session_state.q_index += 1

        # If we just answered the last question, mark complete
        if st.session_state.q_index >= len(QUESTIONS):
            st.session_state.interview_complete = True

            # Append synthetic data if demo mode enabled
            if append_demo:
                # Simulate extraction (we'll fill reasonable guesses)
                # The answers list: consent, sector, employees, years, tools, productivity, barriers, autonomy, control, (last is summary)
                try:
                    sector = st.session_state.answers[1] if len(st.session_state.answers) > 1 else "unknown"
                    employees_str = st.session_state.answers[2] if len(st.session_state.answers) > 2 else "unknown"
                    employees = int(''.join(filter(str.isdigit, employees_str))) if any(c.isdigit() for c in employees_str) else 15
                    years_str = st.session_state.answers[3] if len(st.session_state.answers) > 3 else "unknown"
                    years = int(''.join(filter(str.isdigit, years_str))) if any(c.isdigit() for c in years_str) else 8
                    adoption = "medium"  # default
                    productivity_pct = 15.0  # default
                    barriers = ["cost", "skills"]  # default
                    autonomy_score = 3.0
                    vendor_concern = "neutral"

                    data = {
                        "sector": sector,
                        "employees": employees,
                        "years_in_business": years,
                        "ai_adoption_level": adoption,
                        "productivity_improvement_percent": productivity_pct,
                        "main_barriers": "; ".join(barriers),
                        "digital_autonomy_score": autonomy_score,
                        "vendor_lock_in_concern": vendor_concern,
                        "prolific_pid": "demo_" + datetime.now().strftime("%Y%m%d%H%M%S"),
                        "completed": True
                    }

                    df_new = pd.DataFrame([data])
                    csv_path = "data/responses_demo.csv"
                    if os.path.exists(csv_path):
                        df_new.to_csv(csv_path, mode="a", index=False, header=False)
                    else:
                        df_new.to_csv(csv_path, index=False)
                    st.success("Your anonymised response has been added. Visit the Dashboard to see the update.")
                except Exception as e:
                    st.warning(f"Could not save data: {e}")

            st.balloons()
            st.success("Interview complete. Thank you for your time.")
            # Show last question
            with st.chat_message("assistant"):
                st.markdown(current_q)

else:
    st.info("This interview is finished. Refresh the page to start a new one, or explore the Dashboard.")

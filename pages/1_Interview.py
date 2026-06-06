import streamlit as st
import pandas as pd
from datetime import datetime
import os

st.set_page_config(page_title="Agentic AI Interview")

st.title("AI in je bedrijf – Agentic SME Interview")
st.caption("This is a structural prototype of the conversational AI interviewer. "
           "It follows the exact same sequence and logic as the full LLM‑based version. "
           "In the live PhD project, an AI agent dynamically adapts questions and probes for deeper insights.")

# ----------------------------------------------------------------------
# Define the interview structure with:
#   - question text
#   - expected answer type (used only for display guidance)
#   - a key name for later summary generation
# ----------------------------------------------------------------------
INTERVIEW_STEPS = [
    {
        "question": (
            "Welcome to the AI in je bedrijf SME diagnostic interview. "
            "This conversation is anonymous and will take about 10 minutes. "
            "Your responses will help us understand how small and medium enterprises in the Netherlands adopt AI. "
            "Do you consent to participate? (yes / no)"
        ),
        "key": "consent",
        "hint": "Please answer yes or no."
    },
    {
        "question": "Thank you. What sector does your company operate in? (e.g., manufacturing, logistics, retail, agri‑tech, professional services)",
        "key": "sector",
        "hint": "Please specify your industry."
    },
    {
        "question": "How many people currently work in your organisation?",
        "key": "employees",
        "hint": "Please enter a number (e.g., 12)."
    },
    {
        "question": "How many years has your company been in business?",
        "key": "years_in_business",
        "hint": "Please enter a number (e.g., 8)."
    },
    {
        "question": (
            "What digital tools or AI‑based systems do you currently use? "
            "Examples: inventory management software, customer service chatbot, "
            "predictive maintenance, accounting AI, CRM with analytics, etc. "
            "Please list the most important ones."
        ),
        "key": "tools",
        "hint": "List tools or software, separated by commas."
    },
    {
        "question": (
            "Thinking about the tools you just mentioned, have they improved your company's productivity? "
            "If possible, can you estimate the overall improvement? "
            "(For example: 'about 15% faster order processing' or 'reduced customer response time by 30%'.)"
        ),
        "key": "productivity_improvement",
        "hint": "Describe improvement or give a rough percentage."
    },
    {
        "question": (
            "What is the single biggest obstacle that prevents you from adopting more AI? "
            "Common barriers: cost, lack of internal skills, unclear return on investment, "
            "data quality issues, integration with existing systems, regulatory concerns, "
            "or fear of vendor lock‑in. Feel free to mention more than one."
        ),
        "key": "barriers",
        "hint": "Name one or more obstacles."
    },
    {
        "question": (
            "Who owns the data that your digital tools generate? "
            "If you wanted to switch to a different software vendor tomorrow, "
            "how easy would that be? Do you feel locked in?"
        ),
        "key": "autonomy",
        "hint": "Describe data ownership and switching difficulty."
    },
    {
        "question": (
            "Finally, do you feel in control of the AI tools you use, "
            "or do they sometimes feel like a 'black box' where you don't understand the decisions? "
            "How transparent are the tools to you?"
        ),
        "key": "transparency",
        "hint": "Share your sense of control and transparency."
    }
]

# ----------------------------------------------------------------------
# Session state
# ----------------------------------------------------------------------
if "q_index" not in st.session_state:
    st.session_state.q_index = 0
if "answers" not in st.session_state:
    st.session_state.answers = {}
if "interview_complete" not in st.session_state:
    st.session_state.interview_complete = False

append_demo = st.sidebar.checkbox(
    "Append my response to the demo dataset after completion",
    value=False,
    help="An anonymised summary of your answers will be added to the dashboard data."
)

# ----------------------------------------------------------------------
# Display chat history
# ----------------------------------------------------------------------
for i in range(st.session_state.q_index):
    step = INTERVIEW_STEPS[i]
    with st.chat_message("assistant"):
        st.markdown(step["question"])
    with st.chat_message("user"):
        st.markdown(st.session_state.answers.get(step["key"], "(no answer)"))

# ----------------------------------------------------------------------
# Ask current question if not finished
# ----------------------------------------------------------------------
if not st.session_state.interview_complete:
    current_step = INTERVIEW_STEPS[st.session_state.q_index]
    with st.chat_message("assistant"):
        st.markdown(current_step["question"])
        st.caption(current_step["hint"])

    if prompt := st.chat_input("Type your answer here..."):
        # Save answer
        st.session_state.answers[current_step["key"]] = prompt
        st.session_state.q_index += 1

        # If all questions answered, wrap up
        if st.session_state.q_index >= len(INTERVIEW_STEPS):
            st.session_state.interview_complete = True

            # ----------------------------------------------------------
            # Generate a simple maturity summary based on answers
            # (In the real LLM version, this is dynamic)
            # ----------------------------------------------------------
            sector = st.session_state.answers.get("sector", "unknown")
            employees_str = st.session_state.answers.get("employees", "0")
            employees = int(''.join(filter(str.isdigit, employees_str))) if any(c.isdigit() for c in employees_str) else 15
            years_str = st.session_state.answers.get("years_in_business", "0")
            years = int(''.join(filter(str.isdigit, years_str))) if any(c.isdigit() for c in years_str) else 8
            tools_mentioned = st.session_state.answers.get("tools", "")
            productivity_text = st.session_state.answers.get("productivity_improvement", "")
            barriers_text = st.session_state.answers.get("barriers", "").lower()
            autonomy_text = st.session_state.answers.get("autonomy", "").lower()

            # Simple heuristics to guess adoption/productivity/autonomy levels
            if any(term in tools_mentioned.lower() for term in ["ai", "machine learning", "predictive", "chatbot", "analytics"]):
                adoption = "high" if "predictive" in tools_mentioned.lower() or "ai" in tools_mentioned.lower() else "medium"
            else:
                adoption = "low"

            if "30%" in productivity_text or "40%" in productivity_text or "half" in productivity_text:
                productivity_pct = 30.0
            elif "10%" in productivity_text or "15%" in productivity_text or "20%" in productivity_text:
                productivity_pct = 15.0
            else:
                productivity_pct = 10.0

            # Map barriers
            barrier_mapping = {
                "cost": "cost",
                "skills": "skills",
                "skill": "skills",
                "data quality": "data_quality",
                "integration": "integration",
                "regulation": "regulation",
                "vendor lock": "vendor_lock_in",
                "roi": "cost",
                "trust": "trust"
            }
            extracted_barriers = []
            for key, mapped in barrier_mapping.items():
                if key in barriers_text:
                    extracted_barriers.append(mapped)
            if not extracted_barriers:
                extracted_barriers = ["cost", "skills"]  # default

            if "easy" in autonomy_text or "no lock" in autonomy_text:
                autonomy_score = 4.0
                vendor_concern = "no"
            elif "difficult" in autonomy_text or "locked" in autonomy_text:
                autonomy_score = 2.0
                vendor_concern = "yes"
            else:
                autonomy_score = 3.0
                vendor_concern = "neutral"

            # Final summary message
            summary = (
                f"Interview complete. Based on your responses:\n"
                f"- Sector: {sector}\n"
                f"- AI adoption level: **{adoption}**\n"
                f"- Estimated productivity improvement: **{productivity_pct:.0f}%**\n"
                f"- Main barriers: {', '.join(extracted_barriers)}\n"
                f"- Digital autonomy score: **{autonomy_score:.0f}/5**\n\n"
                f"Thank you for your time. Your completion code is: **AI-SME-2026**"
            )

            with st.chat_message("assistant"):
                st.markdown(summary)
            st.success("Interview complete. Thank you for your time.")
            st.balloons()

            # Append to demo CSV if requested
            if append_demo:
                try:
                    data = {
                        "sector": sector,
                        "employees": employees,
                        "years_in_business": years,
                        "ai_adoption_level": adoption,
                        "productivity_improvement_percent": productivity_pct,
                        "main_barriers": "; ".join(extracted_barriers),
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
else:
    st.info("This interview is finished. Refresh the page to start a new one, or explore the Dashboard.")

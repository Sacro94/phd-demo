import streamlit as st
import openai
import json
import pandas as pd
from datetime import datetime
import os

st.set_page_config(page_title="Agentic AI Interview")

st.title("AI in je bedrijf – Agentic SME Interview")
st.caption("This conversational AI interviewer assesses AI adoption, productivity, barriers, and digital autonomy. All responses are anonymous.")

# --- OpenAI API key -------------------------------------------------
openai.api_key = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))

# --- Prompts --------------------------------------------------------
SYSTEM_PROMPT = """You are an AI interviewer for "AI in je bedrijf", a Dutch national initiative studying AI adoption in small and medium enterprises.

Goal: assess the SME's AI adoption, productivity gains, barriers, and digital autonomy.

Structure:
1. Introduction: greet, explain it takes about 10 minutes and is anonymous. Ask for consent.
2. Company context: sector, number of employees, years in business.
3. Digital tools: what software/AI tools they use. Probe for specifics.
4. Productivity impact: for each tool, ask about productivity improvement (percentage, time saved, cost change). If unsure, gently encourage an estimate.
5. Adoption barriers: ask what stops them from using more AI. Follow up on their answers. Possible barriers: skills, cost, data quality, trust, integration, regulation, vendor lock-in.
6. Digital autonomy: who owns their data? Can they switch vendors easily? Do they feel in control of AI tools? Ask about transparency.
7. Wrap-up: summarise key points, give a simple maturity level (low/medium/high) for Adoption, Productivity, Autonomy. Thank them and provide the completion code: AI-SME-2026.

Rules: professional, friendly, one question at a time. Adapt to their answers. Never ask for personal or company names."""

EXTRACTION_PROMPT = """Extract the following from the interview transcript into a JSON object with these exact keys:
{
  "sector": "...",
  "employees": <integer>,
  "years_in_business": <integer>,
  "ai_adoption_level": "none/low/medium/high",
  "productivity_improvement_percent": <float>,
  "main_barriers": ["barrier1","barrier2",...],
  "digital_autonomy_score": <float 1-5>,
  "vendor_lock_in_concern": "yes/no/neutral"
}
Only use information explicitly stated in the transcript. If not mentioned, use null."""

# --- Session state --------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "system", "content": SYSTEM_PROMPT}]
if "interview_complete" not in st.session_state:
    st.session_state.interview_complete = False

# --- Sidebar option to append demo data ------------------------------
append_demo = st.sidebar.checkbox(
    "Append my response to the demo dataset after completion",
    value=False,
    help="Your anonymised answers will be added to the synthetic dataset for dashboard analysis."
)

# --- Show chat history -----------------------------------------------
for msg in st.session_state.messages:
    if msg["role"] != "system":
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

# --- Handle user input -----------------------------------------------
if not st.session_state.interview_complete:
    if prompt := st.chat_input("Type your answer here..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get assistant reply
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = openai.ChatCompletion.create(
                        model="gpt-4",
                        messages=st.session_state.messages,
                        temperature=0.7
                    )
                    reply = response.choices[0].message.content
                except Exception as e:
                    reply = f"An error occurred: {e}"
                st.markdown(reply)

        st.session_state.messages.append({"role": "assistant", "content": reply})

        # Check for completion code
        if "AI-SME-2026" in reply:
            st.session_state.interview_complete = True

            # Extract structured data if demo mode
            if append_demo:
                try:
                    transcript = "\n".join(
                        [m["content"] for m in st.session_state.messages if m["role"] in ("user", "assistant")]
                    )
                    extract_resp = openai.ChatCompletion.create(
                        model="gpt-4",
                        messages=[
                            {"role": "system", "content": EXTRACTION_PROMPT},
                            {"role": "user", "content": "Transcript:\n" + transcript}
                        ],
                        temperature=0
                    )
                    data = json.loads(extract_resp.choices[0].message.content)
                    data["prolific_pid"] = "demo_" + datetime.now().strftime("%Y%m%d%H%M%S")
                    data["completed"] = True
                    # Append to CSV
                    df_new = pd.DataFrame([data])
                    csv_path = "data/responses_demo.csv"
                    if os.path.exists(csv_path):
                        df_new.to_csv(csv_path, mode="a", index=False, header=False)
                    else:
                        df_new.to_csv(csv_path, index=False)
                    st.success("Your anonymised response has been added. Visit the Dashboard to see the update.")
                except Exception as e:
                    st.warning(f"Could not save data: {e}")

            # Show completion message
            st.balloons()
            st.success("Interview complete. Thank you for your time.")
else:
    st.info("This interview is finished. Refresh the page to start a new one, or explore the Dashboard.")

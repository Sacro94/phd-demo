import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
import plotly.express as px
import os

st.set_page_config(page_title="Econometric Dashboard")

st.title("AI in je bedrijf – Impact Analysis Prototype")
st.caption("Synthetic dataset modelled on Dutch SME demographics | Scripted interview demo data | Built for the 'AI for All' PhD position, UvA, June 2026")

# ------------------------------------------------------------
# 1. LOAD DATA
# ------------------------------------------------------------
SYNTHETIC_PATH = "data/responses_synthetic.csv"
DEMO_PATH = "data/responses_demo.csv"

if not os.path.exists(SYNTHETIC_PATH):
    st.error("Synthetic dataset not found. Please place responses_synthetic.csv in the data folder.")
    st.stop()

df = pd.read_csv(SYNTHETIC_PATH)

if os.path.exists(DEMO_PATH):
    demo = pd.read_csv(DEMO_PATH)
    if set(demo.columns) == set(df.columns):
        df = pd.concat([df, demo], ignore_index=True)
        st.info(f"Including {len(demo)} demo response(s) from the scripted interview.")

# Create numeric adoption variable
adoption_map = {'none': 0, 'low': 1, 'medium': 2, 'high': 3}
df['adoption_num'] = df['ai_adoption_level'].map(adoption_map)

# ------------------------------------------------------------
# 2. DESCRIPTIVE CHARTS
# ------------------------------------------------------------
st.subheader("SME Landscape")

col1, col2 = st.columns(2)
with col1:
    sector_counts = df['sector'].value_counts()
    fig1 = px.bar(sector_counts, title="Firms by Sector",
                  labels={'index': 'Sector', 'value': 'Count'})
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    fig2 = px.histogram(df, x='ai_adoption_level', title="AI Adoption Levels")
    st.plotly_chart(fig2, use_container_width=True)

fig3 = px.scatter(
    df, x='adoption_num', y='productivity_pct',
    color='sector', size='employees',
    hover_data=['years_in_business'],
    title="Productivity Improvement vs. AI Adoption"
)
st.plotly_chart(fig3, use_container_width=True)

all_barriers = df['main_barriers'].str.split('; ').explode()
barrier_counts = all_barriers.value_counts()
fig4 = px.bar(barrier_counts, title="Most Cited Adoption Barriers",
              labels={'index': 'Barrier', 'value': 'Count'})
st.plotly_chart(fig4, use_container_width=True)

# ------------------------------------------------------------
# 3. REGRESSION ANALYSIS (IMPROVED FORMATTING)
# ------------------------------------------------------------
st.subheader("Regression Analysis")
st.markdown("*The models below use cross‑sectional data (synthetic + demo). They identify associations, not causal effects.*")

# Prepare data
df_model = df.dropna(subset=['productivity_pct', 'adoption_num', 'employees', 'years_in_business']).copy()

# ============================================================
# OLS: Productivity ~ Adoption + Employees + Years
# ============================================================
st.write("**OLS: Productivity ~ Adoption + Number of Employees + Years in Business**")

X_ols = df_model[['adoption_num', 'employees', 'years_in_business']]
X_ols = sm.add_constant(X_ols)
y_ols = df_model['productivity_pct']
model_ols = sm.OLS(y_ols, X_ols).fit()

# Extract coefficients into a clean DataFrame
ols_coef = pd.DataFrame({
    'Coefficient': model_ols.params,
    'Std. Error': model_ols.bse,
    't‑statistic': model_ols.tvalues,
    'P‑value': model_ols.pvalues
}).reset_index().rename(columns={'index': 'Variable'})

# Round for display
ols_coef[['Coefficient', 'Std. Error', 't‑statistic', 'P‑value']] = ols_coef[
    ['Coefficient', 'Std. Error', 't‑statistic', 'P‑value']
].round(4)

st.dataframe(ols_coef, use_container_width=True, hide_index=True)

# Model fit metrics in a clean row
col1, col2, col3, col4 = st.columns(4)
col1.metric("R‑squared", f"{model_ols.rsquared:.3f}")
col2.metric("Adj. R‑squared", f"{model_ols.rsquared_adj:.3f}")
col3.metric("F‑statistic", f"{model_ols.fvalue:.2f}")
col4.metric("Observations", f"{int(model_ols.nobs)}")

# ============================================================
# LOGIT: High Adoption ~ Digital Autonomy + Employees
# ============================================================
st.write("**Logit: High Adoption (medium/high) ~ Digital Autonomy Score + Employees**")

df_model['high_adoption'] = (df_model['adoption_num'] >= 2).astype(int)
X_logit = df_model[['digital_autonomy_score', 'employees']]
X_logit = sm.add_constant(X_logit)
y_logit = df_model['high_adoption']

try:
    model_logit = sm.Logit(y_logit, X_logit).fit(disp=0)
    # Extract coefficients
    logit_coef = pd.DataFrame({
        'Coefficient': model_logit.params,
        'Std. Error': model_logit.bse,
        'z‑statistic': model_logit.tvalues,
        'P‑value': model_logit.pvalues
    }).reset_index().rename(columns={'index': 'Variable'})
    logit_coef[['Coefficient', 'Std. Error', 'z‑statistic', 'P‑value']] = logit_coef[
        ['Coefficient', 'Std. Error', 'z‑statistic', 'P‑value']
    ].round(4)

    st.dataframe(logit_coef, use_container_width=True, hide_index=True)
    st.caption(f"Pseudo R‑squared: {model_logit.prsquared:.3f}  |  Log‑Likelihood: {model_logit.llf:.2f}  |  Observations: {int(model_logit.nobs)}")

except np.linalg.LinAlgError:
    st.warning(
        "The logistic regression could not be estimated because of perfect separation or a singular matrix. "
        "This is common with small, homogeneous samples. The full PhD project, with thousands of observations "
        "and greater variation, will reliably estimate such models."
    )
except Exception as e:
    st.warning(f"Logit model could not be estimated: {e}")

# ------------------------------------------------------------
# 4. DIGITAL AUTONOMY
# ------------------------------------------------------------
st.subheader("Digital Autonomy Score Distribution")
fig5 = px.histogram(df, x='digital_autonomy_score', nbins=10,
                    title="Autonomy Scores (1 = low, 5 = high)")
st.plotly_chart(fig5, use_container_width=True)

# ------------------------------------------------------------
# 5. FROM CORRELATION TO CAUSALITY
# ------------------------------------------------------------
st.subheader("From Correlation to Causality: The Full PhD Project Design")
st.markdown("""
The analyses above use a small, cross‑sectional synthetic dataset together with any demo responses from the scripted interview. The full PhD project will deploy an **LLM‑driven agentic survey** to **10,000 Dutch SMEs** through the **AI in je bedrijf** network and track them over time.

**Causal identification strategy:**

- **Difference‑in‑Differences (DiD):** Compare productivity trajectories of SMEs that adopt AI between two survey waves (treatment) versus comparable firms that do not adopt (control).
- **Alternative approaches:** Instrumental variables (e.g., distance to a digital innovation hub, historical IT investment) and regression discontinuity if a programme eligibility cutoff exists.
- **Survey validation:** A randomised sub‑sample will take both the agentic AI interview and a traditional fixed questionnaire, allowing comparison of response depth, completion rates, and data quality.

**Expected contributions:**

The project will produce the first large‑scale, causally identified evidence on AI‑driven productivity gains in European SMEs. Results will directly inform the Dutch national AI adoption strategy and contribute open‑source, auditable survey tools for future research.
""")

st.markdown("---")
st.caption("Qudus Oladeji | qudus.oladeji@um6p.ma")

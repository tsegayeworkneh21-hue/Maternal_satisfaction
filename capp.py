"""
Streamlit deployment app — Final 10-Feature Maternal Satisfaction Model
=========================================================================
Deploys the exact model trained in secondaphi_fixed.ipynb:
  Maternal_Satisfaction_Logistic_Model_10Features.pkl

Run with:  streamlit run streamlit_deploy_final_model.py
Place "Maternal_Satisfaction_Logistic_Model_10Features.pkl" in the same
folder as this script (it's what the notebook's joblib.dump() cell saves).
"""

import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

st.set_page_config(page_title="Maternal Satisfaction Predictor", page_icon="🤰", layout="centered")

MODEL_PATH = Path(__file__).parent / "Maternal_Satisfaction_Logistic_Model_10Features.pkl"

# ---------------------------------------------------------------------------
# Human-readable labels <-> the exact raw SPSS codes the model was trained on
# (pulled directly from the dataset's value labels - the model expects these
# exact numeric codes, since the notebook never decoded them to strings)
# ---------------------------------------------------------------------------
FEATURE_LABELS = {
    "q411owners": {
        "question": "Facility ownership",
        "options": {"Private": 1.0, "Government": 2.0},
    },
    "q112waitpl": {
        "question": "Was a waiting area available?",
        "options": {"Yes": 1.0, "No": 2.0},
    },
    "q301greeti": {
        "question": "Was she greeted by the professional on arrival?",
        "options": {"Yes": 1.0, "No": 2.0},
    },
    "q113privac": {
        "question": "Was gender privacy maintained during physical examination?",
        "options": {"Yes": 1.0, "No": 2.0},
    },
    "q111timeta": {
        "question": "Time taken to get a professional's attention",
        "options": {"Less than 1 hour": 1.0, "More than 1 hour": 2.0},
    },
    "q110reason": {
        "question": "Reason for visiting the hospital",
        "options": {"Planned": 1.0, "Referred": 2.0},
    },
    "q108reside": {
        "question": "Residence",
        "options": {"Urban": 1.0, "Rural": 2.0},
    },
    "q302respec": {
        "question": "Was she treated respectfully during delivery?",
        "options": {"Yes": 1.0, "No": 2.0},
    },
    "q102mstatu": {
        "question": "Marital status",
        "options": {"Single": 1.0, "Married": 2.0, "Cohabited": 3.0,
                     "Widowed": 4.0, "Divorced": 5.0},
    },
}
NUMERIC_FEATURE = "q1091distk"  # distance to facility, in km — entered directly, no mapping needed

FEATURE_ORDER = ["q411owners", "q112waitpl", "q301greeti", "q113privac", "q111timeta",
                  "q110reason", "q108reside", "q302respec", "q1091distk", "q102mstatu"]


@st.cache_resource
def load_model(path: Path):
    return joblib.load(path)


st.title("🤰 Maternal Satisfaction Predictor")
st.caption("Final 10-feature logistic regression model — Bahir Dar delivery care satisfaction study")

if not MODEL_PATH.exists():
    st.error(
        f"Model file not found: `{MODEL_PATH.name}`\n\n"
        "Place `Maternal_Satisfaction_Logistic_Model_10Features.pkl` in the same folder as "
        "this script (this is the file created by the notebook's `joblib.dump(...)` cell)."
    )
    st.stop()

model = load_model(MODEL_PATH)

st.sidebar.header("⚙️ Settings")
threshold = st.sidebar.slider(
    "Decision threshold", min_value=0.05, max_value=0.95, value=0.50, step=0.05,
    help="Raising this makes the model more conservative about predicting 'satisfied' "
         "(higher precision, lower recall). Lowering it does the opposite."
)
st.sidebar.caption(
    "Default 0.50 balances precision/recall. Try 0.65+ if you specifically need fewer "
    "false 'satisfied' predictions."
)

st.subheader("Enter the mother's details")

user_choices = {}
cols = st.columns(2)
for i, feature in enumerate(FEATURE_ORDER):
    if feature == NUMERIC_FEATURE:
        continue
    info = FEATURE_LABELS[feature]
    target = cols[i % 2]
    user_choices[feature] = target.selectbox(info["question"], list(info["options"].keys()), key=feature)

distance_km = st.number_input(
    "Distance from home to the facility (km)", min_value=0.0, max_value=500.0, value=5.0, step=1.0
)

if st.button("Predict satisfaction", type="primary", use_container_width=True):
    row = {}
    for feature in FEATURE_ORDER:
        if feature == NUMERIC_FEATURE:
            row[feature] = distance_km
        else:
            chosen_label = user_choices[feature]
            row[feature] = FEATURE_LABELS[feature]["options"][chosen_label]

    input_df = pd.DataFrame([row])[FEATURE_ORDER]  # exact column order the model was trained on

    proba_satisfied = model.predict_proba(input_df)[0, 1]
    prediction = "Satisfied 😊" if proba_satisfied >= threshold else "Not satisfied 😟"

    st.markdown("---")
    st.subheader("Result")
    c1, c2 = st.columns(2)
    c1.metric("Prediction", prediction)
    c2.metric("Predicted probability of satisfaction", f"{proba_satisfied*100:.1f}%")
    st.progress(min(max(proba_satisfied, 0.0), 1.0))

    with st.expander("Show the exact values sent to the model"):
        st.dataframe(input_df, use_container_width=True)

st.markdown("---")
st.caption(
    "Model: Logistic Regression · Features: facility ownership, waiting area, greeting, "
    "privacy, time to attention, visit reason, residence, respectful treatment, distance (km), "
    "marital status."
)

import streamlit as st
import numpy as np
import pandas as pd
import joblib, json, shap
import matplotlib.pyplot as plt

st.set_page_config(page_title="CardioAI Heart Disease Predictor", layout="wide")

@st.cache_resource
def load_artefacts():
    model   = joblib.load("model.pkl")
    scaler  = joblib.load("scaler.pkl")
    columns = joblib.load("feature_columns.pkl")
    with open("sample_patient.json") as f:
        sample = json.load(f)
    return model, scaler, columns, sample

model, scaler, feat_columns, sample = load_artefacts()

CATEGORICAL = ["cp", "restecg", "slope", "thal"]
CONTINUOUS  = ["age", "trestbps", "chol", "thalach", "oldpeak", "ca"]

st.title("💓 CardioAI — Heart Disease Risk Predictor")
st.markdown("*Decision-support tool for community cardiologists. Not a substitute for clinical judgement.*")
st.divider()

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Patient Feature Input")
    age      = st.number_input("Age (years) [20–80]",      min_value=20, max_value=80,  value=int(sample.get("age", 55)))
    sex      = st.selectbox("Sex", options=[0, 1], format_func=lambda x: "Female (0)" if x == 0 else "Male (1)", index=int(sample.get("sex", 1)))
    cp       = st.selectbox("Chest Pain Type [0–3]", [0,1,2,3], index=int(sample.get("cp", 0)),
                             format_func=lambda x: {0:"Typical Angina",1:"Atypical",2:"Non-Anginal",3:"Asymptomatic"}[x])
    trestbps = st.number_input("Resting BP (mmHg) [90–200]", min_value=90, max_value=200, value=int(sample.get("trestbps", 130)))
    chol     = st.number_input("Cholesterol (mg/dl) [100–600]", min_value=100, max_value=600, value=int(sample.get("chol", 250)))
    fbs      = st.selectbox("Fasting Blood Sugar > 120", [0,1], index=int(sample.get("fbs",0)), format_func=lambda x: "No (0)" if x==0 else "Yes (1)")
    restecg  = st.selectbox("Resting ECG [0–2]", [0,1,2], index=int(sample.get("restecg",0)))
    thalach  = st.number_input("Max Heart Rate [70–210]",  min_value=70, max_value=210, value=int(sample.get("thalach", 150)))
    exang    = st.selectbox("Exercise-Induced Angina", [0,1], index=int(sample.get("exang",0)), format_func=lambda x: "No" if x==0 else "Yes")
    oldpeak  = st.number_input("ST Depression (oldpeak) [0.0–6.2]", min_value=0.0, max_value=6.2, value=float(sample.get("oldpeak", 1.0)), step=0.1)
    slope    = st.selectbox("ST Slope [0–2]", [0,1,2], index=int(sample.get("slope",1)))
    ca       = st.number_input("Major Vessels (fluoroscopy) [0–3]", min_value=0, max_value=3, value=int(sample.get("ca",0)))
    thal     = st.selectbox("Thal [1–3]", [1,2,3], index=int(sample.get("thal",3))-1,
                            format_func=lambda x: {1:"Normal",2:"Fixed Defect",3:"Reversible Defect"}[x])

    predict_btn = st.button("🔍 Predict", use_container_width=True, type="primary")

with col2:
    st.subheader("Prediction Results")
    if predict_btn:
        raw = pd.DataFrame([{"age":age,"sex":sex,"cp":cp,"trestbps":trestbps,"chol":chol,
                              "fbs":fbs,"restecg":restecg,"thalach":thalach,"exang":exang,
                              "oldpeak":oldpeak,"slope":slope,"ca":ca,"thal":thal}])
        enc = pd.get_dummies(raw, columns=CATEGORICAL)
        enc = enc.reindex(columns=feat_columns, fill_value=0)
        enc[CONTINUOUS] = scaler.transform(enc[CONTINUOUS])

        prob    = model.predict_proba(enc)[0][1]
        pred    = int(prob >= 0.5)
        conf    = prob if pred == 1 else 1 - prob

        if pred == 1:
            st.error(f"🔴 **Disease Present** — Confidence: {conf*100:.1f}%")
        else:
            st.success(f"🟢 **No Disease Detected** — Confidence: {conf*100:.1f}%")

        st.metric("Risk Score", f"{prob*100:.1f}%")

        # SHAP feature importance bar chart
        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(enc)
        feat_imp  = pd.Series(np.abs(shap_vals[0]), index=feat_columns).nlargest(3)

        fig, ax = plt.subplots(figsize=(5, 3))
        feat_imp[::-1].plot.barh(ax=ax, color="#E57373")
        ax.set_title("Top 3 Driving Features (SHAP)")
        ax.set_xlabel("Absolute SHAP Value")
        st.pyplot(fig)

        # Plain-English explanation
        top3 = feat_imp.index.tolist()
        st.info(
            f"This patient's highest-risk indicators are **{top3[0]}**, **{top3[1]}**, and **{top3[2]}**. "
            "Elevated ST depression (oldpeak) and reduced maximum heart rate are classical markers of "
            "myocardial ischaemia under stress. Please refer for a formal stress echocardiogram if not "
            "already completed."
        )
    else:
        st.info("Fill in the patient features and click **Predict** to see the risk assessment.")
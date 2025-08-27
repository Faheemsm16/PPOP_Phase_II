# app.py — Streamlit front-end (Personalized PPOP)
import numpy as np
import pandas as pd
import streamlit as st
import joblib, xgboost as xgb
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import os
import io

# Import utils
from utils.explainability import shap_values_rf_classifier, shap_barplot_matplotlib, get_feature_names_from_ct
from utils.reporting import build_pdf_report

# ---------------- Config ----------------
st.set_page_config(page_title="PPOP – Hemophilia Decision Support", layout="centered")
st.title("PPOP – Hemophilia Decision Support")

# ---------------- Paths ----------------
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")

# ---------------- Load artifacts ----------------
@st.cache_resource
def load_artifacts():
    preproc_xgb = joblib.load(os.path.join(MODEL_DIR, "preproc_xgb.pkl"))
    xgb_booster = xgb.Booster()
    xgb_booster.load_model(os.path.join(MODEL_DIR, "xgb_booster.json"))

    preproc_rf  = joblib.load(os.path.join(MODEL_DIR, "preproc_rf.pkl"))
    rf_model    = joblib.load(os.path.join(MODEL_DIR, "rf_model.pkl"))
    with open(os.path.join(MODEL_DIR, "rf_threshold.txt")) as f:
        best_thr = float(f.read().strip())

    lstm_model  = load_model(os.path.join(MODEL_DIR, "lstm_model.keras"), compile=False)
    lstm_scaler = joblib.load(os.path.join(MODEL_DIR, "lstm_scaler.pkl"))

    return preproc_xgb, xgb_booster, preproc_rf, rf_model, best_thr, lstm_model, lstm_scaler

preproc_xgb, xgb_booster, preproc_rf, rf_model, best_thr, lstm_model, lstm_scaler = load_artifacts()

# ---------------- Helper: PK Half-life ----------------
def estimate_half_life(factor_type, severity, age, weight, sex, custom_half_life=None):
    if custom_half_life is not None:
        return custom_half_life

    # Base (population PK)
    t_half = 12 if factor_type == "Factor VIII" else 24

    # Severity adjustment
    if severity == "Mild":
        t_half += 2
    elif severity == "Severe":
        t_half -= 2

    # Age adjustment
    if age < 18:
        t_half *= 0.9
    elif age > 50:
        t_half *= 1.1

    # Weight adjustment
    if weight < 40:
        t_half *= 0.9
    elif weight > 80:
        t_half *= 1.05

    # Sex adjustment
    if sex == "Female":
        t_half *= 1.05

    return round(max(6.0, t_half), 1)

def pk_decay_series(factor_now: float, k_per_hr: float, horizon_h: int, step_h: int = 6):
    times = np.arange(0, horizon_h + step_h, step_h)
    preds = factor_now * np.exp(-k_per_hr * times)
    return times, preds

# ---------------- Sidebar Inputs ----------------
with st.sidebar:
    st.header("Patient Profile")
    patient_name = st.text_input("Patient Name", value="")
    age = st.slider("Age", 1, 80, 30)
    weight = st.slider("Weight (kg)", 10, 120, 60)
    height = st.slider("Height (cm)", 50, 200, 170)
    sex = st.selectbox("Sex", ["Male", "Female"], index=0)

    st.markdown("---")
    st.header("Hemophilia Profile")
    factor_type = st.selectbox("Factor Type", ["Factor VIII", "Factor IX"], index=0)
    severity = st.selectbox("Severity", ["Mild", "Moderate", "Severe"], index=2)
    inhibitor_status = st.selectbox("Inhibitor Present?", ["No", "Yes"], index=0)

    st.markdown("---")
    st.header("Infusion Details")
    brand = st.text_input("Factor Brand (e.g., Advate, Hemlibra, BeneFIX)")
    route = st.selectbox("Infusion Route", ["IV", "SC"], index=0)
    dose = st.selectbox("Infusion Dose (IU)", [250, 500, 1000, 1500, 2000, 2500], index=4)
    time_since = st.slider("Time since last infusion (hr)", 0, 168, 24)

    st.markdown("---")
    st.header("PK Profile (Half-life)")
    use_custom_half_life = st.checkbox("Enter Patient-Specific Half-life")
    custom_half_life = None
    if use_custom_half_life:
        custom_half_life = st.number_input("Half-life (hours, from lab PK study)",
                                           min_value=4.0, max_value=72.0, step=0.5, value=12.0)

    st.markdown("---")
    st.header("Current Situation")
    activity_level = st.selectbox("Activity Level",
                                  ["Rest", "Normal daily", "Moderate exercise", "Intense exercise"])
    trauma = st.checkbox("Recent Trauma / Surgery?")
    bleed_history = st.text_area("Recent Bleeding Events (if any)")

    st.markdown("---")
    st.header("Prediction Settings")
    horizon = st.slider("Prediction Horizon (hours)", 24, 168, 72, step=24)
    enforce_decay = st.checkbox("Enforce monotonic decay (no new infusion)", value=True)

# ---------------- Prediction ----------------
if st.button("Predict"):
    # Assemble input row for models
    row = {
        "patient_name": patient_name,
        "age": age, "weight_kg": weight, "height_cm": height, "sex": sex,
        "factor_type": factor_type, "severity": severity, "inhibitor": inhibitor_status,
        "infusion_brand": brand, "route": route,
        "infusion_dose_IU": dose, "time_since_last_infusion_hr": time_since,
        "activity_level": activity_level, "trauma": trauma, "bleed_history": bleed_history,
        "tsi_sq": time_since ** 2,
        "log_dose": np.log1p(dose),
        "dose_per_kg": dose / weight
    }

    # Factor level (XGB)
    input_df = pd.DataFrame([row])
    dmat = xgb.DMatrix(preproc_xgb.transform(input_df))
    factor_now = float(xgb_booster.predict(dmat)[0])

    # Bleed risk (RF) – include extra features
    row_rf = dict(row); row_rf["factor_level_IU_dL"] = factor_now
    X_proc_rf = preproc_rf.transform(pd.DataFrame([row_rf]))
    proba = float(rf_model.predict_proba(X_proc_rf)[:, 1][0])
    risk_label = "High" if proba >= best_thr else "Low"

    # Temporal prediction
    t_half = estimate_half_life(factor_type, severity, age, weight, sex, custom_half_life)
    k = np.log(2.0) / t_half
    hours, pk_only = pk_decay_series(factor_now, k, horizon, step_h=6)

    if enforce_decay:
        preds = pk_only.tolist()
    else:
        # Hybrid LSTM + PK (anchored)
        seq_features = ["infusion_dose_IU", "time_since_last_infusion_hr", "factor_level_IU_dL"]
        start_unscaled = np.array([[dose, time_since, factor_now]])
        start_scaled = lstm_scaler.transform(pd.DataFrame(start_unscaled, columns=seq_features))[0]
        seq_window = np.tile(start_scaled, (5, 1))
        current_seq = seq_window.reshape(1, seq_window.shape[0], seq_window.shape[1])

        preds = [factor_now]
        last_pred = factor_now
        alpha = 0.6

        for i in range(horizon // 6):
            pred_scaled = lstm_model.predict(current_seq, verbose=0).flatten()[0]
            inv_row = lstm_scaler.inverse_transform(pd.DataFrame(
                [[dose, time_since + (i+1)*6, pred_scaled]], columns=seq_features
            ))
            lstm_pred = float(inv_row[0, -1])
            pk_target = float(factor_now * np.exp(-k * ((i+1) * 6)))
            blended = alpha * pk_target + (1 - alpha) * lstm_pred
            final_pred = min(last_pred, blended)
            preds.append(final_pred)
            last_pred = final_pred
            next_unscaled = np.array([[dose, time_since + (i+1)*6, final_pred]])
            next_scaled = lstm_scaler.transform(pd.DataFrame(next_unscaled, columns=seq_features))[0]
            current_seq = np.vstack([current_seq[0, 1:], next_scaled]).reshape(
                1, seq_window.shape[0], seq_window.shape[1])

        preds = pd.Series(preds).rolling(window=2, min_periods=1).mean().tolist()

    # Save in session
    st.session_state.update({
        "prediction_done": True,
        "row": row,
        "factor_now": factor_now,
        "risk_label": risk_label,
        "proba": proba,
        "preds": preds,
        "hours": hours,
        "horizon": horizon,
        "enforce_decay": enforce_decay,
        "t_half": t_half,
        "X_proc_rf": X_proc_rf
    })

# ---------------- Display Results ----------------
if st.session_state.get("prediction_done", False):
    st.subheader("Point Predictions")
    st.info(f"Factor level (now): **{st.session_state.factor_now:.2f} IU/dL**")
    st.info(f"Bleed Risk: **{st.session_state.risk_label}** "
            f"(p = {st.session_state.proba*100:.1f}%, threshold = {best_thr:.2f})")

    # Graph
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(st.session_state.hours, st.session_state.preds, marker="o")
    ax.axhline(y=50, color="r", linestyle="--", label="Safe Threshold (50 IU/dL)")
    ax.set_xlabel("Time Ahead (hours)")
    ax.set_ylabel("Predicted Factor Level (IU/dL)")
    ax.set_title("Temporal Prediction")
    ax.legend()
    st.pyplot(fig)

    st.markdown(
        f"""
        **Interpretation:**  
        - Current factor level: **{st.session_state.factor_now:.1f} IU/dL**  
        - Using half-life: **{st.session_state.t_half:.1f} hours**  
        - Blue curve shows predicted decline; red line = 50 IU/dL safety threshold.  
        - When blue falls below red, risk of bleed rises.  
        """
    )

    if st.checkbox("Show Explainability"):
        shap_vals = shap_values_rf_classifier(rf_model, st.session_state.X_proc_rf)
        feature_names = get_feature_names_from_ct(preproc_rf)
        fig_exp = shap_barplot_matplotlib(shap_vals[0], feature_names)

        st.subheader("Explainability")
        st.pyplot(fig_exp)

        # Save fig in session for PDF export
        st.session_state.last_shap_fig = fig_exp


    if st.checkbox("Generate PDF Report"):
        patient_meta = st.session_state.row
        snapshot = {
            "factor_now": st.session_state.factor_now,
            "risk_label": st.session_state.risk_label,
            "risk_p": st.session_state.proba
        }

        # Save trend figure
        buf_img_trend = io.BytesIO()
        fig.savefig(buf_img_trend, format="png")
        buf_img_trend.seek(0)

        # Save SHAP figure if available
        shap_png = None
        if "last_shap_fig" in st.session_state:
            buf_img_shap = io.BytesIO()
            st.session_state.last_shap_fig.savefig(buf_img_shap, format="png")
            buf_img_shap.seek(0)
            shap_png = buf_img_shap.getvalue()

        # Build report
        buf = build_pdf_report(
            patient_meta,
            snapshot,
            st.session_state.horizon,
            st.session_state.enforce_decay,
            trend_png_bytes=buf_img_trend.getvalue(),
            shap_png_bytes=shap_png
        )

        st.download_button("Download Report", buf,
                        file_name="hemophilia_report.pdf",
                        mime="application/pdf")
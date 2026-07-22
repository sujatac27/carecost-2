"""
CareCost AI — Streamlit Dashboard
============================================================
Phase 7 of the project plan. A single-page app matching the mockup in the
project brief: procedure/insurance/ZIP inputs -> predicted cost, nearby
provider comparison, and an AI-generated explanation.

Run locally (after training the model and, optionally, starting the API):
    streamlit run frontend/app.py

This talks directly to the same prediction/decision-engine/explainability
modules used by the FastAPI backend (via `src/`), so it works standalone
without the API running. Set API_BASE_URL to instead call a deployed
FastAPI backend over HTTP.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from features import build_model_matrix, TARGET_COL
from decision_engine import rank_providers, potential_savings
from explainability import ExplainabilityEngine, template_explanation, llm_explanation

st.set_page_config(page_title="CareCost AI", page_icon="🏥", layout="wide")

MODEL_PATH = ROOT / "models" / "carecost_model.joblib"
HOSPITALS_PATH = ROOT / "data" / "hospitals.csv"
CLAIMS_PATH = ROOT / "data" / "claims_clean.csv"


# ---------------------------------------------------------------------------
# Cached data / model loading
# ---------------------------------------------------------------------------

@st.cache_resource
def load_model_bundle():
    if not MODEL_PATH.exists():
        st.error(
            "No trained model found. Run the training notebook "
            "(`notebooks/carecost_ai_colab.ipynb`) or `python src/train.py` first."
        )
        st.stop()
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_reference_data():
    hospitals = pd.read_csv(HOSPITALS_PATH, dtype={"zip_code": str, "hospital_id": str})
    claims = pd.read_csv(CLAIMS_PATH, dtype={"zip_code": str, "cpt_code": str, "hospital_id": str})
    procedures = (
        claims[["cpt_code", "procedure_description", "procedure_category"]]
        .drop_duplicates(subset="cpt_code")
        .sort_values("procedure_description")
    )
    plans = sorted(claims["insurance_plan"].unique())
    return hospitals, claims, procedures, plans


bundle = load_model_bundle()
hospitals_df, claims_df, procedures_df, plan_list = load_reference_data()
explainer = ExplainabilityEngine(bundle["model"], bundle["feature_cols"])


def build_feature_row(cpt_code, hospital_row, insurance_plan, patient_age, deductible_met_pct):
    proc_ref = claims_df[claims_df["cpt_code"] == cpt_code].iloc[0]
    plan_ref = claims_df[claims_df["insurance_plan"] == insurance_plan].iloc[0]
    row = {
        "cpt_code": cpt_code,
        "procedure_category": proc_ref["procedure_category"],
        "insurance_plan_type": plan_ref["insurance_plan_type"],
        "patient_age": patient_age,
        "deductible_met_pct": deductible_met_pct,
        "median_household_income": hospital_row["median_household_income"],
        "cost_of_living_index": hospital_row["cost_of_living_index"],
        "population_density": hospital_row["population_density"],
        "star_rating": hospital_row["star_rating"],
        "readmission_rate_pct": hospital_row["readmission_rate_pct"],
        "patient_satisfaction_score": hospital_row["patient_satisfaction_score"],
        "facility_price_index": hospital_row["facility_price_index"],
        "facility_type": hospital_row["facility_type"],
        "metro_tier": hospital_row["metro_tier"],
        "state": hospital_row["state"],
        "hospital_id": hospital_row["hospital_id"],
        TARGET_COL: 0.0,
    }
    X, _, _, _ = build_model_matrix(pd.DataFrame([row]), freq_maps=bundle["freq_maps"], fit_encoders=False)
    return X.reindex(columns=bundle["feature_cols"], fill_value=0.0)


def predict_row(cpt_code, hospital_row, insurance_plan, patient_age, deductible_met_pct):
    X = build_feature_row(cpt_code, hospital_row, insurance_plan, patient_age, deductible_met_pct)
    Xm = X
    if bundle.get("scaler") is not None and bundle["model_name"] == "linear_regression":
        Xm = pd.DataFrame(bundle["scaler"].transform(X), columns=X.columns)
    point = float(np.clip(bundle["model"].predict(Xm)[0], 0, None))
    low = float(np.clip(bundle["lower_model"].predict(X)[0], 0, None))
    high = float(np.clip(bundle["upper_model"].predict(X)[0], 0, None))
    return X, point, min(low, point), max(high, point)


# ---------------------------------------------------------------------------
# Sidebar — inputs
# ---------------------------------------------------------------------------

st.title("🏥 CareCost AI")
st.caption("AI-powered healthcare cost prediction & financial decision engine — "
           "financial transparency, not medical advice.")
st.warning(
    "**Demo data notice:** Facility names shown are real hospitals, but every cost, "
    "star rating, and readmission figure is **simulated** for this demo — not that "
    "hospital's actual published pricing or real CMS quality score. "
    "Procedure costs are anchored to real 2025-2026 Medicare fee-schedule reference "
    "rates; hospital-specific numbers are illustrative only.",
    icon="⚠️",
)

with st.sidebar:
    st.header("Your Estimate")
    proc_label_to_cpt = {
        f"{row.procedure_description} ({row.cpt_code})": row.cpt_code
        for row in procedures_df.itertuples()
    }
    proc_label = st.selectbox("Procedure", list(proc_label_to_cpt.keys()))
    cpt_code = proc_label_to_cpt[proc_label]

    insurance_plan = st.selectbox("Insurance Plan", plan_list)
    zip_code = st.selectbox("ZIP Code", sorted(hospitals_df["zip_code"].unique()))

    with st.expander("Advanced (age & deductible)"):
        patient_age = st.slider("Patient age", 0, 100, 38)
        deductible_met_pct = st.slider("% of deductible already met this year", 0, 100, 15)

    st.divider()
    st.subheader("Ranking weights")
    w_cost = st.slider("Cost", 0.0, 1.0, 0.50)
    w_quality = st.slider("Quality (star rating)", 0.0, 1.0, 0.25)
    w_readmit = st.slider("Readmission risk", 0.0, 1.0, 0.15)
    w_dist = st.slider("Distance (n/a in demo data)", 0.0, 1.0, 0.10)

    run = st.button("Get Estimate", type="primary", use_container_width=True)


# ---------------------------------------------------------------------------
# Main panel
# ---------------------------------------------------------------------------

if run:
    candidates = hospitals_df[hospitals_df["zip_code"] == zip_code]
    if candidates.empty:
        candidates = hospitals_df.head(10)

    rows = []
    for _, h in candidates.iterrows():
        _, point, low, high = predict_row(cpt_code, h, insurance_plan, patient_age, deductible_met_pct)
        rows.append({
            "hospital_id": h["hospital_id"], "hospital_name": h["hospital_name"],
            "predicted_cost": point, "cost_low": low, "cost_high": high,
            "star_rating": h["star_rating"], "readmission_rate_pct": h["readmission_rate_pct"],
        })
    cand_df = pd.DataFrame(rows)
    weights = {"cost": w_cost, "quality": w_quality, "readmission": w_readmit, "distance": w_dist}
    ranked = rank_providers(cand_df, weights=weights)
    top = ranked[0]
    savings = potential_savings(ranked)

    total_bill_estimate = top.predicted_cost / 0.35 if top.predicted_cost > 0 else 0.0
    insurance_pays = max(0.0, total_bill_estimate - top.predicted_cost)

    # --- Top metric row ------------------------------------------------------
    c1, c2, c3 = st.columns(3)
    c1.metric("Estimated Total Bill", f"${total_bill_estimate:,.0f}")
    c2.metric("Insurance Pays", f"${insurance_pays:,.0f}")
    c3.metric("You Pay", f"${top.predicted_cost:,.0f}",
              help=f"90% confidence interval: ${top.cost_low:,.0f} – ${top.cost_high:,.0f}")
    st.caption(f"Best-value facility: **{top.hospital_name}** "
               f"(★ {top.star_rating} · value score {top.value_score}/100)")

    st.divider()

    # --- Provider comparison ---------------------------------------------------
    left, right = st.columns([1.3, 1])

    with left:
        st.subheader("Nearby Providers")
        table = pd.DataFrame([{
            "Provider": p.hospital_name,
            "You Pay": f"${p.predicted_cost:,.0f}",
            "Range": f"${p.cost_low:,.0f}–${p.cost_high:,.0f}",
            "Stars": "⭐" * max(1, round(p.star_rating)),
            "Value Score": p.value_score,
        } for p in ranked])
        st.dataframe(table, use_container_width=True, hide_index=True)

        fig = go.Figure()
        names = [p.hospital_name[:28] for p in ranked]
        costs = [p.predicted_cost for p in ranked]
        colors = ["#2E7D32" if p.hospital_id == top.hospital_id else "#5B8DEF" for p in ranked]
        fig.add_trace(go.Bar(
            x=costs, y=names, orientation="h", marker_color=colors,
            error_x=dict(type="data", symmetric=False,
                         array=[p.cost_high - p.predicted_cost for p in ranked],
                         arrayminus=[p.predicted_cost - p.cost_low for p in ranked]),
        ))
        fig.update_layout(title="Your Estimated Out-of-Pocket Cost by Provider",
                           xaxis_title="USD", height=420, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.subheader("AI Explanation")
        top_hosp_row = candidates[candidates["hospital_id"] == top.hospital_id].iloc[0]
        X, point, low, high = predict_row(cpt_code, top_hosp_row, insurance_plan,
                                           patient_age, deductible_met_pct)
        drivers = explainer.top_drivers(X, top_n=4)

        use_llm = bool(os.environ.get("OPENAI_API_KEY"))
        explanation = (llm_explanation(drivers, point, proc_label, comparison_savings=savings)
                       if use_llm else
                       template_explanation(drivers, point, comparison_savings=savings))
        st.info(explanation)

        st.subheader("What's Driving Your Cost")
        drv_df = pd.DataFrame([{"Factor": d.readable_label, "Impact ($)": d.shap_value} for d in drivers])
        fig2 = px.bar(drv_df, x="Impact ($)", y="Factor", orientation="h",
                      color="Impact ($)", color_continuous_scale=["#2E7D32", "#C62828"])
        fig2.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10), showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)

else:
    st.info("Set your procedure, insurance plan, and ZIP code in the sidebar, "
            "then click **Get Estimate**.")

    st.subheader("Cost Landscape (demo dataset)")
    fig = px.histogram(claims_df, x="patient_responsibility", nbins=60,
                        title="Distribution of Patient Out-of-Pocket Costs (All Claims)")
    fig.update_layout(xaxis_title="Patient Responsibility ($)", yaxis_title="Claim Count")
    st.plotly_chart(fig, use_container_width=True)

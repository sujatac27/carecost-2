"""
CareCost AI — FastAPI Backend
============================================================
Phase 6 of the project plan.

Endpoints:
    POST /predict   -> single-facility (or best-match) cost estimate
    POST /compare    -> ranked list of nearby providers by value score
    GET  /hospitals  -> facility lookup/search
    POST /explain    -> SHAP-driven natural-language cost explanation

Run locally:
    uvicorn backend.main:app --reload --port 8000

Then visit http://localhost:8000/docs for interactive Swagger UI.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from features import build_model_matrix, TARGET_COL          # noqa: E402
from decision_engine import rank_providers, potential_savings  # noqa: E402
from explainability import ExplainabilityEngine, template_explanation, llm_explanation  # noqa: E402

from schemas import (                                          # noqa: E402
    PredictRequest, PredictResponse, CostBreakdown,
    CompareRequest, CompareResponse,
    HospitalInfo, ExplainRequest, ExplainResponse,
)

app = FastAPI(
    title="CareCost AI",
    description="AI-powered healthcare cost prediction & financial decision engine",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in production
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = ROOT / "models" / "carecost_model.joblib"
HOSPITALS_PATH = ROOT / "data" / "hospitals.csv"
CLAIMS_PATH = ROOT / "data" / "claims_clean.csv"

_state: dict = {}


@app.on_event("startup")
def load_artifacts():
    """Loads the trained model bundle and reference tables once at startup."""
    if not MODEL_PATH.exists():
        raise RuntimeError(
            f"Model file not found at {MODEL_PATH}. Run the training notebook / "
            f"`python src/train.py` first to generate it."
        )
    bundle = joblib.load(MODEL_PATH)
    _state["model"] = bundle["model"]
    _state["model_name"] = bundle["model_name"]
    _state["feature_cols"] = bundle["feature_cols"]
    _state["freq_maps"] = bundle["freq_maps"]
    _state["lower_model"] = bundle["lower_model"]
    _state["upper_model"] = bundle["upper_model"]
    _state["scaler"] = bundle.get("scaler")

    _state["hospitals"] = pd.read_csv(HOSPITALS_PATH, dtype={"zip_code": str, "hospital_id": str})
    claims = pd.read_csv(CLAIMS_PATH, dtype={"zip_code": str, "cpt_code": str, "hospital_id": str})
    _state["procedures"] = (
        claims[["cpt_code", "procedure_description", "procedure_category"]]
        .drop_duplicates(subset="cpt_code")
        .set_index("cpt_code")
    )
    # Per-CPT median financial ratios, used to build a synthetic candidate row
    # for hospitals that have no historical claims for a given procedure yet.
    _state["claims_reference"] = claims

    _state["explainer"] = ExplainabilityEngine(
        _state["model"], _state["feature_cols"], background_data=None
    )
    print(f"[startup] Loaded model '{_state['model_name']}' with "
          f"{len(_state['feature_cols'])} features and {len(_state['hospitals'])} hospitals.")


def _build_feature_row(cpt_code: str, hospital_row: pd.Series, insurance_plan: str,
                        patient_age: int, deductible_met_pct: float) -> pd.DataFrame:
    """Assembles a single-row dataframe in the exact schema the model was trained
    on, then runs it through the same feature engineering pipeline used in training
    (fit_encoders=False so we reuse the training-time frequency maps)."""
    claims_ref = _state["claims_reference"]
    proc_rows = claims_ref[claims_ref["cpt_code"] == cpt_code]
    if proc_rows.empty:
        raise HTTPException(404, f"Unknown CPT/HCPCS code: {cpt_code}")
    plan_rows = claims_ref[claims_ref["insurance_plan"] == insurance_plan]
    if plan_rows.empty:
        raise HTTPException(404, f"Unknown insurance plan: {insurance_plan}")

    proc_ref = proc_rows.iloc[0]
    plan_ref = plan_rows.iloc[0]

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
        TARGET_COL: 0.0,  # placeholder, dropped before prediction
    }
    df = pd.DataFrame([row])
    X, _, _, _ = build_model_matrix(df, freq_maps=_state["freq_maps"], fit_encoders=False)
    # Ensure column order/set matches training exactly (missing dummy cols -> 0)
    X = X.reindex(columns=_state["feature_cols"], fill_value=0.0)
    return X


def _predict_for_hospital(cpt_code, hospital_row, insurance_plan, patient_age, deductible_met_pct):
    X = _build_feature_row(cpt_code, hospital_row, insurance_plan, patient_age, deductible_met_pct)
    Xm = X
    if _state["scaler"] is not None and _state["model_name"] == "linear_regression":
        Xm = pd.DataFrame(_state["scaler"].transform(X), columns=X.columns)
    point = float(np.clip(_state["model"].predict(Xm)[0], 0, None))
    low = float(np.clip(_state["lower_model"].predict(X)[0], 0, None))
    high = float(np.clip(_state["upper_model"].predict(X)[0], 0, None))
    low, high = min(low, point), max(high, point)
    return X, point, low, high


def _hospitals_near_zip(zip_code: str, limit: int = 15) -> pd.DataFrame:
    """In this synthetic dataset, ZIP -> hospitals is a direct lookup. In a real
    deployment, swap this for a radius search using the Google Maps / Census API
    (see README for real-data integration notes)."""
    hospitals = _state["hospitals"]
    same_zip = hospitals[hospitals["zip_code"] == zip_code]
    if not same_zip.empty:
        return pd.concat([same_zip, hospitals[hospitals["zip_code"] != zip_code].head(limit)]) \
            .head(limit)
    return hospitals.head(limit)


@app.get("/")
def root():
    return {"service": "CareCost AI", "status": "healthy", "model": _state.get("model_name")}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    """Estimates total cost, insurance-paid amount, and patient out-of-pocket
    responsibility for a single procedure — at a specified hospital, or at the
    single best-value nearby facility if none is specified."""
    candidates = _hospitals_near_zip(req.zip_code)
    if req.hospital_id:
        candidates = candidates[candidates["hospital_id"] == req.hospital_id]
        if candidates.empty:
            raise HTTPException(404, f"Unknown hospital_id: {req.hospital_id}")

    rows = []
    for _, h in candidates.iterrows():
        _, point, low, high = _predict_for_hospital(
            req.cpt_code, h, req.insurance_plan, req.patient_age, req.deductible_met_pct)
        rows.append({
            "hospital_id": h["hospital_id"], "hospital_name": h["hospital_name"],
            "predicted_cost": point, "cost_low": low, "cost_high": high,
            "star_rating": h["star_rating"], "readmission_rate_pct": h["readmission_rate_pct"],
        })
    cand_df = pd.DataFrame(rows)
    ranked = rank_providers(cand_df)
    top = ranked[0]

    proc_desc = _state["procedures"].loc[req.cpt_code, "procedure_description"]
    savings = potential_savings(ranked, chosen_hospital_id=req.hospital_id)

    explainer = _state["explainer"]
    top_hosp_row = candidates[candidates["hospital_id"] == top.hospital_id].iloc[0]
    X, point, low, high = _predict_for_hospital(
        req.cpt_code, top_hosp_row, req.insurance_plan, req.patient_age, req.deductible_met_pct)
    drivers = explainer.top_drivers(X, top_n=3)
    explanation = template_explanation(drivers, point, comparison_savings=savings)

    # Approximate insurance-paid split using the plan's average reimbursement rate
    total_bill_estimate = point / 0.35 if point > 0 else 0.0  # rough allowed-amount back-out
    insurance_pays = max(0.0, total_bill_estimate - point)

    breakdown = CostBreakdown(
        hospital_id=top.hospital_id, hospital_name=top.hospital_name,
        estimated_total_bill=round(total_bill_estimate, 2),
        insurance_pays=round(insurance_pays, 2),
        you_pay=top.predicted_cost, you_pay_low=top.cost_low, you_pay_high=top.cost_high,
        star_rating=top.star_rating, readmission_rate_pct=top.readmission_rate_pct,
        value_score=top.value_score,
    )
    return PredictResponse(procedure_description=proc_desc, primary_estimate=breakdown,
                            ai_explanation=explanation)


@app.post("/compare", response_model=CompareResponse)
def compare(req: CompareRequest):
    """Ranks nearby providers for the given procedure by a weighted value score
    (cost, quality, readmission risk, distance) — the financial decision engine."""
    candidates = _hospitals_near_zip(req.zip_code, limit=15)
    rows = []
    for _, h in candidates.iterrows():
        _, point, low, high = _predict_for_hospital(
            req.cpt_code, h, req.insurance_plan, req.patient_age, req.deductible_met_pct)
        rows.append({
            "hospital_id": h["hospital_id"], "hospital_name": h["hospital_name"],
            "predicted_cost": point, "cost_low": low, "cost_high": high,
            "star_rating": h["star_rating"], "readmission_rate_pct": h["readmission_rate_pct"],
        })
    cand_df = pd.DataFrame(rows)
    weights = {"cost": req.weight_cost, "quality": req.weight_quality,
               "readmission": req.weight_readmission, "distance": req.weight_distance}
    ranked = rank_providers(cand_df, weights=weights)[:req.max_results]
    savings = potential_savings(ranked)

    proc_desc = _state["procedures"].loc[req.cpt_code, "procedure_description"]
    options = [
        CostBreakdown(
            hospital_id=p.hospital_id, hospital_name=p.hospital_name,
            estimated_total_bill=round(p.predicted_cost / 0.35, 2),
            insurance_pays=round(max(0.0, p.predicted_cost / 0.35 - p.predicted_cost), 2),
            you_pay=p.predicted_cost, you_pay_low=p.cost_low, you_pay_high=p.cost_high,
            star_rating=p.star_rating, readmission_rate_pct=p.readmission_rate_pct,
            value_score=p.value_score,
        ) for p in ranked
    ]
    return CompareResponse(procedure_description=proc_desc, options=options, potential_savings=savings)


@app.get("/hospitals", response_model=list[HospitalInfo])
def hospitals(zip_code: str | None = None, state: str | None = None, limit: int = 20):
    """Looks up facilities, optionally filtered by ZIP or state."""
    df = _state["hospitals"]
    if zip_code:
        df = df[df["zip_code"] == zip_code]
    if state:
        df = df[df["state"] == state.upper()]
    if df.empty:
        raise HTTPException(404, "No hospitals found for the given filters")
    return [
        HospitalInfo(**{k: row[k] for k in HospitalInfo.model_fields.keys()})
        for _, row in df.head(limit).iterrows()
    ]


@app.post("/explain", response_model=ExplainResponse)
def explain(req: ExplainRequest):
    """Returns the SHAP-driven explanation for a specific hospital + plan combo."""
    hospitals_df = _state["hospitals"]
    match = hospitals_df[hospitals_df["hospital_id"] == req.hospital_id]
    if match.empty:
        raise HTTPException(404, f"Unknown hospital_id: {req.hospital_id}")
    h = match.iloc[0]

    X, point, low, high = _predict_for_hospital(
        req.cpt_code, h, req.insurance_plan, req.patient_age, req.deductible_met_pct)
    drivers = _state["explainer"].top_drivers(X, top_n=4)
    driver_dicts = [{"feature": d.readable_label, "impact": round(d.shap_value, 2),
                      "direction": d.direction} for d in drivers]

    proc_desc = _state["procedures"].loc[req.cpt_code, "procedure_description"]
    if req.use_llm:
        text = llm_explanation(drivers, point, proc_desc)
    else:
        text = template_explanation(drivers, point)

    return ExplainResponse(predicted_cost=round(point, 2), top_drivers=driver_dicts, explanation=text)

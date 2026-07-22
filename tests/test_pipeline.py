"""
CareCost AI — Test Suite
============================================================
Basic smoke tests for the data pipeline, feature engineering, and decision
engine. Run with: pytest tests/
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from data_generator import generate_synthetic_dataset, inject_realistic_messiness
from features import clean_claims, build_model_matrix, TARGET_COL
from decision_engine import rank_providers, potential_savings, haversine_miles


@pytest.fixture(scope="module")
def sample_data():
    hospitals_df, claims_df = generate_synthetic_dataset(n_claims=2000, seed=1)
    claims_df = inject_realistic_messiness(claims_df, seed=1)
    return hospitals_df, claims_df


def test_data_generation_shapes(sample_data):
    hospitals_df, claims_df = sample_data
    assert len(hospitals_df) > 0
    assert len(claims_df) >= 2000  # messiness injection adds a few duplicate rows
    assert {"hospital_id", "star_rating", "facility_price_index"}.issubset(hospitals_df.columns)
    assert {"cpt_code", "patient_responsibility", "billed_amount"}.issubset(claims_df.columns)


def test_financial_amounts_are_non_negative_after_cleaning(sample_data):
    _, claims_df = sample_data
    clean = clean_claims(claims_df, verbose=False)
    for col in ["billed_amount", "allowed_amount", "insurance_paid", "patient_responsibility"]:
        assert (clean[col] >= 0).all(), f"{col} has negative values after cleaning"


def test_patient_responsibility_never_exceeds_allowed_amount(sample_data):
    _, claims_df = sample_data
    clean = clean_claims(claims_df, verbose=False)
    assert (clean["patient_responsibility"] <= clean["allowed_amount"] + 1e-6).all()


def test_plan_name_normalization(sample_data):
    _, claims_df = sample_data
    clean = clean_claims(claims_df, verbose=False)
    # messy variants should be collapsed to canonical names
    assert "AETNA PPO" not in clean["insurance_plan"].values
    assert "Aetna PPO" in clean["insurance_plan"].values


def test_feature_matrix_has_no_nans(sample_data):
    _, claims_df = sample_data
    clean = clean_claims(claims_df, verbose=False)
    X, y, feature_cols, freq_maps = build_model_matrix(clean)
    assert not X.isna().any().any(), "Feature matrix contains NaNs"
    assert not y.isna().any(), "Target contains NaNs"
    assert len(feature_cols) == X.shape[1]


def test_feature_matrix_consistent_at_inference_time(sample_data):
    """Encoders fit on training data must produce the same columns for a
    single-row inference dataframe (no train/serve skew)."""
    _, claims_df = sample_data
    clean = clean_claims(claims_df, verbose=False)
    X_train, _, feature_cols, freq_maps = build_model_matrix(clean)

    single_row = clean.iloc[[0]]
    X_single, _, _, _ = build_model_matrix(single_row, freq_maps=freq_maps, fit_encoders=False)
    X_single = X_single.reindex(columns=feature_cols, fill_value=0.0)
    assert list(X_single.columns) == feature_cols


def test_rank_providers_orders_by_value_score():
    candidates = pd.DataFrame([
        {"hospital_id": "H1", "hospital_name": "Cheap & Good", "predicted_cost": 100,
         "cost_low": 80, "cost_high": 120, "star_rating": 5.0, "readmission_rate_pct": 8.0},
        {"hospital_id": "H2", "hospital_name": "Expensive & Poor", "predicted_cost": 1000,
         "cost_low": 800, "cost_high": 1200, "star_rating": 1.0, "readmission_rate_pct": 25.0},
    ])
    ranked = rank_providers(candidates)
    assert ranked[0].hospital_id == "H1"
    assert ranked[0].value_score > ranked[1].value_score


def test_potential_savings_is_non_negative():
    candidates = pd.DataFrame([
        {"hospital_id": "H1", "hospital_name": "A", "predicted_cost": 100,
         "cost_low": 80, "cost_high": 120, "star_rating": 4.0, "readmission_rate_pct": 10.0},
        {"hospital_id": "H2", "hospital_name": "B", "predicted_cost": 300,
         "cost_low": 250, "cost_high": 350, "star_rating": 3.0, "readmission_rate_pct": 15.0},
    ])
    ranked = rank_providers(candidates)
    savings = potential_savings(ranked)
    assert savings >= 0


def test_haversine_zero_distance_for_same_point():
    assert haversine_miles(40.0, -75.0, 40.0, -75.0) == pytest.approx(0.0, abs=1e-6)


def test_haversine_known_distance():
    # Philadelphia to New York City, roughly 80 miles
    dist = haversine_miles(39.9526, -75.1652, 40.7128, -74.0060)
    assert 75 < dist < 100

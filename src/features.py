"""
CareCost AI — Data Cleaning & Feature Engineering
============================================================
Phase 1 (cleaning) and Phase 3 (feature engineering) of the project plan.

Cleaning handles the realistic messiness injected by data_generator.py, which
mirrors what real CMS/hospital machine-readable files require:
    - inconsistent text casing / whitespace in categorical fields
    - missing values
    - duplicate claim rows
    - extreme chargemaster outliers

Feature engineering converts procedure codes, hospitals, insurance plans,
and geography into model-ready features (one-hot / target / frequency
encodings + engineered ratios).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Phase 1: Cleaning
# ---------------------------------------------------------------------------

def clean_claims(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Cleans a raw claims dataframe produced by the data generator (or real CMS data)."""
    df = df.copy()
    n_start = len(df)

    # 1. Normalize categorical text (strip whitespace, standardize casing)
    text_cols = ["insurance_plan", "hospital_name", "procedure_description",
                 "facility_type", "state", "metro_tier"]
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    # 2. Collapse known plan-name variants back to a canonical name
    plan_canonical_map = {
        "AETNA PPO": "Aetna PPO", "aetna ppo": "Aetna PPO", "Aetna  PPO": "Aetna PPO",
        "BCBS PPO": "Blue Cross Blue Shield PPO",
        "Blue Cross Blue Shield  PPO": "Blue Cross Blue Shield PPO",
    }
    df["insurance_plan"] = df["insurance_plan"].replace(plan_canonical_map)

    # 3. Drop exact duplicate claim rows (billing re-submissions)
    n_dupes = df.duplicated(subset=["claim_id"]).sum()
    df = df.drop_duplicates(subset=["claim_id"], keep="first")

    # 4. Impute missing numeric values with group-wise medians (more accurate
    #    than a global median — e.g. satisfaction scores vary a lot by facility type)
    if "patient_satisfaction_score" in df.columns:
        df["patient_satisfaction_score"] = df.groupby("facility_type")["patient_satisfaction_score"] \
            .transform(lambda s: s.fillna(s.median()))
    if "deductible_met_pct" in df.columns:
        df["deductible_met_pct"] = df["deductible_met_pct"].fillna(df["deductible_met_pct"].median())

    # 5. Cap extreme chargemaster outliers using the IQR rule (winsorize, don't drop —
    #    dropping would bias the model toward under-predicting expensive facilities)
    n_outliers = 0
    if "billed_amount" in df.columns:
        for cat, group in df.groupby("procedure_category"):
            q1, q3 = group["billed_amount"].quantile([0.25, 0.75])
            iqr = q3 - q1
            upper = q3 + 3 * iqr  # 3x IQR = conservative outlier cap for skewed billing data
            outlier_idx = group[group["billed_amount"] > upper].index
            n_outliers += len(outlier_idx)
            df.loc[outlier_idx, "billed_amount"] = upper

    # 6. Recompute allowed/patient/insurance amounts that could exceed the (now capped) billed amount
    if {"billed_amount", "allowed_amount"}.issubset(df.columns):
        df["allowed_amount"] = np.minimum(df["allowed_amount"], df["billed_amount"])
        df["patient_responsibility"] = np.minimum(df["patient_responsibility"], df["allowed_amount"])
        df["insurance_paid"] = df["allowed_amount"] - df["patient_responsibility"]

    # 7. Drop rows with impossible/negative financial values
    numeric_cols = ["billed_amount", "allowed_amount", "insurance_paid", "patient_responsibility"]
    numeric_cols = [c for c in numeric_cols if c in df.columns]
    before = len(df)
    df = df[(df[numeric_cols] >= 0).all(axis=1)]
    n_negative = before - len(df)

    df = df.reset_index(drop=True)

    if verbose:
        print("Cleaning summary")
        print("-" * 40)
        print(f"Rows in:              {n_start:,}")
        print(f"Duplicates removed:   {n_dupes:,}")
        print(f"Outliers capped:      {n_outliers:,}")
        print(f"Negative rows dropped:{n_negative:,}")
        print(f"Missing values fixed: patient_satisfaction_score, deductible_met_pct")
        print(f"Rows out:             {len(df):,}")

    return df


# ---------------------------------------------------------------------------
# Phase 3: Feature Engineering
# ---------------------------------------------------------------------------

TARGET_COL = "patient_responsibility"

RAW_NUMERIC_FEATURES = [
    "patient_age", "deductible_met_pct", "median_household_income",
    "cost_of_living_index", "population_density", "star_rating",
    "readmission_rate_pct", "patient_satisfaction_score", "facility_price_index",
]

RAW_CATEGORICAL_FEATURES = [
    "cpt_code", "procedure_category", "insurance_plan_type",
    "facility_type", "metro_tier", "state",
]


def engineer_features(df: pd.DataFrame, fit_encoders: bool = True,
                       freq_maps: dict | None = None) -> tuple[pd.DataFrame, dict]:
    """
    Converts raw claim/hospital fields into model-ready features.

    Encoding strategy (chosen deliberately, not just one-hot everywhere):
      - Low-cardinality categoricals (facility_type, metro_tier, insurance_plan_type)
        -> one-hot encoded (tree models + linear models both handle this well)
      - High-cardinality categoricals (cpt_code, state) -> frequency encoding
        (keeps dimensionality low, avoids sparse one-hot explosion, and frequency
        itself is informative: common procedures tend to be more price-competitive)
      - Engineered ratio features that capture pricing dynamics directly

    Parameters
    ----------
    fit_encoders : if True, computes frequency maps from this df (training time).
                   if False, reuses `freq_maps` passed in (inference time) so that
                   unseen categories don't leak information or crash encoding.
    """
    df = df.copy()
    freq_maps = freq_maps or {}

    # --- Engineered ratio / interaction features -------------------------------------
    df["price_vs_national_avg"] = df["facility_price_index"]  # already a ratio to national baseline
    df["deductible_remaining_frac"] = 1 - (df["deductible_met_pct"] / 100.0)
    df["quality_per_dollar_proxy"] = df["star_rating"] / (df["facility_price_index"] + 0.01)
    df["is_high_deductible_stage"] = (df["deductible_met_pct"] < 30).astype(int)
    df["urban_flag"] = (df["metro_tier"] == "Urban").astype(int)

    # --- Frequency encoding for high-cardinality columns ------------------------------
    for col in ["cpt_code", "state", "hospital_id"]:
        if col not in df.columns:
            continue
        map_key = f"{col}_freq"
        if fit_encoders:
            freq = df[col].value_counts(normalize=True)
            freq_maps[map_key] = freq
        freq = freq_maps.get(map_key, pd.Series(dtype=float))
        df[map_key] = df[col].map(freq).fillna(0.0)

    # --- One-hot encode low-cardinality categoricals -----------------------------------
    onehot_cols = ["procedure_category", "insurance_plan_type", "facility_type", "metro_tier"]
    df = pd.get_dummies(df, columns=[c for c in onehot_cols if c in df.columns], prefix=onehot_cols)

    return df, freq_maps


def build_model_matrix(df: pd.DataFrame, freq_maps: dict | None = None,
                        fit_encoders: bool = True):
    """
    Full pipeline: engineer features, then select the final X / y matrices used
    for model training or inference. Returns (X, y, feature_names, freq_maps).
    """
    engineered, freq_maps = engineer_features(df, fit_encoders=fit_encoders, freq_maps=freq_maps)

    engineered_extra = [
        "price_vs_national_avg", "deductible_remaining_frac",
        "quality_per_dollar_proxy", "is_high_deductible_stage", "urban_flag",
        "cpt_code_freq", "state_freq", "hospital_id_freq",
    ]
    base_numeric = [c for c in RAW_NUMERIC_FEATURES if c in engineered.columns]
    onehot_cols = [c for c in engineered.columns if c.startswith((
        "procedure_category_", "insurance_plan_type_", "facility_type_", "metro_tier_"
    ))]
    feature_cols = base_numeric + [c for c in engineered_extra if c in engineered.columns] + onehot_cols
    feature_cols = list(dict.fromkeys(feature_cols))  # de-dupe, preserve order

    X = engineered[feature_cols].astype(float)
    y = engineered[TARGET_COL].astype(float) if TARGET_COL in engineered.columns else None

    return X, y, feature_cols, freq_maps

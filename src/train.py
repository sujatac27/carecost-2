"""
CareCost AI — End-to-End Training Script
============================================================
Runs the full pipeline: generate/load data -> clean -> engineer features ->
train & evaluate models -> train quantile (confidence-interval) models ->
save the winning bundle to models/carecost_model.joblib.

Usage:
    python src/train.py                  # uses existing data/claims_raw.csv if present
    python src/train.py --regenerate      # regenerates synthetic data first
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import joblib
import pandas as pd

from data_generator import generate_synthetic_dataset, inject_realistic_messiness
from features import clean_claims, build_model_matrix
from modeling import train_test_prepare, train_all_models, best_model, train_quantile_models

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"


def main(regenerate: bool = False, n_claims: int = 45000):
    DATA_DIR.mkdir(exist_ok=True)
    MODELS_DIR.mkdir(exist_ok=True)
    t0 = time.time()

    raw_path = DATA_DIR / "claims_raw.csv"
    hospitals_path = DATA_DIR / "hospitals.csv"

    if regenerate or not raw_path.exists():
        print(f"Generating {n_claims:,} synthetic claims...")
        hospitals_df, claims_df = generate_synthetic_dataset(n_claims=n_claims)
        claims_df = inject_realistic_messiness(claims_df)
        hospitals_df.to_csv(hospitals_path, index=False)
        claims_df.to_csv(raw_path, index=False)
    else:
        print(f"Loading existing data from {raw_path}")
        claims_df = pd.read_csv(raw_path, dtype={"zip_code": str, "cpt_code": str, "hospital_id": str})

    print("\nCleaning data...")
    clean_df = clean_claims(claims_df)
    clean_df.to_csv(DATA_DIR / "claims_clean.csv", index=False)

    print("\nEngineering features...")
    X, y, feature_cols, freq_maps = build_model_matrix(clean_df)
    print(f"Feature matrix: {X.shape[0]:,} rows x {X.shape[1]} features")

    X_train, X_test, y_train, y_test = train_test_prepare(X, y)

    print()
    results = train_all_models(X_train, y_train, X_test, y_test)
    best = best_model(results)
    print(f"\nBest model: {best.name}  (test MAE ${best.mae:.2f}, R² {best.r2:.3f})")

    print("\nTraining confidence-interval (quantile) models...")
    lower_model, upper_model = train_quantile_models(X_train, y_train)

    bundle = {
        "model": best.model,
        "model_name": best.name.lower().replace(" ", "_").replace("*", ""),
        "feature_cols": feature_cols,
        "freq_maps": freq_maps,
        "lower_model": lower_model,
        "upper_model": upper_model,
        "scaler": getattr(best, "scaler", None),
        "results": results,
    }
    out_path = MODELS_DIR / "carecost_model.joblib"
    joblib.dump(bundle, out_path)
    print(f"\nSaved model bundle to {out_path}")
    print(f"Total pipeline time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--regenerate", action="store_true", help="Regenerate synthetic data")
    parser.add_argument("--n-claims", type=int, default=45000)
    args = parser.parse_args()
    main(regenerate=args.regenerate, n_claims=args.n_claims)

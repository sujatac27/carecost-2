"""
CareCost AI — Financial Decision Engine (Provider Optimization)
============================================================
This is the "exceptional" layer from the project brief: instead of only
answering "what will this cost?", it answers "what is the financially
optimal choice?"

For a given procedure + patient context, it:
  1. Predicts patient out-of-pocket cost at every candidate facility
     (using the trained regression model)
  2. Combines predicted cost with quality (star rating), readmission risk,
     and distance into a single weighted "value score"
  3. Ranks providers so the app can recommend the best overall choice —
     not just the cheapest one.

This mirrors how insurers build "steerage" / network-tiering models and how
asset managers build multi-factor scoring models (weighted combination of
normalized signals) — same math, different inputs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass
from math import radians, sin, cos, asin, sqrt


DEFAULT_WEIGHTS = {
    "cost": 0.50,        # predicted out-of-pocket cost (lower is better)
    "quality": 0.25,      # CMS star rating (higher is better)
    "readmission": 0.15,  # readmission rate (lower is better)
    "distance": 0.10,     # travel distance (lower is better)
}


@dataclass
class ProviderScore:
    hospital_id: str
    hospital_name: str
    predicted_cost: float
    cost_low: float
    cost_high: float
    star_rating: float
    readmission_rate_pct: float
    distance_miles: float | None
    value_score: float  # 0-100, higher = better overall choice


def haversine_miles(lat1, lon1, lat2, lon2) -> float:
    """Great-circle distance between two lat/lon points, in miles."""
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    return 2 * 6371 * asin(sqrt(a)) * 0.621371  # km -> miles


def _normalize(series: pd.Series, invert: bool = False) -> pd.Series:
    """Min-max normalize to [0, 1]. If invert=True, lower raw values score higher
    (used for cost, readmission rate, distance — where lower is better)."""
    if series.max() == series.min():
        return pd.Series(0.5, index=series.index)
    norm = (series - series.min()) / (series.max() - series.min())
    return 1 - norm if invert else norm


def rank_providers(candidates: pd.DataFrame, weights: dict | None = None) -> list[ProviderScore]:
    """
    candidates must contain columns:
        hospital_id, hospital_name, predicted_cost, cost_low, cost_high,
        star_rating, readmission_rate_pct, and optionally distance_miles.

    Returns providers ranked best-to-worst by a weighted composite value score.
    This is a multi-factor scoring model — the same technique used for
    multi-factor equity ranking or vendor/counterparty scoring in finance:
    normalize each signal to a common scale, apply business-chosen weights,
    sum to a single comparable score.
    """
    weights = weights or DEFAULT_WEIGHTS
    df = candidates.copy()

    cost_score = _normalize(df["predicted_cost"], invert=True)
    quality_score = _normalize(df["star_rating"], invert=False)
    readmission_score = _normalize(df["readmission_rate_pct"], invert=True)

    if "distance_miles" in df.columns and df["distance_miles"].notna().any():
        distance_score = _normalize(df["distance_miles"].fillna(df["distance_miles"].median()), invert=True)
    else:
        distance_score = pd.Series(0.5, index=df.index)  # neutral if unknown

    composite = (
        weights["cost"] * cost_score +
        weights["quality"] * quality_score +
        weights["readmission"] * readmission_score +
        weights["distance"] * distance_score
    )
    df["value_score"] = (composite * 100).round(1)
    df = df.sort_values("value_score", ascending=False)

    return [
        ProviderScore(
            hospital_id=row.hospital_id,
            hospital_name=row.hospital_name,
            predicted_cost=round(row.predicted_cost, 2),
            cost_low=round(row.cost_low, 2),
            cost_high=round(row.cost_high, 2),
            star_rating=row.star_rating,
            readmission_rate_pct=row.readmission_rate_pct,
            distance_miles=getattr(row, "distance_miles", None),
            value_score=row.value_score,
        )
        for row in df.itertuples(index=False)
    ]


def potential_savings(ranked: list[ProviderScore], chosen_hospital_id: str | None = None) -> float:
    """
    Computes the dollar savings available by switching from either:
      - the hospital the user specified (chosen_hospital_id), or
      - the most expensive in-network option (if none specified)
    to the best value_score option. Used to power the
    "Choosing X could save you $Y" line in the AI explanation.
    """
    if not ranked:
        return 0.0
    best = ranked[0]
    if chosen_hospital_id:
        current = next((p for p in ranked if p.hospital_id == chosen_hospital_id), ranked[-1])
    else:
        current = max(ranked, key=lambda p: p.predicted_cost)
    return max(0.0, round(current.predicted_cost - best.predicted_cost, 2))

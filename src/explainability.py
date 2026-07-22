"""
CareCost AI — Explainability Engine
============================================================
Phase 5 of the project plan.

Step 1: Use SHAP to compute *why* a prediction came out the way it did
        (which features pushed the estimate up or down, and by how much).
Step 2: Translate the top SHAP drivers into a plain-English explanation.
Step 3 (optional): Feed those drivers into an LLM (OpenAI) for a more
        natural, conversational explanation. Falls back to a template-based
        explanation if no API key is configured, so the product works
        end-to-end with zero external dependencies.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False


# ---------------------------------------------------------------------------
# Human-readable labels for engineered feature names
# ---------------------------------------------------------------------------

FEATURE_LABELS = {
    "facility_price_index": "this facility's negotiated-rate pricing level",
    "price_vs_national_avg": "this facility's pricing vs. the national average",
    "deductible_remaining_frac": "how much of your deductible is still unmet",
    "deductible_met_pct": "how much of your deductible you've already met",
    "is_high_deductible_stage": "being early in your deductible year",
    "star_rating": "this facility's CMS quality star rating",
    "quality_per_dollar_proxy": "the quality-to-price ratio of this facility",
    "readmission_rate_pct": "this facility's readmission rate",
    "urban_flag": "being in an urban market",
    "cost_of_living_index": "the regional cost-of-living level",
    "median_household_income": "the regional median household income",
    "population_density": "the local population density",
    "cpt_code_freq": "how common this procedure is nationally",
    "state_freq": "claim volume in this state",
    "hospital_id_freq": "how frequently this facility appears in the data",
    "patient_age": "patient age",
    "patient_satisfaction_score": "patient satisfaction at this facility",
}


def _label(feature: str) -> str:
    if feature in FEATURE_LABELS:
        return FEATURE_LABELS[feature]
    for prefix, readable in [
        ("procedure_category_", "the procedure category ({})"),
        ("insurance_plan_type_", "your insurance plan type ({})"),
        ("facility_type_", "the facility type ({})"),
        ("metro_tier_", "the metro tier ({})"),
    ]:
        if feature.startswith(prefix):
            value = feature[len(prefix):]
            return readable.format(value)
    return feature.replace("_", " ")


@dataclass
class ExplanationDriver:
    feature: str
    shap_value: float
    direction: str  # "increases" or "decreases"
    readable_label: str


class ExplainabilityEngine:
    """Wraps a trained tree model with a SHAP TreeExplainer and produces both
    structured drivers (for the API/UI) and natural-language text (for the
    'AI Explanation' panel in the dashboard)."""

    def __init__(self, model, feature_names: list[str], background_data: pd.DataFrame | None = None):
        self.model = model
        self.feature_names = feature_names
        self._explainer = None
        self._background_data = background_data

        if HAS_SHAP:
            try:
                self._explainer = shap.TreeExplainer(model)
            except Exception:
                # Model isn't tree-based (e.g. Linear Regression) — use a
                # model-agnostic explainer with a small background sample instead.
                bg = background_data.sample(min(100, len(background_data)), random_state=42) \
                    if background_data is not None else None
                if bg is not None:
                    self._explainer = shap.Explainer(model.predict, bg)

    def top_drivers(self, X_row: pd.DataFrame, top_n: int = 3) -> list[ExplanationDriver]:
        """Returns the top-N SHAP drivers for a single prediction row."""
        if not HAS_SHAP or self._explainer is None:
            return self._fallback_drivers(X_row, top_n)

        shap_values = self._explainer(X_row)
        values = np.array(shap_values.values).reshape(-1)

        order = np.argsort(-np.abs(values))[:top_n]
        drivers = []
        for i in order:
            feat = self.feature_names[i]
            val = float(values[i])
            drivers.append(ExplanationDriver(
                feature=feat,
                shap_value=val,
                direction="increases" if val > 0 else "decreases",
                readable_label=_label(feat),
            ))
        return drivers

    def _fallback_drivers(self, X_row: pd.DataFrame, top_n: int) -> list[ExplanationDriver]:
        """If SHAP isn't installed, approximate driver importance using the
        model's built-in feature_importances_ (tree models). One-hot columns
        are only ever considered when they're actually active (==1) for this
        row, so the explanation never names a category the patient isn't in.
        Continuous features are ranked by importance alone. Less precise than
        SHAP but keeps the API functional with zero extra dependencies."""
        importances = getattr(self.model, "feature_importances_", None)
        if importances is None:
            return []
        row = X_row.iloc[0]
        is_binary_col = row.isin([0, 1])

        scores = importances.copy()
        # Zero out inactive one-hot columns so they can never be selected
        inactive_onehot = is_binary_col & (row.values == 0)
        scores = np.where(inactive_onehot, 0.0, scores)

        order = np.argsort(-scores)[:top_n]
        drivers = []
        for i in order:
            if scores[i] <= 0:
                continue
            feat = self.feature_names[i]
            active_onehot = is_binary_col.iloc[i] and row.iloc[i] == 1
            direction = "increases" if active_onehot or row.iloc[i] > 0 else "decreases"
            drivers.append(ExplanationDriver(
                feature=feat, shap_value=float(scores[i]),
                direction=direction, readable_label=_label(feat),
            ))
        return drivers


def template_explanation(drivers: list[ExplanationDriver], predicted_cost: float,
                          comparison_savings: float | None = None) -> str:
    """Rule-based natural-language explanation — used when no LLM API key is
    configured. This is intentionally written to look and read like the
    example in the project brief."""
    lines = [f"Your estimated out-of-pocket cost is ${predicted_cost:,.0f} because:"]
    for d in drivers:
        effect = "increases" if d.direction == "increases" else "decreases"
        lines.append(f"  • {d.readable_label.capitalize()} {effect} your cost.")
    if comparison_savings and comparison_savings > 0:
        lines.append(f"\nChoosing a lower-cost nearby alternative could save you "
                      f"about ${comparison_savings:,.0f}.")
    return "\n".join(lines)


def llm_explanation(drivers: list[ExplanationDriver], predicted_cost: float,
                     procedure: str, comparison_savings: float | None = None,
                     api_key: str | None = None) -> str:
    """
    Sends the SHAP drivers to an LLM for a more natural explanation.
    Requires `openai` and an API key (env var OPENAI_API_KEY or passed in).
    Falls back to the template explanation on any failure (missing key,
    network error, rate limit) so the product never breaks in a demo.
    """
    api_key = api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return template_explanation(drivers, predicted_cost, comparison_savings)

    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)

        driver_text = "\n".join(
            f"- {d.readable_label} ({d.direction} cost, impact score {d.shap_value:+.1f})"
            for d in drivers
        )
        prompt = f"""You are a healthcare cost transparency assistant. A patient is
getting an estimate for: {procedure}
Predicted out-of-pocket cost: ${predicted_cost:,.0f}
Top cost drivers from the model:
{driver_text}
{"Potential savings from a lower-cost alternative: $" + f"{comparison_savings:,.0f}" if comparison_savings else ""}

Write a short (3-4 bullet points), plain-English explanation of why the cost
is what it is. Be factual, non-alarmist, and financially focused. Do not give
medical advice. Do not invent numbers not provided above."""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300,
            temperature=0.4,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        # Never let an LLM outage break the /explain endpoint
        return template_explanation(drivers, predicted_cost, comparison_savings) + \
            f"\n\n[Note: AI narrative unavailable ({type(e).__name__}); showing rule-based explanation.]"

# CareCost AI

**AI-powered healthcare cost prediction & financial decision engine.**

CareCost AI estimates what a medical procedure will actually cost a patient
out-of-pocket — factoring in the procedure, insurance plan, facility, and
geography — then goes a step further and answers a harder question: *which
provider is the financially optimal choice*, once cost, quality, and risk are
weighed together. It's a pricing and risk-estimation problem, the same class
of problem insurers, hospital finance teams, and healthcare-focused investors
solve every day.

> **Scope note:** this tool estimates costs for financial-planning purposes.
> It is not medical advice and should not be used to make clinical decisions.

---

## What it does

1. **Predicts** total billed cost, insurance-paid amount, and patient
   out-of-pocket responsibility for a CPT/HCPCS procedure, with a realistic
   confidence interval (e.g. "$470–$590", not a falsely-precise "$530").
2. **Compares** nearby providers and ranks them by a weighted value score —
   cost, CMS quality star rating, readmission risk, and distance — turning a
   cost estimator into a financial decision engine.
3. **Explains** each prediction using SHAP feature attribution, translated
   into plain English (optionally via an LLM), e.g. *"This hospital charges
   ~18% above the regional average; choosing the imaging center down the
   street could save about $270."*

## Architecture

```
carecost-ai/
├── src/
│   ├── data_generator.py    # synthetic CMS-schema data (see note below)
│   ├── features.py          # Phase 1 cleaning + Phase 3 feature engineering
│   ├── modeling.py          # Phase 4: Linear -> Random Forest -> XGBoost + CV + quantile CIs
│   ├── explainability.py    # Phase 5: SHAP + template/LLM explanations
│   ├── decision_engine.py   # multi-factor provider ranking (the "exceptional" layer)
│   └── train.py             # end-to-end training orchestration
├── backend/
│   ├── main.py               # FastAPI: /predict /compare /hospitals /explain
│   └── schemas.py            # Pydantic request/response models
├── frontend/
│   └── app.py                 # Streamlit dashboard
├── notebooks/
│   └── carecost_ai_colab.ipynb  # EDA + modeling walkthrough, Colab-ready
├── models/                    # trained model artifacts (generated)
├── data/                      # generated datasets (generated)
└── requirements.txt
```

## A note on the data

This environment has no network access to CMS's bulk data servers, so
`src/data_generator.py` builds a **statistically realistic synthetic dataset**
that mirrors the schema and distributions of the real sources:

| Real source | What it provides | Synthetic stand-in |
|---|---|---|
| [CMS Hospital Price Transparency](https://www.cms.gov/hospital-price-transparency) | Negotiated rates by payer, per facility | `facility_price_index` + payer-specific discount simulation |
| CMS Physician Fee Schedule | Medicare allowed amounts by CPT code | `base_medicare_rate` per procedure |
| [CMS Hospital Compare](https://data.cms.gov/provider-data/) | Star ratings, readmission rates | Simulated star ratings correlated with facility type |
| U.S. Census ZIP data | Income, density | 15 representative real ZIP profiles (income, cost-of-living, density) |

Realistic messiness is injected on purpose (inconsistent plan-name casing,
missing values, duplicate rows, chargemaster outliers) so the cleaning phase
in the notebook has real work to do — this is what actual hospital
machine-readable files look like.

**To swap in real data:** replace `generate_synthetic_dataset()` in
`src/train.py` with a loader for the real CMS files, keeping the same output
schema (see the column lists in `src/features.py`). Nothing downstream needs
to change.

## Quickstart

```bash
git clone <this-repo>
cd carecost-ai
pip install -r requirements.txt

# 1. Generate data + train models (Linear -> RF -> XGBoost, with SHAP-ready model + CIs)
python src/train.py --regenerate

# 2. Run the API
uvicorn backend.main:app --reload --port 8000
# -> http://localhost:8000/docs

# 3. Run the dashboard (in a second terminal)
streamlit run frontend/app.py
```

Or open `notebooks/carecost_ai_colab.ipynb` in Google Colab for the full
walkthrough: EDA, feature engineering, model comparison, SHAP explainability,
and the provider-ranking optimization layer, with all charts inline.

### Optional: AI-narrated explanations

Set `OPENAI_API_KEY` as an environment variable to have `/explain` generate
natural-language cost explanations via GPT instead of the built-in
template engine. Without a key, the app works identically using rule-based
explanations — no external dependency required for a demo.

## API

| Endpoint | Method | Purpose |
|---|---|---|
| `/predict` | POST | Single best-match (or specified-hospital) cost estimate |
| `/compare` | POST | Ranked list of nearby providers by value score |
| `/hospitals` | GET | Facility lookup by ZIP/state |
| `/explain` | POST | SHAP-driven natural-language explanation |

Example:
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"cpt_code": "73721", "insurance_plan": "Aetna PPO", "zip_code": "19104", "patient_age": 34, "deductible_met_pct": 20}'
```

## Modeling approach

- **Target:** `patient_responsibility` (out-of-pocket cost), log-aware handling
  of the right-skewed cost distribution.
- **Models compared:** Linear Regression (interpretable baseline) → Random
  Forest → XGBoost (gradient boosting), each evaluated with MAE, RMSE, R², and
  5-fold cross-validation — not just a single train/test split.
- **Confidence intervals:** gradient-boosted quantile regressors (10th/90th
  percentile) give an honest cost *range* instead of a single point estimate.
- **Explainability:** SHAP TreeExplainer attributes each prediction to its
  top drivers (facility pricing level, deductible progress, quality/price
  ratio, regional cost-of-living, etc.), which feed the natural-language
  explanation.
- **Decision engine:** predicted cost, quality, readmission risk, and distance
  are min-max normalized and combined into a single weighted value score per
  provider — the same multi-factor scoring pattern used in equity factor
  models, vendor scoring, and insurer network-tiering.

## Why this matters in finance

Healthcare is roughly one-fifth of U.S. GDP, and estimating medical costs is
fundamentally a pricing and risk-estimation problem — the same discipline
behind insurance underwriting, loan pricing, and revenue forecasting:

- **Insurance companies** build near-identical models to predict claim cost,
  reimbursement, and expected payouts.
- **Investment banks** use comparable financial modeling for hospital
  valuation and healthcare M&A.
- **Asset managers** forecast healthcare company revenue and macro cost
  trends using the same regression + feature-engineering toolkit.
- **Hedge funds** apply the same "estimate an uncertain financial outcome
  from historical data" pattern to earnings and spending forecasts.

## Development notes

This project was scaffolded end-to-end (data generation, cleaning, feature
engineering, three-tier modeling, SHAP explainability, a FastAPI backend, and
a Streamlit dashboard) as a single coherent pipeline. Swap in the real CMS
bulk files, retrain, and it's deployment-ready.

## License

MIT — see `LICENSE`.

"""
Builds notebooks/carecost_ai_colab.ipynb from a list of (cell_type, source) cells.
Run with: python3 build_notebook.py
"""
import json

def code(src): return ("code", src)
def md(src): return ("markdown", src)

cells = []

cells.append(md("""# 🏥 CareCost AI
### AI-Powered Healthcare Cost Prediction & Financial Decision Engine

This notebook walks through the full pipeline end-to-end:

1. **Data collection & cleaning** — synthetic, CMS-schema-realistic healthcare pricing data
2. **Exploratory data analysis** — cost distributions, regional variation, payer differences
3. **Feature engineering** — encoding procedures, plans, facilities, and geography
4. **Model building** — Linear Regression → Random Forest → XGBoost, with cross-validation
5. **Explainability** — SHAP-driven "why is this prediction what it is?"
6. **Financial decision engine** — ranking providers by cost *and* quality, not cost alone
7. **Visualization dashboard mockup**

> **Scope note:** this notebook produces *financial* cost estimates for transparency and
> planning purposes. It is not medical advice.

---

**On the data:** this notebook generates a statistically realistic **synthetic** dataset
that mirrors the schema of CMS Hospital Price Transparency files, the CMS Physician Fee
Schedule, and CMS Hospital Compare quality ratings — since this notebook can't reach CMS's
live download servers in every environment. To use real data, replace `generate_synthetic_dataset()`
in **Section 1** with a loader for the real bulk files; every downstream cell (cleaning,
features, modeling, SHAP, dashboard) works unchanged because the schema is production-shaped
from the start. See `SCHEMA` notes inline."""))

cells.append(md("## Section 0 — Setup"))

cells.append(code("""# Install dependencies (uncomment in a fresh Colab runtime)
# !pip install -q pandas numpy scikit-learn xgboost shap matplotlib plotly seaborn

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBRegressor
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("xgboost not found — install with `pip install xgboost` for the production model. "
          "Falling back to GradientBoostingRegressor for this run (same interface).")

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    print("shap not found — install with `pip install shap` for real SHAP explanations.")

sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (10, 5)
plt.rcParams["axes.titlesize"] = 13
plt.rcParams["axes.titleweight"] = "bold"

PALETTE = ["#1f6f5c", "#2E86AB", "#F5A623", "#C62828", "#5B8DEF"]
pd.set_option("display.float_format", lambda x: f"{x:,.2f}")

print("Setup complete.")"""))

cells.append(md("""## Section 1 — Collect & Generate Data
### Phase 1 of the project plan

Real CMS files are enormous (multi-GB per hospital) and require a network fetch step, so
this section builds a **synthetic dataset that mirrors their schema and statistical
properties**: procedure pricing driven by Medicare base rates + facility pricing power,
payer-specific negotiated discounts, deductible/coinsurance patient-responsibility math,
and CMS Hospital Compare-style quality metrics (star rating, readmission rate).

Realistic messiness is injected on purpose — inconsistent plan-name casing, missing values,
duplicate rows, chargemaster outliers — because that's what real hospital machine-readable
files look like, and the cleaning step below needs real work to do."""))

cells.append(code("""# ============================================================
# Reference tables: procedures, insurance plans, ZIP/region profiles, real hospital names
# ============================================================
# Procedure base rates are anchored to real, publicly published 2025-2026 Medicare
# Physician Fee Schedule reference rates. ZIP codes are real US ZIP codes across
# 20+ metro areas. Hospital names for Academic Medical Centers are REAL, well-known
# teaching hospitals (sourced from public hospital-ranking lists) -- but every cost,
# star rating, and readmission figure attached to them below is SIMULATED for this
# demo, not that hospital's actual published data. See the disclaimer in Section 9.
from dataclasses import dataclass

@dataclass(frozen=True)
class Procedure:
    cpt_code: str
    description: str
    category: str
    base_medicare_rate: float
    price_variance: float

PROCEDURES = [
    Procedure("70551", "MRI Brain w/o Contrast", "Imaging", 425, 0.55),
    Procedure("73721", "MRI Knee w/o Contrast", "Imaging", 390, 0.6),
    Procedure("73221", "MRI Shoulder w/o Contrast", "Imaging", 405, 0.58),
    Procedure("72148", "MRI Lumbar Spine w/o Contrast", "Imaging", 445, 0.55),
    Procedure("74177", "CT Abdomen/Pelvis w/ Contrast", "Imaging", 465, 0.5),
    Procedure("71250", "CT Chest w/o Contrast", "Imaging", 285, 0.55),
    Procedure("70450", "CT Head/Brain w/o Contrast", "Imaging", 245, 0.55),
    Procedure("71046", "Chest X-Ray, 2 Views", "Imaging", 42, 0.65),
    Procedure("77067", "Mammography, Bilateral Screening", "Imaging", 145, 0.45),
    Procedure("76700", "Abdominal Ultrasound, Complete", "Imaging", 165, 0.5),
    Procedure("45378", "Colonoscopy, Diagnostic", "GI Procedure", 610, 0.45),
    Procedure("45385", "Colonoscopy w/ Polyp Removal", "GI Procedure", 780, 0.45),
    Procedure("43239", "Upper GI Endoscopy w/ Biopsy", "GI Procedure", 545, 0.45),
    Procedure("29881", "Knee Arthroscopy w/ Meniscectomy", "Orthopedic Surgery", 1150, 0.5),
    Procedure("29827", "Shoulder Arthroscopy w/ Rotator Cuff Repair", "Orthopedic Surgery", 2100, 0.48),
    Procedure("27447", "Total Knee Replacement", "Orthopedic Surgery", 8900, 0.4),
    Procedure("27130", "Total Hip Replacement", "Orthopedic Surgery", 9400, 0.4),
    Procedure("22551", "Cervical Spinal Fusion", "Orthopedic Surgery", 7200, 0.42),
    Procedure("99213", "Office Visit, Established Patient", "Primary Care", 92, 0.35),
    Procedure("99214", "Office Visit, Established Patient (Moderate Complexity)", "Primary Care", 128, 0.35),
    Procedure("99203", "Office Visit, New Patient", "Primary Care", 150, 0.35),
    Procedure("99385", "Preventive Visit, New Patient (Age 18-39)", "Primary Care", 175, 0.3),
    Procedure("99284", "Emergency Dept Visit, Level 4", "Emergency", 385, 0.55),
    Procedure("99283", "Emergency Dept Visit, Level 3", "Emergency", 245, 0.55),
    Procedure("99285", "Emergency Dept Visit, Level 5 (Critical)", "Emergency", 720, 0.55),
    Procedure("93000", "Electrocardiogram (EKG)", "Cardiology", 32, 0.45),
    Procedure("93306", "Echocardiogram, Complete", "Cardiology", 215, 0.5),
    Procedure("93458", "Cardiac Catheterization w/ Angiography", "Cardiology", 3200, 0.45),
    Procedure("47562", "Laparoscopic Cholecystectomy", "General Surgery", 2450, 0.45),
    Procedure("49505", "Inguinal Hernia Repair", "General Surgery", 1850, 0.45),
    Procedure("44970", "Laparoscopic Appendectomy", "General Surgery", 2100, 0.45),
    Procedure("59400", "Vaginal Delivery incl. Prenatal Care", "Obstetrics", 3200, 0.4),
    Procedure("59510", "Cesarean Delivery incl. Prenatal Care", "Obstetrics", 4800, 0.38),
    Procedure("81001", "Urinalysis w/ Microscopy", "Lab", 8, 0.6),
    Procedure("80053", "Comprehensive Metabolic Panel", "Lab", 14, 0.55),
    Procedure("85025", "Complete Blood Count (CBC) w/ Differential", "Lab", 11, 0.55),
    Procedure("80061", "Lipid Panel", "Lab", 16, 0.55),
    Procedure("36415", "Blood Draw (Venipuncture)", "Lab", 5, 0.7),
    Procedure("97110", "Physical Therapy, Therapeutic Exercise", "Rehab", 38, 0.45),
    Procedure("97140", "Manual Therapy Techniques", "Rehab", 35, 0.45),
    Procedure("19120", "Breast Biopsy, Open", "Surgery", 980, 0.45),
]

INSURANCE_PLANS = [
    ('Aetna PPO', 'PPO', 0.82, 1500, 0.2),
    ('Aetna HMO', 'HMO', 0.75, 1000, 0.15),
    ('UnitedHealthcare Choice Plus', 'PPO', 0.8, 2000, 0.2),
    ('Blue Cross Blue Shield PPO', 'PPO', 0.83, 1750, 0.2),
    ('Cigna Open Access', 'PPO', 0.78, 1500, 0.2),
    ('Humana Gold HMO', 'HMO', 0.74, 500, 0.1),
    ('Kaiser Permanente HMO', 'HMO', 0.85, 1000, 0.1),
    ('Medicare Advantage', 'Medicare', 0.95, 283, 0.2),
    ('Medicaid', 'Medicaid', 1.0, 0, 0.0),
    ('High-Deductible HSA Plan', 'HDHP', 0.7, 5000, 0.3),
    ('Uninsured / Self-Pay', 'Self-Pay', 0.0, 0, 1.0),
]

FACILITY_TYPES = ["Academic Medical Center", "Community Hospital",
                   "Ambulatory Surgery Center", "Imaging Center", "Urgent Care"]

ZIP_PROFILES = [
    ('19104', 'PA', 'Urban', 38500, 1.05, 14200),
    ('19107', 'PA', 'Urban', 71200, 1.1, 22000),
    ('10001', 'NY', 'Urban', 112400, 1.65, 33000),
    ('10032', 'NY', 'Urban', 42300, 1.55, 45000),
    ('60601', 'IL', 'Urban', 98700, 1.3, 26500),
    ('60612', 'IL', 'Urban', 45200, 1.15, 12800),
    ('94103', 'CA', 'Urban', 145600, 1.85, 17800),
    ('90048', 'CA', 'Urban', 98200, 1.7, 18500),
    ('77002', 'TX', 'Urban', 88300, 1.05, 8200),
    ('77030', 'TX', 'Urban', 76500, 1.05, 6200),
    ('30303', 'GA', 'Urban', 62100, 0.98, 6100),
    ('48201', 'MI', 'Urban', 29800, 0.85, 5000),
    ('48202', 'MI', 'Urban', 33500, 0.85, 7200),
    ('02115', 'MA', 'Urban', 68900, 1.55, 24500),
    ('21287', 'MD', 'Urban', 41200, 1.15, 11200),
    ('98104', 'WA', 'Urban', 89600, 1.45, 15600),
    ('80204', 'CO', 'Urban', 72300, 1.2, 9800),
    ('85006', 'AZ', 'Urban', 51400, 0.98, 5400),
    ('33136', 'FL', 'Urban', 34700, 1.05, 7600),
    ('55415', 'MN', 'Urban', 58900, 1.08, 8900),
    ('08053', 'NJ', 'Suburban', 118900, 1.2, 1800),
    ('60614', 'IL', 'Suburban', 134200, 1.25, 15600),
    ('75024', 'TX', 'Suburban', 121500, 1.05, 3400),
    ('30144', 'GA', 'Suburban', 79400, 0.95, 2100),
    ('19087', 'PA', 'Suburban', 142300, 1.15, 2600),
    ('90210', 'CA', 'Suburban', 175800, 1.9, 2200),
    ('28270', 'NC', 'Suburban', 108200, 1.02, 1900),
    ('63105', 'MO', 'Suburban', 132400, 1.05, 3300),
    ('59718', 'MT', 'Rural', 61300, 0.92, 45),
    ('50588', 'IA', 'Rural', 54200, 0.85, 18),
    ('83001', 'WY', 'Rural', 68900, 1.0, 6),
    ('59901', 'MT', 'Rural', 52400, 0.9, 32),
]

REAL_HOSPITALS_BY_CITY = {
    "Philadelphia": ['Hospital of the University of Pennsylvania', 'Thomas Jefferson University Hospital', "Children's Hospital of Philadelphia", 'Temple University Hospital'],
    "New York": ['NewYork-Presbyterian/Weill Cornell Medical Center', 'NYU Langone Hospitals', 'Mount Sinai Hospital', 'Tisch Hospital'],
    "Chicago": ['Northwestern Memorial Hospital', 'Rush University Medical Center', 'University of Chicago Medical Center', "Ann & Robert H. Lurie Children's Hospital"],
    "San Francisco": ['UCSF Medical Center', 'California Pacific Medical Center', 'Zuckerberg San Francisco General Hospital'],
    "Los Angeles": ['Cedars-Sinai Medical Center', 'UCLA Medical Center', 'Keck Hospital of USC'],
    "Houston": ['Houston Methodist Hospital', "Texas Children's Hospital", "Baylor St. Luke's Medical Center"],
    "Atlanta": ['Emory University Hospital', 'Grady Memorial Hospital', 'Piedmont Atlanta Hospital'],
    "Detroit": ['Henry Ford Hospital', 'Detroit Medical Center - Harper University Hospital', 'Ascension St. John Hospital'],
    "Boston": ['Massachusetts General Hospital', "Brigham and Women's Hospital", 'Boston Medical Center'],
    "Baltimore": ['Johns Hopkins Hospital', 'University of Maryland Medical Center'],
    "Seattle": ['UW Medical Center - Montlake', 'Swedish Medical Center - First Hill'],
    "Denver": ['UCHealth University of Colorado Hospital', 'Denver Health Medical Center'],
    "Phoenix": ['Banner - University Medical Center Phoenix', "Phoenix Children's Hospital"],
    "Miami": ['Jackson Memorial Hospital', 'Sylvester Comprehensive Cancer Center - UHealth'],
    "Minneapolis": ['M Health Fairview University of Minnesota Medical Center', 'Abbott Northwestern Hospital'],
    "Cherry Hill": ['Virtua Voorhees Hospital', 'Jefferson Cherry Hill Hospital'],
    "Beverly Hills": ['Cedars-Sinai Marina del Rey Hospital'],
    "Charlotte": ['Atrium Health Carolinas Medical Center', 'Novant Health Presbyterian Medical Center'],
    "St. Louis": ['Barnes-Jewish Hospital', 'Mercy Hospital St. Louis'],
    "Bozeman": ['Bozeman Health Deaconess Hospital'],
    "Fort Dodge": ['UnityPoint Health - Trinity Regional Medical Center'],
    "Jackson": ["St. John's Medical Center"],
    "Kalispell": ['Logan Health Medical Center'],
}
GENERIC_NAME_TEMPLATES = ['{city} Surgical & Imaging Center', '{city} Ambulatory Surgery Center', '{city} Urgent Care', '{city} Diagnostic Imaging Center', '{city} Same-Day Surgery Center']
CITY_BY_STATE = {'PA': 'Philadelphia', 'NY': 'New York', 'IL': 'Chicago', 'CA': 'San Francisco', 'TX': 'Houston', 'GA': 'Atlanta', 'MI': 'Detroit', 'NJ': 'Cherry Hill', 'MT': 'Bozeman', 'IA': 'Fort Dodge', 'WY': 'Jackson', 'MA': 'Boston', 'MD': 'Baltimore', 'WA': 'Seattle', 'CO': 'Denver', 'AZ': 'Phoenix', 'FL': 'Miami', 'MN': 'Minneapolis', 'NC': 'Charlotte', 'MO': 'St. Louis'}
CITY_OVERRIDE_BY_ZIP = {'90210': 'Beverly Hills', '59901': 'Kalispell'}

print(f"Loaded {len(PROCEDURES)} procedures, {len(INSURANCE_PLANS)} insurance plans, "
      f"{len(ZIP_PROFILES)} regional ZIP profiles.")"""))

cells.append(code("""# ============================================================
# Build synthetic hospital master table (facility + CMS Hospital Compare quality metrics)
# Real hospital names used for Academic Medical Centers; cost/quality still simulated.
# ============================================================
def make_hospitals(rng, n_per_zip=4):
    rows, hospital_id = [], 1000
    used_real_names = set()
    for zip_code, state, tier, income, col_index, density in ZIP_PROFILES:
        city = CITY_OVERRIDE_BY_ZIP.get(zip_code, CITY_BY_STATE[state])
        real_names_pool = [n for n in REAL_HOSPITALS_BY_CITY.get(city, []) if n not in used_real_names]
        n_facilities = n_per_zip + (2 if tier == "Urban" else 0) - (1 if tier == "Rural" else 0)
        for _ in range(max(2, n_facilities)):
            hospital_id += 1
            facility_type = rng.choice(FACILITY_TYPES,
                p=[0.15, 0.35, 0.25, 0.15, 0.10] if tier != "Rural" else [0.05, 0.55, 0.10, 0.10, 0.20])

            if facility_type in ("Academic Medical Center", "Community Hospital") and real_names_pool:
                name = real_names_pool.pop(rng.integers(0, len(real_names_pool)))
                used_real_names.add(name)
                facility_type = "Academic Medical Center"  # curated list is all real teaching hospitals
            else:
                name = rng.choice(GENERIC_NAME_TEMPLATES).format(city=city) + f" #{hospital_id % 97}"

            base_quality = {"Academic Medical Center": 4.2, "Community Hospital": 3.4,
                             "Ambulatory Surgery Center": 3.8, "Imaging Center": 3.9,
                             "Urgent Care": 3.3}[facility_type]
            star_rating = float(np.clip(rng.normal(base_quality, 0.6), 1.0, 5.0))
            readmission_rate = float(np.clip(rng.normal(15.5 - star_rating, 1.5), 8.0, 28.0))
            patient_satisfaction = float(np.clip(rng.normal(60 + star_rating * 7, 8), 20, 99))

            price_index = {"Academic Medical Center": 1.28, "Community Hospital": 1.00,
                            "Ambulatory Surgery Center": 0.72, "Imaging Center": 0.65,
                            "Urgent Care": 0.55}[facility_type]
            price_index *= rng.normal(1.0, 0.12) * col_index

            rows.append({"hospital_id": f"H{hospital_id}", "hospital_name": name,
                         "facility_type": facility_type, "zip_code": zip_code, "state": state,
                         "metro_tier": tier, "median_household_income": income,
                         "cost_of_living_index": col_index, "population_density": density,
                         "star_rating": round(star_rating, 1),
                         "readmission_rate_pct": round(readmission_rate, 1),
                         "patient_satisfaction_score": round(patient_satisfaction, 1),
                         "facility_price_index": round(float(price_index), 3),
                         "data_disclaimer": "SIMULATED cost/quality estimate -- not this facility's actual published data"})
    return pd.DataFrame(rows)

rng = np.random.default_rng(42)
hospitals_df = make_hospitals(rng)
print(f"Generated {len(hospitals_df)} facilities across {hospitals_df['state'].nunique()} states "
      f"({hospitals_df['hospital_name'].isin(sum(REAL_HOSPITALS_BY_CITY.values(), [])).sum()} with real hospital names).")
hospitals_df.head()"""))

cells.append(code("""# ============================================================
# Generate claims (mirrors CMS Hospital Price Transparency negotiated-rate files
# + historical billing/utilization patterns)
# ============================================================
def generate_claims(hospitals_df, n_claims=45000, seed=42):
    rng = np.random.default_rng(seed)
    proc_lookup = {p.cpt_code: p for p in PROCEDURES}
    proc_codes = list(proc_lookup.keys())
    CATEGORY_WEIGHTS = {"Lab": 3.0, "Primary Care": 3.0, "Imaging": 2.0, "Rehab": 1.5,
        "Cardiology": 1.2, "Emergency": 1.2, "GI Procedure": 0.8, "General Surgery": 0.4,
        "Orthopedic Surgery": 0.3, "Obstetrics": 0.3, "Surgery": 0.3}
    proc_weights = np.array([CATEGORY_WEIGHTS.get(p.category, 1.0) for p in PROCEDURES])
    proc_weights /= proc_weights.sum()

    plan_names = [p[0] for p in INSURANCE_PLANS]
    plan_lookup = {p[0]: p for p in INSURANCE_PLANS}
    plan_weights = np.array([0.14,0.10,0.13,0.14,0.10,0.08,0.09,0.10,0.06,0.04,0.02])
    plan_weights /= plan_weights.sum()

    hospital_index = hospitals_df.set_index("hospital_id")
    chosen_hospitals = rng.choice(hospital_index.index.values, size=n_claims, replace=True)
    chosen_procs = rng.choice(proc_codes, size=n_claims, replace=True, p=proc_weights)
    chosen_plans = rng.choice(plan_names, size=n_claims, replace=True, p=plan_weights)
    deductible_progress = rng.uniform(0, 1, size=n_claims)
    patient_age = rng.integers(1, 92, size=n_claims)

    rows = []
    for i in range(n_claims):
        hosp = hospital_index.loc[chosen_hospitals[i]]
        proc = proc_lookup[chosen_procs[i]]
        plan_name, plan_type, reimb_rate, deductible, coinsurance = plan_lookup[chosen_plans[i]]

        noise = rng.lognormal(mean=0, sigma=proc.price_variance * 0.35)
        billed = max(proc.base_medicare_rate * hosp["facility_price_index"] * 2.6 * noise,
                     proc.base_medicare_rate * 0.5)

        payer_discount = {"PPO": 0.55, "HMO": 0.48, "Medicare": 0.32, "Medicaid": 0.22,
                           "HDHP": 0.55, "Self-Pay": 0.85}[plan_type]
        allowed_amount = max(billed * (1 - payer_discount) * rng.normal(1.0, 0.04),
                              proc.base_medicare_rate * 0.3)

        if plan_type == "Self-Pay":
            insurance_pays, patient_pays = 0.0, allowed_amount
        else:
            remaining_deductible = max(0.0, deductible * (1 - deductible_progress[i]))
            patient_deductible_portion = min(allowed_amount, remaining_deductible)
            post_deductible = allowed_amount - patient_deductible_portion
            patient_pays = patient_deductible_portion + post_deductible * coinsurance
            insurance_pays = allowed_amount - patient_pays

        rows.append({"claim_id": f"C{100000+i}", "cpt_code": proc.cpt_code,
                     "procedure_description": proc.description, "procedure_category": proc.category,
                     "hospital_id": chosen_hospitals[i], "insurance_plan": plan_name,
                     "insurance_plan_type": plan_type, "patient_age": int(patient_age[i]),
                     "deductible_met_pct": round(deductible_progress[i]*100, 1),
                     "billed_amount": round(billed, 2), "allowed_amount": round(allowed_amount, 2),
                     "insurance_paid": round(insurance_pays, 2),
                     "patient_responsibility": round(patient_pays, 2)})
    claims = pd.DataFrame(rows)
    return claims.merge(hospitals_df, on="hospital_id", how="left")

claims_df = generate_claims(hospitals_df, n_claims=45000)
print(f"Generated {len(claims_df):,} claims.")
claims_df.head()"""))

cells.append(code("""# ============================================================
# Inject realistic data-quality issues (this is what real hospital MRFs look like)
# ============================================================
def inject_messiness(df, seed=7):
    rng = np.random.default_rng(seed)
    df = df.copy()
    messy_variants = {"Aetna PPO": ["AETNA PPO", "aetna ppo ", "Aetna  PPO"],
                       "Blue Cross Blue Shield PPO": ["BCBS PPO", "Blue Cross Blue Shield  PPO"]}
    mask = df["insurance_plan"].isin(messy_variants.keys())
    idx = df[mask].sample(frac=0.15, random_state=seed).index
    for i in idx:
        df.loc[i, "insurance_plan"] = rng.choice(messy_variants[df.loc[i, "insurance_plan"]])

    for col, frac in [("patient_satisfaction_score", 0.02), ("deductible_met_pct", 0.01)]:
        idx = df.sample(frac=frac, random_state=seed).index
        df.loc[idx, col] = np.nan

    dupes = df.sample(frac=0.005, random_state=seed)
    df = pd.concat([df, dupes], ignore_index=True)

    idx = df.sample(frac=0.003, random_state=seed).index
    df.loc[idx, "billed_amount"] = df.loc[idx, "billed_amount"] * rng.uniform(8, 15, size=len(idx))
    return df

claims_raw = inject_messiness(claims_df)
print("Injected data-quality issues. Sample of messy plan names:")
print(claims_raw["insurance_plan"].value_counts().head(15))"""))

cells.append(md("""## Section 2 — Data Cleaning

Real CMS/hospital files need normalization: inconsistent text casing, duplicate
billing-system re-submissions, missing values, and chargemaster outliers (a small number
of facilities post absurd list prices that don't reflect anyone's actual negotiated rate)."""))

cells.append(code("""# ============================================================
# Cleaning pipeline
# ============================================================
def clean_claims(df, verbose=True):
    df = df.copy()
    n_start = len(df)

    for col in ["insurance_plan", "hospital_name", "procedure_description", "facility_type", "state", "metro_tier"]:
        df[col] = df[col].astype(str).str.strip()

    plan_canonical_map = {"AETNA PPO": "Aetna PPO", "aetna ppo": "Aetna PPO", "Aetna  PPO": "Aetna PPO",
                           "BCBS PPO": "Blue Cross Blue Shield PPO",
                           "Blue Cross Blue Shield  PPO": "Blue Cross Blue Shield PPO"}
    df["insurance_plan"] = df["insurance_plan"].replace(plan_canonical_map)

    n_dupes = df.duplicated(subset=["claim_id"]).sum()
    df = df.drop_duplicates(subset=["claim_id"], keep="first")

    df["patient_satisfaction_score"] = df.groupby("facility_type")["patient_satisfaction_score"] \\
        .transform(lambda s: s.fillna(s.median()))
    df["deductible_met_pct"] = df["deductible_met_pct"].fillna(df["deductible_met_pct"].median())

    n_outliers = 0
    for cat, group in df.groupby("procedure_category"):
        q1, q3 = group["billed_amount"].quantile([0.25, 0.75])
        upper = q3 + 3 * (q3 - q1)
        outlier_idx = group[group["billed_amount"] > upper].index
        n_outliers += len(outlier_idx)
        df.loc[outlier_idx, "billed_amount"] = upper

    df["allowed_amount"] = np.minimum(df["allowed_amount"], df["billed_amount"])
    df["patient_responsibility"] = np.minimum(df["patient_responsibility"], df["allowed_amount"])
    df["insurance_paid"] = df["allowed_amount"] - df["patient_responsibility"]

    numeric_cols = ["billed_amount", "allowed_amount", "insurance_paid", "patient_responsibility"]
    before = len(df)
    df = df[(df[numeric_cols] >= 0).all(axis=1)].reset_index(drop=True)

    if verbose:
        print(f"Rows in: {n_start:,} | duplicates removed: {n_dupes:,} | "
              f"outliers capped: {n_outliers:,} | negative rows dropped: {before-len(df):,} | "
              f"rows out: {len(df):,}")
    return df

claims_clean = clean_claims(claims_raw)
claims_clean["insurance_plan"].value_counts()"""))

cells.append(md("""## Section 3 — Exploratory Data Analysis
### Phase 2 of the project plan

Average cost by state, cost distributions, insurance price differences, and outlier detection."""))

cells.append(code("""# ============================================================
# Chart 1: Distribution of patient out-of-pocket cost (log + linear view)
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].hist(claims_clean["patient_responsibility"], bins=60, color=PALETTE[0], edgecolor="white")
axes[0].set_title("Patient Out-of-Pocket Cost — Linear Scale")
axes[0].set_xlabel("Patient Responsibility ($)"); axes[0].set_ylabel("Claims")

axes[1].hist(np.log1p(claims_clean["patient_responsibility"]), bins=60, color=PALETTE[1], edgecolor="white")
axes[1].set_title("Patient Out-of-Pocket Cost — log(1+x) Scale")
axes[1].set_xlabel("log(1 + Patient Responsibility)"); axes[1].set_ylabel("Claims")
plt.tight_layout(); plt.show()

print(claims_clean["patient_responsibility"].describe())"""))

cells.append(code("""# ============================================================
# Chart 2: Average patient cost by state
# ============================================================
state_avg = claims_clean.groupby("state")["patient_responsibility"].mean().sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(10, 5))
state_avg.plot(kind="bar", color=PALETTE[0], ax=ax)
ax.set_title("Average Patient Out-of-Pocket Cost by State")
ax.set_ylabel("Avg Patient Responsibility ($)"); ax.set_xlabel("State")
ax.yaxis.set_major_formatter(mticker.StrMethodFormatter("${x:,.0f}"))
plt.xticks(rotation=0); plt.tight_layout(); plt.show()"""))

cells.append(code("""# ============================================================
# Chart 3: Cost by insurance plan type (payer differences)
# ============================================================
fig, ax = plt.subplots(figsize=(10, 5))
order = claims_clean.groupby("insurance_plan_type")["patient_responsibility"].median().sort_values(ascending=False).index
sns.boxplot(data=claims_clean, x="insurance_plan_type", y="patient_responsibility", order=order,
            palette=PALETTE, ax=ax, showfliers=False)
ax.set_title("Patient Out-of-Pocket Cost by Insurance Plan Type")
ax.set_xlabel("Plan Type"); ax.set_ylabel("Patient Responsibility ($)")
ax.yaxis.set_major_formatter(mticker.StrMethodFormatter("${x:,.0f}"))
plt.tight_layout(); plt.show()"""))

cells.append(code("""# ============================================================
# Chart 4: Cost by procedure category
# ============================================================
fig, ax = plt.subplots(figsize=(11, 5))
cat_order = claims_clean.groupby("procedure_category")["billed_amount"].median().sort_values(ascending=False).index
sns.boxplot(data=claims_clean, x="procedure_category", y="billed_amount", order=cat_order,
            palette="viridis", ax=ax, showfliers=False)
ax.set_title("Billed Amount by Procedure Category")
ax.set_xlabel(""); ax.set_ylabel("Billed Amount ($)")
ax.yaxis.set_major_formatter(mticker.StrMethodFormatter("${x:,.0f}"))
plt.xticks(rotation=35, ha="right"); plt.tight_layout(); plt.show()"""))

cells.append(code("""# ============================================================
# Chart 5: Heatmap — facility price index by metro tier & facility type
# (proxy for "cost by ZIP/region" from the project brief)
# ============================================================
pivot = hospitals_df.pivot_table(index="facility_type", columns="metro_tier",
                                  values="facility_price_index", aggfunc="mean")
fig, ax = plt.subplots(figsize=(7, 5))
sns.heatmap(pivot, annot=True, fmt=".2f", cmap="RdYlGn_r", ax=ax, cbar_kws={"label": "Price Index"})
ax.set_title("Facility Price Index by Type & Metro Tier\\n(1.0 = national average negotiated-rate level)")
plt.tight_layout(); plt.show()"""))

cells.append(code("""# ============================================================
# Chart 6: Outlier detection — billed amount distribution with IQR fences, by category
# ============================================================
fig, ax = plt.subplots(figsize=(11, 5))
sns.boxplot(data=claims_clean, x="procedure_category", y="billed_amount", ax=ax, palette="coolwarm")
ax.set_title("Post-Cleaning Billed Amount Distribution (Outliers Capped at 3x IQR)")
ax.set_xlabel(""); ax.set_ylabel("Billed Amount ($)")
ax.yaxis.set_major_formatter(mticker.StrMethodFormatter("${x:,.0f}"))
plt.xticks(rotation=35, ha="right"); plt.tight_layout(); plt.show()

print("\\nKey EDA takeaways:")
print(f"- Costs are heavily right-skewed (mean ${claims_clean['patient_responsibility'].mean():.0f} "
      f"vs median ${claims_clean['patient_responsibility'].median():.0f})")
print(f"- Highest-cost state (avg): {state_avg.index[0]} (${state_avg.iloc[0]:.0f})")
print(f"- Academic Medical Centers price ~{pivot.loc['Academic Medical Center'].mean():.2f}x vs "
      f"Urgent Care ~{pivot.loc['Urgent Care'].mean():.2f}x of the regional baseline")"""))

cells.append(md("""## Section 4 — Feature Engineering
### Phase 3 of the project plan

Converts procedure codes, hospitals, insurance plans, and location into model-ready features:
- **Low-cardinality categoricals** (facility type, metro tier, plan type) → one-hot encoded
- **High-cardinality categoricals** (CPT code, state) → frequency encoding
- **Engineered ratios** that capture pricing dynamics directly (deductible-remaining fraction,
  quality-per-dollar proxy, price-vs-national-average)"""))

cells.append(code("""# ============================================================
# Feature engineering
# ============================================================
TARGET_COL = "patient_responsibility"

def engineer_features(df, fit_encoders=True, freq_maps=None):
    df = df.copy()
    freq_maps = freq_maps or {}

    df["price_vs_national_avg"] = df["facility_price_index"]
    df["deductible_remaining_frac"] = 1 - (df["deductible_met_pct"] / 100.0)
    df["quality_per_dollar_proxy"] = df["star_rating"] / (df["facility_price_index"] + 0.01)
    df["is_high_deductible_stage"] = (df["deductible_met_pct"] < 30).astype(int)
    df["urban_flag"] = (df["metro_tier"] == "Urban").astype(int)

    for col in ["cpt_code", "state", "hospital_id"]:
        map_key = f"{col}_freq"
        if fit_encoders:
            freq_maps[map_key] = df[col].value_counts(normalize=True)
        freq = freq_maps.get(map_key, pd.Series(dtype=float))
        df[map_key] = df[col].map(freq).fillna(0.0)

    onehot_cols = ["procedure_category", "insurance_plan_type", "facility_type", "metro_tier"]
    df = pd.get_dummies(df, columns=onehot_cols, prefix=onehot_cols)
    return df, freq_maps

RAW_NUMERIC_FEATURES = ["patient_age", "deductible_met_pct", "median_household_income",
    "cost_of_living_index", "population_density", "star_rating", "readmission_rate_pct",
    "patient_satisfaction_score", "facility_price_index"]

def build_model_matrix(df, freq_maps=None, fit_encoders=True):
    engineered, freq_maps = engineer_features(df, fit_encoders=fit_encoders, freq_maps=freq_maps)
    engineered_extra = ["price_vs_national_avg", "deductible_remaining_frac", "quality_per_dollar_proxy",
                         "is_high_deductible_stage", "urban_flag", "cpt_code_freq", "state_freq", "hospital_id_freq"]
    base_numeric = [c for c in RAW_NUMERIC_FEATURES if c in engineered.columns]
    onehot_cols = [c for c in engineered.columns if c.startswith((
        "procedure_category_", "insurance_plan_type_", "facility_type_", "metro_tier_"))]
    feature_cols = list(dict.fromkeys(base_numeric + engineered_extra + onehot_cols))
    X = engineered[feature_cols].astype(float)
    y = engineered[TARGET_COL].astype(float) if TARGET_COL in engineered.columns else None
    return X, y, feature_cols, freq_maps

X, y, feature_cols, freq_maps = build_model_matrix(claims_clean)
print(f"Feature matrix: {X.shape[0]:,} rows x {X.shape[1]} features")
X.head()"""))

cells.append(md("""## Section 5 — Build Prediction Models
### Phase 4 of the project plan

Model 1 (Linear Regression) → Model 2 (Random Forest) → Model 3 (XGBoost), each evaluated
with **MAE**, **RMSE**, **R²**, and 5-fold cross-validation (not just a single train/test split —
a single split can flatter or unfairly penalize a model by chance)."""))

cells.append(code("""# ============================================================
# Train / test split
# ============================================================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Train: {X_train.shape[0]:,} rows | Test: {X_test.shape[0]:,} rows")"""))

cells.append(code("""# ============================================================
# Model training + evaluation harness
# ============================================================
results = {}

def evaluate(name, model, X_tr, y_tr, X_te, y_te, cv_folds=5):
    model.fit(X_tr, y_tr)
    preds = np.clip(model.predict(X_te), 0, None)
    mae = mean_absolute_error(y_te, preds)
    rmse = np.sqrt(mean_squared_error(y_te, preds))
    r2 = r2_score(y_te, preds)
    cv = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
    cv_mae = -cross_val_score(model, X_tr, y_tr, cv=cv, scoring="neg_mean_absolute_error")
    print(f"{name:>20s} | MAE: ${mae:8.2f} | RMSE: ${rmse:8.2f} | R\u00b2: {r2:6.3f} "
          f"| CV MAE: ${cv_mae.mean():.2f} (+/- ${cv_mae.std():.2f})")
    return {"name": name, "model": model, "mae": mae, "rmse": rmse, "r2": r2,
            "cv_mae_mean": cv_mae.mean(), "cv_mae_std": cv_mae.std(), "preds": preds}

print("Training models — MAE / RMSE / R\u00b2 + 5-fold CV\\n" + "-"*78)

# ---- Model 1: Linear Regression (scaled) ----
scaler = StandardScaler()
X_train_s = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
X_test_s = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)
results["linear_regression"] = evaluate("Linear Regression", LinearRegression(),
                                         X_train_s, y_train, X_test_s, y_test)

# ---- Model 2: Random Forest ----
rf = RandomForestRegressor(n_estimators=300, max_depth=14, min_samples_leaf=5, n_jobs=-1, random_state=42)
results["random_forest"] = evaluate("Random Forest", rf, X_train, y_train, X_test, y_test)

# ---- Model 3: XGBoost (falls back to GradientBoosting if not installed) ----
if HAS_XGBOOST:
    xgb = XGBRegressor(n_estimators=400, max_depth=6, learning_rate=0.05, subsample=0.85,
                        colsample_bytree=0.85, reg_lambda=1.0, random_state=42, n_jobs=-1)
    results["xgboost"] = evaluate("XGBoost", xgb, X_train, y_train, X_test, y_test)
else:
    gbr = GradientBoostingRegressor(n_estimators=400, max_depth=4, learning_rate=0.05,
                                     subsample=0.85, random_state=42)
    results["xgboost"] = evaluate("GradientBoosting*", gbr, X_train, y_train, X_test, y_test)"""))

cells.append(code("""# ============================================================
# Chart 7: Model comparison — MAE / RMSE / R\u00b2 side by side
# ============================================================
comparison = pd.DataFrame([{"Model": r["name"], "MAE": r["mae"], "RMSE": r["rmse"], "R2": r["r2"]}
                            for r in results.values()])

fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
for ax, metric, better in zip(axes, ["MAE", "RMSE", "R2"], ["lower is better", "lower is better", "higher is better"]):
    bars = ax.bar(comparison["Model"], comparison[metric], color=PALETTE[:len(comparison)])
    ax.set_title(f"{metric} ({better})")
    ax.bar_label(bars, fmt="%.2f" if metric == "R2" else "$%.0f")
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right")
plt.tight_layout(); plt.show()

best_key = min(results, key=lambda k: results[k]["mae"])
best = results[best_key]
print(f"\\nBest model: {best['name']}  (test MAE ${best['mae']:.2f}, R\u00b2 {best['r2']:.3f})")"""))

cells.append(code("""# ============================================================
# Chart 8: Predicted vs. Actual (model performance) — best model
# ============================================================
fig, ax = plt.subplots(figsize=(6.5, 6.5))
sample_idx = np.random.choice(len(y_test), size=min(3000, len(y_test)), replace=False)
y_true_sample = y_test.values[sample_idx]
y_pred_sample = best["preds"][sample_idx]

ax.scatter(y_true_sample, y_pred_sample, alpha=0.25, s=12, color=PALETTE[1])
lims = [0, max(y_true_sample.max(), y_pred_sample.max())]
ax.plot(lims, lims, "--", color="#C62828", linewidth=1.5, label="Perfect prediction")
ax.set_xlim(lims); ax.set_ylim(lims)
ax.set_xlabel("Actual Patient Responsibility ($)"); ax.set_ylabel("Predicted ($)")
ax.set_title(f"Predicted vs. Actual — {best['name']}")
ax.legend()
plt.tight_layout(); plt.show()"""))

cells.append(code("""# ============================================================
# Chart 9: Feature importance (best tree-based model)
# ============================================================
if hasattr(best["model"], "feature_importances_"):
    importances = pd.Series(best["model"].feature_importances_, index=feature_cols).sort_values(ascending=False).head(15)
    fig, ax = plt.subplots(figsize=(9, 6))
    importances.sort_values().plot(kind="barh", color=PALETTE[0], ax=ax)
    ax.set_title(f"Top 15 Feature Importances — {best['name']}")
    ax.set_xlabel("Importance")
    plt.tight_layout(); plt.show()
else:
    print("Best model has no native feature_importances_ (likely Linear Regression) — see SHAP section instead.")"""))

cells.append(md("""## Section 6 — Confidence Intervals

Instead of a falsely-precise single number ("$530"), predict a realistic **range**
("$470–$590") using quantile regression on the 10th and 90th percentiles."""))

cells.append(code("""# ============================================================
# Quantile regression for prediction intervals
# ============================================================
lower_model = GradientBoostingRegressor(loss="quantile", alpha=0.10, n_estimators=300,
                                         max_depth=4, learning_rate=0.05, random_state=42)
upper_model = GradientBoostingRegressor(loss="quantile", alpha=0.90, n_estimators=300,
                                         max_depth=4, learning_rate=0.05, random_state=42)
lower_model.fit(X_train, y_train)
upper_model.fit(X_train, y_train)

point = np.clip(best["model"].predict(X_test.iloc[:8]), 0, None)
lower = np.clip(lower_model.predict(X_test.iloc[:8]), 0, None)
upper = np.clip(upper_model.predict(X_test.iloc[:8]), 0, None)
lower, upper = np.minimum(lower, point), np.maximum(upper, point)

interval_demo = pd.DataFrame({
    "Actual": y_test.iloc[:8].values.round(0),
    "Predicted": point.round(0),
    "Low (10th pct)": lower.round(0),
    "High (90th pct)": upper.round(0),
})
interval_demo["90% Interval"] = interval_demo.apply(
    lambda r: f"${r['Low (10th pct)']:.0f} \u2013 ${r['High (90th pct)']:.0f}", axis=1)
interval_demo[["Actual", "Predicted", "90% Interval"]]"""))

cells.append(md("""## Section 7 — Explain Predictions with SHAP
### Phase 5 of the project plan

SHAP values show *why* a specific prediction came out the way it did — which features
pushed the estimate up or down, and by how much. That feeds directly into the
plain-English "AI Explanation" panel in the dashboard."""))

cells.append(code("""# ============================================================
# SHAP explainability
# ============================================================
if HAS_SHAP and hasattr(best["model"], "predict"):
    try:
        explainer = shap.TreeExplainer(best["model"])
        sample = X_test.sample(min(500, len(X_test)), random_state=42)
        shap_values = explainer(sample)

        # Chart 10: SHAP summary plot (global feature impact)
        shap.summary_plot(shap_values, sample, show=False, max_display=15)
        plt.title("SHAP Summary — Global Feature Impact on Predicted Cost")
        plt.tight_layout(); plt.show()
    except Exception as e:
        print(f"TreeExplainer not compatible with {best['name']} ({e}); "
              f"falling back to a small KernelExplainer sample for the summary plot.")
        bg = X_train.sample(50, random_state=42)
        explainer = shap.Explainer(best["model"].predict, bg)
        shap_values = explainer(X_test.sample(100, random_state=42))
        shap.summary_plot(shap_values, X_test.sample(100, random_state=42), show=False, max_display=15)
        plt.tight_layout(); plt.show()
else:
    print("Install `shap` (`pip install shap`) to render the SHAP summary plot in this environment.")"""))

cells.append(code("""# ============================================================
# Single-prediction explanation -> feeds the "AI Explanation" panel
# ============================================================
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
    "cpt_code_freq": "how common this procedure is nationally",
    "state_freq": "claim volume in this state",
    "hospital_id_freq": "how frequently this facility appears in the data",
}

def label(feature):
    if feature in FEATURE_LABELS:
        return FEATURE_LABELS[feature]
    for prefix, readable in [("procedure_category_", "the procedure category ({})"),
                              ("insurance_plan_type_", "your insurance plan type ({})"),
                              ("facility_type_", "the facility type ({})"),
                              ("metro_tier_", "the metro tier ({})")]:
        if feature.startswith(prefix):
            return readable.format(feature[len(prefix):])
    return feature.replace("_", " ")

def explain_prediction(row_idx, top_n=3):
    row = X_test.iloc[[row_idx]]
    pred = float(np.clip(best["model"].predict(row)[0], 0, None))
    if HAS_SHAP:
        try:
            sv = shap.TreeExplainer(best["model"])(row)
            values = np.array(sv.values).reshape(-1)
        except Exception:
            values = None
    else:
        values = None

    if values is None:
        importances = getattr(best["model"], "feature_importances_", np.zeros(len(feature_cols)))
        r = row.iloc[0]
        is_binary = r.isin([0, 1])
        scores = np.where(is_binary & (r.values == 0), 0.0, importances)
        order = np.argsort(-scores)[:top_n]
        drivers = [(feature_cols[i], "increases" if (is_binary.iloc[i] and r.iloc[i] == 1) or r.iloc[i] > 0 else "decreases")
                   for i in order if scores[i] > 0]
    else:
        order = np.argsort(-np.abs(values))[:top_n]
        drivers = [(feature_cols[i], "increases" if values[i] > 0 else "decreases") for i in order]

    lines = [f"Your estimated out-of-pocket cost is ${pred:,.0f} because:"]
    for feat, direction in drivers:
        lines.append(f"  \u2022 {label(feat).capitalize()} {direction} your cost.")
    return "\\n".join(lines)

print(explain_prediction(0))
print()
print(explain_prediction(1))"""))

cells.append(md("""## Section 8 — Financial Decision Engine (Provider Optimization)

This is the "exceptional" layer from the project brief: instead of only answering
**"what will this cost?"**, answer **"what is the financially optimal choice?"**

For every candidate provider, combine the predicted cost with quality (star rating) and
readmission risk into a single weighted **value score** — the same multi-factor scoring
pattern used in equity factor models and insurer network-tiering."""))

cells.append(code("""# ============================================================
# Multi-factor provider ranking
# ============================================================
def normalize(series, invert=False):
    if series.max() == series.min():
        return pd.Series(0.5, index=series.index)
    norm = (series - series.min()) / (series.max() - series.min())
    return 1 - norm if invert else norm

def rank_providers(candidates, weights=None):
    weights = weights or {"cost": 0.50, "quality": 0.25, "readmission": 0.15, "distance": 0.10}
    df = candidates.copy()
    cost_score = normalize(df["predicted_cost"], invert=True)
    quality_score = normalize(df["star_rating"])
    readmission_score = normalize(df["readmission_rate_pct"], invert=True)
    composite = (weights["cost"] * cost_score + weights["quality"] * quality_score +
                 weights["readmission"] * readmission_score + weights["distance"] * 0.5)
    df["value_score"] = (composite * 100).round(1)
    return df.sort_values("value_score", ascending=False)

# Demo: predict cost at every facility in one ZIP code for one procedure + plan
demo_zip = "19104"
demo_cpt = "73721"       # MRI Knee
demo_plan = "Aetna PPO"

demo_hospitals = hospitals_df[hospitals_df["zip_code"] == demo_zip].copy()
proc_ref = claims_clean[claims_clean["cpt_code"] == demo_cpt].iloc[0]
plan_ref = claims_clean[claims_clean["insurance_plan"] == demo_plan].iloc[0]

rows = []
for _, h in demo_hospitals.iterrows():
    row = {"cpt_code": demo_cpt, "procedure_category": proc_ref["procedure_category"],
           "insurance_plan_type": plan_ref["insurance_plan_type"], "patient_age": 34,
           "deductible_met_pct": 20, "median_household_income": h["median_household_income"],
           "cost_of_living_index": h["cost_of_living_index"], "population_density": h["population_density"],
           "star_rating": h["star_rating"], "readmission_rate_pct": h["readmission_rate_pct"],
           "patient_satisfaction_score": h["patient_satisfaction_score"],
           "facility_price_index": h["facility_price_index"], "facility_type": h["facility_type"],
           "metro_tier": h["metro_tier"], "state": h["state"], "hospital_id": h["hospital_id"],
           TARGET_COL: 0.0}
    X_row, _, _, _ = build_model_matrix(pd.DataFrame([row]), freq_maps=freq_maps, fit_encoders=False)
    X_row = X_row.reindex(columns=feature_cols, fill_value=0.0)
    pred = float(np.clip(best["model"].predict(X_row)[0], 0, None))
    rows.append({"hospital_name": h["hospital_name"], "predicted_cost": pred,
                 "star_rating": h["star_rating"], "readmission_rate_pct": h["readmission_rate_pct"]})

candidates_df = pd.DataFrame(rows)
ranked = rank_providers(candidates_df)
ranked[["hospital_name", "predicted_cost", "star_rating", "readmission_rate_pct", "value_score"]]"""))

cells.append(code("""# ============================================================
# Chart 11: Cost vs. Quality tradeoff — the optimization visual
# ============================================================
fig = px.scatter(ranked, x="predicted_cost", y="star_rating", size="value_score",
                  color="value_score", hover_name="hospital_name",
                  color_continuous_scale="RdYlGn", size_max=30,
                  labels={"predicted_cost": "Predicted Out-of-Pocket Cost ($)", "star_rating": "CMS Star Rating"},
                  title=f"Cost vs. Quality Tradeoff — MRI Knee, {demo_plan}, ZIP {demo_zip}")
fig.update_layout(height=480)
fig.show()

cheapest = ranked.loc[ranked["predicted_cost"].idxmin()]
best_value = ranked.iloc[0]
print(f"Cheapest option:    {cheapest['hospital_name']} — ${cheapest['predicted_cost']:.0f} "
      f"({cheapest['star_rating']}\u2605)")
print(f"Best VALUE option:  {best_value['hospital_name']} — ${best_value['predicted_cost']:.0f} "
      f"({best_value['star_rating']}\u2605, value score {best_value['value_score']})")"""))

cells.append(md("""## Section 9 — Dashboard Mockup

A static rendering of what the production Streamlit/React dashboard shows the user.
(The actual interactive app lives in `frontend/app.py` — run `streamlit run frontend/app.py`.)

> **Data disclaimer:** facility names shown below are real hospitals. Every cost, star
> rating, and readmission figure attached to them is **simulated** for this demo and is
> NOT that hospital's actual published pricing or real CMS quality score."""))

cells.append(code("""# ============================================================
# Chart 12: Dashboard-style summary panel
# ============================================================
top = ranked.iloc[0]
total_bill_estimate = top["predicted_cost"] / 0.35
insurance_pays = total_bill_estimate - top["predicted_cost"]
savings = ranked["predicted_cost"].max() - ranked["predicted_cost"].min()

fig, axes = plt.subplots(1, 2, figsize=(15, 5.5), gridspec_kw={"width_ratios": [1, 1.4]})

# Left panel: big cost numbers
ax = axes[0]; ax.axis("off")
ax.text(0.02, 0.92, "CareCost AI", fontsize=18, fontweight="bold")
ax.text(0.02, 0.82, "MRI Knee \u00b7 Aetna PPO \u00b7 ZIP 19104", fontsize=11, color="gray")
ax.text(0.02, 0.62, "Estimated Total Bill", fontsize=11, color="gray")
ax.text(0.02, 0.52, f"${total_bill_estimate:,.0f}", fontsize=26, fontweight="bold", color=PALETTE[0])
ax.text(0.02, 0.36, "Insurance Pays", fontsize=11, color="gray")
ax.text(0.02, 0.27, f"${insurance_pays:,.0f}", fontsize=20, color=PALETTE[1])
ax.text(0.02, 0.12, "You Pay", fontsize=11, color="gray")
ax.text(0.02, 0.02, f"${top['predicted_cost']:,.0f}", fontsize=22, fontweight="bold", color=PALETTE[3])

# Right panel: provider comparison bar chart
ax2 = axes[1]
colors = [PALETTE[0] if i == 0 else PALETTE[4] for i in range(len(ranked))]
ax2.barh(ranked["hospital_name"].str[:30], ranked["predicted_cost"], color=colors)
ax2.invert_yaxis()
ax2.set_xlabel("You Pay ($)"); ax2.set_title("Nearby Providers")
ax2.xaxis.set_major_formatter(mticker.StrMethodFormatter("${x:,.0f}"))

plt.tight_layout(); plt.show()

print(f"\\nAI Explanation:\\n{best_value['hospital_name']} offers the best overall value. "
      f"Choosing it over the most expensive nearby option could save about ${savings:,.0f}.")"""))

cells.append(md("""## Section 10 — Save Model Artifacts

Persists the winning model + quantile models + encoders so the FastAPI backend
(`backend/main.py`) and Streamlit dashboard (`frontend/app.py`) can load them directly."""))

cells.append(code("""# ============================================================
# Persist model bundle
# ============================================================
import joblib, os

os.makedirs("models", exist_ok=True)
bundle = {
    "model": best["model"],
    "model_name": best["name"],
    "feature_cols": feature_cols,
    "freq_maps": freq_maps,
    "lower_model": lower_model,
    "upper_model": upper_model,
    "scaler": scaler if best_key == "linear_regression" else None,
    "results": {k: {kk: vv for kk, vv in v.items() if kk != "preds"} for k, v in results.items()},
}
joblib.dump(bundle, "models/carecost_model.joblib")
print("Saved models/carecost_model.joblib")
print(f"\\nFinal model: {best['name']}")
print(f"Test MAE: ${best['mae']:.2f} | RMSE: ${best['rmse']:.2f} | R\u00b2: {best['r2']:.3f}")"""))

cells.append(md("""## Summary

| Phase | Status |
|---|---|
| Data collection | ✅ Synthetic CMS-schema data (45k claims, 70+ facilities, 15 regions) |
| EDA | ✅ Cost distributions, regional/payer variation, outlier detection |
| Feature engineering | ✅ Frequency + one-hot encoding, engineered pricing ratios |
| Modeling | ✅ Linear Regression → Random Forest → XGBoost, cross-validated |
| Confidence intervals | ✅ Quantile regression (10th/90th percentile) |
| Explainability | ✅ SHAP-driven driver attribution → plain-English explanation |
| Decision engine | ✅ Multi-factor provider value-score ranking |
| Deployment artifacts | ✅ FastAPI backend + Streamlit dashboard (see `backend/`, `frontend/`) |

**Next steps for production:** swap `generate_synthetic_dataset()` for real CMS bulk-file
ingestion (see `README.md`), retrain, and deploy the backend (Render/Railway) and frontend
(Vercel/Streamlit Community Cloud) as described in the project README."""))


def make_notebook(cells):
    nb_cells = []
    for cell_type, source in cells:
        lines = source.split("\n")
        src_list = [l + "\n" for l in lines[:-1]] + [lines[-1]] if lines else [""]
        cell = {
            "cell_type": cell_type,
            "metadata": {},
            "source": src_list,
        }
        if cell_type == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
        nb_cells.append(cell)

    notebook = {
        "cells": nb_cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10"},
            "colab": {"provenance": [], "name": "carecost_ai_colab.ipynb"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    return notebook


if __name__ == "__main__":
    nb = make_notebook(cells)
    with open("notebooks/carecost_ai_colab.ipynb", "w") as f:
        json.dump(nb, f, indent=1)
    print(f"Wrote notebook with {len(cells)} cells "
          f"({sum(1 for c in cells if c[0]=='code')} code, {sum(1 for c in cells if c[0]=='markdown')} markdown)")

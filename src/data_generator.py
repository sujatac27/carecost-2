"""
CareCost AI — Synthetic Healthcare Pricing Data Generator
============================================================
Generates a statistically realistic dataset that mirrors the SCHEMA and
DISTRIBUTIONS of real CMS data sources:

    - CMS Hospital Price Transparency (negotiated rates by payer)
    - CMS Physician Fee Schedule (Medicare allowed amounts)
    - CMS Hospital Compare (quality star ratings, readmission rates)
    - U.S. Census ZIP-level demographics (median income, population density)

WHY SYNTHETIC DATA:
This environment has no network access, so the real CMS bulk files
(multi-GB JSON/CSV per hospital) cannot be downloaded here. Every
function below is written so that swapping in real data requires
changing ONLY the ingestion step — the schema, column names, and
downstream pipeline (EDA -> features -> model -> API) are built to be
production-shaped from day one.

TO SWAP IN REAL DATA LATER:
    1. Download hospital MRFs from https://www.cms.gov/hospital-price-transparency
    2. Download the Physician Fee Schedule from the CMS PFS Look-Up Tool
    3. Download Hospital Compare star ratings from data.cms.gov
    4. Map their columns to the schema in `SCHEMA.md`
    5. Replace `generate_synthetic_dataset()` with `load_real_cms_data()`

Author: CareCost AI
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List


# ---------------------------------------------------------------------------
# Reference tables (mirror real-world CPT/HCPCS + payer + facility structures)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Procedure:
    cpt_code: str
    description: str
    category: str
    base_medicare_rate: float   # approximate Medicare allowed amount (national avg)
    price_variance: float       # how much negotiated rates vary around base (facility-driven)


PROCEDURES: List[Procedure] = [
    # Base rates below are anchored to real, publicly published 2025-2026 Medicare
    # Physician Fee Schedule national payment amounts and CMS reference data where
    # available (e.g. total knee/hip replacement ~$1,300-$2,000 professional-fee
    # component per recent CMS fee-schedule studies; Medicare Part B deductible
    # $283 and Part A deductible $1,736 for 2026). Values represent an approximate
    # *total episode cost* (facility + professional), not the isolated physician
    # fee, since that's what patients actually see on a bill.
    Procedure("70551", "MRI Brain w/o Contrast", "Imaging", 425, 0.55),
    Procedure("73721", "MRI Knee w/o Contrast", "Imaging", 390, 0.60),
    Procedure("73221", "MRI Shoulder w/o Contrast", "Imaging", 405, 0.58),
    Procedure("72148", "MRI Lumbar Spine w/o Contrast", "Imaging", 445, 0.55),
    Procedure("74177", "CT Abdomen/Pelvis w/ Contrast", "Imaging", 465, 0.50),
    Procedure("71250", "CT Chest w/o Contrast", "Imaging", 285, 0.55),
    Procedure("70450", "CT Head/Brain w/o Contrast", "Imaging", 245, 0.55),
    Procedure("71046", "Chest X-Ray, 2 Views", "Imaging", 42, 0.65),
    Procedure("77067", "Mammography, Bilateral Screening", "Imaging", 145, 0.45),
    Procedure("76700", "Abdominal Ultrasound, Complete", "Imaging", 165, 0.50),
    Procedure("45378", "Colonoscopy, Diagnostic", "GI Procedure", 610, 0.45),
    Procedure("45385", "Colonoscopy w/ Polyp Removal", "GI Procedure", 780, 0.45),
    Procedure("43239", "Upper GI Endoscopy w/ Biopsy", "GI Procedure", 545, 0.45),
    Procedure("29881", "Knee Arthroscopy w/ Meniscectomy", "Orthopedic Surgery", 1150, 0.50),
    Procedure("29827", "Shoulder Arthroscopy w/ Rotator Cuff Repair", "Orthopedic Surgery", 2100, 0.48),
    Procedure("27447", "Total Knee Replacement", "Orthopedic Surgery", 8900, 0.40),
    Procedure("27130", "Total Hip Replacement", "Orthopedic Surgery", 9400, 0.40),
    Procedure("22551", "Cervical Spinal Fusion", "Orthopedic Surgery", 7200, 0.42),
    Procedure("99213", "Office Visit, Established Patient", "Primary Care", 92, 0.35),
    Procedure("99214", "Office Visit, Established Patient (Moderate Complexity)", "Primary Care", 128, 0.35),
    Procedure("99203", "Office Visit, New Patient", "Primary Care", 150, 0.35),
    Procedure("99385", "Preventive Visit, New Patient (Age 18-39)", "Primary Care", 175, 0.30),
    Procedure("99284", "Emergency Dept Visit, Level 4", "Emergency", 385, 0.55),
    Procedure("99283", "Emergency Dept Visit, Level 3", "Emergency", 245, 0.55),
    Procedure("99285", "Emergency Dept Visit, Level 5 (Critical)", "Emergency", 720, 0.55),
    Procedure("93000", "Electrocardiogram (EKG)", "Cardiology", 32, 0.45),
    Procedure("93306", "Echocardiogram, Complete", "Cardiology", 215, 0.50),
    Procedure("93458", "Cardiac Catheterization w/ Angiography", "Cardiology", 3200, 0.45),
    Procedure("47562", "Laparoscopic Cholecystectomy", "General Surgery", 2450, 0.45),
    Procedure("49505", "Inguinal Hernia Repair", "General Surgery", 1850, 0.45),
    Procedure("44970", "Laparoscopic Appendectomy", "General Surgery", 2100, 0.45),
    Procedure("59400", "Vaginal Delivery incl. Prenatal Care", "Obstetrics", 3200, 0.40),
    Procedure("59510", "Cesarean Delivery incl. Prenatal Care", "Obstetrics", 4800, 0.38),
    Procedure("81001", "Urinalysis w/ Microscopy", "Lab", 8, 0.60),
    Procedure("80053", "Comprehensive Metabolic Panel", "Lab", 14, 0.55),
    Procedure("85025", "Complete Blood Count (CBC) w/ Differential", "Lab", 11, 0.55),
    Procedure("80061", "Lipid Panel", "Lab", 16, 0.55),
    Procedure("36415", "Blood Draw (Venipuncture)", "Lab", 5, 0.70),
    Procedure("97110", "Physical Therapy, Therapeutic Exercise", "Rehab", 38, 0.45),
    Procedure("97140", "Manual Therapy Techniques", "Rehab", 35, 0.45),
    Procedure("19120", "Breast Biopsy, Open", "Surgery", 980, 0.45),
]

INSURANCE_PLANS = [
    # name, plan_type, reimbursement_rate, annual_deductible, coinsurance_rate
    # Medicare Advantage's deductible is anchored to the real 2026 Medicare Part B
    # annual deductible ($283, per the CMS 2026 Part B premium/deductible notice).
    ("Aetna PPO", "PPO", 0.82, 1500, 0.20),
    ("Aetna HMO", "HMO", 0.75, 1000, 0.15),
    ("UnitedHealthcare Choice Plus", "PPO", 0.80, 2000, 0.20),
    ("Blue Cross Blue Shield PPO", "PPO", 0.83, 1750, 0.20),
    ("Cigna Open Access", "PPO", 0.78, 1500, 0.20),
    ("Humana Gold HMO", "HMO", 0.74, 500, 0.10),
    ("Kaiser Permanente HMO", "HMO", 0.85, 1000, 0.10),
    ("Medicare Advantage", "Medicare", 0.95, 283, 0.20),
    ("Medicaid", "Medicaid", 1.00, 0, 0.00),
    ("High-Deductible HSA Plan", "HDHP", 0.70, 5000, 0.30),
    ("Uninsured / Self-Pay", "Self-Pay", 0.00, 0, 1.00),
]

FACILITY_TYPES = ["Academic Medical Center", "Community Hospital",
                   "Ambulatory Surgery Center", "Imaging Center", "Urgent Care"]

# 32 representative real ZIP codes across metro / suburban / rural tiers,
# spanning more regions of the country
ZIP_PROFILES = [
    # zip, state, metro_tier, median_income, cost_of_living_index, population_density
    ("19104", "PA", "Urban", 38500, 1.05, 14200),
    ("19107", "PA", "Urban", 71200, 1.10, 22000),
    ("10001", "NY", "Urban", 112400, 1.65, 33000),
    ("10032", "NY", "Urban", 42300, 1.55, 45000),
    ("60601", "IL", "Urban", 98700, 1.30, 26500),
    ("60612", "IL", "Urban", 45200, 1.15, 12800),
    ("94103", "CA", "Urban", 145600, 1.85, 17800),
    ("90048", "CA", "Urban", 98200, 1.70, 18500),
    ("77002", "TX", "Urban", 88300, 1.05, 8200),
    ("77030", "TX", "Urban", 76500, 1.05, 6200),
    ("30303", "GA", "Urban", 62100, 0.98, 6100),
    ("48201", "MI", "Urban", 29800, 0.85, 5000),
    ("48202", "MI", "Urban", 33500, 0.85, 7200),
    ("02115", "MA", "Urban", 68900, 1.55, 24500),
    ("21287", "MD", "Urban", 41200, 1.15, 11200),
    ("98104", "WA", "Urban", 89600, 1.45, 15600),
    ("80204", "CO", "Urban", 72300, 1.20, 9800),
    ("85006", "AZ", "Urban", 51400, 0.98, 5400),
    ("33136", "FL", "Urban", 34700, 1.05, 7600),
    ("55415", "MN", "Urban", 58900, 1.08, 8900),
    ("08053", "NJ", "Suburban", 118900, 1.20, 1800),
    ("60614", "IL", "Suburban", 134200, 1.25, 15600),
    ("75024", "TX", "Suburban", 121500, 1.05, 3400),
    ("30144", "GA", "Suburban", 79400, 0.95, 2100),
    ("19087", "PA", "Suburban", 142300, 1.15, 2600),
    ("90210", "CA", "Suburban", 175800, 1.90, 2200),
    ("28270", "NC", "Suburban", 108200, 1.02, 1900),
    ("63105", "MO", "Suburban", 132400, 1.05, 3300),
    ("59718", "MT", "Rural", 61300, 0.92, 45),
    ("50588", "IA", "Rural", 54200, 0.85, 18),
    ("83001", "WY", "Rural", 68900, 1.00, 6),
    ("59901", "MT", "Rural", 52400, 0.90, 32),
]

# ---------------------------------------------------------------------------
# Real hospital names, by city — used for "Academic Medical Center" and
# "Community Hospital" facility types. Sourced from public hospital-ranking
# lists (Becker's Hospital Review, U.S. News Best Hospitals).
#
# IMPORTANT DISCLAIMER: these are REAL hospital names, but every cost, star
# rating, and readmission figure attached to them in this dataset is
# SIMULATED for demo purposes — it is NOT that hospital's actual published
# pricing or actual CMS quality score. Every UI surface (dashboard, API
# responses, README) must carry this disclaimer so numbers are never mistaken
# for a real hospital's real data. Ambulatory Surgery Centers, Imaging
# Centers, and Urgent Care facilities keep generic template names since
# these smaller facility types aren't well-known named entities anyway.
# ---------------------------------------------------------------------------
REAL_HOSPITALS_BY_CITY = {
    "Philadelphia": ["Hospital of the University of Pennsylvania", "Thomas Jefferson University Hospital",
                     "Children's Hospital of Philadelphia", "Temple University Hospital"],
    "New York": ["NewYork-Presbyterian/Weill Cornell Medical Center", "NYU Langone Hospitals",
                 "Mount Sinai Hospital", "Tisch Hospital"],
    "Chicago": ["Northwestern Memorial Hospital", "Rush University Medical Center",
                "University of Chicago Medical Center", "Ann & Robert H. Lurie Children's Hospital"],
    "San Francisco": ["UCSF Medical Center", "California Pacific Medical Center",
                       "Zuckerberg San Francisco General Hospital"],
    "Los Angeles": ["Cedars-Sinai Medical Center", "UCLA Medical Center", "Keck Hospital of USC"],
    "Houston": ["Houston Methodist Hospital", "Texas Children's Hospital", "Baylor St. Luke's Medical Center"],
    "Atlanta": ["Emory University Hospital", "Grady Memorial Hospital", "Piedmont Atlanta Hospital"],
    "Detroit": ["Henry Ford Hospital", "Detroit Medical Center - Harper University Hospital",
                "Ascension St. John Hospital"],
    "Boston": ["Massachusetts General Hospital", "Brigham and Women's Hospital", "Boston Medical Center"],
    "Baltimore": ["Johns Hopkins Hospital", "University of Maryland Medical Center"],
    "Seattle": ["UW Medical Center - Montlake", "Swedish Medical Center - First Hill"],
    "Denver": ["UCHealth University of Colorado Hospital", "Denver Health Medical Center"],
    "Phoenix": ["Banner - University Medical Center Phoenix", "Phoenix Children's Hospital"],
    "Miami": ["Jackson Memorial Hospital", "Sylvester Comprehensive Cancer Center - UHealth"],
    "Minneapolis": ["M Health Fairview University of Minnesota Medical Center", "Abbott Northwestern Hospital"],
    "Cherry Hill": ["Virtua Voorhees Hospital", "Jefferson Cherry Hill Hospital"],
    "Beverly Hills": ["Cedars-Sinai Marina del Rey Hospital"],
    "Charlotte": ["Atrium Health Carolinas Medical Center", "Novant Health Presbyterian Medical Center"],
    "St. Louis": ["Barnes-Jewish Hospital", "Mercy Hospital St. Louis"],
    "Bozeman": ["Bozeman Health Deaconess Hospital"],
    "Fort Dodge": ["UnityPoint Health - Trinity Regional Medical Center"],
    "Jackson": ["St. John's Medical Center"],
    "Kalispell": ["Logan Health Medical Center"],
}

GENERIC_NAME_TEMPLATES = [
    "{city} Surgical & Imaging Center", "{city} Ambulatory Surgery Center",
    "{city} Urgent Care", "{city} Diagnostic Imaging Center", "{city} Same-Day Surgery Center",
]

CITY_BY_STATE = {
    "PA": "Philadelphia", "NY": "New York", "IL": "Chicago", "CA": "San Francisco",
    "TX": "Houston", "GA": "Atlanta", "MI": "Detroit", "NJ": "Cherry Hill",
    "MT": "Bozeman", "IA": "Fort Dodge", "WY": "Jackson",
    "MA": "Boston", "MD": "Baltimore", "WA": "Seattle", "CO": "Denver",
    "AZ": "Phoenix", "FL": "Miami", "MN": "Minneapolis", "NC": "Charlotte", "MO": "St. Louis",
}
# Some ZIP profiles reuse a state (e.g. two MT rows, two CA/PA/etc. rows) —
# this secondary map lets those rows pull a different city than the default
# for that state, so facility names stay geographically varied.
CITY_OVERRIDE_BY_ZIP = {
    "90210": "Beverly Hills",
    "59901": "Kalispell",
}


def _make_hospitals(rng: np.random.Generator, n_per_zip: int = 4) -> pd.DataFrame:
    """Builds a synthetic hospital/facility master table with quality metrics.

    Facility NAMES are real (sourced from public hospital-ranking lists), but
    every financial and quality metric attached to them is simulated for demo
    purposes — see the `data_disclaimer` column and REAL_HOSPITALS_BY_CITY note
    above. This keeps names realistic while making it unmistakable that the
    numbers aren't that hospital's actual real-world figures.
    """
    rows = []
    hospital_id = 1000
    used_real_names = set()

    for zip_code, state, tier, income, col_index, density in ZIP_PROFILES:
        city = CITY_OVERRIDE_BY_ZIP.get(zip_code, CITY_BY_STATE[state])
        real_names_pool = [n for n in REAL_HOSPITALS_BY_CITY.get(city, []) if n not in used_real_names]

        n_facilities = n_per_zip + (2 if tier == "Urban" else 0) - (1 if tier == "Rural" else 0)
        for _ in range(max(2, n_facilities)):
            hospital_id += 1
            facility_type = rng.choice(
                FACILITY_TYPES,
                p=[0.15, 0.35, 0.25, 0.15, 0.10] if tier != "Rural" else [0.05, 0.55, 0.10, 0.10, 0.20]
            )

            # Real-named hospitals in our curated list are all major academic/
            # teaching institutions in reality, so give them that facility_type
            # for factual accuracy (rather than a randomly-assigned type).
            if facility_type in ("Academic Medical Center", "Community Hospital") and real_names_pool:
                name = real_names_pool.pop(rng.integers(0, len(real_names_pool)))
                used_real_names.add(name)
                facility_type = "Academic Medical Center"
            else:
                name = rng.choice(GENERIC_NAME_TEMPLATES).format(city=city) + f" #{hospital_id % 97}"

            # Quality metrics: academic centers skew higher quality but higher price
            base_quality = {"Academic Medical Center": 4.2, "Community Hospital": 3.4,
                             "Ambulatory Surgery Center": 3.8, "Imaging Center": 3.9,
                             "Urgent Care": 3.3}[facility_type]
            star_rating = float(np.clip(rng.normal(base_quality, 0.6), 1.0, 5.0))
            readmission_rate = float(np.clip(rng.normal(15.5 - star_rating, 1.5), 8.0, 28.0))
            patient_satisfaction = float(np.clip(rng.normal(60 + star_rating * 7, 8), 20, 99))

            # Price index: how far this facility's negotiated rates sit vs regional avg
            price_index = {"Academic Medical Center": 1.28, "Community Hospital": 1.00,
                            "Ambulatory Surgery Center": 0.72, "Imaging Center": 0.65,
                            "Urgent Care": 0.55}[facility_type]
            price_index *= rng.normal(1.0, 0.12)
            price_index *= col_index  # regional cost-of-living adjustment

            rows.append({
                "hospital_id": f"H{hospital_id}",
                "hospital_name": name,
                "facility_type": facility_type,
                "zip_code": zip_code,
                "state": state,
                "metro_tier": tier,
                "median_household_income": income,
                "cost_of_living_index": col_index,
                "population_density": density,
                "star_rating": round(star_rating, 1),
                "readmission_rate_pct": round(readmission_rate, 1),
                "patient_satisfaction_score": round(patient_satisfaction, 1),
                "facility_price_index": round(float(price_index), 3),
                "data_disclaimer": "SIMULATED cost/quality estimate — not this facility's actual published data",
            })
    return pd.DataFrame(rows)


def generate_synthetic_dataset(n_claims: int = 45000, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generates two linked tables:
      1. hospitals_df  — facility master data + CMS Hospital Compare quality metrics
      2. claims_df      — line-item claims resembling CMS Hospital Price Transparency
                           negotiated-rate files + historical utilization/billing data

    Returns
    -------
    (hospitals_df, claims_df)
    """
    rng = np.random.default_rng(seed)
    hospitals_df = _make_hospitals(rng)

    proc_lookup = {p.cpt_code: p for p in PROCEDURES}
    proc_codes = list(proc_lookup.keys())
    # Procedures aren't uniformly likely in real claims data — labs, office visits,
    # and imaging are far more common than major surgery. Weight by category
    # (auto-scales to however many procedures are in PROCEDURES, unlike a
    # hand-typed per-code list that would silently break if the list changes).
    CATEGORY_WEIGHTS = {
        "Lab": 3.0, "Primary Care": 3.0, "Imaging": 2.0, "Rehab": 1.5,
        "Cardiology": 1.2, "Emergency": 1.2, "GI Procedure": 0.8,
        "General Surgery": 0.4, "Orthopedic Surgery": 0.3, "Obstetrics": 0.3,
        "Surgery": 0.3,
    }
    proc_weights = np.array([CATEGORY_WEIGHTS.get(p.category, 1.0) for p in PROCEDURES])
    proc_weights = proc_weights / proc_weights.sum()

    plan_names = [p[0] for p in INSURANCE_PLANS]
    plan_lookup = {p[0]: p for p in INSURANCE_PLANS}
    plan_weights = np.array([0.14, 0.10, 0.13, 0.14, 0.10, 0.08, 0.09, 0.10, 0.06, 0.04, 0.02])
    plan_weights = plan_weights / plan_weights.sum()

    hospital_ids = hospitals_df["hospital_id"].values
    hospital_index = hospitals_df.set_index("hospital_id")

    rows = []
    chosen_hospitals = rng.choice(hospital_ids, size=n_claims, replace=True)
    chosen_procs = rng.choice(proc_codes, size=n_claims, replace=True, p=proc_weights)
    chosen_plans = rng.choice(plan_names, size=n_claims, replace=True, p=plan_weights)

    # patient-level randomness: how much of the deductible has already been met this year
    deductible_progress = rng.uniform(0, 1, size=n_claims)
    patient_age = rng.integers(1, 92, size=n_claims)

    for i in range(n_claims):
        hosp = hospital_index.loc[chosen_hospitals[i]]
        proc = proc_lookup[chosen_procs[i]]
        plan_name, plan_type, reimb_rate, deductible, coinsurance = plan_lookup[chosen_plans[i]]

        # --- Billed amount: base Medicare rate * facility price index * procedure variance
        noise = rng.lognormal(mean=0, sigma=proc.price_variance * 0.35)
        billed = proc.base_medicare_rate * hosp["facility_price_index"] * 2.6 * noise
        billed = max(billed, proc.base_medicare_rate * 0.5)

        # --- Negotiated/allowed amount: payer-specific discount off billed charges
        payer_discount = {"PPO": 0.55, "HMO": 0.48, "Medicare": 0.32,
                           "Medicaid": 0.22, "HDHP": 0.55, "Self-Pay": 0.85}[plan_type]
        allowed_amount = billed * (1 - payer_discount) * rng.normal(1.0, 0.04)
        allowed_amount = max(allowed_amount, proc.base_medicare_rate * 0.3)

        # --- Insurance pays vs patient responsibility
        if plan_type == "Self-Pay":
            insurance_pays = 0.0
            remaining_deductible = 0.0
            patient_pays = allowed_amount
        else:
            remaining_deductible = max(0.0, deductible * (1 - deductible_progress[i]))
            patient_deductible_portion = min(allowed_amount, remaining_deductible)
            post_deductible_amount = allowed_amount - patient_deductible_portion
            patient_coinsurance = post_deductible_amount * coinsurance
            patient_pays = patient_deductible_portion + patient_coinsurance
            insurance_pays = allowed_amount - patient_pays

        rows.append({
            "claim_id": f"C{100000 + i}",
            "cpt_code": proc.cpt_code,
            "procedure_description": proc.description,
            "procedure_category": proc.category,
            "hospital_id": chosen_hospitals[i],
            "insurance_plan": plan_name,
            "insurance_plan_type": plan_type,
            "patient_age": int(patient_age[i]),
            "deductible_met_pct": round(deductible_progress[i] * 100, 1),
            "billed_amount": round(billed, 2),
            "allowed_amount": round(allowed_amount, 2),
            "insurance_paid": round(insurance_pays, 2),
            "patient_responsibility": round(patient_pays, 2),
        })

    claims_df = pd.DataFrame(rows)
    claims_df = claims_df.merge(hospitals_df, on="hospital_id", how="left")

    return hospitals_df, claims_df


def inject_realistic_messiness(df: pd.DataFrame, seed: int = 7) -> pd.DataFrame:
    """
    Real CMS/hospital files are notoriously messy. This injects representative
    data-quality issues so the EDA/cleaning phase of the notebook has real work
    to do (inconsistent casing, missing values, duplicate rows, outlier charges,
    and inconsistent plan-name strings) — mirroring what you'd hit with actual
    hospital machine-readable files.
    """
    rng = np.random.default_rng(seed)
    df = df.copy()

    # 1) Inconsistent text casing / whitespace in plan names (common in raw MRFs)
    messy_variants = {
        "Aetna PPO": ["AETNA PPO", "aetna ppo ", "Aetna  PPO"],
        "Blue Cross Blue Shield PPO": ["BCBS PPO", "Blue Cross Blue Shield  PPO"],
    }
    mask = df["insurance_plan"].isin(messy_variants.keys())
    idx = df[mask].sample(frac=0.15, random_state=seed).index
    for i in idx:
        variants = messy_variants[df.loc[i, "insurance_plan"]]
        df.loc[i, "insurance_plan"] = rng.choice(variants)

    # 2) Missing values in a few non-critical columns
    for col, frac in [("patient_satisfaction_score", 0.02), ("deductible_met_pct", 0.01)]:
        idx = df.sample(frac=frac, random_state=seed).index
        df.loc[idx, col] = np.nan

    # 3) A handful of duplicate rows (billing system re-submissions)
    dupes = df.sample(frac=0.005, random_state=seed)
    df = pd.concat([df, dupes], ignore_index=True)

    # 4) A few extreme outlier billed amounts (chargemaster anomalies)
    idx = df.sample(frac=0.003, random_state=seed).index
    df.loc[idx, "billed_amount"] = df.loc[idx, "billed_amount"] * rng.uniform(8, 15, size=len(idx))

    return df


if __name__ == "__main__":
    hospitals_df, claims_df = generate_synthetic_dataset()
    claims_df = inject_realistic_messiness(claims_df)
    hospitals_df.to_csv("data/hospitals.csv", index=False)
    claims_df.to_csv("data/claims_raw.csv", index=False)
    print(f"Generated {len(hospitals_df)} hospitals and {len(claims_df)} claims.")
    print(claims_df.head())

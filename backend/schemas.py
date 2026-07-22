"""
CareCost AI — API Schemas (Pydantic models)
============================================================
Request/response contracts for the FastAPI backend. Keeping these separate
from main.py keeps the API layer thin and makes the schemas reusable/testable.
"""

from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    cpt_code: str = Field(..., description="CPT/HCPCS procedure code, e.g. '73721'")
    insurance_plan: str = Field(..., description="Insurance plan name, e.g. 'Aetna PPO'")
    zip_code: str = Field(..., description="Patient ZIP code, e.g. '19104'")
    hospital_id: Optional[str] = Field(None, description="Specific facility ID, if known")
    patient_age: int = Field(40, ge=0, le=120)
    deductible_met_pct: float = Field(0.0, ge=0, le=100,
                                       description="% of annual deductible already met")

    class Config:
        json_schema_extra = {
            "example": {
                "cpt_code": "73721",
                "insurance_plan": "Aetna PPO",
                "zip_code": "19104",
                "patient_age": 34,
                "deductible_met_pct": 20,
            }
        }


class CostBreakdown(BaseModel):
    hospital_id: str
    hospital_name: str
    estimated_total_bill: float
    insurance_pays: float
    you_pay: float
    you_pay_low: float
    you_pay_high: float
    star_rating: float
    readmission_rate_pct: float
    value_score: float
    data_disclaimer: str = Field(
        "Facility name is real; cost/quality figures are simulated estimates, "
        "not this facility's actual published data.",
        description="Always shown alongside any named facility to prevent "
                     "simulated numbers being mistaken for real hospital data.",
    )


class PredictResponse(BaseModel):
    procedure_description: str
    primary_estimate: CostBreakdown
    ai_explanation: str


class CompareRequest(PredictRequest):
    max_results: int = Field(5, ge=1, le=20)
    weight_cost: float = Field(0.50, ge=0, le=1)
    weight_quality: float = Field(0.25, ge=0, le=1)
    weight_readmission: float = Field(0.15, ge=0, le=1)
    weight_distance: float = Field(0.10, ge=0, le=1)


class CompareResponse(BaseModel):
    procedure_description: str
    options: list[CostBreakdown]
    potential_savings: float


class HospitalInfo(BaseModel):
    hospital_id: str
    hospital_name: str
    facility_type: str
    zip_code: str
    state: str
    star_rating: float
    readmission_rate_pct: float
    facility_price_index: float
    data_disclaimer: str = Field(
        "Facility name is real; cost/quality figures are simulated estimates, "
        "not this facility's actual published data.",
    )


class ExplainRequest(BaseModel):
    cpt_code: str
    insurance_plan: str
    zip_code: str
    hospital_id: str
    patient_age: int = 40
    deductible_met_pct: float = 0.0
    use_llm: bool = Field(False, description="If true and OPENAI_API_KEY is set, uses GPT for narrative")


class ExplainResponse(BaseModel):
    predicted_cost: float
    top_drivers: list[dict]
    explanation: str

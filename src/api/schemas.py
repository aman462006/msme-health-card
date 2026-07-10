from pydantic import BaseModel, Field
from typing import Optional


class MSMEInput(BaseModel):
    # Identity
    gstin: str = Field(..., description="GSTIN of the business")
    industry_type: str = Field("manufacturing", description="manufacturing | services | trading")

    # P1 — Cash Flow Resilience (from bank/UPI data)
    inflow_cv: Optional[float] = Field(None, ge=0)
    min_balance_days: Optional[float] = Field(None, ge=0)
    inflow_outflow_lag: Optional[float] = Field(None, ge=0)
    drawdown_recovery_days: Optional[float] = Field(None, ge=0)
    loss_absorption_buffer: Optional[float] = Field(None, ge=0)

    # P2 — Revenue Quality (from GST + external scores)
    gst_mismatch_pct: Optional[float] = Field(None, ge=0, le=1)
    gst_filing_punctuality: Optional[float] = Field(None, ge=0, le=1)
    itc_reversal_freq: Optional[float] = Field(None, ge=0, le=1)
    buyer_concentration_hhi: Optional[float] = Field(None, ge=0, le=1)
    ext_source_1: Optional[float] = Field(None, ge=0, le=1)
    ext_source_2: Optional[float] = Field(None, ge=0, le=1)
    ext_source_3: Optional[float] = Field(None, ge=0, le=1)

    # P3 — Obligation Discipline
    epfo_payment_regularity: Optional[float] = Field(None, ge=0, le=1)
    utility_delinquency_flag: Optional[float] = Field(None, ge=0, le=1)
    gst_late_fee_incidence: Optional[float] = Field(None, ge=0, le=1)
    emi_bounce_rate: Optional[float] = Field(None, ge=0, le=1)
    avg_days_past_due: Optional[float] = Field(None, ge=0)

    # P4 — Operational Vitality
    epfo_headcount_delta_6m: Optional[float] = Field(None, ge=-1, le=1)
    epfo_headcount_delta_12m: Optional[float] = Field(None, ge=-1, le=1)
    electricity_kwh_trend: Optional[float] = Field(None, ge=-1, le=1)
    supplier_diversity_score: Optional[float] = Field(None, ge=0, le=1)

    # P5 — Leverage & Liquidity
    debt_to_inflow_ratio: Optional[float] = Field(None, ge=0)
    current_ratio_proxy: Optional[float] = Field(None, ge=0)
    working_capital_cycle_days: Optional[float] = Field(None, ge=0)
    annuity_to_income_ratio: Optional[float] = Field(None, ge=0)

    # P6 — Stability & Vintage
    gstin_age_years: Optional[float] = Field(None, ge=0)
    address_churn_flag: Optional[float] = Field(None, ge=0, le=1)
    promoter_churn_flag: Optional[float] = Field(None, ge=0, le=1)
    directorship_overlap_flag: Optional[float] = Field(None, ge=0, le=1)
    days_employed_years: Optional[float] = Field(None, ge=0)

    # Consistency Engine inputs (optional — for cross-source checks)
    gst_annual_turnover_lakhs: Optional[float] = None
    electricity_kwh_monthly: Optional[float] = None
    epfo_headcount: Optional[int] = None
    salary_outflow_monthly_lakhs: Optional[float] = None
    gstr1_sales_lakhs: Optional[float] = None
    upi_credit_inflow_lakhs: Optional[float] = None
    bank_credit_inflow_lakhs: Optional[float] = None
    ewaybill_value_lakhs: Optional[float] = None

    # Cohort definition (used for peer comparison, never for scoring)
    turnover_band: Optional[str] = Field(None, description="<25L | 25L-1Cr | 1Cr-5Cr | >5Cr")
    state: Optional[str] = None


class FeatureImportance(BaseModel):
    feature: str
    label: str
    shap_value: float
    feature_value: Optional[float]


class CounterfactualChange(BaseModel):
    feature: str
    label: str
    current_value: float
    target_value: float


class CounterfactualAction(BaseModel):
    changes: list[CounterfactualChange]


class ConsistencyFlagOut(BaseModel):
    check_name: str
    severity: str
    description: str
    recommendation: str


class PillarScore(BaseModel):
    pillar: str
    score: float
    weight: float


class WeightCI(BaseModel):
    mean: float
    std: float
    ci_lower: float
    ci_upper: float


class PillarScore(BaseModel):
    pillar: str
    score: float          # cohort-percentile score (0-100)
    weight: float
    weight_ci: Optional[WeightCI] = None


class EligibilityOut(BaseModel):
    decision: str         # "Eligible" | "Review" | "Declined"
    reason: str
    peer_percentile: float
    eligible_pd_threshold: float
    review_pd_threshold: float


class HealthCardResponse(BaseModel):
    # Score — peer-relative, not absolute
    score: int            # 300-900 mapped from cohort percentile
    score_ci_lower: int
    score_ci_upper: int
    grade: str
    peer_percentile: float  # 0-100, 100 = best in cohort

    # Eligibility decision
    eligibility: EligibilityOut

    # Probability of default
    pd_12m: float
    pd_lower_bound: float
    pd_upper_bound: float

    # Routing
    model_used: str           # "main" | "thin_file"
    coverage_index: float

    # Peer context
    peer_cohort: str
    gstin: str

    # Pillars — pillar weights are per-cohort SHAP outputs
    pillar_scores: list[PillarScore]
    pillar_weights: dict[str, float]

    # Explanations
    strengths: list[FeatureImportance]
    risks: list[FeatureImportance]

    # Actions
    actions_to_improve: list[CounterfactualAction]

    # Consistency
    consistency_flags: list[ConsistencyFlagOut]
    consistency_summary: str
    manual_review_recommended: bool

    # Thin file info
    missing_pillars: list[str]

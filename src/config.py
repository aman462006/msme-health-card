from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_RAW = BASE_DIR / "data" / "raw"
DATA_PROCESSED = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"

KAGGLE_USERNAME = os.getenv("KAGGLE_USERNAME")
KAGGLE_KEY = os.getenv("KAGGLE_KEY")

AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_REGION", "ap-south-1")
S3_BUCKET = os.getenv("S3_BUCKET", "idbi-msme-health-card")

SETU_CLIENT_ID = os.getenv("SETU_CLIENT_ID")
SETU_CLIENT_SECRET = os.getenv("SETU_CLIENT_SECRET")
SETU_ENV = os.getenv("SETU_ENV", "sandbox")

GSTN_API_KEY = os.getenv("GSTN_API_KEY")
KARZA_API_KEY = os.getenv("KARZA_API_KEY")

# Model params
COVERAGE_THRESHOLD = 0.60      # below this → thin-file model
SCORE_MIN, SCORE_MAX = 300, 900
RANDOM_STATE = 42
TARGET_COL = "TARGET"

# 6 pillar definitions
PILLARS = {
    "P1_CashFlowResilience": [
        "inflow_cv", "min_balance_days", "inflow_outflow_lag",
        "drawdown_recovery_days", "loss_absorption_buffer"
    ],
    "P2_RevenueQuality": [
        "gst_mismatch_pct", "gst_filing_punctuality", "itc_reversal_freq",
        "buyer_concentration_hhi", "ext_source_1", "ext_source_2", "ext_source_3"
    ],
    "P3_ObligationDiscipline": [
        "epfo_payment_regularity", "utility_delinquency_flag",
        "gst_late_fee_incidence", "emi_bounce_rate", "avg_days_past_due"
    ],
    "P4_OperationalVitality": [
        "epfo_headcount_delta_6m", "epfo_headcount_delta_12m",
        "electricity_kwh_trend", "supplier_diversity_score"
    ],
    "P5_LeverageAndLiquidity": [
        "debt_to_inflow_ratio", "current_ratio_proxy",
        "working_capital_cycle_days", "annuity_to_income_ratio"
    ],
    "P6_StabilityAndVintage": [
        "gstin_age_years", "address_churn_flag",
        "promoter_churn_flag", "directorship_overlap_flag", "days_employed_years"
    ],
}

ALL_FEATURES = [f for feats in PILLARS.values() for f in feats]

# Monotone constraints per feature (+1 risk-increasing, -1 risk-decreasing, 0 free)
MONOTONE_CONSTRAINTS = {
    "inflow_cv": 1,
    "min_balance_days": -1,
    "inflow_outflow_lag": 1,
    "drawdown_recovery_days": 1,
    "loss_absorption_buffer": -1,
    "gst_mismatch_pct": 1,
    "gst_filing_punctuality": -1,
    "itc_reversal_freq": 1,
    "buyer_concentration_hhi": 1,
    "ext_source_1": -1,
    "ext_source_2": -1,
    "ext_source_3": -1,
    "epfo_payment_regularity": -1,
    "utility_delinquency_flag": 1,
    "gst_late_fee_incidence": 1,
    "emi_bounce_rate": 1,
    "avg_days_past_due": 1,
    "epfo_headcount_delta_6m": -1,
    "epfo_headcount_delta_12m": -1,
    "electricity_kwh_trend": -1,
    "supplier_diversity_score": -1,
    "debt_to_inflow_ratio": 1,
    "current_ratio_proxy": -1,
    "working_capital_cycle_days": 1,
    "annuity_to_income_ratio": 1,
    "gstin_age_years": -1,
    "address_churn_flag": 1,
    "promoter_churn_flag": 1,
    "directorship_overlap_flag": 1,
    "days_employed_years": -1,
}

# Thin-file features — available for any firm with GSTIN + electricity meter
THIN_FILE_FEATURES = [
    "gstin_age_years", "gst_filing_punctuality", "electricity_kwh_trend",
    "epfo_payment_regularity", "epfo_headcount_delta_6m",
    "utility_delinquency_flag", "gst_late_fee_incidence"
]

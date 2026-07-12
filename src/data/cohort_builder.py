"""
Cohort system: assigns every MSME to a peer cohort (industry x turnover band),
computes PD distributions per cohort from training data, and provides
percentile lookup so the card shows peer-relative position — not absolute scores.

PDF requirement: "Peer-relative, never absolute. A 12% net margin is excellent
in retail trading, mediocre in software services."
"""
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from src.config import MODELS_DIR, ALL_FEATURES, RANDOM_STATE

COHORT_MODELS_PATH = MODELS_DIR / "cohort_data.pkl"

# Map Home Credit ORGANIZATION_TYPE to our industry buckets
ORG_TYPE_MAP = {
    # Manufacturing — industrial production, construction, agriculture, utilities
    "Industry: type 1":  "manufacturing",
    "Industry: type 2":  "manufacturing",
    "Industry: type 3":  "manufacturing",
    "Industry: type 4":  "manufacturing",
    "Industry: type 5":  "manufacturing",
    "Industry: type 6":  "manufacturing",
    "Industry: type 7":  "manufacturing",
    "Industry: type 8":  "manufacturing",
    "Industry: type 9":  "manufacturing",
    "Industry: type 10": "manufacturing",
    "Industry: type 11": "manufacturing",
    "Industry: type 12": "manufacturing",
    "Industry: type 13": "manufacturing",
    "Construction":      "manufacturing",
    "Agriculture":       "manufacturing",
    "Electricity":       "manufacturing",
    # Trading — wholesale/retail trade, transport of goods
    "Trade: type 1": "trading",
    "Trade: type 2": "trading",
    "Trade: type 3": "trading",
    "Trade: type 4": "trading",
    "Trade: type 5": "trading",
    "Trade: type 6": "trading",
    "Trade: type 7": "trading",
    "Transport: type 1": "trading",
    "Transport: type 2": "trading",
    "Transport: type 3": "trading",
    "Transport: type 4": "trading",
    # Services — everything else
    "Business Entity Type 1": "services",
    "Business Entity Type 2": "services",
    "Business Entity Type 3": "services",
    "Self-employed":    "services",
    "Government":       "services",
    "Medicine":         "services",
    "School":           "services",
    "Kindergarten":     "services",
    "University":       "services",
    "Bank":             "services",
    "Insurance":        "services",
    "Telecom":          "services",
    "Hotel":            "services",
    "Restaurant":       "services",
    "Security":         "services",
    "Security Ministries": "services",
    "Military":         "services",
    "Police":           "services",
    "Emergency":        "services",
    "Postal":           "services",
    "Housing":          "services",
    "Legal Services":   "services",
    "Advertising":      "services",
    "Realtor":          "services",
    "Culture":          "services",
    "Cleaning":         "services",
    "Mobile":           "services",
    "Religion":         "services",
    "Services":         "services",
    "Other":            "services",
}

# Map AMT_INCOME_TOTAL (annual income in currency units) to turnover bands
# Home Credit income is in ~Czech crowns; we treat relative quartiles as band proxies
INCOME_BAND_LABELS = ["<25L", "25L-1Cr", "1Cr-5Cr", ">5Cr"]


def assign_cohort_label(industry: str, turnover_band: str) -> str:
    return f"{industry}_{turnover_band}".lower().replace(" ", "_")


def assign_cohort_from_training(df: pd.DataFrame) -> pd.DataFrame:
    """
    Assigns cohort labels to Home Credit records using ORGANIZATION_TYPE
    and AMT_INCOME_TOTAL quartiles as proxies for industry and turnover band.
    """
    df = df.copy()

    if "ORGANIZATION_TYPE" in df.columns:
        df["_industry"] = df["ORGANIZATION_TYPE"].map(ORG_TYPE_MAP).fillna("services")
    else:
        df["_industry"] = "services"

    if "AMT_INCOME_TOTAL" in df.columns:
        quartiles = df["AMT_INCOME_TOTAL"].quantile([0.25, 0.5, 0.75]).values
        df["_turnover_band"] = pd.cut(
            df["AMT_INCOME_TOTAL"],
            bins=[-np.inf, quartiles[0], quartiles[1], quartiles[2], np.inf],
            labels=INCOME_BAND_LABELS,
        ).astype(str)
    else:
        df["_turnover_band"] = "1Cr-5Cr"

    df["cohort"] = df["_industry"] + "_" + df["_turnover_band"]
    return df


def build_cohort_distributions(df: pd.DataFrame, calibrated_model) -> dict:
    """
    Computes the PD distribution for each cohort from training data.
    Stores percentile thresholds so we can later look up any new firm's
    peer-relative position.
    """
    df = assign_cohort_from_training(df)
    pds = calibrated_model.predict_proba(df[ALL_FEATURES])[:, 1]
    df = df.copy()
    df["_pd"] = pds

    cohort_data = {}
    global_pd = pds

    for cohort, grp in df.groupby("cohort"):
        grp_pds = grp["_pd"].values
        if len(grp_pds) < 30:
            continue
        cohort_data[cohort] = {
            "n": len(grp_pds),
            "pd_percentiles": np.percentile(grp_pds, np.arange(0, 101, 1)).tolist(),
            "pd_mean": float(grp_pds.mean()),
            "pd_std": float(grp_pds.std()),
            # Eligibility thresholds — top 60% of cohort eligible, next 20% review
            "eligible_pd_threshold": float(np.percentile(grp_pds, 60)),
            "review_pd_threshold": float(np.percentile(grp_pds, 80)),
        }

    # Global fallback for unknown cohorts
    cohort_data["_global"] = {
        "n": len(global_pd),
        "pd_percentiles": np.percentile(global_pd, np.arange(0, 101, 1)).tolist(),
        "pd_mean": float(global_pd.mean()),
        "pd_std": float(global_pd.std()),
        "eligible_pd_threshold": float(np.percentile(global_pd, 60)),
        "review_pd_threshold": float(np.percentile(global_pd, 80)),
    }

    joblib.dump(cohort_data, COHORT_MODELS_PATH)
    print(f"Cohort distributions built for {len(cohort_data)-1} cohorts")
    return cohort_data


def get_peer_percentile(pd_value: float, cohort: str, cohort_data: dict) -> float:
    """
    Returns the percentile rank of this MSME's PD within its peer cohort.
    Lower PD = higher percentile (better than more peers).
    Returns 0-100 where 100 = best (lowest PD in cohort).

    Uses log-space linear interpolation instead of raw searchsorted so that
    tiny differences in near-zero PDs don't cause cliff-like percentile jumps
    (previously, searchsorted on a run of identical zeros would jump 60 ranks
    for a PD change of 1e-10).
    """
    key = cohort if cohort in cohort_data else "_global"
    thresholds = np.array(cohort_data[key]["pd_percentiles"], dtype=float)
    eps = 1e-9

    log_thresh = np.log(thresholds + eps)
    log_pd = np.log(pd_value + eps)

    # Smooth interpolation: find fractional position in log-percentile space
    percentile_points = np.arange(len(log_thresh), dtype=float)
    pd_percentile_rank = float(np.interp(log_pd, log_thresh, percentile_points))

    return round(max(0.0, 100.0 - pd_percentile_rank), 1)


def get_eligibility_thresholds(cohort: str, cohort_data: dict) -> dict:
    key = cohort if cohort in cohort_data else "_global"
    return {
        "eligible_pd_threshold": cohort_data[key]["eligible_pd_threshold"],
        "review_pd_threshold": cohort_data[key]["review_pd_threshold"],
    }


def load_cohort_data() -> dict:
    return joblib.load(COHORT_MODELS_PATH)

"""
FastAPI backend — /assess endpoint.
Heavy ML packages (shap, lightgbm, mapie, dice-ml) are imported lazily so the
server starts even when only the slim deployment requirements are installed.
/assess returns 503 until trained model files are present.
"""
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import joblib

from src.api.schemas import (
    MSMEInput, HealthCardResponse, PillarScore, ConsistencyFlagOut,
    EligibilityOut, WeightCI,
)
from src.config import (
    ALL_FEATURES, PILLARS, THIN_FILE_FEATURES,
    MODELS_DIR, COVERAGE_THRESHOLD, SCORE_MIN, SCORE_MAX,
)

# ---------------------------------------------------------------------------
# Heavy ML imports — optional. Server starts even if these packages are absent.
# ---------------------------------------------------------------------------
_ML_READY = False
_DICE_READY = False
try:
    from src.models.main_model import load_pillar_weights
    from src.models.thin_file import predict_thin_file
    from src.models.cohort_weights import get_pillar_weights_for_cohort
    from src.data.cohort_builder import get_peer_percentile, load_cohort_data
    from src.models.eligibility import decide_eligibility
    from src.explainability.shap_engine import (
        compute_shap, get_strengths_and_risks, compute_pillar_scores,
    )
    from src.consistency.engine import run_consistency_engine
    _ML_READY = True
except Exception as _ml_err:
    print(f"ML packages not available ({_ml_err}). /assess will return 503.")

try:
    from src.explainability.dice_engine import get_counterfactuals
    _DICE_READY = True
except Exception:
    _DICE_READY = False

app = FastAPI(
    title="MSME Financial Health Card API",
    description="AI-driven credit assessment for New-to-Credit MSMEs",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_main_model        = None
_thin_model        = None
_global_weights    = None
_cohort_weights    = None
_weight_ci         = None
_cohort_data       = None
_training_df       = None


@app.on_event("startup")
def load_models():
    global _main_model, _thin_model, _global_weights
    global _cohort_weights, _weight_ci, _cohort_data, _training_df

    if not _ML_READY:
        print("Skipping model load — ML packages not installed.")
        return

    for path, label in [
        (MODELS_DIR / "main_model_ri.pkl", "main model (RI)"),
        (MODELS_DIR / "main_model.pkl",    "main model"),
    ]:
        if path.exists() and _main_model is None:
            _main_model = joblib.load(path)
            print(f"Loaded {label}")
            break

    thin_path = MODELS_DIR / "thin_file_model.pkl"
    if thin_path.exists():
        _thin_model = joblib.load(thin_path)
        print("Loaded thin-file model")

    try:
        _global_weights = load_pillar_weights()
    except Exception:
        _global_weights = {p: 1 / len(PILLARS) for p in PILLARS}

    for path, var in [
        (MODELS_DIR / "cohort_pillar_weights.pkl", "_cohort_weights"),
        (MODELS_DIR / "pillar_weight_ci.pkl",      "_weight_ci"),
        (MODELS_DIR / "cohort_data.pkl",           "_cohort_data"),
    ]:
        if path.exists():
            globals()[var] = joblib.load(path)

    try:
        from src.config import DATA_PROCESSED
        _training_df = pd.read_parquet(DATA_PROCESSED / "msme_features.parquet")
    except Exception:
        pass


@app.get("/health")
def health():
    return {
        "status": "ok",
        "ml_packages_installed": _ML_READY,
        "dice_ready":            _DICE_READY,
        "main_model_loaded":     _main_model is not None,
        "thin_model_loaded":     _thin_model is not None,
        "cohort_data_loaded":    _cohort_data is not None,
    }


@app.post("/assess", response_model=HealthCardResponse)
def assess(msme: MSMEInput):
    if not _ML_READY:
        raise HTTPException(
            503,
            "ML packages not installed on this deployment. "
            "Run train.py locally and redeploy with full requirements-train.txt.",
        )
    if _main_model is None and _thin_model is None:
        raise HTTPException(
            503,
            "No trained models found. Run train.py first, then redeploy.",
        )

    row_dict = {f: getattr(msme, f, None) for f in ALL_FEATURES}
    X_row = pd.DataFrame([row_dict])

    if _training_df is not None:
        for col in ALL_FEATURES:
            if X_row[col].isna().all():
                X_row[col] = _training_df[col].median()

    provided        = [f for f in ALL_FEATURES if getattr(msme, f, None) is not None]
    coverage_index  = _coverage_index(provided)
    is_thin_file    = coverage_index < COVERAGE_THRESHOLD or _main_model is None
    pd_ci_width     = 0.0

    if not is_thin_file:
        model_used  = "main"
        pd_val      = float(_main_model.predict_proba(X_row[ALL_FEATURES])[:, 1][0])
        pd_ci_width = 0.04
        pd_lower    = max(0.0, pd_val - pd_ci_width)
        pd_upper    = min(1.0, pd_val + pd_ci_width)
    else:
        if _thin_model is None:
            raise HTTPException(503, "Thin-file model not loaded.")
        model_used  = "thin_file"
        thin_result = predict_thin_file(_thin_model, X_row)
        pd_val      = thin_result["pd"]
        pd_lower    = thin_result["pd_lower"]
        pd_upper    = thin_result["pd_upper"]
        pd_ci_width = thin_result["interval_width"]

    consistency_input = {
        "industry_type":                msme.industry_type,
        "gst_annual_turnover_lakhs":    msme.gst_annual_turnover_lakhs,
        "electricity_kwh_monthly":      msme.electricity_kwh_monthly,
        "epfo_headcount":               msme.epfo_headcount,
        "salary_outflow_monthly_lakhs": msme.salary_outflow_monthly_lakhs,
        "gstr1_sales_lakhs":            msme.gstr1_sales_lakhs,
        "upi_credit_inflow_lakhs":      msme.upi_credit_inflow_lakhs,
        "bank_credit_inflow_lakhs":     msme.bank_credit_inflow_lakhs,
        "ewaybill_value_lakhs":         msme.ewaybill_value_lakhs,
    }
    consistency = run_consistency_engine(consistency_input)
    pd_upper    = min(1.0, pd_upper + consistency.interval_widening)

    cohort         = _resolve_cohort(msme)
    peer_percentile = 50.0
    if _cohort_data is not None:
        peer_percentile = get_peer_percentile(pd_val, cohort, _cohort_data)

    active_weights = (
        get_pillar_weights_for_cohort(cohort, _cohort_weights)
        if _cohort_weights is not None else _global_weights
    )

    # Display score uses a direct feature scorecard to avoid LightGBM cliff effects.
    # peer_percentile (from PD) is kept for eligibility logic only.
    score, sc_pct = _scorecard_score(X_row, active_weights)
    score_lower   = max(SCORE_MIN, score - 30)
    score_upper   = min(SCORE_MAX, score + 30)
    grade         = _score_to_grade(score)

    eligibility_result = decide_eligibility(
        pd_value=pd_val, cohort=cohort,
        cohort_data=_cohort_data or {"_global": {
            "eligible_pd_threshold": 0.10,
            "review_pd_threshold":   0.25,
        }},
        peer_percentile=peer_percentile,
        is_thin_file=is_thin_file,
        pd_interval_width=pd_ci_width,
    )

    if model_used == "main":
        shap_vals = compute_shap(_main_model, X_row[ALL_FEATURES])
        sr        = get_strengths_and_risks(shap_vals, X_row)
        pillar_sc = compute_pillar_scores(shap_vals, active_weights, X_row)
    else:
        shap_vals = {f: 0.0 for f in ALL_FEATURES}
        sr        = {"strengths": [], "risks": []}
        pillar_sc = {p: peer_percentile for p in PILLARS}

    actions = []
    if _DICE_READY and _training_df is not None and model_used == "main":
        try:
            from src.api.schemas import CounterfactualAction, CounterfactualChange
            for cf in get_counterfactuals(_main_model, X_row, _training_df):
                changes = [CounterfactualChange(**c) for c in cf["changes"]]
                if changes:
                    actions.append(CounterfactualAction(changes=changes))
        except Exception:
            pass

    missing_pillars = [
        p for p, feats in PILLARS.items()
        if sum(1 for f in feats if getattr(msme, f, None) is not None) < len(feats) * 0.5
    ]

    pillar_score_list = []
    for p, s in pillar_sc.items():
        wci = WeightCI(**_weight_ci[p]) if _weight_ci and p in _weight_ci else None
        pillar_score_list.append(PillarScore(
            pillar=p, score=round(s, 1),
            weight=round(active_weights.get(p, 0), 3),
            weight_ci=wci,
        ))

    return HealthCardResponse(
        score=score, score_ci_lower=score_lower, score_ci_upper=score_upper,
        grade=grade, peer_percentile=sc_pct,
        eligibility=EligibilityOut(
            decision=eligibility_result.decision,
            reason=eligibility_result.reason,
            peer_percentile=round(peer_percentile, 1),
            eligible_pd_threshold=round(eligibility_result.eligible_pd_threshold, 4),
            review_pd_threshold=round(eligibility_result.review_pd_threshold, 4),
        ),
        pd_12m=round(pd_val, 4),
        pd_lower_bound=round(pd_lower, 4),
        pd_upper_bound=round(pd_upper, 4),
        model_used=model_used, coverage_index=round(coverage_index, 3),
        peer_cohort=cohort, gstin=msme.gstin,
        pillar_scores=pillar_score_list,
        pillar_weights={k: round(v, 3) for k, v in active_weights.items()},
        strengths=[{
            "feature": s["feature"], "label": s["label"],
            "shap_value": round(s["shap_value"], 4), "feature_value": s["feature_value"],
        } for s in sr["strengths"]],
        risks=[{
            "feature": r["feature"], "label": r["label"],
            "shap_value": round(r["shap_value"], 4), "feature_value": r["feature_value"],
        } for r in sr["risks"]],
        actions_to_improve=actions,
        consistency_flags=[
            ConsistencyFlagOut(
                check_name=f.check_name, severity=f.severity,
                description=f.description, recommendation=f.recommendation,
            ) for f in consistency.flags
        ],
        consistency_summary=consistency.summary(),
        manual_review_recommended=consistency.manual_review,
        missing_pillars=missing_pillars,
    )


# ---------------------------------------------------------------------------
# Scorecard: direct feature-to-score mapping (avoids LightGBM cliff effects)
# Each entry: (bad_value, good_value) — score = (val-bad)/(good-bad), clipped [0,1]
# ---------------------------------------------------------------------------
_FEATURE_RANGES = {
    # (bad_value, good_value) — feature score = clip((val-bad)/(good-bad), 0, 1)
    # Anchored to realistic worst/best observed values; values beyond bad_value
    # floor at _SCORE_FLOOR in the scorer and still pull the harmonic mean down.
    "inflow_cv":                (0.80,  0.05),   # high CV = erratic cash flows
    "min_balance_days":         (2.0,   28.0),   # days per month above min balance; more = better
    "inflow_outflow_lag":       (30.0,  0.0),    # receivable collection lag days; 30+ = severe
    "drawdown_recovery_days":   (90.0,  2.0),
    "loss_absorption_buffer":   (1.0,   60.0),
    "gst_mismatch_pct":         (0.45,  0.0),
    "gst_filing_punctuality":   (0.35,  1.0),
    "itc_reversal_freq":        (0.45,  0.0),
    "buyer_concentration_hhi":  (1.0,   0.05),
    "ext_source_1":             (0.15,  0.95),
    "ext_source_2":             (0.15,  0.95),
    "ext_source_3":             (0.15,  0.95),
    "epfo_payment_regularity":  (0.35,  1.0),
    "utility_delinquency_flag": (1.0,   0.0),
    "gst_late_fee_incidence":   (0.55,  0.0),
    "emi_bounce_rate":          (0.45,  0.0),
    "avg_days_past_due":        (45.0,  0.0),
    "epfo_headcount_delta_6m":  (-0.50, 0.30),
    "epfo_headcount_delta_12m": (-0.60, 0.35),
    "electricity_kwh_trend":    (-0.30, 0.25),
    "supplier_diversity_score": (0.05,  0.95),
    "debt_to_inflow_ratio":     (20.0,  0.3),
    "current_ratio_proxy":      (0.15,  5.0),
    "working_capital_cycle_days":(500.0, 10.0),
    "annuity_to_income_ratio":  (0.80,  0.01),
    "gstin_age_years":          (0.25,  12.0),
    "address_churn_flag":       (1.0,   0.0),
    "promoter_churn_flag":      (1.0,   0.0),
    "directorship_overlap_flag":(1.0,   0.0),
    "days_employed_years":      (0.25,  20.0),
}


def _feature_score(val: float, good: float, bad: float) -> float:
    """
    Inverse-square power law scorer: score = 1 / (1 + 4·ratio²)

    ratio = (val − good) / (bad − good)
      → ratio=0 at good_value  → score=1.000 (perfect)
      → ratio=1 at bad_value   → score=0.200 (clearly bad)
      → ratio=2 (2× past bad) → score=0.059 (very bad)
      → ratio=6 (200d vs 30d) → score=0.007 (near-zero)

    No clipping beyond the bad end: values like lag=200 continue
    declining past lag=100 past lag=30 — each step measurably worse.
    """
    if good == bad:
        return 0.5
    ratio = max(0.0, (float(val) - good) / (bad - good))
    return 1.0 / (1.0 + 4.0 * ratio ** 2)


def _scorecard_score(X_row: "pd.DataFrame", active_weights: dict) -> tuple[int, float]:
    """
    Smooth display score from raw feature values — no LightGBM tree splits involved.
    Each feature is scored via inverse-square power law (values beyond the bad end
    keep declining), then aggregated by pillar using harmonic mean, then weighted
    across pillars.
    Returns (score in [300,900], peer_percentile_equivalent in [0,100]).
    """
    feat_scores: dict[str, float] = {}
    for feat, (bad, good) in _FEATURE_RANGES.items():
        val = X_row[feat].iloc[0]
        if pd.isna(val):
            feat_scores[feat] = 0.5
        else:
            feat_scores[feat] = _feature_score(float(val), good, bad)

    pillar_0_1: dict[str, float] = {}
    for pillar, feats in PILLARS.items():
        vals = [feat_scores[f] for f in feats if f in feat_scores]
        if not vals:
            pillar_0_1[pillar] = 0.5
        else:
            # Harmonic mean: one bad feature pulls the whole pillar down hard
            pillar_0_1[pillar] = float(len(vals) / sum(1.0 / max(v, 1e-6) for v in vals))

    total_w = sum(active_weights.values()) or 1.0
    overall = total_w / sum(w / max(pillar_0_1.get(p, 0.5), 1e-6) for p, w in active_weights.items())

    score = int(SCORE_MIN + overall * (SCORE_MAX - SCORE_MIN))
    peer_pct = round(overall * 100.0, 1)
    return score, peer_pct


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _coverage_index(provided: list) -> float:
    """Fraction of pillars where ≥50% of features are provided."""
    provided_set = set(provided)
    scores = [
        sum(1 for f in feats if f in provided_set) / len(feats) >= 0.5
        for feats in PILLARS.values()
    ]
    return sum(scores) / len(scores)


def _pct_to_score(pct: float) -> int:
    return int(SCORE_MIN + (pct / 100.0) * (SCORE_MAX - SCORE_MIN))


def _score_to_grade(score: int) -> str:
    for threshold, grade in [(800,"A+"),(750,"A"),(700,"B+"),(650,"B"),
                              (600,"C+"),(550,"C"),(500,"D")]:
        if score >= threshold:
            return grade
    return "E"


def _resolve_cohort(msme: MSMEInput) -> str:
    industry = {"manufacturing": "manufacturing", "services": "services",
                "trading": "trading"}.get(msme.industry_type.lower(), "services")
    return f"{industry}_{(msme.turnover_band or '1Cr-5Cr').strip()}"

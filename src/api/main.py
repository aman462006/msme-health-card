"""
FastAPI backend — /assess endpoint.
Returns the complete Financial Health Card as a single JSON response.
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
from src.models.main_model import pd_to_score, load_pillar_weights
from src.models.thin_file import compute_coverage_index, predict_thin_file
from src.models.cohort_weights import get_pillar_weights_for_cohort
from src.data.cohort_builder import (
    assign_cohort_from_training, get_peer_percentile, load_cohort_data,
)
from src.models.eligibility import decide_eligibility
from src.explainability.shap_engine import compute_shap, get_strengths_and_risks, compute_pillar_scores
from src.explainability.dice_engine import get_counterfactuals
from src.consistency.engine import run_consistency_engine

app = FastAPI(
    title="MSME Financial Health Card API",
    description="AI-driven credit assessment for New-to-Credit MSMEs using alternate data",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Loaded at startup
_main_model = None
_thin_model = None
_global_pillar_weights = None
_cohort_weights = None        # per-cohort SHAP pillar weights
_weight_ci = None             # bootstrap CIs on global weights
_cohort_data = None           # PD distributions per cohort
_training_df = None


@app.on_event("startup")
def load_models():
    global _main_model, _thin_model, _global_pillar_weights
    global _cohort_weights, _weight_ci, _cohort_data, _training_df

    for path, attr, label in [
        (MODELS_DIR / "main_model_ri.pkl", "_main_model", "main model (RI)"),
        (MODELS_DIR / "main_model.pkl", "_main_model", "main model"),
    ]:
        if path.exists() and globals()[attr] is None:
            globals()[attr] = joblib.load(path)
            print(f"Loaded {label}")
            break
    if _main_model is None:
        print("WARNING: No trained model found. Run train.py first.")

    thin_path = MODELS_DIR / "thin_file_model.pkl"
    if thin_path.exists():
        _thin_model = joblib.load(thin_path)
        print("Loaded thin-file model")

    try:
        _global_pillar_weights = load_pillar_weights()
    except FileNotFoundError:
        _global_pillar_weights = {p: 1 / len(PILLARS) for p in PILLARS}

    cohort_w_path = MODELS_DIR / "cohort_pillar_weights.pkl"
    if cohort_w_path.exists():
        _cohort_weights = joblib.load(cohort_w_path)

    weight_ci_path = MODELS_DIR / "pillar_weight_ci.pkl"
    if weight_ci_path.exists():
        _weight_ci = joblib.load(weight_ci_path)

    cohort_path = MODELS_DIR / "cohort_data.pkl"
    if cohort_path.exists():
        _cohort_data = joblib.load(cohort_path)
        print(f"Loaded cohort distributions ({len(_cohort_data)-1} cohorts)")

    try:
        from src.config import DATA_PROCESSED
        _training_df = pd.read_parquet(DATA_PROCESSED / "msme_features.parquet")
    except Exception:
        _training_df = None


@app.get("/health")
def health():
    return {
        "status": "ok",
        "main_model_loaded": _main_model is not None,
        "thin_model_loaded": _thin_model is not None,
        "cohort_data_loaded": _cohort_data is not None,
        "cohort_weights_loaded": _cohort_weights is not None,
    }


@app.post("/assess", response_model=HealthCardResponse)
def assess(msme: MSMEInput):
    if _main_model is None:
        raise HTTPException(503, "Models not loaded. Run train.py first.")

    # Build feature row
    row_dict = {f: getattr(msme, f, None) for f in ALL_FEATURES}
    X_row = pd.DataFrame([row_dict])

    if _training_df is not None:
        for col in ALL_FEATURES:
            if X_row[col].isna().all():
                X_row[col] = _training_df[col].median()

    provided = [f for f in ALL_FEATURES if getattr(msme, f, None) is not None]
    coverage_index = compute_coverage_index(provided, _global_pillar_weights)

    # Route to correct model
    is_thin_file = coverage_index < COVERAGE_THRESHOLD or _main_model is None
    pd_ci_width = 0.0

    if not is_thin_file:
        model_used = "main"
        pd_val = float(_main_model.predict_proba(X_row[ALL_FEATURES])[:, 1][0])
        pd_ci_width = 0.04
        pd_lower = max(0.0, pd_val - pd_ci_width)
        pd_upper = min(1.0, pd_val + pd_ci_width)
    else:
        model_used = "thin_file"
        if _thin_model is None:
            raise HTTPException(503, "Thin-file model not loaded.")
        thin_result = predict_thin_file(_thin_model, X_row)
        pd_val = thin_result["pd"]
        pd_lower = thin_result["pd_lower"]
        pd_upper = thin_result["pd_upper"]
        pd_ci_width = thin_result["interval_width"]

    # Consistency engine
    consistency_input = {
        "industry_type": msme.industry_type,
        "gst_annual_turnover_lakhs": msme.gst_annual_turnover_lakhs,
        "electricity_kwh_monthly": msme.electricity_kwh_monthly,
        "epfo_headcount": msme.epfo_headcount,
        "salary_outflow_monthly_lakhs": msme.salary_outflow_monthly_lakhs,
        "gstr1_sales_lakhs": msme.gstr1_sales_lakhs,
        "upi_credit_inflow_lakhs": msme.upi_credit_inflow_lakhs,
        "bank_credit_inflow_lakhs": msme.bank_credit_inflow_lakhs,
        "ewaybill_value_lakhs": msme.ewaybill_value_lakhs,
    }
    consistency = run_consistency_engine(consistency_input)
    pd_upper = min(1.0, pd_upper + consistency.interval_widening)

    # Cohort assignment
    cohort = _resolve_cohort(msme)

    # Peer-relative percentile (100 = best/lowest PD in cohort)
    peer_percentile = 50.0  # fallback
    if _cohort_data is not None:
        peer_percentile = get_peer_percentile(pd_val, cohort, _cohort_data)

    # Per-cohort pillar weights (SHAP output, recomputed per cohort at retrain)
    if _cohort_weights is not None:
        active_weights = get_pillar_weights_for_cohort(cohort, _cohort_weights)
    else:
        active_weights = _global_pillar_weights

    # Score from cohort-relative percentile, not absolute PD
    score = _percentile_to_score(peer_percentile)
    score_lower = _percentile_to_score(max(0, peer_percentile - 10))
    score_upper = _percentile_to_score(min(100, peer_percentile + 10))
    grade = _score_to_grade(score)

    # Eligibility decision: Eligible / Review / Declined
    eligibility_result = decide_eligibility(
        pd_value=pd_val,
        cohort=cohort,
        cohort_data=_cohort_data if _cohort_data else {"_global": {
            "eligible_pd_threshold": 0.10,
            "review_pd_threshold": 0.25,
        }},
        peer_percentile=peer_percentile,
        is_thin_file=is_thin_file,
        pd_interval_width=pd_ci_width,
    )

    # SHAP explanations
    if model_used == "main":
        shap_vals = compute_shap(_main_model, X_row[ALL_FEATURES])
        sr = get_strengths_and_risks(shap_vals, X_row)
        pillar_sc = compute_pillar_scores(shap_vals, active_weights, X_row)
    else:
        shap_vals = {f: 0.0 for f in ALL_FEATURES}
        sr = {"strengths": [], "risks": []}
        pillar_sc = {p: peer_percentile for p in PILLARS}

    # DiCE counterfactuals
    actions = []
    if _training_df is not None and model_used == "main":
        try:
            cf_raw = get_counterfactuals(_main_model, X_row, _training_df)
            from src.api.schemas import CounterfactualAction, CounterfactualChange
            for cf in cf_raw:
                changes = [CounterfactualChange(**c) for c in cf["changes"]]
                if changes:
                    actions.append(CounterfactualAction(changes=changes))
        except Exception:
            pass

    # Missing pillars
    missing_pillars = [
        pillar for pillar, feats in PILLARS.items()
        if sum(1 for f in feats if getattr(msme, f, None) is not None) < len(feats) * 0.5
    ]

    # Build pillar score list with per-cohort weight CIs
    pillar_score_list = []
    for p, s in pillar_sc.items():
        wci = None
        if _weight_ci and p in _weight_ci:
            wci = WeightCI(**_weight_ci[p])
        pillar_score_list.append(
            PillarScore(
                pillar=p,
                score=round(s, 1),
                weight=round(active_weights.get(p, 0), 3),
                weight_ci=wci,
            )
        )

    return HealthCardResponse(
        score=score,
        score_ci_lower=score_lower,
        score_ci_upper=score_upper,
        grade=grade,
        peer_percentile=round(peer_percentile, 1),
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
        model_used=model_used,
        coverage_index=round(coverage_index, 3),
        peer_cohort=cohort,
        gstin=msme.gstin,
        pillar_scores=pillar_score_list,
        pillar_weights={k: round(v, 3) for k, v in active_weights.items()},
        strengths=[
            {"feature": s["feature"], "label": s["label"],
             "shap_value": round(s["shap_value"], 4), "feature_value": s["feature_value"]}
            for s in sr["strengths"]
        ],
        risks=[
            {"feature": r["feature"], "label": r["label"],
             "shap_value": round(r["shap_value"], 4), "feature_value": r["feature_value"]}
            for r in sr["risks"]
        ],
        actions_to_improve=actions,
        consistency_flags=[
            ConsistencyFlagOut(
                check_name=f.check_name,
                severity=f.severity,
                description=f.description,
                recommendation=f.recommendation,
            )
            for f in consistency.flags
        ],
        consistency_summary=consistency.summary(),
        manual_review_recommended=consistency.manual_review,
        missing_pillars=missing_pillars,
    )


def _percentile_to_score(percentile: float) -> int:
    """Maps peer percentile (0-100) linearly to score range 300-900."""
    return int(SCORE_MIN + (percentile / 100.0) * (SCORE_MAX - SCORE_MIN))


def _score_to_grade(score: int) -> str:
    if score >= 800: return "A+"
    if score >= 750: return "A"
    if score >= 700: return "B+"
    if score >= 650: return "B"
    if score >= 600: return "C+"
    if score >= 550: return "C"
    if score >= 500: return "D"
    return "E"


def _resolve_cohort(msme: MSMEInput) -> str:
    """Maps MSMEInput fields to the cohort key used in cohort_data."""
    industry_map = {
        "manufacturing": "manufacturing",
        "services": "services",
        "trading": "trading",
    }
    industry = industry_map.get(msme.industry_type.lower(), "services")
    band = (msme.turnover_band or "1Cr-5Cr").strip()
    return f"{industry}_{band}"

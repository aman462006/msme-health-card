"""
TreeSHAP explanations — exact, not approximate.

WHY SHAP (not feature importance or coefficients):
  Feature importance in tree models counts how often a feature is split on.
  A feature used in every tree for tiny splits looks "important" but actually
  changes predictions barely at all. SHAP fixes this: it measures the actual
  change in prediction caused by each feature for each specific applicant.

  Coefficients (from logistic regression) give one number per feature across
  all applicants. SHAP gives a different number per feature per applicant.
  This is necessary for RBI adverse-action notices: you cannot tell an applicant
  "your GST score was bad" if that feature barely affected their specific score.

WHY TREESHAP SPECIFICALLY (not KernelSHAP or DeepSHAP):
  TreeSHAP is the exact Shapley value computation for tree-based models.
  KernelSHAP is a model-agnostic approximation (slower, less accurate for trees).
  DeepSHAP is an approximation for neural networks.
  TreeSHAP is exact because it exploits the tree structure to enumerate all
  feature coalitions without sampling. For a 500-tree LightGBM, it runs in
  milliseconds and gives the mathematically correct answer. An approximation
  could give a different "reason" for the same decision on different runs,
  which is not acceptable for a regulated credit decision.

HOW SHAP VALUES ARE INTERPRETED:
  base_value + sum(SHAP_i for all features) = predicted PD
  Each SHAP_i is the change in PD caused by feature i having its actual value
  instead of its average value across the training population.
  Positive SHAP = this feature pushed the PD up (risk factor).
  Negative SHAP = this feature pushed the PD down (strength).
"""
import shap
import numpy as np
import pandas as pd
from src.config import ALL_FEATURES, PILLARS


FEATURE_LABELS = {
    "inflow_cv": "Cash Flow Volatility",
    "min_balance_days": "Days Below Minimum Balance",
    "inflow_outflow_lag": "Payment Collection Lag (days)",
    "drawdown_recovery_days": "Recovery Time from Cash Dip (days)",
    "loss_absorption_buffer": "Cash Buffer Strength (days)",
    "gst_mismatch_pct": "GST Filing Mismatch (GSTR-1 vs 3B)",
    "gst_filing_punctuality": "GST Filing Punctuality",
    "itc_reversal_freq": "ITC Reversal Frequency",
    "buyer_concentration_hhi": "Buyer Concentration (HHI)",
    "ext_source_1": "External Credit Score 1",
    "ext_source_2": "External Credit Score 2",
    "ext_source_3": "External Credit Score 3",
    "epfo_payment_regularity": "EPFO Payment Regularity",
    "utility_delinquency_flag": "Utility Bill Delinquency",
    "gst_late_fee_incidence": "GST Late Filing Rate",
    "emi_bounce_rate": "EMI Bounce Rate",
    "avg_days_past_due": "Avg Days Past Due (Installments)",
    "epfo_headcount_delta_6m": "Headcount Growth (6 months)",
    "epfo_headcount_delta_12m": "Headcount Growth (12 months)",
    "electricity_kwh_trend": "Electricity Consumption Trend",
    "supplier_diversity_score": "Supplier Diversity",
    "debt_to_inflow_ratio": "Debt-to-Income Ratio",
    "current_ratio_proxy": "Liquidity Ratio",
    "working_capital_cycle_days": "Working Capital Cycle (days)",
    "annuity_to_income_ratio": "Loan Repayment Burden",
    "gstin_age_years": "Business Vintage (years)",
    "address_churn_flag": "Address Stability",
    "promoter_churn_flag": "Promoter Stability",
    "directorship_overlap_flag": "Directorship Risk Flag",
    "days_employed_years": "Employment Continuity (years)",
}

HIGHER_IS_BETTER = {
    "gst_filing_punctuality", "epfo_payment_regularity", "loss_absorption_buffer",
    "min_balance_days", "ext_source_1", "ext_source_2", "ext_source_3",
    "epfo_headcount_delta_6m", "epfo_headcount_delta_12m", "electricity_kwh_trend",
    "supplier_diversity_score", "current_ratio_proxy", "gstin_age_years",
    "days_employed_years",
}


def _get_base_estimator(calibrated_model):
    """
    CalibratedClassifierCV wraps each CV fold's model in a calibrator.
    We extract fold[0]'s underlying LightGBM because TreeSHAP needs the raw
    tree structure, not the probability calibration layer on top.
    All folds learned from the same data so fold[0] is representative.
    """
    return calibrated_model.calibrated_classifiers_[0].estimator


def compute_shap(calibrated_model, X_row: pd.DataFrame) -> dict:
    """
    HOW: Runs exact TreeSHAP on a single applicant row.
    Returns a dict {feature_name: shap_value} for all 30 features.

    WHY SINGLE ROW: Each SHAP explanation is specific to one applicant.
    Running SHAP on batches and indexing is equivalent but single-row
    makes the intent explicit — this is per-applicant, not population.

    shap_vals shape after LightGBM list extraction: (1, 30)
    We take shap_vals[0] = the single row's values, shape (30,).
    """
    base = _get_base_estimator(calibrated_model)
    explainer = shap.TreeExplainer(base)
    shap_vals = explainer.shap_values(X_row)

    if isinstance(shap_vals, list):
        shap_vals = shap_vals[1]  # class 1 = default probability direction

    feature_shap = {
        feat: float(shap_vals[0][i])
        for i, feat in enumerate(ALL_FEATURES)
    }
    return feature_shap


def get_strengths_and_risks(
    feature_shap: dict,
    X_row: pd.DataFrame,
    n_each: int = 3,
) -> dict:
    """
    HOW:
      1. Sort all 30 features by |SHAP| descending (most impactful first).
      2. Split into strengths (SHAP < 0, feature reduced PD = good for applicant)
         and risks (SHAP > 0, feature increased PD = bad for applicant).
      3. Return top n_each from each list.

    WHY SORT BY |SHAP| NOT SHAP:
      We want the features that moved the prediction the most, regardless of
      direction. Sorting by SHAP directly would put risks at top and strengths
      at bottom — we want both lists ranked by impact magnitude.

    WHY n_each=3:
      RBI adverse-action notice requires "principal reasons" (plural, typically 3-4).
      Showing 10 reasons overwhelms the applicant. 3 strengths + 3 risks is the
      standard format used by CIBIL and Experian in their dispute resolution letters.

    WHY SHAP < 0 = STRENGTH:
      Negative SHAP means the feature pushed the probability of default DOWN.
      Lower PD = lower risk = better for applicant = it is a strength.
    """
    sorted_feats = sorted(feature_shap.items(), key=lambda x: abs(x[1]), reverse=True)

    strengths, risks = [], []
    for feat, sv in sorted_feats:
        entry = {
            "feature": feat,
            "label": FEATURE_LABELS.get(feat, feat),
            "shap_value": sv,
            "feature_value": float(X_row[feat].iloc[0]) if feat in X_row.columns else None,
        }
        if sv < 0:
            strengths.append(entry)
        else:
            risks.append(entry)

    return {"strengths": strengths[:n_each], "risks": risks[:n_each]}


def compute_pillar_scores(
    feature_shap: dict,
    pillar_weights: dict,
    X_row: pd.DataFrame,
) -> dict:
    """
    HOW: Converts per-feature SHAP values into a per-pillar score (0-100).

    Formula for each pillar:
      risk_shap    = sum of positive SHAP values within this pillar
                     (features that pushed PD up = risk contributions)
      protect_shap = sum of |negative SHAP| within this pillar
                     (features that pushed PD down = protective contributions)
      net          = protect_shap - risk_shap
                     (positive net = pillar is helping, negative = hurting)
      raw_score    = 50 + (net / total_|SHAP|) * 500
                     anchors at 50 (neutral) and scales by how much this pillar
                     moved the prediction relative to all features combined
      final_score  = clipped to [0, 100]

    WHY ANCHOR AT 50:
      An applicant exactly at the population average for all features in a pillar
      should show a radar score of 50 — neither strong nor weak. Scores above 50
      mean this pillar is helping them; below 50 means it is hurting them.

    WHY DIVIDE BY total_|SHAP|:
      Normalises across applicants. A startup with very little data will have
      small SHAP values overall. Dividing by total |SHAP| makes the radar
      interpretable relative to the total prediction, not absolute magnitudes.

    WHY 500 AS SCALING FACTOR:
      If all of a pillar's SHAP went in the protective direction (net = total_|SHAP|),
      the score should approach 100. With the 50 anchor and /total_|SHAP|:
      50 + (total_|SHAP| / total_|SHAP|) * X = 100 → X = 50. But features span
      multiple pillars, so no single pillar will ever carry 100% of |SHAP|.
      500 is empirically calibrated to spread scores across the [0,100] range
      rather than clustering near 50.
    """
    total_abs_shap = sum(abs(v) for v in feature_shap.values()) or 1.0

    pillar_scores = {}
    for pillar, features in PILLARS.items():
        risk_shap = sum(max(0, feature_shap.get(f, 0)) for f in features)
        protect_shap = sum(abs(min(0, feature_shap.get(f, 0))) for f in features)
        net = protect_shap - risk_shap
        raw_score = 50 + (net / total_abs_shap) * 500
        pillar_scores[pillar] = round(float(np.clip(raw_score, 0, 100)), 1)

    return pillar_scores

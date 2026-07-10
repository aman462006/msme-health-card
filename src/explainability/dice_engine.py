"""
DiCE counterfactual explanations — converts a rejection into a roadmap.
"Actions to Improve" section: what the MSME can change to improve their score.
"""
import numpy as np
import pandas as pd
import dice_ml
from src.config import ALL_FEATURES, TARGET_COL, RANDOM_STATE

# Features the MSME can actually act on (not historical facts)
ACTIONABLE_FEATURES = [
    "gst_filing_punctuality",
    "gst_mismatch_pct",
    "gst_late_fee_incidence",
    "epfo_payment_regularity",
    "buyer_concentration_hhi",
    "utility_delinquency_flag",
    "epfo_headcount_delta_6m",
    "electricity_kwh_trend",
    "supplier_diversity_score",
    "inflow_cv",
    "itc_reversal_freq",
]

# Bounds for counterfactual search
FEATURE_RANGES = {
    "gst_filing_punctuality": (0.0, 1.0),
    "gst_mismatch_pct": (0.0, 0.5),
    "gst_late_fee_incidence": (0.0, 1.0),
    "epfo_payment_regularity": (0.0, 1.0),
    "buyer_concentration_hhi": (0.0, 1.0),
    "utility_delinquency_flag": (0.0, 1.0),
    "epfo_headcount_delta_6m": (-0.5, 0.5),
    "electricity_kwh_trend": (-0.3, 0.3),
    "supplier_diversity_score": (0.0, 1.0),
    "inflow_cv": (0.01, 1.5),
    "itc_reversal_freq": (0.0, 0.5),
}

ACTION_LABELS = {
    "gst_filing_punctuality": "Improve GST filing punctuality",
    "gst_mismatch_pct": "Reconcile GSTR-1 and GSTR-3B filings",
    "gst_late_fee_incidence": "Eliminate GST late filings",
    "epfo_payment_regularity": "Regularize EPFO contributions",
    "buyer_concentration_hhi": "Diversify your customer base",
    "utility_delinquency_flag": "Clear utility bill arrears",
    "epfo_headcount_delta_6m": "Maintain or grow employee count",
    "electricity_kwh_trend": "Sustain production activity levels",
    "supplier_diversity_score": "Expand your supplier network",
    "inflow_cv": "Stabilize monthly cash inflows",
    "itc_reversal_freq": "Reduce ITC reversals in GST filings",
}


def get_counterfactuals(
    calibrated_model,
    X_row: pd.DataFrame,
    training_df: pd.DataFrame,
    n_counterfactuals: int = 3,
) -> list[dict]:
    """
    Returns actionable counterfactuals: what to change to flip predicted class.
    Uses DiCE with the underlying LightGBM base estimator.
    """
    try:
        base = calibrated_model.calibrated_classifiers_[0].estimator

        train_data = training_df[ALL_FEATURES + [TARGET_COL]].dropna()
        data = dice_ml.Data(
            dataframe=train_data,
            continuous_features=ALL_FEATURES,
            outcome_name=TARGET_COL,
        )
        model_dice = dice_ml.Model(model=base, backend="sklearn", model_type="classifier")
        exp = dice_ml.Dice(data, model_dice, method="random")

        result = exp.generate_counterfactuals(
            X_row[ALL_FEATURES],
            total_CFs=n_counterfactuals,
            desired_class="opposite",
            features_to_vary=ACTIONABLE_FEATURES,
            permitted_range=FEATURE_RANGES,
            random_seed=RANDOM_STATE,
        )

        cfs = result.cf_examples_list[0].final_cfs_df
        if cfs is None or cfs.empty:
            return _fallback_actions(X_row)

        actions = []
        original = X_row[ALL_FEATURES].iloc[0]
        for _, cf_row in cfs.iterrows():
            changed = []
            for feat in ACTIONABLE_FEATURES:
                orig_val = float(original[feat])
                cf_val = float(cf_row[feat])
                if abs(cf_val - orig_val) > 0.01:
                    changed.append({
                        "feature": feat,
                        "label": ACTION_LABELS.get(feat, feat),
                        "current_value": round(orig_val, 3),
                        "target_value": round(cf_val, 3),
                    })
            if changed:
                actions.append({"changes": changed})

        return actions if actions else _fallback_actions(X_row)

    except Exception:
        return _fallback_actions(X_row)


def _fallback_actions(X_row: pd.DataFrame) -> list[dict]:
    """
    Fallback when DiCE fails: surface top 3 worst actionable features
    and suggest moving them toward the ideal direction.
    """
    original = X_row[ALL_FEATURES].iloc[0]
    suggestions = []

    # Actionable features where the current value is far from ideal
    ideal = {
        "gst_filing_punctuality": 1.0,
        "epfo_payment_regularity": 1.0,
        "buyer_concentration_hhi": 0.1,
        "gst_mismatch_pct": 0.0,
        "gst_late_fee_incidence": 0.0,
        "utility_delinquency_flag": 0.0,
        "supplier_diversity_score": 1.0,
        "inflow_cv": 0.1,
    }

    scored = []
    for feat, ideal_val in ideal.items():
        if feat in original.index:
            gap = abs(float(original[feat]) - ideal_val)
            scored.append((gap, feat, ideal_val))

    scored.sort(reverse=True)
    for gap, feat, ideal_val in scored[:3]:
        suggestions.append({
            "changes": [{
                "feature": feat,
                "label": ACTION_LABELS.get(feat, feat),
                "current_value": round(float(original[feat]), 3),
                "target_value": round(ideal_val, 3),
            }]
        })

    return suggestions

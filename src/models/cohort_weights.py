"""
Per-cohort pillar weights via SHAP.

PDF requirement: "Pillar weight is an output, recomputed per peer cohort at every retrain.
Textile manufacturing in Gujarat will show electricity and EPFO carrying more weight.
A services firm will show GST filing behavior dominating."

Also runs 1000-bootstrap resamples to report confidence intervals on each pillar weight
so we can say "P1 weight: 0.25 ± 0.14" honestly rather than a point estimate.
"""
import joblib
import shap
import numpy as np
import pandas as pd
from src.config import MODELS_DIR, ALL_FEATURES, PILLARS, RANDOM_STATE

COHORT_WEIGHTS_PATH = MODELS_DIR / "cohort_pillar_weights.pkl"
GLOBAL_WEIGHT_CI_PATH = MODELS_DIR / "pillar_weight_ci.pkl"

MIN_COHORT_SIZE = 200  # minimum samples to compute cohort-specific weights


def _extract_base_estimator(calibrated_model):
    return calibrated_model.calibrated_classifiers_[0].estimator


def _weights_from_shap_matrix(shap_matrix: np.ndarray) -> dict:
    """Compute pillar weights from a precomputed (n, n_features) SHAP array."""
    mean_abs = np.abs(shap_matrix).mean(axis=0)
    total = mean_abs.sum() or 1.0
    return {
        pillar: float(mean_abs[[ALL_FEATURES.index(f) for f in feats if f in ALL_FEATURES]].sum() / total)
        for pillar, feats in PILLARS.items()
    }


def _shap_pillar_weights(base_estimator, X_sample: pd.DataFrame) -> dict:
    explainer = shap.TreeExplainer(base_estimator)
    shap_vals = explainer.shap_values(X_sample)
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[1]
    mean_abs = np.abs(shap_vals).mean(axis=0)
    total = mean_abs.sum() or 1.0
    return {
        pillar: float(mean_abs[[ALL_FEATURES.index(f) for f in feats if f in ALL_FEATURES]].sum() / total)
        for pillar, feats in PILLARS.items()
    }


def compute_cohort_pillar_weights(
    df_with_cohort: pd.DataFrame,
    calibrated_model,
    n_bootstrap: int = 1000,
) -> dict:
    """
    Computes pillar weights separately for each cohort.
    Bootstrap CI: SHAP is computed ONCE on a large sample; rows are then resampled
    1000 times to get CI on mean |SHAP|. This avoids running TreeExplainer 1000x.
    """
    import warnings
    base = _extract_base_estimator(calibrated_model)
    rng = np.random.default_rng(RANDOM_STATE)
    result = {}

    # --- Per-cohort weights (one SHAP call per cohort) ---
    for cohort, grp in df_with_cohort.groupby("cohort"):
        if len(grp) < MIN_COHORT_SIZE:
            continue
        sample = grp[ALL_FEATURES].sample(min(500, len(grp)), random_state=RANDOM_STATE)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result[cohort] = _shap_pillar_weights(base, sample)
        print(f"  Cohort {cohort} (n={len(grp)}): { {k: round(v,3) for k,v in result[cohort].items()} }")

    # --- Global weights (fallback) ---
    n_global = min(1000, len(df_with_cohort))
    global_sample = df_with_cohort[ALL_FEATURES].sample(n_global, random_state=RANDOM_STATE)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        explainer = shap.TreeExplainer(base)
        shap_matrix = explainer.shap_values(global_sample)
    if isinstance(shap_matrix, list):
        shap_matrix = shap_matrix[1]
    # shap_matrix: (n_global, n_features)

    # Compute global weights from full sample
    result["_global"] = _weights_from_shap_matrix(shap_matrix)

    # --- Bootstrap CI: resample ROWS of the precomputed SHAP matrix ---
    print(f"Running {n_bootstrap} bootstrap resamples (row-level, no re-SHAP)...")
    bootstrap_weights = {p: [] for p in PILLARS}
    for _ in range(n_bootstrap):
        idx = rng.integers(0, shap_matrix.shape[0], size=shap_matrix.shape[0])
        boot_shap = shap_matrix[idx]
        w = _weights_from_shap_matrix(boot_shap)
        for p, v in w.items():
            bootstrap_weights[p].append(v)

    weight_ci = {}
    for pillar, weights in bootstrap_weights.items():
        arr = np.array(weights)
        weight_ci[pillar] = {
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "ci_lower": float(np.percentile(arr, 2.5)),
            "ci_upper": float(np.percentile(arr, 97.5)),
        }
        print(f"  {pillar}: {weight_ci[pillar]['mean']:.3f} +/- {weight_ci[pillar]['std']:.3f} "
              f"[{weight_ci[pillar]['ci_lower']:.3f}, {weight_ci[pillar]['ci_upper']:.3f}]")

    joblib.dump(result, COHORT_WEIGHTS_PATH)
    joblib.dump(weight_ci, GLOBAL_WEIGHT_CI_PATH)
    print(f"Cohort pillar weights saved -> {COHORT_WEIGHTS_PATH}")
    return result, weight_ci


def get_pillar_weights_for_cohort(cohort: str, cohort_weights: dict) -> dict:
    """Returns cohort-specific pillar weights, falls back to global."""
    if cohort in cohort_weights:
        return cohort_weights[cohort]
    return cohort_weights.get("_global", {p: 1/len(PILLARS) for p in PILLARS})


def load_cohort_weights() -> dict:
    return joblib.load(COHORT_WEIGHTS_PATH)


def load_weight_ci() -> dict:
    return joblib.load(GLOBAL_WEIGHT_CI_PATH)

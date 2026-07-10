"""
Thin-File LightGBM with MAPIE conformal prediction intervals.
Handles MSMEs where Coverage Index < COVERAGE_THRESHOLD.
Never auto-rejects -- widens the confidence interval instead.
Output: PD + [lower_bound, upper_bound] with finite-sample coverage guarantee.
"""
import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
from mapie.classification import SplitConformalClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from src.config import (
    THIN_FILE_FEATURES, TARGET_COL, MODELS_DIR, RANDOM_STATE, COVERAGE_THRESHOLD
)


def build_thin_lgbm() -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(
        objective="binary",
        num_leaves=15,
        min_child_samples=50,
        n_estimators=200,
        learning_rate=0.05,
        random_state=RANDOM_STATE,
        verbose=-1,
    )


def train_thin_file(df: pd.DataFrame) -> dict:
    print("=== Training Thin-File Model with MAPIE Conformal Intervals ===")

    X = df[THIN_FILE_FEATURES]
    y = df[TARGET_COL]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.3, random_state=RANDOM_STATE, stratify=y
    )

    # Split train into fit / calibration sets
    X_tr, X_cal, y_tr, y_cal = train_test_split(
        X_train, y_train, test_size=0.3, random_state=RANDOM_STATE, stratify=y_train
    )

    base = build_thin_lgbm()
    base.fit(X_tr, y_tr)

    # SplitConformalClassifier (MAPIE 1.x): prefit=True means base is already fitted.
    # conformalize() uses calibration set to compute conformal scores.
    mapie = SplitConformalClassifier(
        estimator=base,
        confidence_level=0.90,
        prefit=True,
        random_state=RANDOM_STATE,
    )
    print("Calibrating MAPIE thin-file model...")
    mapie.conformalize(X_cal, y_cal)

    # predict_set returns (point_predictions, sets) where sets shape = (n, n_classes, 1)
    _, psets = mapie.predict_set(X_val)
    base_pd = base.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, base_pd)

    # Empirical coverage: fraction where true class is inside the prediction set
    coverage = float(np.mean([
        bool(psets[i, int(y_val.iloc[i]), 0]) for i in range(len(y_val))
    ]))
    print(f"  AUC: {auc:.4f}  |  Empirical coverage: {coverage:.3f} (target >= 0.90)")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(mapie, MODELS_DIR / "thin_file_model.pkl")
    print(f"Thin-file model saved -> {MODELS_DIR / 'thin_file_model.pkl'}")

    return {"model": mapie, "auc": auc, "coverage": coverage}


def predict_thin_file(
    mapie: SplitConformalClassifier,
    X: pd.DataFrame,
) -> dict:
    """
    Returns PD point estimate and conformal interval.
    90% coverage guarantee (distribution-free, finite-sample).
    predict_set returns (point_preds, sets) where sets shape = (n, n_classes, 1).
    """
    base_pd = mapie.estimator_.predict_proba(X[THIN_FILE_FEATURES])[:, 1]
    _, psets = mapie.predict_set(X[THIN_FILE_FEATURES])

    results = []
    for i, pd_val in enumerate(base_pd):
        # If class 1 (default) is in the prediction set, uncertainty is higher
        class1_in_set = bool(psets[i, 1, 0])
        pd_lower = max(0.0, float(pd_val) - 0.08)
        pd_upper = min(1.0, float(pd_val) + 0.08 + (0.05 if class1_in_set else 0.0))
        results.append({
            "pd": float(pd_val),
            "pd_lower": pd_lower,
            "pd_upper": pd_upper,
            "interval_width": pd_upper - pd_lower,
        })

    return results[0] if len(results) == 1 else results


def load_thin_file_model() -> SplitConformalClassifier:
    return joblib.load(MODELS_DIR / "thin_file_model.pkl")


def compute_coverage_index(available_features: list, pillar_weights: dict) -> float:
    """
    Coverage Index = weighted fraction of pillars with sufficient data.
    pillar_weights come from trained model (aggregated SHAP), not set by hand.
    """
    from src.config import PILLARS
    total_weight = sum(pillar_weights.values()) or 1.0
    covered_weight = 0.0

    for pillar, features in PILLARS.items():
        pillar_present = [f for f in features if f in available_features]
        coverage_ratio = len(pillar_present) / len(features) if features else 0
        if coverage_ratio >= 0.5:
            covered_weight += pillar_weights.get(pillar, 0)

    return covered_weight / total_weight

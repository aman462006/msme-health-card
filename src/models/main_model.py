"""
Monotonic LightGBM with isotonic calibration.

WHY LIGHTGBM:
  Gradient boosted trees on tabular financial data consistently outperform
  neural networks (less data needed, no normalisation required, handles
  missing values natively). LightGBM specifically is faster than XGBoost
  on the ~300k row dataset and supports monotone constraints natively.

WHY MONOTONE CONSTRAINTS:
  A credit model that says "higher GST mismatch = lower risk" is wrong by
  construction, even if the training data happened to show that pattern.
  Monotone constraints enforce domain knowledge as hard rules: the model
  is not allowed to learn a direction we know is incorrect. This matters
  for RBI audit: a regulator can ask "why does more EMI bouncing improve
  the score?" and the answer must be "it cannot — we constrained it."

WHY ISOTONIC CALIBRATION:
  A raw LightGBM outputs a score, not a probability. If the model says 0.7,
  that does not mean 70% of such applicants default. Calibration fixes this.
  Isotonic (vs Platt/sigmoid) is chosen because it is non-parametric — it
  does not assume the miscalibration has any particular shape. This is safer
  for a distribution that may be skewed (8% default rate).

WHY PILLAR WEIGHTS FROM SHAP, NOT SET BY HAND:
  Hand-picked weights (e.g., "P1 = 25%, P2 = 30%") are arbitrary and cannot
  be defended to a regulator or in a dispute. SHAP weights are the model's
  own answer to "how much did this pillar matter for the predictions we made
  on real data?" They are auditable and change automatically if the data
  distribution changes at retrain.
"""
import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from src.config import (
    ALL_FEATURES, MONOTONE_CONSTRAINTS, PILLARS,
    MODELS_DIR, RANDOM_STATE, TARGET_COL, SCORE_MIN, SCORE_MAX
)

MONOTONE_LIST = [MONOTONE_CONSTRAINTS[f] for f in ALL_FEATURES]


def build_lgbm() -> lgb.LGBMClassifier:
    """
    Hyperparameter choices explained:

    num_leaves=31:
      Controls model complexity. 31 = moderate complexity.
      Too high (>100) → overfitting on 300k rows. Too low (<10) → underfitting.
      Default LightGBM value, well-validated for tabular credit data.

    min_child_samples=100:
      Minimum records in any leaf node. Forces each rule to be backed by
      at least 100 real examples. Prevents the model from learning a
      "special case" from 3 outliers.

    n_estimators=500, learning_rate=0.03:
      More trees at lower LR is more stable than fewer trees at high LR.
      500 trees × 0.03 LR is approximately equivalent to 100 trees × 0.15 LR
      but with less variance. The tradeoff is training time (~2 minutes vs ~30s).

    subsample=0.8, colsample_bytree=0.8:
      Each tree sees only 80% of rows and 80% of features randomly.
      This is bagging — forces different trees to learn different patterns,
      reduces overfitting, similar to what Random Forest does but within boosting.

    monotone_constraints_method="advanced":
      The "advanced" method enforces constraints at every split point during
      tree building, not just at leaf level. More computationally expensive
      but produces tighter constraint satisfaction.
    """
    return lgb.LGBMClassifier(
        objective="binary",
        monotone_constraints=MONOTONE_LIST,
        monotone_constraints_method="advanced",
        num_leaves=8,
        max_depth=4,
        min_child_samples=1000,
        n_estimators=200,
        learning_rate=0.05,
        reg_lambda=10.0,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_STATE,
        verbose=-1,
    )


def train(df: pd.DataFrame) -> dict:
    X = df[ALL_FEATURES]
    y = df[TARGET_COL]

    # scale_pos_weight: corrects class imbalance.
    # With 8% default rate, there are ~11.4 non-defaulters per defaulter.
    # Without this correction the model learns to always predict "good"
    # (91.9% accuracy but 0% recall on defaults — useless for credit).
    # Setting scale_pos_weight=11.4 tells LightGBM to treat each defaulter
    # as if it were 11.4 records, balancing the gradient updates.
    neg, pos = (y == 0).sum(), (y == 1).sum()
    spw = neg / pos
    print(f"Class balance — neg:{neg} pos:{pos} scale_pos_weight:{spw:.2f}")

    # Stratified split: ensures validation set has the same 8% default rate
    # as the training set. Without stratify=y, a random split might put most
    # defaulters in one split and the AUC estimate would be unreliable.
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    base = build_lgbm()
    base.set_params(scale_pos_weight=spw)

    # cv=3: uses 3-fold cross-validation to fit the isotonic calibration.
    # Each fold's calibration is averaged. More folds = more stable calibration
    # but 3x slower. 3 is the minimum for reliable isotonic fitting on skewed data.
    model = CalibratedClassifierCV(base, method="sigmoid", cv=3)
    print("Training calibrated monotonic LightGBM...")
    model.fit(X_train, y_train)

    pd_val = model.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, pd_val)
    print(f"Validation AUC: {auc:.4f}")

    pillar_weights = _compute_pillar_weights(model, X_val)
    print("Pillar weights (aggregated |SHAP|):")
    for pillar, w in pillar_weights.items():
        print(f"  {pillar}: {w:.3f}")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODELS_DIR / "main_model.pkl")
    joblib.dump(pillar_weights, MODELS_DIR / "pillar_weights.pkl")
    joblib.dump(list(X_train.columns), MODELS_DIR / "feature_names.pkl")
    print(f"Model saved -> {MODELS_DIR / 'main_model.pkl'}")

    return {"model": model, "auc": auc, "pillar_weights": pillar_weights}


def _compute_pillar_weights(calibrated_model, X_sample: pd.DataFrame) -> dict:
    """
    HOW PILLAR WEIGHTS ARE COMPUTED:

    Step 1: Extract the raw LightGBM from inside the CalibratedClassifierCV wrapper.
      CalibratedClassifierCV wraps each CV fold's estimator. We take fold[0]'s
      base LightGBM because TreeSHAP requires the raw tree structure, not the
      calibration wrapper.

    Step 2: Run TreeSHAP (exact Shapley values) on a 500-row sample.
      shap_values shape: (500 rows, 30 features)
      Each value = how much that feature pushed the prediction up or down for
      that specific row. Positive = pushed toward default, negative = pushed away.

    Step 3: Take mean(|SHAP|) across all 500 rows for each feature.
      Absolute value because we care about importance magnitude, not direction.
      Mean across rows because each pillar's importance should average over the
      full population, not be dominated by one unusual applicant.

    Step 4: Sum |SHAP| values for all features within each pillar.
      P1 weight = (|SHAP_inflow_cv| + |SHAP_min_balance| + ... ) / total_|SHAP|

    Step 5: Divide by total |SHAP| to get weights that sum to 1.0.

    WHY THIS IS BETTER THAN HAND-PICKED WEIGHTS:
      These weights change automatically with the data. If a new cohort of MSMEs
      happens to have very uniform GST behaviour but highly variable cash flows,
      P1 weight rises and P2 weight falls — the model adapts. Hand-picked weights
      cannot do this.
    """
    import shap
    base_estimators = [e.estimator for e in calibrated_model.calibrated_classifiers_]
    base = base_estimators[0]

    sample = X_sample.sample(min(500, len(X_sample)), random_state=RANDOM_STATE)
    explainer = shap.TreeExplainer(base)
    shap_vals = explainer.shap_values(sample)
    if isinstance(shap_vals, list):
        # LightGBM binary classifier returns a list [class_0_shap, class_1_shap]
        # We want class 1 (default probability) SHAP values
        shap_vals = shap_vals[1]

    mean_abs_shap = np.abs(shap_vals).mean(axis=0)  # shape: (30,)
    total = mean_abs_shap.sum()

    pillar_weights = {}
    for pillar, features in PILLARS.items():
        idxs = [ALL_FEATURES.index(f) for f in features if f in ALL_FEATURES]
        pillar_weights[pillar] = float(mean_abs_shap[idxs].sum() / total)
    return pillar_weights


def pd_to_score(pd_value: float) -> int:
    """
    Maps calibrated PD [0, 1] to display score [300, 900].

    WHY 300-900 RANGE:
      Mirrors the CIBIL credit score range familiar to Indian bankers and
      borrowers. A score of 750+ is widely understood as "good credit" in
      India. This range makes the card immediately legible without explanation.

    WHY LINEAR MAPPING:
      PD → Score is a direct inverse linear transform: lower PD = higher score.
      Non-linear mapping (e.g., log scale) would compress differences at the
      extremes and make score changes feel arbitrary. Linear is more explainable.

    NOTE: In the updated API, scores are mapped from peer_percentile, not PD
    directly, so this function is used only as a fallback.
    """
    score = SCORE_MAX - (pd_value * (SCORE_MAX - SCORE_MIN))
    return int(np.clip(score, SCORE_MIN, SCORE_MAX))


def load_model():
    return joblib.load(MODELS_DIR / "main_model.pkl")


def load_pillar_weights() -> dict:
    return joblib.load(MODELS_DIR / "pillar_weights.pkl")


def load_feature_names() -> list:
    return joblib.load(MODELS_DIR / "feature_names.pkl")

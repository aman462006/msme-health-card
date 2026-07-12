"""
Fuzzy parcelling reject inference.

THE PROBLEM THIS SOLVES (survivorship bias):
  Any bank that has been lending for years only has outcome data (repaid/defaulted)
  for applicants it APPROVED. The rejected applicants were never given loans, so
  we never found out if they would have repaid. If we train only on approved loans,
  we are training the model to replicate the OLD credit officer's decisions — which
  is exactly the bias we want to overcome for new-to-credit MSMEs.

  Example: Old IDBI system rejected all firms without 3 years of ITR history.
  If we train only on approved (ITR-having) firms, our model learns that ITR history
  is essential. A GST-compliant, EPFO-regular firm without ITR will always be rejected
  even if it is perfectly creditworthy. Reject inference fixes this.

THE FUZZY PARCELLING ALGORITHM:
  Step 1: Train initial model on approved-and-labeled population.
  Step 2: Score the historically rejected population using this model.
          Each rejected applicant gets a probability of default (PD).
  Step 3: Duplicate each rejected record TWICE:
          - Once labeled TARGET=0 (good) with sample_weight = (1 - PD)
            "We think there's a (1-PD) chance this person would have repaid"
          - Once labeled TARGET=1 (bad) with sample_weight = PD
            "We think there's a PD chance this person would have defaulted"
  Step 4: Retrain on approved + both copies of rejected (weighted).
          The model now learns from the inferred population, not just approved.
  Step 5: Repeat 2-3 times. Each iteration refines the PD estimates for
          the rejected population based on the improved model.

WHY "FUZZY" (not hard assignment):
  Alternative (hard parcelling): assign each reject as definitely good or bad
  based on a 50% threshold. Problem: this introduces a hard cliff — a PD of
  0.49 becomes "definitely good" and 0.51 becomes "definitely bad", doubling
  the signal. Fuzzy assignment treats every reject as a weighted mixture,
  preserving uncertainty and producing more stable training.

WHY 3 ITERATIONS:
  Empirically, 2-3 iterations converge for most datasets. More iterations risk
  amplifying initial errors (if the first model has a wrong region, it reinforces
  wrong labels in iterations 2 and 3). 3 is the standard from the academic
  literature on reject inference (Feelders 2000, Hand & Henley 1997).
"""
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.calibration import CalibratedClassifierCV

from src.config import ALL_FEATURES, TARGET_COL, MODELS_DIR, RANDOM_STATE
from src.models.main_model import build_lgbm


def run_reject_inference(
    df_approved: pd.DataFrame,
    df_rejected: pd.DataFrame,
    n_iterations: int = 3,
) -> dict:
    """
    df_approved: approved loans with real TARGET labels (0=repaid, 1=defaulted)
    df_rejected: historically rejected applicants — no TARGET column, just features
    Returns: retrained calibrated model with reject inference applied
    """
    print("=== Reject Inference: Fuzzy Parcelling ===")

    X_app = df_approved[ALL_FEATURES]
    y_app = df_approved[TARGET_COL]

    neg, pos = (y_app == 0).sum(), (y_app == 1).sum()
    spw = neg / pos  # class imbalance correction (same as main model)

    # Step 1: initial model on approved-only
    base = build_lgbm()
    base.set_params(scale_pos_weight=spw)
    model = CalibratedClassifierCV(base, method="sigmoid", cv=3)
    model.fit(X_app, y_app)

    X_rej = df_rejected[ALL_FEATURES].copy()

    for iteration in range(1, n_iterations + 1):
        print(f"  Iteration {iteration}/{n_iterations}...")

        # Step 2: score rejects with current model
        pd_rej = model.predict_proba(X_rej)[:, 1]

        # Step 3: duplicate with complementary soft labels
        # Good copy: weight = how confident we are they would have repaid
        rej_good = X_rej.copy()
        rej_good[TARGET_COL] = 0
        rej_good["_weight"] = 1 - pd_rej

        # Bad copy: weight = how confident we are they would have defaulted
        rej_bad = X_rej.copy()
        rej_bad[TARGET_COL] = 1
        rej_bad["_weight"] = pd_rej

        # Approved records get weight=1.0 (we know their true outcome)
        df_app_w = df_approved[ALL_FEATURES + [TARGET_COL]].copy()
        df_app_w["_weight"] = 1.0

        # Step 4: combine and retrain
        df_union = pd.concat([df_app_w, rej_good, rej_bad], ignore_index=True)
        X_union = df_union[ALL_FEATURES]
        y_union = df_union[TARGET_COL]
        w_union = df_union["_weight"]

        base_new = build_lgbm()
        base_new.set_params(scale_pos_weight=spw)
        model = CalibratedClassifierCV(base_new, method="sigmoid", cv=3)
        model.fit(X_union, y_union, sample_weight=w_union)
        # Step 5 loops back to Step 2 automatically

    # Final evaluation
    # AUC on approved-only: ground truth comparison (we know these real outcomes)
    X_train, X_val, y_train, y_val = train_test_split(
        X_app, y_app, test_size=0.2, random_state=RANDOM_STATE, stratify=y_app
    )
    pd_val = model.predict_proba(X_val)[:, 1]
    auc_approved = roc_auc_score(y_val, pd_val)

    # AUC on rejected: soft labels from last iteration as ground truth proxy
    # This AUC is not real — it is circular (we made the soft labels using the model).
    # It is shown only to confirm the model assigns consistent PDs to rejects.
    pd_rej_final = model.predict_proba(X_rej)[:, 1]
    y_rej_soft = pd_rej
    auc_inferred = roc_auc_score((y_rej_soft > 0.5).astype(int), pd_rej_final)

    print(f"  AUC on approved-only (ground truth): {auc_approved:.4f}")
    print(f"  AUC on inferred rejected population: {auc_inferred:.4f}")
    print(f"  Gap (be honest about this in the deck): {abs(auc_approved - auc_inferred):.4f}")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODELS_DIR / "main_model_ri.pkl")
    print(f"Reject-inference model saved -> {MODELS_DIR / 'main_model_ri.pkl'}")

    return {"model": model, "auc_approved": auc_approved, "auc_inferred": auc_inferred}


def build_synthetic_rejected(df_approved: pd.DataFrame, frac: float = 0.30) -> pd.DataFrame:
    """
    WHY THIS EXISTS:
      Real reject inference needs the bank's historical reject file — applicants
      who were denied and whose features were recorded. IDBI has this file but it
      is not in the Kaggle dataset. This function creates a stand-in.

    HOW IT SIMULATES REJECTS:
      Old credit systems rejected disproportionately more high-risk applicants
      (TARGET=1) than low-risk ones, but also rejected many creditworthy MSMEs
      (the bias we are trying to fix). We simulate this by sampling:
        - 60% from high-risk approved (TARGET=1): the true rejects the old system caught
        - 40% from low-risk approved (TARGET=0): the good applicants the old system
          incorrectly rejected due to thin-file or informal economy reasons
      Features only — no TARGET column, as a real reject file would have.

    REPLACE WITH:
      IDBI's actual rejected application file from the last 5-7 years.
      Columns needed: same 30 features (or whatever subset was recorded at the time).
    """
    high_risk = df_approved[df_approved[TARGET_COL] == 1]
    low_risk = df_approved[df_approved[TARGET_COL] == 0]

    n_total = int(len(df_approved) * frac)
    n_high = int(n_total * 0.6)
    n_low = n_total - n_high

    rejected = pd.concat([
        high_risk.sample(min(n_high, len(high_risk)), random_state=RANDOM_STATE),
        low_risk.sample(min(n_low, len(low_risk)), random_state=RANDOM_STATE),
    ], ignore_index=True)

    return rejected[ALL_FEATURES]  # no TARGET — rejects have no known outcome

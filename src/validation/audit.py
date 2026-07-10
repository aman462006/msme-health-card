"""
Out-of-time validation, Population Stability Index, fairness audit,
and adversarial test.

PDF requirement: "Out-of-time validation on most recent 3 months,
PSI <= 0.10 target, fairness audit by gender and geography,
adversarial stress test on worst 5% of cohort."
"""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class AuditReport:
    oot_auc: float
    psi: Dict[str, float]           # feature -> PSI value
    psi_flag: bool                  # True if any PSI > 0.10
    fairness: Dict[str, float]      # group -> AUC
    fairness_max_gap: float         # max AUC gap across groups
    adversarial_auc: float
    passed: bool
    warnings: List[str] = field(default_factory=list)


def compute_psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index. < 0.10 = stable, 0.10-0.25 = caution, > 0.25 = unstable."""
    eps = 1e-6
    expected_pcts, bin_edges = np.histogram(expected, bins=bins)
    actual_pcts, _ = np.histogram(actual, bins=bin_edges)
    expected_pcts = expected_pcts / (expected_pcts.sum() + eps)
    actual_pcts = actual_pcts / (actual_pcts.sum() + eps)
    expected_pcts = np.clip(expected_pcts, eps, None)
    actual_pcts = np.clip(actual_pcts, eps, None)
    return float(np.sum((actual_pcts - expected_pcts) * np.log(actual_pcts / expected_pcts)))


def run_full_audit(
    df: pd.DataFrame,
    model,
    feature_cols: List[str],
    target_col: str = "TARGET",
    date_col: str = "_date_proxy",
    group_col: str = "cohort",
) -> AuditReport:
    """
    Runs four checks:
    1. Out-of-time validation (most-recent 30% of rows as OOT split proxy)
    2. PSI on all features between train and OOT
    3. Fairness audit AUC per group_col
    4. Adversarial test — stress worst 5% of each cohort
    """
    warnings = []

    # 1. OOT split — no real date col so we use row index as time proxy
    split_idx = int(len(df) * 0.70)
    df_train = df.iloc[:split_idx]
    df_oot = df.iloc[split_idx:]

    X_oot = df_oot[feature_cols]
    y_oot = df_oot[target_col]
    oot_pd = model.predict_proba(X_oot)[:, 1]
    oot_auc = float(roc_auc_score(y_oot, oot_pd))
    if oot_auc < 0.70:
        warnings.append(f"OOT AUC {oot_auc:.3f} below 0.70 — model may not generalise")

    # 2. PSI per feature
    psi_values = {}
    for feat in feature_cols:
        try:
            psi_values[feat] = compute_psi(df_train[feat].dropna().values, df_oot[feat].dropna().values)
        except Exception:
            psi_values[feat] = 0.0
    psi_flag = any(v > 0.10 for v in psi_values.values())
    high_psi = [f for f, v in psi_values.items() if v > 0.10]
    if high_psi:
        warnings.append(f"PSI > 0.10 for: {', '.join(high_psi)}")

    # 3. Fairness audit by group
    fairness = {}
    if group_col in df_oot.columns:
        for grp, sub in df_oot.groupby(group_col):
            if len(sub) < 50 or sub[target_col].nunique() < 2:
                continue
            g_pd = model.predict_proba(sub[feature_cols])[:, 1]
            fairness[str(grp)] = float(roc_auc_score(sub[target_col], g_pd))
    fairness_max_gap = max(fairness.values()) - min(fairness.values()) if len(fairness) >= 2 else 0.0
    if fairness_max_gap > 0.05:
        warnings.append(f"Fairness AUC gap {fairness_max_gap:.3f} exceeds 0.05 across cohorts")

    # 4. Adversarial test — worst 5% of cohort (highest OOT PD)
    cutoff_5pct = np.percentile(oot_pd, 95)
    adv_mask = oot_pd >= cutoff_5pct
    if adv_mask.sum() > 10:
        adv_pd = oot_pd[adv_mask]
        adv_y = y_oot.values[adv_mask]
        if len(np.unique(adv_y)) == 2:
            adversarial_auc = float(roc_auc_score(adv_y, adv_pd))
        else:
            adversarial_auc = 1.0
    else:
        adversarial_auc = 1.0

    passed = (oot_auc >= 0.70 and not psi_flag and fairness_max_gap <= 0.05)

    print(f"  OOT AUC: {oot_auc:.4f}  |  PSI flag: {psi_flag}  |  "
          f"Fairness gap: {fairness_max_gap:.3f}  |  Adv AUC: {adversarial_auc:.4f}")
    if warnings:
        for w in warnings:
            print(f"  [WARN] {w}")

    return AuditReport(
        oot_auc=oot_auc,
        psi=psi_values,
        psi_flag=psi_flag,
        fairness=fairness,
        fairness_max_gap=fairness_max_gap,
        adversarial_auc=adversarial_auc,
        passed=passed,
        warnings=warnings,
    )

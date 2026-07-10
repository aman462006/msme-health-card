"""
Full training pipeline — run this once after downloading datasets.
Produces: main_model_ri.pkl, thin_file_model.pkl, pillar_weights.pkl,
          cohort_data.pkl, cohort_pillar_weights.pkl, pillar_weight_ci.pkl
"""
import sys
sys.path.insert(0, ".")

from src.data.feature_engineering import build_and_save
from src.models.main_model import train
from src.models.thin_file import train_thin_file
from src.models.reject_inference import run_reject_inference, build_synthetic_rejected
from src.data.cohort_builder import assign_cohort_from_training, build_cohort_distributions
from src.models.cohort_weights import compute_cohort_pillar_weights
from src.validation.audit import run_full_audit
from src.config import ALL_FEATURES, TARGET_COL, MODELS_DIR


def main():
    print("=" * 60)
    print("MSME Financial Health Card -- Training Pipeline")
    print("=" * 60)

    print("\n[1/7] Building features from Home Credit + synthetic MSME overlay...")
    df = build_and_save()

    print("\n[2/7] Training main Monotonic LightGBM with isotonic calibration...")
    result = train(df)
    main_model = result["model"]
    print(f"      Main model AUC: {result['auc']:.4f}")

    print("\n[3/7] Running reject inference (fuzzy parcelling, 3 iterations)...")
    df_rejected = build_synthetic_rejected(df, frac=0.30)
    ri_result = run_reject_inference(df, df_rejected, n_iterations=3)
    ri_model = ri_result["model"]
    print(f"      AUC on approved: {ri_result['auc_approved']:.4f}")
    print(f"      AUC on inferred: {ri_result['auc_inferred']:.4f}")

    print("\n[4/7] Building cohort PD distributions (peer-relative scoring)...")
    df_with_cohort = assign_cohort_from_training(df)
    cohort_data = build_cohort_distributions(df_with_cohort, ri_model)
    print(f"      Built {len(cohort_data)-1} cohort distributions")

    print("\n[5/7] Computing per-cohort pillar weights via SHAP + bootstrap CI...")
    cohort_weights, weight_ci = compute_cohort_pillar_weights(
        df_with_cohort, ri_model, n_bootstrap=1000
    )
    print(f"      Pillar weights computed for {len(cohort_weights)-1} cohorts + global")

    print("\n[6/7] Running validation audit (OOT, PSI, fairness, adversarial)...")
    audit = run_full_audit(
        df_with_cohort, ri_model,
        feature_cols=ALL_FEATURES,
        target_col=TARGET_COL,
        group_col="cohort",
    )
    print(f"      OOT AUC: {audit.oot_auc:.4f}  |  PSI flag: {audit.psi_flag}"
          f"  |  Fairness gap: {audit.fairness_max_gap:.3f}")
    if audit.warnings:
        for w in audit.warnings:
            print(f"      [WARN] {w}")
    print(f"      Audit {'PASSED' if audit.passed else 'FAILED (see warnings)'}")

    print("\n[7/7] Training thin-file model with MAPIE conformal intervals...")
    thin_result = train_thin_file(df)
    print(f"      Thin-file AUC: {thin_result['auc']:.4f}")
    print(f"      Empirical coverage: {thin_result['coverage']:.3f}")

    print("\n" + "=" * 60)
    print("Training complete. Models saved to ./models/")
    print("  main_model_ri.pkl       -- monotonic LightGBM + isotonic calibration + RI")
    print("  thin_file_model.pkl     -- MAPIE conformal classifier")
    print("  pillar_weights.pkl      -- global SHAP pillar weights")
    print("  cohort_data.pkl         -- PD distributions per cohort")
    print("  cohort_pillar_weights.pkl  -- per-cohort SHAP pillar weights")
    print("  pillar_weight_ci.pkl    -- bootstrap 95% CI on global pillar weights")
    print("\nRun: uvicorn src.api.main:app --reload --port 8000")
    print("=" * 60)


if __name__ == "__main__":
    main()

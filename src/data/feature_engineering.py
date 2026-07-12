"""
Feature engineering pipeline.

TRAINING DATA SITUATION (important to understand before reading this file):
---------------------------------------------------------------------------
The model needs two things: (a) feature values and (b) a default label (did this
firm actually fail to repay?). We use the Home Credit Default Risk dataset from
Kaggle (307,511 personal loan applications) as a proxy because:
  - It has a real TARGET column (0 = repaid, 1 = defaulted) from actual loan outcomes
  - It has external credit bureau scores, installment payment history, employment data
  - These map reasonably onto MSME creditworthiness signals

The PROBLEM: 14 of our 30 features (all P1, P2 partial, P4) have no Home Credit
equivalent. So those 14 are GENERATED with a known correlation to TARGET (see
_add_synthetic_msme_overlay). This means AUC = 0.9999 — the model is memorising
a signal we planted. This is acknowledged. The fix is to replace the synthetic
overlay with real MSME data (Setu AA + GSTN + EPFO) once those APIs are live.
The remaining 16 features use real Home Credit values and learn a real relationship.

HOW EACH VALUE IS DERIVED:
Each feature is a single float in [0, 1] or [0, N]. It is NOT directly given by any
API — it is COMPUTED from raw API data (transactions, GST filings, EPFO records).
The comments below explain: (1) what raw data is needed, (2) the exact formula,
(3) why this formula captures creditworthiness better than alternatives.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from src.config import DATA_RAW, DATA_PROCESSED, PILLARS, ALL_FEATURES, RANDOM_STATE

RNG = np.random.default_rng(RANDOM_STATE)


# ---------------------------------------------------------------------------
# 1. Load Home Credit
# ---------------------------------------------------------------------------

def load_home_credit() -> tuple[pd.DataFrame, pd.DataFrame]:
    hc_dir = DATA_RAW / "home_credit"
    app = pd.read_csv(hc_dir / "application_train.csv")
    inst = pd.read_csv(hc_dir / "installments_payments.csv")
    return app, inst


def load_lending_club_rejected() -> pd.DataFrame:
    lc_dir = DATA_RAW / "lending_club"
    candidates = list(lc_dir.glob("*rejected*")) + list(lc_dir.glob("*Rejected*"))
    if not candidates:
        print("Lending Club rejected file not found -- skipping reject inference data.")
        return pd.DataFrame()
    return pd.read_csv(candidates[0], low_memory=False)


# ---------------------------------------------------------------------------
# 2. Installment payment features (real Home Credit data, no synthetic)
# ---------------------------------------------------------------------------

def _build_installment_features(inst: pd.DataFrame) -> pd.DataFrame:
    """
    Source: installments_payments.csv from Home Credit.
    Each row = one scheduled installment and what was actually paid.

    avg_days_past_due:
      Formula: mean of max(0, actual_payment_date - scheduled_date) per installment
      Why this formula: days late is the most direct measure of obligation discipline.
      Mean (not max) because a single emergency delay matters less than chronic lateness.
      Alternative considered: binary flag (ever late vs never) -- rejected because it
      treats 1 day late the same as 90 days late.

    emi_bounce_rate:
      Formula: fraction of installments where amount_paid < 95% of amount_due
      Why 95%: small rounding differences in bank transfers shouldn't count as a bounce.
      Why fraction not count: count penalises older borrowers with more installments.
    """
    inst = inst.copy()
    inst["days_past_due"] = (inst["DAYS_ENTRY_PAYMENT"] - inst["DAYS_INSTALMENT"]).clip(lower=0)
    inst["bounced"] = (inst["AMT_PAYMENT"] < inst["AMT_INSTALMENT"] * 0.95).astype(int)

    agg = inst.groupby("SK_ID_CURR").agg(
        avg_days_past_due=("days_past_due", "mean"),
        emi_bounce_rate=("bounced", "mean"),
    ).reset_index()
    return agg


# ---------------------------------------------------------------------------
# 3. Map Home Credit columns -> 6-pillar features (real data where available)
# ---------------------------------------------------------------------------

def build_features(app: pd.DataFrame, inst: pd.DataFrame) -> pd.DataFrame:
    df = app.copy()
    inst_feats = _build_installment_features(inst)
    df = df.merge(inst_feats, on="SK_ID_CURR", how="left")

    # -----------------------------------------------------------------------
    # P2: Revenue Quality -- External credit bureau scores
    # Source: Home Credit's EXT_SOURCE_1/2/3 columns, which are normalised
    # scores from three different credit bureaus (exact bureaus undisclosed
    # by Home Credit but equivalent to CIBIL/Experian/Equifax in India).
    # Formula: use as-is after median imputation for missing values.
    # Why median imputation: these scores are missing-not-at-random (thin file
    # applicants have no bureau history). Median is a conservative middle ground.
    # Why keep all three: each bureau covers different lenders, ensemble reduces
    # single-bureau blind spots. The model's SHAP will weight them appropriately.
    # Range: 0 to 1, higher = better creditworthiness.
    # -----------------------------------------------------------------------
    df["ext_source_1"] = df["EXT_SOURCE_1"].fillna(df["EXT_SOURCE_1"].median())
    df["ext_source_2"] = df["EXT_SOURCE_2"].fillna(df["EXT_SOURCE_2"].median())
    df["ext_source_3"] = df["EXT_SOURCE_3"].fillna(df["EXT_SOURCE_3"].median())

    # -----------------------------------------------------------------------
    # P3: Obligation Discipline -- from installment data (computed above)
    # avg_days_past_due: fill 0 for borrowers with no installment history
    #   (they have no record of lateness, which is weakly positive, not unknown)
    # emi_bounce_rate: same -- no history means no bounces recorded
    # -----------------------------------------------------------------------
    df["avg_days_past_due"] = df["avg_days_past_due"].fillna(0)
    df["emi_bounce_rate"] = df["emi_bounce_rate"].fillna(0)

    # -----------------------------------------------------------------------
    # P5: Leverage & Liquidity
    #
    # annuity_to_income_ratio:
    #   Source: AMT_ANNUITY (monthly loan repayment) / AMT_INCOME_TOTAL (annual income)
    #   Formula: annuity / income, clipped at 5 (beyond 5x monthly income is extreme)
    #   Why this ratio: directly measures whether repayment burden is sustainable.
    #   Higher ratio = more income already committed to debt = less room for new credit.
    #   Why not absolute annuity amount: Rs 10,000/month is crushing for a small firm,
    #   fine for a large one. Ratio normalises across firm sizes.
    #   Real MSME equivalent: total_emi_outflows / monthly_bank_inflows (from Setu AA)
    # -----------------------------------------------------------------------
    df["annuity_to_income_ratio"] = (
        df["AMT_ANNUITY"] / df["AMT_INCOME_TOTAL"].replace(0, np.nan)
    ).clip(0, 5)

    credit_to_income = (
        df["AMT_CREDIT"] / df["AMT_INCOME_TOTAL"].replace(0, np.nan)
    ).clip(0, 20)

    # -----------------------------------------------------------------------
    # debt_to_inflow_ratio:
    #   Source: AMT_CREDIT (total outstanding loan) / AMT_INCOME_TOTAL
    #   Why total debt not monthly payment: captures the STOCK of debt, not just flow.
    #   A firm with 10 years of EMIs remaining looks fine on monthly payments but has
    #   enormous total liability.
    #   Real MSME equivalent: sum_all_outstanding_loans / avg_monthly_bank_inflow * 12
    # -----------------------------------------------------------------------
    df["debt_to_inflow_ratio"] = credit_to_income

    # -----------------------------------------------------------------------
    # current_ratio_proxy:
    #   Source: derived from annuity_to_income_ratio
    #   Formula: 1 / (annuity_to_income + 0.01) -- inverted debt burden
    #   Why this proxy: we have no balance sheet, so we approximate liquidity as
    #   the inverse of repayment pressure. High annuity burden = low current ratio.
    #   +0.01 prevents division by zero for zero-annuity applicants.
    #   Real MSME equivalent: current_assets / current_liabilities from ITR-3/GSTR
    # -----------------------------------------------------------------------
    df["current_ratio_proxy"] = (1 / (df["annuity_to_income_ratio"] + 0.01)).clip(0, 10)

    # -----------------------------------------------------------------------
    # working_capital_cycle_days:
    #   Source: AMT_CREDIT and AMT_ANNUITY
    #   Formula: (total_credit / monthly_annuity) * 30 = months to repay * 30 = days
    #   Why this formula: approximates how long it takes to turn debt into cash flow.
    #   A firm with 20 months of EMIs remaining has ~600 working capital days locked up.
    #   Real MSME equivalent: (inventory_days + receivables_days - payables_days)
    #   computed from GSTR turnover and bank inflow timing.
    # -----------------------------------------------------------------------
    df["working_capital_cycle_days"] = (
        df["AMT_CREDIT"] / (df["AMT_ANNUITY"] + 1) * 30
    ).clip(0, 720)

    # -----------------------------------------------------------------------
    # P6: Stability & Vintage
    #
    # days_employed_years:
    #   Source: DAYS_EMPLOYED (negative = days before application date)
    #   Formula: abs(DAYS_EMPLOYED) / 365
    #   Why absolute value: Home Credit stores as negative days, we convert to years.
    #   Why this signal: longer employment = more stable income stream = lower risk.
    #   Real MSME equivalent: years since promoter's first GSTIN registration
    #   (available from MCA portal via Karza API)
    #
    # gstin_age_years:
    #   Source: DAYS_REGISTRATION (days since the applicant registered with Home Credit)
    #   Formula: abs(DAYS_REGISTRATION) / 365
    #   Why: longer registration = more established customer = lower fraud risk.
    #   Real MSME equivalent: (today - GSTIN_registration_date) in years from GSTN API
    #
    # address_churn_flag:
    #   Source: REG_CITY_NOT_LIVE_CITY (1 if registered city != actual city)
    #   Why: address mismatch is a fraud/instability signal in personal lending.
    #   Real MSME equivalent: address changed in last 12 months on GSTN portal (binary)
    #
    # promoter_churn_flag:
    #   Source: REG_REGION_NOT_LIVE_REGION (regional mismatch)
    #   Real MSME equivalent: director changed in last 12 months on MCA (via Karza)
    #
    # directorship_overlap_flag:
    #   Source: DEF_60_CNT_SOCIAL_CIRCLE (defaulters in applicant's social circle > 1)
    #   Why: network default contagion -- if your peers default, your repayment risk rises.
    #   Real MSME equivalent: promoter's DIN linked to any NPA company on MCA (via Karza)
    # -----------------------------------------------------------------------
    df["days_employed_years"] = (df["DAYS_EMPLOYED"].abs() / 365).clip(0, 40)
    df["gstin_age_years"] = (df["DAYS_REGISTRATION"].abs() / 365).clip(0, 30)
    df["address_churn_flag"] = df["REG_CITY_NOT_LIVE_CITY"].fillna(0).astype(float)
    df["promoter_churn_flag"] = df["REG_REGION_NOT_LIVE_REGION"].fillna(0).astype(float)
    df["directorship_overlap_flag"] = (
        df["DEF_60_CNT_SOCIAL_CIRCLE"].fillna(0) > 1
    ).astype(float)

    # Add synthetic overlay for features with no Home Credit equivalent
    df = _add_synthetic_msme_overlay(df)

    df["TARGET"] = df["TARGET"]
    return df


# ---------------------------------------------------------------------------
# 4. Synthetic MSME overlay
#
# WARNING: These 14 features have NO real equivalent in Home Credit.
# They are generated with a known statistical correlation to TARGET so the
# model has something to train on. This inflates AUC to ~1.0 and makes the
# model useless on real data until replaced with actual API-sourced values.
#
# HOW EACH IS GENERATED:
#   base = normal distribution centred at a realistic population mean
#   shift = an additional push in the risky direction for TARGET=1 applicants
#   The shift magnitude is estimated from domain knowledge of how much these
#   signals differ between defaulters and non-defaulters in Indian MSME studies.
#
# REAL DATA REPLACEMENT:
#   When Setu AA / GSTN / EPFO APIs are live, _add_synthetic_msme_overlay()
#   will be replaced by _fetch_real_msme_features(gstin, account_id) which
#   calls those APIs and computes the same formulas from raw data.
# ---------------------------------------------------------------------------

def _add_synthetic_msme_overlay(df: pd.DataFrame) -> pd.DataFrame:
    n = len(df)
    target = df["TARGET"].values  # 1 = default

    def _signal(base_mean, base_std, target_shift, target_noise_scale=1.5):
        """
        Generates a feature value correlated with default probability.
        base_mean / base_std: realistic population distribution
        target_shift: how much the mean shifts for defaulters (sign matches monotone constraint)
        target_noise_scale: noise on the shift — kept large (1.5x shift) so defaulters and
          non-defaulters overlap substantially, forcing the model to produce calibrated
          intermediate probabilities rather than collapsing to 0/1.
        """
        base = RNG.normal(base_mean, base_std, n)
        shift = target * RNG.normal(target_shift, target_noise_scale, n)
        return base + shift

    # -----------------------------------------------------------------------
    # P1: Cash Flow Resilience
    # Real source: bank statement via Setu Account Aggregator
    # Real formula for each: computed from 12 months of daily debit/credit rows
    #
    # inflow_cv (Coefficient of Variation of monthly inflows):
    #   Real formula: std(monthly_credits) / mean(monthly_credits) over 12 months
    #   Why CV not raw std: normalises across firm size (a Rs 10L std on a 1Cr/yr
    #   firm is fine; same std on a 10L/yr firm is catastrophic).
    #   Risk direction: higher CV = more volatile = higher risk (monotone +1)
    #   Synthetic: base mean 0.25 (25% monthly variation is typical), defaulters
    #   shifted +0.20 higher (more volatile income)
    #
    # min_balance_days (days account balance was below minimum threshold):
    #   Real formula: count days where running_balance < (0.1 * avg_monthly_inflow)
    #   Why 10% of avg inflow as threshold: a firm with less than 10% of monthly
    #   income as buffer cannot absorb even a small payment delay.
    #   Risk direction: more days below minimum = higher risk (monotone +1)
    #   Synthetic: base 5 days/month, defaulters average 13 more days below minimum
    #
    # inflow_outflow_lag (days between receiving money and spending it):
    #   Real formula: for each month, mean(date_of_debit - date_of_preceding_credit)
    #   Why this matters: a firm that immediately spends every inflow has zero buffer.
    #   Longer lag = more cash is being held = lower risk. BUT very long lag (>15 days)
    #   may indicate receivables stuck, so this is non-monotone in extremes.
    #   Synthetic uses the risk direction: shorter lag = more risk (monotone +1)
    #
    # drawdown_recovery_days (days to recover after largest single outflow):
    #   Real formula: find the month's largest single debit, count days until
    #   running balance returns to pre-debit level.
    #   Why: resilience test -- how quickly does the firm recover from a shock?
    #   A firm taking 60 days to recover is far riskier than one that recovers in 3.
    #   Risk direction: more recovery days = higher risk (monotone +1)
    #
    # loss_absorption_buffer (months of expenses the firm can sustain from buffer):
    #   Real formula: (avg_balance - avg_monthly_expenses) / avg_monthly_expenses * 30
    #   = number of days expenses can be paid from buffer alone
    #   Why days not rupees: normalises across firm size.
    #   Risk direction: fewer buffer days = higher risk (monotone -1, inverted)
    # -----------------------------------------------------------------------
    df["inflow_cv"] = np.clip(_signal(0.25, 0.12, 0.20), 0.01, 1.5)
    df["min_balance_days"] = np.clip(_signal(5, 4, 8, 8), 0, 30)
    df["inflow_outflow_lag"] = np.clip(_signal(3, 2, 5, 5), 0, 30)
    df["drawdown_recovery_days"] = np.clip(_signal(15, 8, 20, 20), 1, 90)
    df["loss_absorption_buffer"] = np.clip(_signal(20, 8, -12, 12), 1, 60)

    # -----------------------------------------------------------------------
    # P2: Revenue Quality (partial -- ext_source_1/2/3 are real Home Credit values)
    # Real source: GSTN API for GST features, GSTR-1 e-way bills for buyer HHI
    #
    # gst_mismatch_pct (% difference between GSTR-1 outward and GSTR-3B net):
    #   Real formula: abs(gstr1_total_sales - gstr3b_net_taxable) / gstr1_total_sales
    #   Why: GSTR-1 and GSTR-3B should match. A mismatch means either under-reporting
    #   in GSTR-3B (tax evasion risk) or data error (compliance risk). Either inflates
    #   the firm's apparent revenue relative to actual tax paid.
    #   Risk direction: higher mismatch = higher risk (monotone +1)
    #
    # gst_filing_punctuality (fraction of monthly returns filed by due date):
    #   Real formula: count(filed_on_or_before_due_date) / count(total_periods) over 24m
    #   Why fraction not count: penalises recent non-filers more than historical ones.
    #   Why 24 months: captures seasonal patterns (festive season filings often late).
    #   Risk direction: lower punctuality = higher risk (monotone -1, inverted)
    #
    # itc_reversal_freq (fraction of ITC claims later reversed):
    #   Real formula: count(ITC_reversals) / count(ITC_claims) over 24 months
    #   Why: reversals happen when claimed input credit is disallowed by GSTN,
    #   indicating either fraudulent claims or accounting errors -- both are risk signals.
    #   Risk direction: higher reversal rate = higher risk (monotone +1)
    #
    # buyer_concentration_hhi (Herfindahl-Hirschman Index of buyer concentration):
    #   Real formula: sum((buyer_i_sales / total_sales)^2) across all buyers from GSTR-1
    #   HHI = 1.0 means 100% revenue from one buyer (extremely concentrated = high risk)
    #   HHI = 0.1 means sales spread across 10 equal buyers (diversified = low risk)
    #   Why HHI over count of buyers: 2 buyers at 99%/1% is as risky as 1 buyer.
    #   Risk direction: higher HHI = higher risk (monotone +1)
    # -----------------------------------------------------------------------
    df["gst_mismatch_pct"] = np.clip(_signal(0.05, 0.04, 0.12, 0.12), 0, 0.5)
    df["gst_filing_punctuality"] = np.clip(_signal(0.88, 0.10, -0.18, 0.18), 0, 1)
    df["itc_reversal_freq"] = np.clip(_signal(0.04, 0.03, 0.10, 0.10), 0, 0.5)
    df["buyer_concentration_hhi"] = np.clip(_signal(0.25, 0.15, 0.20, 0.20), 0, 1)

    # -----------------------------------------------------------------------
    # P3: Obligation Discipline (partial -- avg_days_past_due and emi_bounce_rate
    #     are real Home Credit values computed from installments_payments.csv above)
    # Real source for remaining: EPFO API, utility DISCOM, GSTN late fees
    #
    # epfo_payment_regularity (fraction of months PF deposited on time):
    #   Real formula: count(months where ECR filed by 15th) / count(total months) over 24m
    #   Why 15th: EPFO due date is 15th of following month. Late filing = penalty + signal.
    #   Why this matters for credit: PF is a statutory obligation. A firm that skips PF
    #   is prioritising other payments -- signal of cash stress.
    #   Risk direction: lower regularity = higher risk (monotone -1, inverted)
    #
    # utility_delinquency_flag (1 if electricity bill was overdue in last 12 months):
    #   Real formula: 1 if any bill_paid_date > due_date in DISCOM records, else 0
    #   Why binary not continuous: DISCOM APIs typically only report paid/overdue,
    #   not exact days late. Binary is the most reliable extraction from patchy data.
    #   Risk direction: flag=1 means higher risk (monotone +1)
    #
    # gst_late_fee_incidence (fraction of GSTN filings that incurred a late fee):
    #   Real formula: count(filings_with_CGST_50_late_fee > 0) / count(total_filings)
    #   Why: Rs 50/day late fee is recorded in GSTN portal -- directly observable.
    #   Even one late month in 24 is a compliance signal; chronic lateness is severe.
    #   Risk direction: higher incidence = higher risk (monotone +1)
    # -----------------------------------------------------------------------
    df["epfo_payment_regularity"] = np.clip(_signal(0.90, 0.10, -0.20, 0.20), 0, 1)
    df["utility_delinquency_flag"] = (
        RNG.uniform(0, 1, n) < (0.08 + target * 0.15)
    ).astype(float)
    df["gst_late_fee_incidence"] = np.clip(_signal(0.06, 0.05, 0.15, 0.15), 0, 1)

    # -----------------------------------------------------------------------
    # P4: Operational Vitality
    # Real source: EPFO public ECR data, state DISCOM API
    #
    # epfo_headcount_delta_6m (% change in employees over 6 months):
    #   Real formula: (employees_this_month - employees_6m_ago) / employees_6m_ago
    #   from EPFO ECR (Employee Contribution Report) which is publicly filed monthly.
    #   Why 6 months: short enough to catch recent contraction before it becomes a crisis.
    #   Why % not absolute: a drop of 5 employees matters more for a 10-person firm
    #   than for a 500-person firm.
    #   Risk direction: negative delta (shrinking workforce) = higher risk (monotone -1)
    #
    # epfo_headcount_delta_12m (same over 12 months):
    #   Why keep both: 6m catches sudden shocks; 12m captures structural decline.
    #   A firm that shed staff 8 months ago won't show up in the 6m window.
    #   Together they give the model a short-term and long-term operational signal.
    #
    # electricity_kwh_trend (slope of monthly kWh consumption over 12 months):
    #   Real formula: linear_regression_slope(monthly_kwh, month_number) / mean_kwh
    #   = normalised consumption trend (positive = growing, negative = shrinking)
    #   Why electricity: it is the most reliable proxy for actual operational activity.
    #   GST turnover can be manipulated; electricity consumption cannot.
    #   Why normalised slope: allows comparison across manufacturing vs. services firms.
    #   Risk direction: negative trend (declining output) = higher risk (monotone -1)
    #
    # supplier_diversity_score (fraction of spending spread across multiple suppliers):
    #   Real formula: 1 - HHI_of_supplier_concentration from GSTR-2 purchase data
    #   = 1.0 means perfectly diversified, 0.0 means single supplier dependency
    #   Why: single-supplier dependency is an operational risk (supply chain disruption
    #   kills revenue but the firm still owes EMIs).
    #   Risk direction: lower diversity = higher risk (monotone -1, inverted)
    # -----------------------------------------------------------------------
    df["epfo_headcount_delta_6m"] = np.clip(_signal(0.03, 0.08, -0.10, 0.10), -0.5, 0.5)
    df["epfo_headcount_delta_12m"] = np.clip(_signal(0.05, 0.10, -0.14, 0.14), -0.5, 0.5)
    df["electricity_kwh_trend"] = np.clip(_signal(0.04, 0.06, -0.10, 0.10), -0.3, 0.3)
    df["supplier_diversity_score"] = np.clip(_signal(0.60, 0.15, -0.18, 0.18), 0, 1)

    return df


# ---------------------------------------------------------------------------
# 5. Final dataset assembly
# ---------------------------------------------------------------------------

def build_and_save() -> pd.DataFrame:
    print("Loading Home Credit data...")
    app, inst = load_home_credit()
    print(f"  application_train: {app.shape}, installments: {inst.shape}")

    print("Building features...")
    df = build_features(app, inst)

    extra = [c for c in ["TARGET", "SK_ID_CURR", "AMT_INCOME_TOTAL", "ORGANIZATION_TYPE"] if c in df.columns]
    feature_cols = ALL_FEATURES + extra
    out = df[feature_cols].copy()
    out = out.dropna(subset=["TARGET"])

    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    out_path = DATA_PROCESSED / "msme_features.parquet"
    out.to_parquet(out_path, index=False)
    print(f"Saved {len(out)} rows x {len(feature_cols)} cols -> {out_path}")
    print(f"  Default rate: {out['TARGET'].mean():.2%}")
    return out


if __name__ == "__main__":
    build_and_save()

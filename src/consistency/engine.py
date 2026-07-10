"""
Consistency Engine — cross-source triangulation.

WHY THIS EXISTS SEPARATELY FROM THE SCORE:
  The ML model scores based on individual feature values. It cannot detect
  when two features contradict each other across different data sources.
  Example: a firm reports Rs 5 crore GST turnover but uses only 200 kWh of
  electricity per month. Those two numbers are individually valid inputs to
  the model, but together they are physically impossible for a manufacturing firm
  (you cannot manufacture Rs 5 crore of goods with a single home-grade meter).

  A merchant can manipulate one source. They cannot simultaneously fake
  electricity bills (metered by the DISCOM), EPFO records (filed with the
  government), GST returns (matched against counterparty filings), and
  bank statements (from the AA framework). Cross-source triangulation
  catches sophisticated fraud that single-source scoring misses.

WHY BESIDE THE SCORE (not inside it):
  If we folded consistency flags into the score, a firm with clean data but one
  data gap would be penalised even when that gap is explainable. Instead:
  - Flags widen the conformal interval (more uncertainty, not automatic decline)
  - Flags trigger manual review (human judgment for edge cases)
  - The score itself remains model-driven (SHAP-auditable, monotone-constrained)
  This preserves the score's RBI-auditability while still catching fraud.

HOW INTERVAL WIDENING WORKS:
  Each failed consistency check adds 0.05 to pd_upper_bound.
  Maximum widening is 0.20 (4 checks all failing).
  This means a firm that fails all 4 checks has its "worst case PD" raised by 20%.
  If their point-estimate PD was 0.08 (eligible), their upper bound becomes 0.28
  (review territory), triggering human inspection.
"""
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ConsistencyFlag:
    check_name: str
    severity: str           # "red" | "amber"
    description: str
    observed: float
    expected_range: tuple[float, float]
    recommendation: str


@dataclass
class ConsistencyResult:
    flags: list[ConsistencyFlag] = field(default_factory=list)
    interval_widening: float = 0.0
    manual_review: bool = False

    @property
    def red_count(self) -> int:
        return sum(1 for f in self.flags if f.severity == "red")

    @property
    def amber_count(self) -> int:
        return sum(1 for f in self.flags if f.severity == "amber")

    def summary(self) -> str:
        return f"{self.red_count} red / {self.amber_count} amber"


def run_consistency_engine(msme_data: dict) -> ConsistencyResult:
    """
    Runs all 4 triangulation checks. Each check is independent — failure in
    one does not prevent others from running. Missing data skips that check
    (we do not penalise for not providing optional data sources).
    """
    result = ConsistencyResult()

    _check_electricity_vs_turnover(msme_data, result)
    _check_epfo_vs_salary(msme_data, result)
    _check_gstr1_vs_bank_inflows(msme_data, result)
    _check_ewaybill_vs_turnover(msme_data, result)

    # Each failed check adds 0.05 to PD upper bound, capped at 0.20
    result.interval_widening = min(0.05 * len(result.flags), 0.20)

    # Two or more red flags require human review — model alone is insufficient
    if result.red_count >= 2:
        result.manual_review = True

    return result


def _check_electricity_vs_turnover(data: dict, result: ConsistencyResult):
    """
    CHECK: Is electricity consumption consistent with reported GST turnover?

    HOW:
      kwh_per_lakh = monthly_kwh / (annual_turnover / 12 / 1_lakh)
      = how many kWh does this firm consume per Rs 1 lakh of monthly revenue

    WHY INDUSTRY-SPECIFIC BANDS:
      Manufacturing consumes far more electricity per rupee of output than
      services (machines, factories). Trading is in between (warehouses, lighting).
      Using a single band would flag every services firm as suspicious and
      clear every manufacturer.

    BANDS (derived from MSME sector energy intensity studies, BEE 2022):
      Manufacturing: 0.5–8.0 kWh per Rs 1L (motors, furnaces, compressors)
      Services:      0.05–1.5 kWh per Rs 1L (computers, lighting, AC)
      Trading:       0.02–1.0 kWh per Rs 1L (warehouse lighting and refrigeration)

    SEVERITY:
      Red: ratio is 3x outside the band (hard to explain legitimately)
      Amber: ratio is outside band but within 3x (could be seasonal or shared meter)

    WHAT THIS CATCHES:
      Low electricity with high turnover: inflated GST returns (paper transactions)
      High electricity with low turnover: energy theft or off-book production
    """
    kwh = data.get("electricity_kwh_monthly")
    turnover = data.get("gst_annual_turnover_lakhs")
    industry = data.get("industry_type", "manufacturing")

    if kwh is None or turnover is None:
        return

    monthly_turnover = turnover / 12
    kwh_per_lakh = kwh / (monthly_turnover + 1)

    bands = {
        "manufacturing": (0.5, 8.0),
        "services": (0.05, 1.5),
        "trading": (0.02, 1.0),
    }
    lo, hi = bands.get(industry, (0.02, 8.0))

    if not (lo <= kwh_per_lakh <= hi):
        severity = "red" if (kwh_per_lakh < lo * 0.3 or kwh_per_lakh > hi * 3) else "amber"
        result.flags.append(ConsistencyFlag(
            check_name="Electricity vs GST Turnover",
            severity=severity,
            description=(
                f"Electricity intensity ({kwh_per_lakh:.2f} kWh/Rs L) is "
                f"{'unusually low' if kwh_per_lakh < lo else 'unusually high'} "
                f"for {industry}. Expected: {lo}-{hi}."
            ),
            observed=kwh_per_lakh,
            expected_range=(lo, hi),
            recommendation=(
                "Verify electricity bills and GST turnover independently. "
                "Low electricity with high turnover may indicate inflated GST figures."
            ),
        ))


def _check_epfo_vs_salary(data: dict, result: ConsistencyResult):
    """
    CHECK: Is the salary outflow consistent with the declared employee count?

    HOW:
      implied_salary_per_employee = (monthly_salary_outflow * 100000) / epfo_headcount

    WHY Rs 8,000 AS LOWER BOUND:
      Minimum wage in most Indian states is Rs 6,000-10,000/month (2024).
      A firm paying less than Rs 8,000 per EPFO-registered employee either has
      ghost employees (inflated headcount) or is violating minimum wage law.

    WHY Rs 2,00,000 AS UPPER BOUND:
      EPFO is mandatory only for employees earning up to Rs 15,000 basic salary,
      but voluntary for higher earners. A firm with average EPFO salary of
      Rs 2L+/month is likely a high-skilled services firm (IT, consulting).
      Beyond Rs 5L average implies either very few senior staff are EPFO-registered
      or the salary outflow figure includes non-salary payments.

    WHAT THIS CATCHES:
      Too low per head: ghost employees on EPFO to look bigger than reality
      Too high per head: salary outflow includes contractor payments or dividends
      being misclassified as wages
    """
    headcount = data.get("epfo_headcount")
    salary_outflow = data.get("salary_outflow_monthly_lakhs")

    if headcount is None or salary_outflow is None or headcount == 0:
        return

    salary_per_head = (salary_outflow * 100000) / headcount

    lo, hi = 8000, 200000
    if not (lo <= salary_per_head <= hi):
        severity = "red" if salary_per_head < 3000 or salary_per_head > 500000 else "amber"
        result.flags.append(ConsistencyFlag(
            check_name="EPFO Headcount vs Salary Outflow",
            severity=severity,
            description=(
                f"Implied salary/employee: Rs {salary_per_head:,.0f}/month. "
                f"Expected: Rs {lo:,}-Rs {hi:,}."
            ),
            observed=salary_per_head,
            expected_range=(lo, hi),
            recommendation=(
                "Cross-check employee count with EPFO portal and salary outflow "
                "from bank statements. Mismatch may indicate ghost employees or "
                "unregistered workforce."
            ),
        ))


def _check_gstr1_vs_bank_inflows(data: dict, result: ConsistencyResult):
    """
    CHECK: Are bank + UPI inflows consistent with GSTR-1 declared sales?

    HOW:
      collection_ratio = (upi_inflows + bank_inflows) / gstr1_sales
      for the same monthly period

    WHY RATIO NOT ABSOLUTE DIFFERENCE:
      A Rs 10L gap is trivial for a Rs 10 crore firm but enormous for a Rs 15L firm.
      Ratio normalises across business sizes.

    WHY ALLOWED RANGE IS 0.4-1.6 (not 1.0):
      Firms rarely collect 100% of sales in the same month. B2B firms may have
      30-60 day payment terms (collection ratio < 1 in any given month).
      Firms collecting advances or clearing old receivables may show collection > 1.
      0.4-1.6 allows for a 2-3 month receivable cycle in both directions.

    WHAT THIS CATCHES:
      Ratio > 1.6: more money coming in than sales declared
        → circular transactions (merchant A pays merchant B pays back A via UPI
          to inflate apparent business activity)
        → collections from off-book sales not reflected in GST
      Ratio < 0.4: much less money coming in than declared sales
        → GST returns inflated relative to actual business
        → large outstanding receivables that may never be collected (bad debt risk)

    WHY UPI + BANK (not just bank):
      UPI settlements appear in bank statements but with a 1-2 day settlement lag
      and are often aggregated. Treating them separately avoids double-counting
      while ensuring we capture the full picture of digital collections.
    """
    gstr1 = data.get("gstr1_sales_lakhs")
    upi = data.get("upi_credit_inflow_lakhs", 0.0) or 0.0
    bank = data.get("bank_credit_inflow_lakhs", 0.0) or 0.0

    if gstr1 is None:
        return

    total_inflow = upi + bank
    if total_inflow == 0:
        return

    collection_ratio = total_inflow / gstr1

    lo, hi = 0.4, 1.6
    if not (lo <= collection_ratio <= hi):
        severity = "red" if collection_ratio > 2.5 or collection_ratio < 0.2 else "amber"
        result.flags.append(ConsistencyFlag(
            check_name="GSTR-1 Sales vs Bank + UPI Inflows",
            severity=severity,
            description=(
                f"Collection ratio {collection_ratio:.2f}x "
                f"({'inflows far exceed' if collection_ratio > hi else 'inflows far below'} "
                f"declared GST sales). Expected: {lo}-{hi}x."
            ),
            observed=collection_ratio,
            expected_range=(lo, hi),
            recommendation=(
                "High ratio may indicate circular UPI transactions inflating inflows. "
                "Low ratio may indicate off-book sales or high receivables. "
                "Request bank statement breakdown by counterparty."
            ),
        ))


def _check_ewaybill_vs_turnover(data: dict, result: ConsistencyResult):
    """
    CHECK: Is the e-way bill value consistent with declared GST turnover?
    (Only applies to manufacturing and trading — services skip this check)

    HOW:
      ewb_ratio = annual_ewaybill_value / annual_gst_turnover

    WHY E-WAY BILLS:
      Every movement of goods worth Rs 50,000+ requires an e-way bill filed
      with the GST portal before transit. A manufacturer or trader moving
      goods MUST generate e-way bills. If they declare high GST turnover but
      have few e-way bills, either:
        1. Goods are not actually moving (phantom sales)
        2. They are splitting shipments below Rs 50,000 to evade e-way requirement
           (itself a compliance violation)

    WHY 0.3-1.5 RANGE:
      Not all sales require e-way bills (intra-state below threshold, exempt goods).
      The ratio can be below 1.0 legitimately. But below 0.3 means less than 30%
      of claimed turnover generated any traceable goods movement — suspicious.
      Above 1.5 means more goods moved than sales declared — possible if inter-firm
      transfers (branch to branch) are counted but is worth verifying.

    WHY ONLY AMBER (not red) HERE:
      E-way bill data is the newest and patchiest of the four checks. The GSTN
      e-way bill portal has data gaps and not all goods categories require bills.
      We flag but do not escalate to red without corroborating evidence.
    """
    ewaybill = data.get("ewaybill_value_lakhs")
    turnover = data.get("gst_annual_turnover_lakhs")
    industry = data.get("industry_type", "manufacturing")

    if ewaybill is None or turnover is None:
        return

    if industry == "services":
        return  # services firms do not move physical goods

    ewb_ratio = ewaybill / (turnover + 1)

    lo, hi = 0.3, 1.5
    if not (lo <= ewb_ratio <= hi):
        result.flags.append(ConsistencyFlag(
            check_name="E-Way Bill Volume vs GST Turnover",
            severity="amber",
            description=(
                f"E-way bill value is {ewb_ratio:.2f}x of GST turnover. "
                f"Expected: {lo}-{hi}x for {industry}."
            ),
            observed=ewb_ratio,
            expected_range=(lo, hi),
            recommendation=(
                "Low e-way bill ratio vs turnover may indicate unverified goods movement. "
                "Verify physical stock movement against e-way bill records."
            ),
        ))

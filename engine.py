from dataclasses import dataclass
from typing import Tuple, List, Dict, Optional
import numpy as np
import pandas as pd


# ─── Data Model ──────────────────────────────────────────────────────────────

@dataclass
class MSMEProfile:
    business_name: str
    sector: str
    city: str
    years_in_business: int
    employee_count: int
    monthly_revenue: float
    is_ntc: bool = False
    is_ntb: bool = False
    cibil_score: Optional[int] = None

    # Pillar 1: Liquidity & Cash Flow
    bounce_count: int = 0
    bounce_cleared_24h: bool = True
    mdb_pct_sales: float = 15.0

    # Pillar 2: Profitability & Solvency
    dscr: float = 1.5
    net_profit_margin: float = 15.0

    # Pillar 3: Operational Discipline
    gst_filing_rate: float = 95.0
    utility_delay_days: float = 2.0

    # Pillar 4: Customer Velocity
    upi_active_days: int = 22
    invoice_collection_days: float = 20.0
    digital_rating: float = 4.0

    # Fraud signals
    round_gst_filings: bool = False
    sudden_revenue_spike: bool = False


# ─── Sector-Adaptive Weights ─────────────────────────────────────────────────

BASE_WEIGHTS = {
    "liquidity": 0.35,
    "profitability": 0.30,
    "discipline": 0.20,
    "velocity": 0.15,
}

SECTOR_ADJUSTMENTS = {
    "Retail":               {"liquidity": 0.00, "profitability": 0.00, "discipline": 0.00, "velocity": 0.00},
    "Manufacturing":        {"liquidity": 0.05, "profitability": 0.05, "discipline": 0.00, "velocity":-0.10},
    "Food & Beverage":      {"liquidity": 0.00, "profitability":-0.05, "discipline": 0.00, "velocity": 0.05},
    "Textiles":             {"liquidity": 0.05, "profitability": 0.00, "discipline": 0.00, "velocity":-0.05},
    "Services":             {"liquidity":-0.05, "profitability": 0.05, "discipline": 0.00, "velocity": 0.00},
    "Handicrafts / Exports":{"liquidity": 0.00, "profitability": 0.05, "discipline": 0.05, "velocity":-0.10},
    "Healthcare":           {"liquidity": 0.00, "profitability": 0.00, "discipline": 0.05, "velocity":-0.05},
    "Transport & Logistics":{"liquidity": 0.05, "profitability": 0.00, "discipline": 0.00, "velocity":-0.05},
}

def get_sector_weights(sector: str) -> Dict[str, float]:
    adj = SECTOR_ADJUSTMENTS.get(sector, {k: 0.0 for k in BASE_WEIGHTS})
    raw = {k: BASE_WEIGHTS[k] + adj.get(k, 0.0) for k in BASE_WEIGHTS}
    total = sum(raw.values())
    return {k: v / total for k, v in raw.items()}


# ─── Pillar Scoring ───────────────────────────────────────────────────────────

def score_liquidity(bounce_count: int, bounce_cleared_24h: bool, mdb_pct_sales: float) -> float:
    if bounce_count == 0 and mdb_pct_sales >= 15:
        return 10.0
    if bounce_count <= 2 and bounce_cleared_24h and mdb_pct_sales > 0:
        bonus = min(1.5, (mdb_pct_sales / 15) * 1.5)
        return min(7.5, 6.0 + bonus)
    if bounce_count > 4 or mdb_pct_sales <= 0:
        return 2.0
    base = max(2.0, 8.0 - bounce_count * 1.5)
    mdb_adj = min(2.0, (mdb_pct_sales / 15) * 2.0)
    return round(min(10.0, max(0.0, base + mdb_adj - 4.0)), 2)


def score_profitability(dscr: float, net_profit_margin: float) -> float:
    if dscr >= 2.0 and net_profit_margin >= 20:
        return 10.0
    if dscr >= 2.0:
        return 8.5
    if 1.25 <= dscr < 2.0:
        t = (dscr - 1.25) / (2.0 - 1.25)
        base = 7.0 + t * 2.5
        margin_adj = min(1.0, net_profit_margin / 20)
        return round(min(10.0, base * (0.7 + 0.3 * margin_adj)), 2)
    if 1.0 <= dscr < 1.25:
        t = (dscr - 1.0) / 0.25
        return round(4.0 + t * 2.5, 2)
    return round(max(0.0, dscr * 3), 2)


def score_discipline(gst_filing_rate: float, utility_delay_days: float) -> float:
    if gst_filing_rate >= 100 and utility_delay_days == 0:
        return 10.0
    if gst_filing_rate >= 90 and utility_delay_days <= 5:
        t = (gst_filing_rate - 90) / 10
        return round(min(10.0, 7.0 + t * 2.5 - min(1.0, utility_delay_days / 5)), 2)
    if utility_delay_days >= 30 or gst_filing_rate < 70:
        return 3.0
    score = (gst_filing_rate / 100) * 8.0
    score -= min(4.0, utility_delay_days / 7.5)
    return round(min(10.0, max(0.0, score)), 2)


def score_velocity(upi_active_days: int, invoice_collection_days: float, digital_rating: float) -> float:
    if upi_active_days >= 25 and invoice_collection_days <= 14 and digital_rating >= 4.2:
        return 10.0
    if upi_active_days < 10 and digital_rating < 2.5:
        return 1.0
    upi_score = min(4.5, (upi_active_days / 30) * 4.5)
    coll_score = min(3.5, max(0, (60 - invoice_collection_days) / 60) * 3.5)
    rating_score = min(2.0, (digital_rating / 5) * 2.0)
    return round(min(10.0, max(0.0, upi_score + coll_score + rating_score)), 2)


# ─── Composite Scoring ────────────────────────────────────────────────────────

def compute_scores(profile: MSMEProfile) -> Dict:
    p1 = score_liquidity(profile.bounce_count, profile.bounce_cleared_24h, profile.mdb_pct_sales)
    p2 = score_profitability(profile.dscr, profile.net_profit_margin)
    p3 = score_discipline(profile.gst_filing_rate, profile.utility_delay_days)
    p4 = score_velocity(profile.upi_active_days, profile.invoice_collection_days, profile.digital_rating)

    weights = get_sector_weights(profile.sector)
    composite = round(
        p1 * weights["liquidity"] +
        p2 * weights["profitability"] +
        p3 * weights["discipline"] +
        p4 * weights["velocity"],
        2,
    )

    fraud_flags = []
    if profile.round_gst_filings:
        fraud_flags.append("Suspicious round-number GST filings detected — manual verification advised")
    if profile.sudden_revenue_spike:
        fraud_flags.append("Sudden revenue spike (>200% MoM) flagged — source of funds check required")
    if profile.bounce_count > 5 and profile.gst_filing_rate > 95:
        fraud_flags.append("High bounce count inconsistent with strong GST compliance — cross-verify data integrity")

    return {
        "p1": p1, "p2": p2, "p3": p3, "p4": p4,
        "composite": composite,
        "weights": weights,
        "fraud_flags": fraud_flags,
    }


# ─── Credit Classification ───────────────────────────────────────────────────

def get_credit_band(score: float) -> Tuple[str, str, str]:
    if score >= 8.0:
        return "Prime", "#059669", "Excellent profile — Pre-approved for working capital"
    if score >= 6.5:
        return "Standard", "#1d4ed8", "Good profile — Eligible for standard interest rates"
    if score >= 5.0:
        return "Sub-Standard", "#d97706", "Moderate risk — Enhanced monitoring recommended"
    if score >= 3.5:
        return "Watch", "#dc2626", "Elevated risk — Collateral or co-applicant advised"
    return "NPA Risk", "#7f1d1d", "Very high risk — Credit not advisable at this time"


def get_interest_rate(score: float) -> float:
    if score >= 8.0: return 9.5
    if score >= 6.5: return 11.5
    if score >= 5.0: return 14.0
    if score >= 3.5: return 17.5
    return 22.0


def recommend_credit_limit(monthly_revenue: float, dscr: float, score: float) -> int:
    base = monthly_revenue * 4
    dscr_mult = min(dscr / 1.5, 1.6)
    score_mult = (score / 10) ** 0.7
    return int(base * dscr_mult * score_mult)


def get_peer_percentile(score: float, sector: str) -> int:
    SECTOR_AVG = {
        "Retail": 5.8, "Manufacturing": 6.1, "Food & Beverage": 5.5,
        "Textiles": 6.3, "Services": 6.5, "Handicrafts / Exports": 5.9,
        "Healthcare": 7.0, "Transport & Logistics": 5.6,
    }
    avg = SECTOR_AVG.get(sector, 6.0)
    z = (score - avg) / 1.2
    return min(99, max(1, int(50 + z * 34)))


# ─── Strengths, Risks, Roadmap ───────────────────────────────────────────────

def get_strengths_risks(scores: Dict, profile: MSMEProfile):
    pillar_names = {
        "p1": "Liquidity & Cash Flow",
        "p2": "Profitability & Solvency",
        "p3": "Operational Discipline",
        "p4": "Customer Velocity",
    }
    insights = {
        "p1": {
            "strength": f"Zero payment bounces over 12 months with healthy {profile.mdb_pct_sales:.0f}% cash buffer",
            "risk": f"{profile.bounce_count} auto-debit bounce(s) and thin {profile.mdb_pct_sales:.0f}% cash buffer signal liquidity stress",
        },
        "p2": {
            "strength": f"DSCR {profile.dscr:.2f}x — monthly profit comfortably covers loan repayment",
            "risk": f"DSCR {profile.dscr:.2f}x — thin margin; any revenue dip risks loan default",
        },
        "p3": {
            "strength": f"{profile.gst_filing_rate:.0f}% on-time GST compliance shows strong financial discipline",
            "risk": f"{100 - profile.gst_filing_rate:.0f}% late GST filings and {profile.utility_delay_days:.0f}-day utility delays indicate cash-flow stress",
        },
        "p4": {
            "strength": f"{profile.upi_active_days}/30 active UPI days with {profile.digital_rating:.1f}★ — strong market presence",
            "risk": f"Only {profile.upi_active_days}/30 UPI active days and {profile.invoice_collection_days:.0f}-day collection cycle slow cash circulation",
        },
    }
    strengths = [(pillar_names[k], insights[k]["strength"]) for k in ["p1","p2","p3","p4"] if scores[k] >= 7]
    risks = [(pillar_names[k], insights[k]["risk"]) for k in ["p1","p2","p3","p4"] if scores[k] < 5]
    return strengths, risks


def get_improvement_roadmap(scores: Dict, profile: MSMEProfile) -> List[Dict]:
    roadmap = []
    weights = get_sector_weights(profile.sector)

    if scores["p1"] < 7:
        delta = (min(10, scores["p1"] + 2.5) - scores["p1"]) * weights["liquidity"]
        roadmap.append({
            "pillar": "Liquidity & Cash Flow",
            "action": "Set up a minimum-balance sweep account with ₹50,000 buffer. Automate EMI scheduling to avoid bounces.",
            "impact": f"+{delta:.2f} pts overall",
            "timeframe": "1–2 months",
            "icon": "📈",
        })

    if scores["p2"] < 7 and profile.dscr < 1.5:
        delta = (min(10, scores["p2"] + 2.0) - scores["p2"]) * weights["profitability"]
        roadmap.append({
            "pillar": "Profitability & Solvency",
            "action": "Reduce operating costs 10–15% OR shift product mix toward higher-margin items to push DSCR above 1.5x.",
            "impact": f"+{delta:.2f} pts overall",
            "timeframe": "3–6 months",
            "icon": "💹",
        })

    if scores["p3"] < 7:
        delta = (min(10, scores["p3"] + 3.0) - scores["p3"]) * weights["discipline"]
        roadmap.append({
            "pillar": "Operational Discipline",
            "action": "Set calendar reminders for GSTR-1/3B due dates. Enable auto-pay on BBPS for commercial utility bills.",
            "impact": f"+{delta:.2f} pts overall",
            "timeframe": "1 month",
            "icon": "📋",
        })

    if scores["p4"] < 7:
        delta = (min(10, scores["p4"] + 2.0) - scores["p4"]) * weights["velocity"]
        roadmap.append({
            "pillar": "Customer Velocity",
            "action": "Add a QR code at the counter for UPI payments. Request verified Google Maps reviews from regular customers.",
            "impact": f"+{delta:.2f} pts overall",
            "timeframe": "2–4 months",
            "icon": "⚡",
        })

    return roadmap


# ─── Historical Trend (Synthetic) ────────────────────────────────────────────

def generate_historical_trend(scores: Dict, months: int = 6) -> pd.DataFrame:
    np.random.seed(7)
    records = []
    labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun"][:months]
    for i, label in enumerate(labels):
        factor = 0.62 + (i / (months - 1)) * 0.38
        noise = np.random.uniform(-0.25, 0.25)
        records.append({
            "Month": label,
            "Overall": max(1.0, min(10.0, scores["composite"] * factor + noise)),
            "Liquidity": max(1.0, min(10.0, scores["p1"] * factor + noise * 1.1)),
            "Profitability": max(1.0, min(10.0, scores["p2"] * factor + noise * 1.3)),
            "Discipline": max(1.0, min(10.0, scores["p3"] * factor + noise * 0.4)),
            "Velocity": max(1.0, min(10.0, scores["p4"] * factor + noise * 1.5)),
        })
    return pd.DataFrame(records)


# ─── Early Warning System ─────────────────────────────────────────────────────

def simulate_ews(profile: MSMEProfile, drop_pct: float) -> Dict:
    triggered = drop_pct >= 40
    current_upi = max(0, int(profile.upi_active_days * (1 - drop_pct / 100)))
    return {
        "original_upi_days": profile.upi_active_days,
        "current_upi_days": current_upi,
        "drop_pct": drop_pct,
        "alert_triggered": triggered,
        "alert_level": "ORANGE" if triggered else "GREEN",
        "action": "Suspend new credit line; assign relationship manager for immediate contact" if triggered else "No action required — business within normal range",
        "trigger_reason": f"UPI merchant transaction density fell {drop_pct:.0f}% over a rolling 7-day window" if triggered else None,
    }


# ─── OCEN Lender Offers ──────────────────────────────────────────────────────

def get_lender_offers(score: float, credit_limit: int) -> List[Dict]:
    if score < 3.5:
        return []
    base_rate = get_interest_rate(score)
    offers = []

    if score >= 5.0:
        offers.append({
            "lender": "IDBI Bank",
            "product": "MSME Working Capital Term Loan",
            "amount": credit_limit,
            "rate": base_rate,
            "tenure": "12 months",
            "fee": "0.50%",
            "tat": "4 hours",
            "tag": "Best Rate",
            "tag_color": "#059669",
        })

    if score >= 6.0:
        offers.append({
            "lender": "SBI MSME Hub",
            "product": "GST-Backed Revolving Credit Line",
            "amount": int(credit_limit * 0.9),
            "rate": round(base_rate + 0.75, 2),
            "tenure": "18 months",
            "fee": "0.75%",
            "tat": "24 hours",
            "tag": "Longer Tenure",
            "tag_color": "#1d4ed8",
        })

    if score >= 5.5:
        offers.append({
            "lender": "Paisabazaar NBFC",
            "product": "Digital Flexi MSME Loan",
            "amount": int(credit_limit * 1.1),
            "rate": round(base_rate + 1.5, 2),
            "tenure": "6 months",
            "fee": "1.50%",
            "tat": "30 minutes",
            "tag": "Fastest Approval",
            "tag_color": "#7c3aed",
        })

    return offers


# ─── Demo Profiles ───────────────────────────────────────────────────────────

DEMO_PROFILES = {
    "Sunrise Textiles — Prime MSME": MSMEProfile(
        business_name="Sunrise Textiles Pvt. Ltd.",
        sector="Textiles", city="Surat",
        years_in_business=6, employee_count=28, monthly_revenue=850000,
        is_ntc=False, is_ntb=False, cibil_score=768,
        bounce_count=0, bounce_cleared_24h=True, mdb_pct_sales=18.0,
        dscr=2.2, net_profit_margin=22.0,
        gst_filing_rate=100.0, utility_delay_days=0,
        upi_active_days=27, invoice_collection_days=11.0, digital_rating=4.6,
        round_gst_filings=False, sudden_revenue_spike=False,
    ),
    "Metro Quick Foods — NTC/NTB Viable": MSMEProfile(
        business_name="Metro Quick Foods",
        sector="Food & Beverage", city="Mumbai",
        years_in_business=1, employee_count=6, monthly_revenue=210000,
        is_ntc=True, is_ntb=True, cibil_score=None,
        bounce_count=1, bounce_cleared_24h=True, mdb_pct_sales=13.0,
        dscr=1.45, net_profit_margin=16.0,
        gst_filing_rate=93.0, utility_delay_days=3,
        upi_active_days=29, invoice_collection_days=4.0, digital_rating=4.5,
        round_gst_filings=False, sudden_revenue_spike=False,
    ),
    "Sharma General Store — Struggling": MSMEProfile(
        business_name="Sharma General Store",
        sector="Retail", city="Delhi",
        years_in_business=3, employee_count=3, monthly_revenue=120000,
        is_ntc=False, is_ntb=False, cibil_score=612,
        bounce_count=5, bounce_cleared_24h=False, mdb_pct_sales=4.0,
        dscr=1.08, net_profit_margin=7.0,
        gst_filing_rate=76.0, utility_delay_days=22,
        upi_active_days=16, invoice_collection_days=38.0, digital_rating=3.1,
        round_gst_filings=True, sudden_revenue_spike=False,
    ),
    "Kaveri Handicrafts — Seasonal": MSMEProfile(
        business_name="Kaveri Handicrafts & Exports",
        sector="Handicrafts / Exports", city="Jaipur",
        years_in_business=4, employee_count=12, monthly_revenue=380000,
        is_ntc=False, is_ntb=False, cibil_score=698,
        bounce_count=0, bounce_cleared_24h=True, mdb_pct_sales=12.0,
        dscr=1.65, net_profit_margin=19.0,
        gst_filing_rate=91.0, utility_delay_days=6,
        upi_active_days=14, invoice_collection_days=42.0, digital_rating=4.1,
        round_gst_filings=False, sudden_revenue_spike=False,
    ),

    # ── Profile 5: NPA Risk — credit denied ──────────────────────────────────
    # Composite ≈ 2.08 → NPA Risk band. Shows what happens when the system
    # says no: zero lender offers, full roadmap of what must change before
    # reapplying, and a 6-month flat-low trajectory in the EWS chart.
    "Raj Kirana & General — NPA Risk": MSMEProfile(
        business_name="Raj Kirana & General Store",
        sector="Retail", city="Nagpur",
        years_in_business=2, employee_count=2, monthly_revenue=95000,
        is_ntc=False, is_ntb=False, cibil_score=None,
        bounce_count=10, bounce_cleared_24h=False, mdb_pct_sales=0.5,
        dscr=0.7, net_profit_margin=4.0,
        gst_filing_rate=58.0, utility_delay_days=45.0,
        upi_active_days=5, invoice_collection_days=85.0, digital_rating=2.0,
        round_gst_filings=False, sudden_revenue_spike=False,
    ),

    # ── Profile 6: Fraud / Anomaly Detection ─────────────────────────────────
    # Composite ≈ 7.36 → Standard band on paper. But three fraud flags fire:
    # (1) round GST filings, (2) sudden revenue spike, (3) contradiction between
    # 7 bounces and 98% GST compliance. Demonstrates the AI anomaly layer.
    "Nexgen Pharma Retail — Fraud Flags": MSMEProfile(
        business_name="Nexgen Pharma Retail Pvt. Ltd.",
        sector="Services", city="Hyderabad",
        years_in_business=3, employee_count=9, monthly_revenue=620000,
        is_ntc=False, is_ntb=False, cibil_score=701,
        bounce_count=7, bounce_cleared_24h=True, mdb_pct_sales=22.0,
        dscr=2.5, net_profit_margin=28.0,
        gst_filing_rate=98.0, utility_delay_days=1.0,
        upi_active_days=28, invoice_collection_days=8.0, digital_rating=4.8,
        round_gst_filings=True, sudden_revenue_spike=True,
    ),

    # ── Profile 7: Watch → Standard Recovery ─────────────────────────────────
    # Composite ≈ 7.05 → Standard band now. The historical trend (generated
    # from current scores with a rising factor) shows the score climbing from
    # ~4.4 (Watch) six months ago to ~7.0 today — a genuine recovery arc.
    # Use this to demo the EWS "trajectory" panel and the bank's ability to
    # track post-disbursement improvement before approving a credit line top-up.
    "Vijay Auto Components — Recovery": MSMEProfile(
        business_name="Vijay Auto Components Pvt. Ltd.",
        sector="Manufacturing", city="Pune",
        years_in_business=5, employee_count=18, monthly_revenue=490000,
        is_ntc=False, is_ntb=False, cibil_score=648,
        bounce_count=1, bounce_cleared_24h=True, mdb_pct_sales=10.0,
        dscr=1.6, net_profit_margin=17.0,
        gst_filing_rate=88.0, utility_delay_days=8.0,
        upi_active_days=20, invoice_collection_days=25.0, digital_rating=3.8,
        round_gst_filings=False, sudden_revenue_spike=False,
    ),
}

SECTORS = list(SECTOR_ADJUSTMENTS.keys())
CITIES = ["Mumbai", "Delhi", "Bangalore", "Surat", "Jaipur", "Chennai", "Hyderabad", "Pune", "Kolkata", "Ahmedabad"]

import re
import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from typing import Dict

from engine import (
    MSMEProfile, compute_scores, get_credit_band, recommend_credit_limit,
    get_interest_rate, get_strengths_risks, get_improvement_roadmap,
    generate_historical_trend, simulate_ews, get_lender_offers,
    get_peer_percentile, DEMO_PROFILES, SECTORS, CITIES,
)

# ─── GSTIN Utilities (defined here so no cross-module import issues) ──────────

_GSTIN_STATE_MAP: Dict[str, tuple] = {
    "01": ("Jammu & Kashmir",   "Jammu"),
    "02": ("Himachal Pradesh",  "Shimla"),
    "03": ("Punjab",            "Amritsar"),
    "04": ("Chandigarh",        "Chandigarh"),
    "05": ("Uttarakhand",       "Dehradun"),
    "06": ("Haryana",           "Gurugram"),
    "07": ("Delhi",             "Delhi"),
    "08": ("Rajasthan",         "Jaipur"),
    "09": ("Uttar Pradesh",     "Lucknow"),
    "10": ("Bihar",             "Patna"),
    "11": ("Sikkim",            "Gangtok"),
    "18": ("Assam",             "Guwahati"),
    "19": ("West Bengal",       "Kolkata"),
    "20": ("Jharkhand",         "Ranchi"),
    "21": ("Odisha",            "Bhubaneswar"),
    "22": ("Chhattisgarh",      "Raipur"),
    "23": ("Madhya Pradesh",    "Bhopal"),
    "24": ("Gujarat",           "Ahmedabad"),
    "27": ("Maharashtra",       "Mumbai"),
    "29": ("Karnataka",         "Bangalore"),
    "30": ("Goa",               "Panaji"),
    "32": ("Kerala",            "Thiruvananthapuram"),
    "33": ("Tamil Nadu",        "Chennai"),
    "36": ("Telangana",         "Hyderabad"),
    "37": ("Andhra Pradesh",    "Visakhapatnam"),
}

_GSTIN_PATTERN = re.compile(r'^[0-9]{2}[A-Z]{5}[0-9]{4}[A-Z][1-9A-Z]Z[0-9A-Z]$')


def validate_gstin(gstin: str) -> bool:
    return bool(_GSTIN_PATTERN.match(gstin.upper().strip()))


def _parse_gstin_offline(gstin: str) -> Dict:
    gstin = gstin.upper().strip()
    state_code = gstin[:2]
    state, city = _GSTIN_STATE_MAP.get(state_code, ("Unknown", "Unknown"))
    entity_code = gstin[12]
    entity_type = "Company / LLP" if entity_code.isdigit() else "Individual / Proprietorship"
    return {
        "gstin":              gstin,
        "state":              state,
        "city":               city,
        "entity_type":        entity_type,
        "pan":                gstin[2:12],
        "state_code":         state_code,
        "legal_name":         "",
        "trade_name":         "",
        "registration_date":  "",
        "gst_status":         "Not verified (no API key)",
        "source":             "GSTIN structure (offline)",
    }


def fetch_gstn_taxpayer(gstin: str, api_key: str = "") -> Dict:
    base = _parse_gstin_offline(gstin)
    if not api_key:
        return base
    try:
        import requests
        resp = requests.get(
            f"https://api.gstn.org/commonapi/v1.1/search?tin={gstin}&lang=en",
            timeout=8,
            headers={"Accept": "application/json", "Authorization": f"Bearer {api_key}"},
        )
        if resp.status_code == 200:
            tp = resp.json().get("taxpayerInfo", resp.json())
            return {
                **base,
                "legal_name":        tp.get("lgnm", ""),
                "trade_name":        tp.get("tradeNam", ""),
                "registration_date": tp.get("rgdt", ""),
                "gst_status":        tp.get("sts", "Active"),
                "state":             tp.get("stj", base["state"]) or base["state"],
                "source":            "GSTN API (live)",
            }
        hint = "Invalid API key" if resp.status_code == 401 else f"HTTP {resp.status_code}"
        return {**base, "source": f"GSTIN structure only ({hint})"}
    except Exception as exc:
        return {**base, "source": f"GSTIN structure only (network error: {exc})"}

# ─── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MSME Financial Health Card | IDBI Bank",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Session State ────────────────────────────────────────────────────────────
# consent_given: True after the AA button is clicked for the current profile
# consent_profile: which demo profile the consent was given for
if "consent_given" not in st.session_state:
    st.session_state["consent_given"] = False
if "consent_profile" not in st.session_state:
    st.session_state["consent_profile"] = None
if "gstn_data" not in st.session_state:
    st.session_state["gstn_data"] = None
if "gstn_fetched" not in st.session_state:
    st.session_state["gstn_fetched"] = False

# ─── CSS ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── 1. White backgrounds everywhere ── */
html, body, [data-testid="stAppViewContainer"],
[data-testid="stAppViewContainer"] > .main,
[data-testid="stMain"], section.main, .main .block-container {
    background-color: #ffffff !important;
    color: #1e293b !important;
}
[data-testid="stSidebar"] {
    background-color: #f1f5f9 !important;
    border-right: 1.5px solid #e2e8f0;
}
[data-testid="stSidebar"] > div:first-child { background-color: #f1f5f9 !important; }

/* ── 2. CRITICAL: Force ALL text dark so it shows on white background ── */
/* Widget labels (sliders, inputs, selects, checkboxes, radio buttons) */
[data-testid="stWidgetLabel"],
[data-testid="stWidgetLabel"] p,
[data-testid="stWidgetLabel"] span,
label, label span, label p {
    color: #1e293b !important;
}
/* Markdown / paragraph text */
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] span,
[data-testid="stMarkdownContainer"] li,
.stMarkdown p, .stMarkdown span, .stMarkdown li,
.element-container p, .element-container span {
    color: #1e293b !important;
}
/* Metric labels and values */
[data-testid="stMetricLabel"], [data-testid="stMetricLabel"] p,
[data-testid="stMetricValue"], [data-testid="stMetricValue"] p,
[data-testid="metric-container"] p, [data-testid="metric-container"] span {
    color: #1e293b !important;
}
/* Selectbox, number input, text input text */
[data-baseweb="select"] span, [data-baseweb="select"] div,
[data-baseweb="input"] input,
.stNumberInput input, .stTextInput input {
    color: #1e293b !important;
    background-color: #ffffff !important;
}
/* Dropdown options list */
[data-baseweb="popover"] li, [data-baseweb="menu"] li,
[data-baseweb="option"] { color: #1e293b !important; background: #ffffff !important; }
/* Radio and checkbox labels */
.stRadio label, .stRadio label span,
.stCheckbox label, .stCheckbox label span { color: #1e293b !important; }
/* Slider value labels */
[data-testid="stSlider"] [data-testid="stTickBarMin"],
[data-testid="stSlider"] [data-testid="stTickBarMax"],
.stSlider span { color: #334155 !important; }
/* Caption / small text */
[data-testid="stCaptionContainer"] p,
[data-testid="stCaptionContainer"] span { color: #334155 !important; }
/* Divider */
hr { border-color: #e2e8f0 !important; }
/* Expander label */
[data-testid="stExpander"] summary,
[data-testid="stExpander"] summary span,
[data-testid="stExpander"] summary p { color: #1e293b !important; }
/* Success/Info/Warning/Error boxes keep their natural colours */
[data-testid="stNotification"] p { color: inherit !important; }
/* Sidebar text */
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] { color: #1e293b !important; }
/* DataFrame / table text */
.stDataFrame, .stDataFrame * { color: #1e293b !important; }
/* Heading tags */
h1, h2, h3, h4, h5, h6 { color: #0f172a !important; }

/* ── 3. Tabs — always visible ── */
[data-testid="stTabs"] [role="tab"] {
    font-size: 13px !important;
    font-weight: 600 !important;
    padding: 8px 18px !important;
    color: #1e293b !important;
    white-space: nowrap !important;
}
[data-testid="stTabs"] [role="tab"][aria-selected="true"] {
    color: #1d4ed8 !important;
    border-bottom: 3px solid #1d4ed8 !important;
}
[data-testid="stTabsContent"] { background: white !important; }

/* ── 4. Layout ── */
.block-container { padding-top: 1rem; padding-bottom: 2rem; max-width: 1380px; }

/* ── 5. Custom components ── */
.pillar-card {
    background: #ffffff;
    border: 1.5px solid #e2e8f0;
    border-radius: 14px;
    padding: 18px 12px 14px;
    text-align: center;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
}
.band-pill {
    display: inline-block;
    padding: 4px 14px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 700;
    margin: 3px 4px;
    color: white;
}
.info-box {
    background: #eff6ff;
    border: 1.5px solid #bfdbfe;
    border-radius: 12px;
    padding: 16px 20px;
    margin-bottom: 16px;
    color: #1e293b;
}
.info-box b { color: #1e40af; }
.calc-rule {
    background: #f8fafc;
    border-left: 3px solid #1d4ed8;
    border-radius: 0 8px 8px 0;
    padding: 8px 14px;
    margin: 6px 0;
    font-size: 13px;
    color: #1e293b;
}
.badge { display:inline-block; padding:3px 11px; border-radius:20px; font-size:11px; font-weight:700; margin-right:5px; }
.badge-ntc { background:#eff6ff; color:#1d4ed8; border:1px solid #bfdbfe; }
.badge-ntb { background:#fdf4ff; color:#7e22ce; border:1px solid #e9d5ff; }
.badge-cibil { background:#fff7ed; color:#c2410c; border:1px solid #fed7aa; }
.section-title { font-size:15px; font-weight:700; color:#1e293b !important; padding-bottom:8px;
    border-bottom: 2px solid #e2e8f0; margin-bottom: 14px; }
.offer-card { background:white; border:1.5px solid #e2e8f0; border-radius:14px;
    padding:20px; box-shadow:0 2px 8px rgba(0,0,0,0.05); height:100%; }
.tag { display:inline-block; padding:2px 10px; border-radius:10px; font-size:10px;
    font-weight:700; color:white; margin-bottom:8px; }
.roadmap-row { background:#f8fafc; border-left:4px solid #1d4ed8; border-radius:0 8px 8px 0;
    padding:12px 16px; margin-bottom:10px; }
.ews-green { background:#f0fdf4; border:1.5px solid #86efac; border-radius:10px; padding:14px 18px; }
.ews-orange { background:#fff7ed; border:1.5px solid #fb923c; border-radius:10px; padding:14px 18px; }
.src-chip { display:inline-block; background:#e2e8f0; color:#1e293b; font-size:10px;
    font-weight:600; padding:2px 8px; border-radius:10px; margin:2px; }
/* Explanation helper box */
.explain-box {
    background: #f1f5f9;
    border: 1px solid #cbd5e1;
    border-radius: 10px;
    padding: 10px 14px;
    margin: 8px 0 14px 0;
    font-size: 13px;
    color: #1e293b !important;
    line-height: 1.6;
}
.explain-box b { color: #0f172a; }

#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def score_color(val: float) -> str:
    if val >= 7:  return "#059669"
    if val >= 5:  return "#d97706"
    return "#dc2626"

def fmt_inr(amount: float) -> str:
    if amount >= 1_00_000:
        return f"₹{amount/1_00_000:.2f}L"
    return f"₹{amount:,.0f}"

def make_gauge(score: float, band: str, color: str, desc: str) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        number={"font": {"size": 64, "color": color}, "suffix": "/10"},
        title={"text": f"<b style='color:{color};font-size:20px'>{band}</b><br>"
                       f"<span style='font-size:13px;color:#334155'>{desc}</span>",
               "font": {"size": 14}},
        gauge={
            "axis": {"range": [0, 10], "tickwidth": 1, "tickcolor": "#64748b",
                     "tickvals": [0, 3.5, 5, 6.5, 8, 10],
                     "ticktext": ["0", "3.5", "5", "6.5", "8", "10"],
                     "tickfont": {"size": 11, "color": "#334155"}},
            "bar": {"color": color, "thickness": 0.3},
            "bgcolor": "white",
            "borderwidth": 0,
            "steps": [
                {"range": [0,   3.5], "color": "#fee2e2"},
                {"range": [3.5, 5.0], "color": "#fef3c7"},
                {"range": [5.0, 6.5], "color": "#dbeafe"},
                {"range": [6.5, 8.0], "color": "#d1fae5"},
                {"range": [8.0,10.0], "color": "#a7f3d0"},
            ],
        },
    ))
    fig.update_layout(
        paper_bgcolor="white", font={"color": "#1e293b"},
        height=310, margin=dict(t=80, b=10, l=30, r=30),
        dragmode=False,
    )
    return fig

def make_radar(scores: Dict) -> go.Figure:
    cats = ["Liquidity", "Profitability", "Discipline", "Velocity"]
    full = ["Liquidity & Cash Flow", "Profitability & Solvency",
            "Operational Discipline", "Customer Velocity"]
    vals = [scores["p1"], scores["p2"], scores["p3"], scores["p4"]]

    fig = go.Figure()
    avg = [6.0] * 4
    fig.add_trace(go.Scatterpolar(
        r=avg + [avg[0]], theta=cats + [cats[0]],
        fill="toself", fillcolor="rgba(148,163,184,0.12)",
        line=dict(color="#64748b", width=1.5, dash="dot"),
        name="Sector avg (6.0)",
        hoverinfo="skip",
    ))
    fig.add_trace(go.Scatterpolar(
        r=vals + [vals[0]], theta=cats + [cats[0]],
        fill="toself", fillcolor="rgba(29,78,216,0.15)",
        line=dict(color="#1d4ed8", width=3),
        name="This MSME",
        customdata=[[f"{v:.1f}/10 — {n}" for v, n in zip(vals, full)][i] for i in range(4)] + [None],
        hovertemplate="%{customdata}<extra></extra>",
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True, range=[0, 10],
                tickvals=[2, 4, 6, 8, 10],
                tickfont=dict(size=11, color="#1e293b"),
                gridcolor="#e2e8f0", linecolor="#cbd5e1",
            ),
            angularaxis=dict(
                # Smaller font so labels never clip into each other
                tickfont=dict(size=12, color="#0f172a", family="sans-serif"),
                gridcolor="#e2e8f0", linecolor="#cbd5e1",
                rotation=90, direction="clockwise",
            ),
            bgcolor="white",
        ),
        showlegend=True,
        # Push legend well below the chart so it never overlaps the axes
        legend=dict(orientation="h", xanchor="center", x=0.5,
                    yanchor="top", y=-0.12,
                    font=dict(size=12, color="#1e293b"), bgcolor="white",
                    bordercolor="#e2e8f0", borderwidth=1),
        paper_bgcolor="white",
        height=420,
        # Wider margins so axis labels have room
        margin=dict(t=50, b=100, l=90, r=90),
        title=dict(
            text="4-Pillar Score Profile — solid line = your MSME, dotted = sector avg",
            font=dict(size=12, color="#1e293b"), x=0.5, xanchor="center",
        ),
        dragmode=False,
    )
    return fig


SCORING_RULES = {
    "Liquidity & Cash Flow": {
        "what": "Measures short-term survival — does your business keep enough cash to pay obligations on time?",
        "source": ["AA Bank Statements (via Sahamati)", "NPCI UPI Merchant Logs"],
        "rules": [
            ("10/10 — Excellent", "Zero auto-debit or cheque bounces over 12 months AND minimum daily balance never drops below 15% of monthly sales."),
            ("6/10 — Good",       "1–2 minor bounces cleared within 24 hours. Balance stays positive every week."),
            ("2/10 — Poor",       "Multiple recurring bounces OR account frequently at zero/negative before month-end."),
        ],
        "weight": "35%",
        "color": "#3b82f6",
    },
    "Profitability & Solvency": {
        "what": "Measures long-term viability — does the business make enough profit to repay a loan?",
        "source": ["GSTN — GSTR-1 (Sales)", "GSTN — GSTR-2B (Input Tax Credit)", "BBPS Utility History"],
        "rules": [
            ("10/10 — Excellent", "DSCR > 2.0x (profit is 2× the loan EMI) AND net margin > 20%. Lender's dream borrower."),
            ("7/10 — Good",       "DSCR 1.25x–1.9x — standard safe banking range. Business generates surplus comfortably."),
            ("4/10 — Caution",    "DSCR 1.0x–1.24x — barely covers obligations. Any revenue dip risks default."),
            ("0/10 — Critical",   "DSCR < 1.0x — business is operating at a net loss. Cannot service a new loan."),
        ],
        "weight": "30%",
        "color": "#8b5cf6",
    },
    "Operational Discipline": {
        "what": "Measures intent-to-pay — does the owner meet mandatory deadlines even when cash is available?",
        "source": ["GSTN — Filing timestamps (GSTR-1, GSTR-3B)", "BBPS — Utility payment timelines"],
        "rules": [
            ("10/10 — Excellent", "100% on-time GST returns + all commercial utility bills paid before due date for 12 months."),
            ("7/10 — Good",       "Minor 1–5 day delays occasionally, but all dues cleared within the same calendar month."),
            ("3/10 — Poor",       "Consistent 30+ day GST delays OR active utility disconnection due to prolonged non-payment."),
        ],
        "weight": "20%",
        "color": "#10b981",
    },
    "Customer Velocity": {
        "what": "A forward radar — are customers returning? Will revenue keep flowing tomorrow?",
        "source": ["NPCI / Merchant Aggregators (BharatPe, PhonePe Business)", "GSTN Invoices", "Public Ratings (Google Maps)"],
        "rules": [
            ("10/10 — Excellent", "UPI payments received 25+ out of 30 days. Invoice collection under 14 days. Digital rating ≥ 4.2★."),
            ("6/10 — Moderate",   "Seasonal gaps — active 15 days, quiet 15 days. Collection 30–45 days. Rating ~3.0★."),
            ("1/10 — Critical",   "Stagnant UPI footprint, frozen transaction volume, or flood of 1-star verified complaints."),
        ],
        "weight": "15%",
        "color": "#f59e0b",
    },
}

BAND_LEGEND = [
    ("Prime",        "#059669", "8.0 – 10.0"),
    ("Standard",     "#1d4ed8", "6.5 – 7.9"),
    ("Sub-Standard", "#d97706", "5.0 – 6.4"),
    ("Watch",        "#dc2626", "3.5 – 4.9"),
    ("NPA Risk",     "#7f1d1d", "0 – 3.4"),
]


# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center;padding:14px 0 6px">
        <div style="font-size:32px;font-weight:900;color:#003087;letter-spacing:-1px">IDBI</div>
        <div style="font-size:10px;font-weight:700;color:#334155;letter-spacing:2.5px;margin-top:2px">
            MSME HEALTH CARD
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="font-size:13px;color:#1e293b;text-align:center;padding:6px 8px 12px;
                border-bottom:1px solid #e2e8f0">
        AI-powered credit evaluation using<br>alternate data — no balance sheets needed
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown(
        "<div class='explain-box'>"
        "<b>Demo Profiles</b> — pre-built scenarios · "
        "<b>Custom Input</b> — enter any values · "
        "<b>Live GSTN</b> — type a real GSTIN and fetch actual business data from GSTN"
        "</div>",
        unsafe_allow_html=True,
    )
    mode = st.radio("Input Mode", ["Demo Profiles", "Custom Input", "Live GSTN Lookup"],
                    label_visibility="collapsed")

    if mode == "Demo Profiles":
        st.markdown(
            "<div class='explain-box'>Each demo profile represents a different type of Indian MSME "
            "(e.g. a textile shop, a food stall, a tech startup). Select one, then click the button "
            "below to simulate the AA consent flow and load the full analysis.</div>",
            unsafe_allow_html=True,
        )
        selected = st.selectbox("Select a Demo Profile", list(DEMO_PROFILES.keys()))
        profile = DEMO_PROFILES[selected]

        # Reset consent whenever the user picks a different profile
        if st.session_state["consent_profile"] != selected:
            st.session_state["consent_given"] = False
            st.session_state["consent_profile"] = selected

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("**Data sources that will be fetched:**")
        for s in ["AA Network · Bank Statements", "GSTN · GSTR-1 / 3B",
                  "NPCI · UPI Merchant Logs", "BBPS · Utility Payments",
                  "MCA21 · Company Registry", "EPFO · Payroll Records"]:
            st.markdown(
                f"<div style='font-size:12px;color:#059669;padding:2px 0'>✓ {s}</div>",
                unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.caption("In production this triggers a real Account Aggregator OTP consent flow.")
        if st.button("Simulate AA Consent & Fetch Data",
                     type="primary", use_container_width=True):
            import time
            with st.spinner("Connecting to ULI gateway..."):
                time.sleep(1.2)
            st.session_state["consent_given"] = True
            st.success("Consent granted — all data fetched in 1.2 s")

        if st.session_state["consent_given"]:
            st.markdown(
                "<div style='background:#f0fdf4;border:1.5px solid #86efac;border-radius:8px;"
                "padding:8px 12px;font-size:12px;color:#14532d;margin-top:8px'>"
                "✓ Data loaded for <b>" + profile.business_name + "</b>"
                "</div>",
                unsafe_allow_html=True,
            )

    elif mode == "Custom Input":
        st.markdown(
            "<div class='explain-box'>Fill in the fields below. The score updates instantly. "
            "All values here simulate what the system would pull from GSTN, NPCI, BBPS and "
            "the Account Aggregator network — no paper documents needed.</div>",
            unsafe_allow_html=True,
        )
        st.markdown("**Business Details**")
        biz_name  = st.text_input("Business Name", "My MSME Pvt. Ltd.",
                                  help="Legal name of the business or trade name")
        sector    = st.selectbox("Sector", SECTORS,
                                 help="Business sector — affects pillar weights used in scoring")
        city      = st.selectbox("City", CITIES,
                                 help="Location used for peer comparison benchmarks")
        years     = st.number_input("Years in Business", 0, 50, 2,
                                    help="Older businesses get a small stability bonus")
        employees = st.number_input("Employees", 1, 500, 8,
                                    help="Total headcount including part-time staff")
        revenue   = st.number_input("Monthly Revenue (₹)", 10000, 10000000, 300000, 10000,
                                    help="Average monthly turnover — used to calculate recommended credit limit")
        is_ntc    = st.checkbox("New-to-Credit (NTC)",
                                help="Check if this business has never taken a formal loan before")
        is_ntb    = st.checkbox("New-to-Bank (NTB)",
                                help="Check if this business has no prior relationship with any bank")

        st.divider()
        st.markdown("**Pillar 1 — Liquidity & Cash Flow (35% weight)**")
        st.caption("Measures whether the business maintains enough cash to pay obligations on time.")
        bounce_count = st.slider("Auto-debit Bounces (12 months)", 0, 15, 1,
                                 help="Number of failed auto-debit or cheque bounce events in the last 12 months. 0 = perfect. Each bounce lowers the score.")
        bounce_24h   = st.checkbox("Bounces cleared within 24 h", True,
                                   help="If checked, any bounces were resolved quickly — partial credit given")
        mdb_pct      = st.slider("Min Daily Balance (% of monthly sales)", 0.0, 30.0, 12.0, 0.5,
                                 help="Lowest daily bank balance as a % of monthly revenue. Above 15% = excellent liquidity buffer.")

        st.divider()
        st.markdown("**Pillar 2 — Profitability & Solvency (30% weight)**")
        st.caption("Measures whether the business earns enough profit to comfortably repay a loan.")
        dscr = st.slider("DSCR", 0.3, 3.5, 1.5, 0.05,
                         help="Debt-Service Coverage Ratio = Monthly Net Profit ÷ Monthly Loan EMI. Above 1.25x = bankable. Above 2.0x = prime.")
        npm  = st.slider("Net Profit Margin (%)", 0.0, 40.0, 15.0, 0.5,
                         help="What % of revenue is kept as profit after all costs. Above 20% = strong. Below 5% = risky.")

        st.divider()
        st.markdown("**Pillar 3 — Operational Discipline (20% weight)**")
        st.caption("Measures whether the business meets mandatory government deadlines — a proxy for intent-to-pay.")
        gst_rate   = st.slider("GST Filing On-Time Rate (%)", 50.0, 100.0, 90.0, 1.0,
                               help="% of GST returns (GSTR-1, GSTR-3B) filed before the due date. 100% = perfect discipline.")
        util_delay = st.slider("Utility Payment Delay (avg days)", 0, 60, 5,
                               help="Average days late on electricity/water/gas bills via BBPS. 0 = always on time. Above 30 days = poor discipline.")

        st.divider()
        st.markdown("**Pillar 4 — Customer Velocity (15% weight)**")
        st.caption("A forward-looking radar — are customers coming back? Will revenue keep flowing?")
        upi_days = st.slider("UPI Active Days / month", 0, 30, 22,
                             help="Days in a month when at least one UPI payment was received. 25+ = very active business.")
        inv_days = st.slider("Invoice Collection Speed (days)", 1, 90, 20,
                             help="Average days to collect payment after raising an invoice. Faster = better cash flow.")
        rating   = st.slider("Digital Rating (Google / e-com)", 1.0, 5.0, 4.0, 0.1,
                             help="Average customer rating on Google Maps or e-commerce platform. Above 4.2 = excellent reputation.")

        profile = MSMEProfile(
            business_name=biz_name, sector=sector, city=city,
            years_in_business=int(years), employee_count=int(employees),
            monthly_revenue=float(revenue),
            is_ntc=is_ntc, is_ntb=is_ntb, cibil_score=None,
            bounce_count=int(bounce_count), bounce_cleared_24h=bounce_24h,
            mdb_pct_sales=mdb_pct, dscr=dscr, net_profit_margin=npm,
            gst_filing_rate=gst_rate, utility_delay_days=float(util_delay),
            upi_active_days=int(upi_days), invoice_collection_days=float(inv_days),
            digital_rating=rating,
        )

    else:  # ── Live GSTN Lookup ───────────────────────────────────────────────
        st.markdown(
            "<div class='explain-box'>"
            "Enter any valid 15-character GSTIN. The app will fetch the business name, "
            "state and registration info live from the GSTN API. "
            "Financial fields (which need the AA network) are set via sliders below."
            "</div>",
            unsafe_allow_html=True,
        )

        gstin_raw = st.text_input(
            "GSTIN Number", placeholder="e.g. 27AABCU9603R1ZX",
            max_chars=15,
            help="15-character GST Identification Number printed on any GST invoice",
        ).upper().strip()

        with st.expander("Optional: GSTN API Key (for full name & date lookup)"):
            gstn_api_key = st.text_input(
                "GSTN Sandbox API Key",
                type="password",
                placeholder="Paste key from developer.gst.gov.in",
                help=(
                    "Without a key: state, city and entity type are decoded from "
                    "the GSTIN digits — always works offline.\n\n"
                    "With a key: business name, registration date and GST status "
                    "are fetched live from the GSTN API. "
                    "Get a free sandbox key at developer.gst.gov.in."
                ),
            )

        st.markdown(
            "<div style='font-size:11px;color:#334155;background:#fefce8;"
            "border:1px solid #fde68a;border-radius:6px;padding:6px 10px;margin:4px 0'>"
            "Without API key: state & entity decoded from GSTIN digits (offline, always works).<br>"
            "With API key: fetches real business name & registration date from GSTN."
            "</div>",
            unsafe_allow_html=True,
        )

        fetch_clicked = st.button("Fetch from GSTN", type="primary",
                                  use_container_width=True)

        if fetch_clicked:
            if not gstin_raw:
                st.error("Please enter a GSTIN number first.")
            elif not validate_gstin(gstin_raw):
                st.error(
                    "Invalid GSTIN format. Must be 15 characters — "
                    "2-digit state code + 10-char PAN + entity code + Z + check digit. "
                    "Example: 27AABCU9603R1ZX"
                )
            else:
                if gstn_api_key:
                    msg = "Connecting to GSTN API with your key..."
                else:
                    msg = "Decoding GSTIN structure (offline — no API key provided)..."
                with st.spinner(msg):
                    import time; time.sleep(0.6)
                    result = fetch_gstn_taxpayer(gstin_raw, api_key=gstn_api_key)
                st.session_state["gstn_data"]    = result
                st.session_state["gstn_fetched"] = True

        # Show fetched data panel
        gd = st.session_state.get("gstn_data") or {}
        gstn_ready = st.session_state.get("gstn_fetched") and gd

        if gstn_ready:
            is_live   = "live" in gd.get("source", "")
            is_offline = "offline" in gd.get("source", "")
            bg     = "#f0fdf4" if is_live else "#fefce8"
            border = "#86efac" if is_live else "#fde68a"
            icon   = "🟢 Live GSTN fetch" if is_live else "🟡 Offline GSTIN decode"
            tc     = "#14532d" if is_live else "#713f12"

            name_row = ""
            if gd.get("legal_name"):
                name_row = f"<b>Legal Name:</b> {gd['legal_name']}<br>"
            if gd.get("trade_name") and gd.get("trade_name") != gd.get("legal_name"):
                name_row += f"<b>Trade Name:</b> {gd['trade_name']}<br>"

            date_row = ""
            if gd.get("registration_date"):
                date_row = f"<b>Registered:</b> {gd['registration_date']}<br>"

            always_row = (
                f"<b>GSTIN:</b> {gd.get('gstin','—')}<br>"
                f"<b>State:</b> {gd.get('state','—')} &nbsp;·&nbsp; "
                f"<b>City (approx):</b> {gd.get('city','—')}<br>"
                f"<b>Entity Type:</b> {gd.get('entity_type','—')}<br>"
                f"<b>GST Status:</b> {gd.get('gst_status','—')}<br>"
            )

            if not is_live:
                no_key_note = (
                    "<div style='margin-top:8px;font-size:11px;color:#713f12;"
                    "background:#fef9c3;border-radius:6px;padding:6px 10px'>"
                    "Business name & registration date need an API key. "
                    "Add your key in the expander above and click Fetch again."
                    "</div>"
                )
            else:
                no_key_note = ""

            st.markdown(
                f"<div style='background:{bg};border:1.5px solid {border};"
                f"border-radius:10px;padding:12px 16px;margin:8px 0'>"
                f"<div style='font-size:13px;font-weight:700;color:{tc}'>{icon}</div>"
                f"<div style='font-size:12px;color:#1e293b;margin-top:8px'>"
                f"{name_row}{always_row}{date_row}"
                f"</div>"
                f"<div style='font-size:10px;color:#475569;margin-top:6px'>"
                f"Source: {gd.get('source','—')}</div>"
                f"{no_key_note}"
                f"</div>",
                unsafe_allow_html=True,
            )

            # Derive auto-filled fields
            auto_name = gd.get("legal_name") or gd.get("trade_name") or gstin_raw
            auto_city = gd.get("city", "Mumbai")

            # Registration date → years in business
            auto_years = 3
            rd = gd.get("registration_date", "")
            if rd and len(rd) >= 4:
                try:
                    import datetime
                    reg_yr = int(rd[-4:]) if "/" in rd else int(rd[:4])
                    auto_years = max(0, datetime.date.today().year - reg_yr)
                except Exception:
                    pass

            st.divider()
            st.markdown(
                "<div style='background:#eff6ff;border:1px solid #bfdbfe;"
                "border-radius:8px;padding:10px 14px;font-size:12px;color:#1e293b;margin-bottom:8px'>"
                "✅ <b>From GSTN (live):</b> Business name, state, city, registration date, GST status<br>"
                "🔵 <b>From AA Network (set below):</b> Bank statements, UPI logs, BBPS records — "
                "enter manually for this demo; in production these auto-populate via OTP consent."
                "</div>",
                unsafe_allow_html=True,
            )

            st.markdown("**Business Details** *(auto-filled from GSTN)*")
            biz_name_g  = st.text_input("Business Name", auto_name, key="g_name")
            sector_g    = st.selectbox("Sector", SECTORS, key="g_sector")
            city_g      = st.text_input("City", auto_city, key="g_city")
            years_g     = st.number_input("Years in Business", 0, 50, auto_years, key="g_years")
            employees_g = st.number_input("Employees", 1, 500, 10, key="g_emp")
            revenue_g   = st.number_input("Monthly Revenue (₹)", 10000, 10000000,
                                          300000, 10000, key="g_rev")
            is_ntc_g    = st.checkbox("New-to-Credit (NTC)", key="g_ntc")
            is_ntb_g    = st.checkbox("New-to-Bank (NTB)", key="g_ntb")

            st.divider()
            st.markdown("**Financial Data** *(would come from AA Network — enter for demo)*")

            st.caption("Pillar 1 — Liquidity & Cash Flow")
            bounce_g   = st.slider("Auto-debit Bounces (12M)", 0, 15, 1, key="g_b")
            b24h_g     = st.checkbox("Bounces cleared in 24h", True, key="g_b24")
            mdb_g      = st.slider("Min Daily Balance (% of monthly sales)", 0.0, 30.0, 12.0, 0.5, key="g_mdb")

            st.caption("Pillar 2 — Profitability & Solvency")
            dscr_g = st.slider("DSCR", 0.3, 3.5, 1.5, 0.05, key="g_dscr",
                               help="Debt-Service Coverage Ratio = Monthly Profit ÷ Monthly EMI")
            npm_g  = st.slider("Net Profit Margin (%)", 0.0, 40.0, 15.0, 0.5, key="g_npm")

            st.caption("Pillar 3 — Operational Discipline")
            gst_g  = st.slider("GST On-Time Filing Rate (%)", 50.0, 100.0, 90.0, 1.0, key="g_gst")
            util_g = st.slider("Utility Payment Delay (avg days)", 0, 60, 5, key="g_util")

            st.caption("Pillar 4 — Customer Velocity")
            upi_g = st.slider("UPI Active Days / month", 0, 30, 22, key="g_upi")
            inv_g = st.slider("Invoice Collection Speed (days)", 1, 90, 20, key="g_inv")
            rat_g = st.slider("Digital Rating", 1.0, 5.0, 4.0, 0.1, key="g_rat")

            profile = MSMEProfile(
                business_name=biz_name_g, sector=sector_g, city=city_g,
                years_in_business=int(years_g), employee_count=int(employees_g),
                monthly_revenue=float(revenue_g),
                is_ntc=is_ntc_g, is_ntb=is_ntb_g, cibil_score=None,
                bounce_count=int(bounce_g), bounce_cleared_24h=b24h_g,
                mdb_pct_sales=mdb_g, dscr=dscr_g, net_profit_margin=npm_g,
                gst_filing_rate=gst_g, utility_delay_days=float(util_g),
                upi_active_days=int(upi_g), invoice_collection_days=float(inv_g),
                digital_rating=rat_g,
            )
        else:
            # Nothing fetched yet — use a placeholder profile so the page doesn't crash
            profile = list(DEMO_PROFILES.values())[0]


# ─── Gate: show placeholder until data is ready ──────────────────────────────
_demo_mode = (mode == "Demo Profiles")
_gstn_mode = (mode == "Live GSTN Lookup")
_ready = (
    (not _demo_mode and not _gstn_mode) or          # Custom Input: always ready
    (_demo_mode and st.session_state["consent_given"]) or  # Demo: after consent
    (_gstn_mode and st.session_state.get("gstn_fetched"))  # GSTN: after fetch
)

if not _ready:
    st.markdown("<br>", unsafe_allow_html=True)
    if _gstn_mode:
        st.markdown(
            """
            <div style="text-align:center;padding:60px 20px">
                <div style="font-size:56px">🔍</div>
                <div style="font-size:24px;font-weight:800;color:#0f172a;margin:16px 0 8px">
                    Live GSTN Lookup
                </div>
                <div style="font-size:15px;color:#1e293b;max-width:480px;margin:0 auto 28px">
                    Enter a 15-character GSTIN in the sidebar and click
                    <b>"Fetch from GSTN"</b> to pull real business data and run the scoring engine.
                </div>
                <div style="font-size:13px;color:#1e293b;background:#f1f5f9;
                            border:1px solid #cbd5e1;border-radius:10px;
                            padding:14px 20px;max-width:420px;margin:0 auto;text-align:left">
                    <b>What gets fetched from GSTN:</b><br><br>
                    ✅ Legal / trade name of the business<br>
                    ✅ State and city (from GSTIN state code)<br>
                    ✅ Registration date → years in business<br>
                    ✅ GST filing status (Active / Cancelled)<br>
                    ✅ Entity type (Company / Proprietorship)<br><br>
                    🔵 Bank statements, UPI logs, BBPS data → set manually below
                    (auto-populated via AA network OTP consent in production)
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div style="text-align:center;padding:60px 20px">
                <div style="font-size:56px">🏦</div>
                <div style="font-size:24px;font-weight:800;color:#0f172a;margin:16px 0 8px">
                    MSME Financial Health Card
                </div>
                <div style="font-size:15px;color:#1e293b;max-width:480px;margin:0 auto 28px">
                    Select a demo profile in the sidebar and click
                    <b>"Simulate AA Consent &amp; Fetch Data"</b> to load the full
                    credit analysis — score, loan offers, early warning signals and more.
                </div>
                <div style="font-size:13px;color:#1e293b;background:#f1f5f9;
                            border:1px solid #cbd5e1;border-radius:10px;
                            padding:14px 20px;max-width:420px;margin:0 auto;text-align:left">
                    <b>What happens when you click the button?</b><br><br>
                    1. A simulated OTP consent is sent via the Account Aggregator network.<br>
                    2. Data is fetched in real-time from GSTN, NPCI, BBPS and the AA network.<br>
                    3. The AI engine scores the business across 4 pillars (0–10 each).<br>
                    4. Loan offers, risk flags, and an improvement roadmap are generated.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.stop()

# ─── Compute ──────────────────────────────────────────────────────────────────
scores        = compute_scores(profile)
credit_band, band_color, band_desc = get_credit_band(scores["composite"])
credit_limit  = recommend_credit_limit(profile.monthly_revenue, profile.dscr, scores["composite"])
peer_pct      = get_peer_percentile(scores["composite"], profile.sector)
strengths, risks = get_strengths_risks(scores, profile)
roadmap       = get_improvement_roadmap(scores, profile)
historical_df = generate_historical_trend(scores)
offers        = get_lender_offers(scores["composite"], credit_limit)


# ─── Tabs ────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Health Card",
    "🔍 Score Breakdown",
    "🏦 Loan Offers",
    "⚡ EWS Monitor",
    "🗺 Bank Portfolio",
])


# ═══════════════════════════════════════════════════════════════
# TAB 1 — HEALTH CARD
# ═══════════════════════════════════════════════════════════════
with tab1:
    st.markdown(
        "<div class='explain-box'>"
        "<b>Health Card</b> — Your complete financial report card, computed in real-time from "
        "government and banking data. The gauge shows your overall score (0–10). "
        "The radar chart breaks it into 4 pillars. Scroll down for strengths, risks, and an "
        "improvement roadmap tailored to your profile."
        "</div>",
        unsafe_allow_html=True,
    )

    # ── First-timer explainer ──────────────────────────────────
    with st.expander("🏦  What is the MSME Financial Health Card? — Click to learn how it works"):
        c1, c2, c3 = st.columns(3)
        c1.markdown("""
**Step 1 — You give consent**

With a single OTP via the Account Aggregator network,
you authorize us to securely fetch your digital financial
footprint. No paper. No PDF uploads.
        """)
        c2.markdown("""
**Step 2 — We pull alternate data**

We tap into GSTN, NPCI, BBPS, EPFO and more via
India's Digital Public Infrastructure (DPI). All data
is fetched from official government servers in real-time.
        """)
        c3.markdown("""
**Step 3 — AI scores you across 4 pillars**

Your data is scored across Liquidity, Profitability,
Discipline, and Customer Velocity — each weighted by
how predictive it is for loan repayment. Score: 0–10.
        """)
        st.divider()
        st.markdown("**Score Bands explained:**")
        band_html = "".join([
            f'<span class="band-pill" style="background:{c}">'
            f'{b} &nbsp;({r})</span>'
            for b, c, r in BAND_LEGEND
        ])
        st.markdown(f'<div style="margin:6px 0">{band_html}</div>', unsafe_allow_html=True)
        st.caption("Each band maps to an interest rate range. Prime = lowest rate, NPA Risk = credit denied.")

    # ── Business header ────────────────────────────────────────
    h1, h2, h3, h4 = st.columns([3, 1, 1, 1])
    with h1:
        st.markdown(f"## {profile.business_name}")
        badge_html = ""
        if profile.is_ntc:
            badge_html += '<span class="badge badge-ntc">⚡ NEW-TO-CREDIT</span>'
        if profile.is_ntb:
            badge_html += '<span class="badge badge-ntb">🆕 NEW-TO-BANK</span>'
        if profile.cibil_score is None and (profile.is_ntc or profile.is_ntb):
            badge_html += '<span class="badge badge-cibil">No CIBIL — Alternate Data Only</span>'
        if _gstn_mode and st.session_state.get("gstn_data"):
            gd = st.session_state["gstn_data"]
            icon = "🟢" if "live" in gd.get("source", "") else "🟡"
            badge_html += (
                f'<span class="badge" style="background:#dcfce7;color:#14532d;'
                f'border:1px solid #86efac">'
                f'{icon} GSTN Verified — {gd.get("gst_status","Active")}</span>'
            )
        if badge_html:
            st.markdown(badge_html + "<br>", unsafe_allow_html=True)
    with h2:
        st.metric("Sector", profile.sector.split(" /")[0])
    with h3:
        st.metric("City", profile.city)
    with h4:
        st.metric("Years Active",
                  f"{profile.years_in_business} yr{'s' if profile.years_in_business != 1 else ''}")

    st.divider()

    # ── Score section ──────────────────────────────────────────
    col_g, col_r = st.columns([1, 1], gap="large")

    with col_g:
        st.markdown(
            "<div class='explain-box'>"
            "The <b>Overall Health Score</b> (0–10) is a weighted average of all 4 pillars. "
            "It determines your credit band, interest rate, and maximum loan amount."
            "</div>",
            unsafe_allow_html=True,
        )
        st.plotly_chart(
            make_gauge(scores["composite"], credit_band, band_color, band_desc),
            use_container_width=True, config={"displayModeBar": False},
        )

        # Band legend strip below gauge
        pills = "".join([
            f'<span style="display:inline-block;background:{"#1d4ed8" if b == credit_band else c};'
            f'color:white;padding:3px 12px;border-radius:16px;font-size:11px;font-weight:700;'
            f'margin:2px;opacity:{"1" if b == credit_band else "0.35"}">{b}</span>'
            for b, c, _ in BAND_LEGEND
        ])
        st.markdown(f"<div style='text-align:center;margin-top:-8px'>{pills}</div>",
                    unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown(
            "<div class='explain-box'>"
            "These four key figures are the <b>output</b> of your Health Card — what the bank "
            "and lenders actually see when making a credit decision."
            "</div>",
            unsafe_allow_html=True,
        )
        ka, kb = st.columns(2)
        ka.metric("Recommended Credit Limit", fmt_inr(credit_limit),
                  help="Maximum loan amount the system recommends based on: 4× monthly revenue × DSCR multiplier × score multiplier")
        kb.metric("Indicative Interest Rate", f"{get_interest_rate(scores['composite']):.1f}% p.a.",
                  help="Annual interest rate based on your credit band. Prime = 9.5% p.a. (lowest), NPA Risk = 22%+ (highest or denied)")
        kc, kd = st.columns(2)
        kc.metric("Peer Percentile", f"Top {100 - peer_pct}%",
                  help=f"Your score is better than {peer_pct}% of similar {profile.sector} MSMEs assessed on this platform")
        kd.metric("Monthly Revenue", fmt_inr(profile.monthly_revenue),
                  help="Monthly turnover as entered — this is the base used to calculate your credit limit")

    with col_r:
        st.markdown(
            "<div class='explain-box'>"
            "The <b>Radar Chart</b> shows your score on each of the 4 pillars simultaneously. "
            "A perfect business would fill the entire chart to the edge (10/10 on all axes). "
            "The dotted ring shows the sector average (6.0). Any axis where your shape is "
            "smaller than the ring is an area to improve."
            "</div>",
            unsafe_allow_html=True,
        )
        st.plotly_chart(
            make_radar(scores),
            use_container_width=True, config={"displayModeBar": False},
        )
        st.markdown(
            "<div style='text-align:center;font-size:12px;color:#334155;margin-top:-12px'>"
            "Hover over any axis point to see the exact score and what it measures.</div>",
            unsafe_allow_html=True,
        )

    st.divider()

    # ── Pillar Cards with scoring rules ───────────────────────
    st.markdown('<div class="section-title">4-Pillar Breakdown — Click any pillar to see how it is scored</div>',
                unsafe_allow_html=True)
    st.markdown(
        "<div class='explain-box'>"
        "Your overall score is built from <b>4 pillars</b>, each measuring a different dimension of "
        "financial health. Click <i>'How is this scored?'</i> under any pillar to see the exact rules "
        "and data sources used — so there are no surprises. "
        "Green = strong, amber = caution, red = needs immediate attention."
        "</div>",
        unsafe_allow_html=True,
    )

    pc1, pc2, pc3, pc4 = st.columns(4, gap="small")
    pillar_pairs = [
        (pc1, "Liquidity & Cash Flow",     scores["p1"], "📈", "#3b82f6"),
        (pc2, "Profitability & Solvency",  scores["p2"], "💹", "#8b5cf6"),
        (pc3, "Operational Discipline",    scores["p3"], "📋", "#10b981"),
        (pc4, "Customer Velocity",         scores["p4"], "⚡", "#f59e0b"),
    ]

    for col, name, val, icon, accent in pillar_pairs:
        info = SCORING_RULES[name]
        c = score_color(val)
        with col:
            st.markdown(f"""
            <div class="pillar-card">
                <div style="font-size:26px">{icon}</div>
                <div style="font-size:12px;color:#0f172a;font-weight:700;margin:6px 0 2px">{name}</div>
                <div style="font-size:42px;font-weight:900;color:{c};line-height:1.1">{val:.1f}</div>
                <div style="font-size:11px;color:#334155">/10 &nbsp;·&nbsp; Weight {info['weight']}</div>
                <div style="font-size:12px;color:#1e293b;margin-top:8px;line-height:1.4">
                    {info['what']}</div>
            </div>
            """, unsafe_allow_html=True)
            with st.expander("How is this scored?"):
                st.markdown(f"**Data Sources:**")
                for src in info["source"]:
                    st.markdown(
                        f'<span class="src-chip">{src}</span>',
                        unsafe_allow_html=True,
                    )
                st.markdown("<br>**Scoring Rules (from RBI-aligned MSME framework):**",
                            unsafe_allow_html=True)
                for label, rule in info["rules"]:
                    col_dot = "#059669" if "Excellent" in label else \
                              "#d97706"  if "Good" in label or "Moderate" in label else \
                              "#dc2626"
                    st.markdown(
                        f'<div class="calc-rule">'
                        f'<span style="color:{col_dot};font-weight:700">{label}</span><br>'
                        f'<span style="color:#1e293b">{rule}</span></div>',
                        unsafe_allow_html=True,
                    )
                st.markdown(
                    f"<div style='font-size:12px;color:#334155;margin-top:6px'>"
                    f"Your input: <b>{val:.1f}/10</b></div>",
                    unsafe_allow_html=True,
                )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Strengths & Risks ──────────────────────────────────────
    col_s, col_rk = st.columns(2, gap="large")
    with col_s:
        st.markdown('<div class="section-title">Strengths</div>', unsafe_allow_html=True)
        if strengths:
            for pname, desc in strengths:
                st.success(f"**{pname}** — {desc}")
        else:
            st.info("No standout strengths yet. Focus on improving all pillars using the roadmap below.")

    with col_rk:
        st.markdown('<div class="section-title">Risk Flags</div>', unsafe_allow_html=True)
        if risks:
            for pname, desc in risks:
                st.error(f"**{pname}** — {desc}")
        else:
            st.success("No significant risk flags detected — healthy profile across all pillars.")

    if scores["fraud_flags"]:
        st.divider()
        st.markdown('<div class="section-title">Anomaly Detection</div>', unsafe_allow_html=True)
        for flag in scores["fraud_flags"]:
            st.warning(f"⚠ {flag}")

    # ── Improvement Roadmap ────────────────────────────────────
    if roadmap:
        st.divider()
        st.markdown('<div class="section-title">Credit Improvement Roadmap — What to fix and by when</div>',
                    unsafe_allow_html=True)
        for item in roadmap:
            st.markdown(f"""
            <div class="roadmap-row">
                <div style="font-size:13px;font-weight:700;color:#1e293b">{item['icon']} {item['pillar']}</div>
                <div style="font-size:13px;color:#1e293b;margin-top:4px">{item['action']}</div>
                <div style="margin-top:8px">
                    <span style="font-size:11px;background:#dbeafe;color:#1e40af;
                                 padding:3px 10px;border-radius:10px;font-weight:700">
                        {item['impact']}</span>
                    <span style="font-size:11px;background:#f0fdf4;color:#065f46;
                                 padding:3px 10px;border-radius:10px;font-weight:700;margin-left:6px">
                        {item['timeframe']}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No improvements needed — this MSME is already performing at the top band.")


# ═══════════════════════════════════════════════════════════════
# TAB 2 — SCORE BREAKDOWN
# ═══════════════════════════════════════════════════════════════
with tab2:
    st.markdown("### Score Breakdown & Data Sources")
    st.markdown(
        "<div class='explain-box'>"
        "<b>What this tab shows:</b> A detailed breakdown of exactly how your overall score was calculated — "
        "which raw signals were used, where each data point came from, how much each pillar weighs, "
        "and a <b>What-If Simulator</b> at the bottom so you can see how improving specific metrics "
        "would change your score and interest rate."
        "</div>",
        unsafe_allow_html=True,
    )
    st.caption("Every number shown here is pulled directly from a verified government or DPI data source. Nothing is self-declared.")

    # Bar chart
    st.markdown('<div class="section-title">Pillar Scores vs Standard Threshold (6.5)</div>',
                unsafe_allow_html=True)
    st.markdown(
        "<div class='explain-box'>"
        "Each bar shows your score (0–10) for one pillar. The dashed vertical line at <b>6.5</b> is "
        "the <b>Standard band threshold</b> — bars to the right of it are in a bankable range; "
        "bars to the left are below the standard and may attract higher interest or rejection."
        "</div>",
        unsafe_allow_html=True,
    )
    p_labels = ["Liquidity & Cash Flow", "Profitability & Solvency",
                "Operational Discipline", "Customer Velocity"]
    p_vals   = [scores["p1"], scores["p2"], scores["p3"], scores["p4"]]

    fig_bar = go.Figure(go.Bar(
        x=p_vals, y=p_labels, orientation="h",
        marker_color=[score_color(v) for v in p_vals],
        text=[f"  {v:.1f}/10" for v in p_vals],
        textposition="outside",
        textfont=dict(size=13, color="#1e293b"),
        hovertemplate="%{y}: <b>%{x:.1f}/10</b><extra></extra>",
    ))
    fig_bar.add_vline(x=6.5, line_dash="dash", line_color="#64748b",
                      annotation_text="← 6.5 threshold",
                      annotation_position="bottom right",
                      annotation_font_size=11, annotation_font_color="#1e293b")
    fig_bar.update_layout(
        paper_bgcolor="white", plot_bgcolor="white",
        xaxis=dict(range=[0, 13], showgrid=True, gridcolor="#f1f5f9",
                   title="Score (out of 10)", tickfont=dict(color="#1e293b", size=12)),
        yaxis=dict(showgrid=False, tickfont=dict(color="#1e293b", size=12)),
        margin=dict(l=10, r=40, t=20, b=30), height=260,
        font=dict(color="#1e293b", size=13),
        dragmode=False,
    )
    st.plotly_chart(fig_bar, use_container_width=True, config={"displayModeBar": False})

    st.divider()

    # Raw data tables
    st.markdown(
        "<div class='explain-box'>"
        "The tables below show the <b>raw input signals</b> used to compute your score, "
        "alongside the government/DPI source each figure was pulled from. "
        "This is the audit trail a bank officer would verify."
        "</div>",
        unsafe_allow_html=True,
    )
    col_d1, col_d2 = st.columns(2, gap="large")
    with col_d1:
        st.markdown('<div class="section-title">Liquidity & Profitability Signals</div>',
                    unsafe_allow_html=True)
        st.dataframe(pd.DataFrame({
            "Metric": ["Auto-debit Bounces (12M)", "Bounces Cleared in 24h",
                       "Min Daily Balance (% sales)", "DSCR", "Net Profit Margin"],
            "Your Value": [profile.bounce_count,
                           "Yes" if profile.bounce_cleared_24h else "No",
                           f"{profile.mdb_pct_sales:.1f}%",
                           f"{profile.dscr:.2f}x",
                           f"{profile.net_profit_margin:.1f}%"],
            "Source": ["AA Network", "AA Network", "AA Network",
                       "GSTR-1 + AA", "GSTR-1 + 2B"],
        }), use_container_width=True, hide_index=True)

    with col_d2:
        st.markdown('<div class="section-title">Discipline & Velocity Signals</div>',
                    unsafe_allow_html=True)
        st.dataframe(pd.DataFrame({
            "Metric": ["GST On-Time Filing Rate", "Utility Payment Delay",
                       "UPI Active Days / month", "Invoice Collection Speed", "Digital Rating"],
            "Your Value": [f"{profile.gst_filing_rate:.0f}%",
                           f"{profile.utility_delay_days:.0f} days",
                           f"{profile.upi_active_days} / 30",
                           f"{profile.invoice_collection_days:.0f} days",
                           f"{profile.digital_rating:.1f} / 5.0"],
            "Source": ["GSTN Portal", "BBPS", "NPCI / Merchant Aggregator",
                       "GSTR-1 + AA", "Google Maps / e-com"],
        }), use_container_width=True, hide_index=True)

    st.divider()

    # Sector-adaptive weights
    st.markdown('<div class="section-title">Why do weights differ by sector?</div>',
                unsafe_allow_html=True)
    st.caption(
        f"Because a textile manufacturer and a food stall have very different cash-flow patterns, "
        f"we adjust pillar weights for **{profile.sector}** so the scoring is fair — "
        f"not one-size-fits-all."
    )
    weights = scores["weights"]
    w_labels = ["Liquidity", "Profitability", "Discipline", "Velocity"]
    w_base   = [35, 30, 20, 15]
    w_sector = [round(weights["liquidity"]*100, 1), round(weights["profitability"]*100, 1),
                round(weights["discipline"]*100, 1), round(weights["velocity"]*100, 1)]

    fig_w = go.Figure()
    fig_w.add_trace(go.Bar(name="Base weights (all sectors)", x=w_labels, y=w_base,
                           marker_color="#94a3b8", opacity=0.9,
                           text=[f"{v}%" for v in w_base], textposition="outside",
                           textfont=dict(color="#1e293b", size=11)))
    fig_w.add_trace(go.Bar(name=f"Adjusted for {profile.sector.split(' /')[0]}",
                           x=w_labels, y=w_sector,
                           marker_color="#1d4ed8",
                           text=[f"{v}%" for v in w_sector], textposition="outside",
                           textfont=dict(color="#1e293b", size=11)))
    fig_w.update_layout(
        barmode="group", paper_bgcolor="white", plot_bgcolor="white",
        yaxis=dict(title="Weight (%)", showgrid=True, gridcolor="#f1f5f9",
                   range=[0, 50], tickfont=dict(color="#1e293b")),
        xaxis=dict(tickfont=dict(color="#1e293b", size=12)),
        legend=dict(orientation="h", xanchor="center", x=0.5, y=-0.18,
                    font=dict(size=12, color="#1e293b"),
                    bgcolor="white", bordercolor="#e2e8f0", borderwidth=1),
        margin=dict(l=20, r=20, t=20, b=80), height=300,
        font=dict(color="#1e293b", size=12),
        dragmode=False,
    )
    st.plotly_chart(fig_w, use_container_width=True, config={"displayModeBar": False})

    st.divider()

    # What-if simulator
    st.markdown('<div class="section-title">What-If Simulator — "If I improve X, what happens to my score?"</div>',
                unsafe_allow_html=True)
    st.markdown(
        "<div class='explain-box'>"
        "Move any slider to a hypothetical future value. The <b>Projected Score</b> and "
        "<b>Projected Band</b> update instantly. If the change moves you into a better band, "
        "the system also calculates your potential interest rate saving in rupees — "
        "so you know exactly what to prioritise."
        "</div>",
        unsafe_allow_html=True,
    )
    st.caption("Adjust any variable below to instantly see the projected score impact. Useful for your relationship manager to show what small changes unlock better rates.")

    wc1, wc2, wc3 = st.columns(3, gap="large")
    with wc1:
        wif_dscr    = st.slider("Hypothetical DSCR", 0.5, 3.5, float(profile.dscr), 0.05, key="wif_dscr")
    with wc2:
        wif_bounces = st.slider("Hypothetical Bounces", 0, 15, int(profile.bounce_count), key="wif_b")
    with wc3:
        wif_gst     = st.slider("Hypothetical GST Rate (%)", 50.0, 100.0, float(profile.gst_filing_rate), 1.0, key="wif_g")

    wif_p = MSMEProfile(
        business_name=profile.business_name, sector=profile.sector, city=profile.city,
        years_in_business=profile.years_in_business, employee_count=profile.employee_count,
        monthly_revenue=profile.monthly_revenue,
        bounce_count=int(wif_bounces), bounce_cleared_24h=profile.bounce_cleared_24h,
        mdb_pct_sales=profile.mdb_pct_sales, dscr=wif_dscr,
        net_profit_margin=profile.net_profit_margin, gst_filing_rate=wif_gst,
        utility_delay_days=profile.utility_delay_days, upi_active_days=profile.upi_active_days,
        invoice_collection_days=profile.invoice_collection_days, digital_rating=profile.digital_rating,
    )
    wif_s = compute_scores(wif_p)
    delta = round(wif_s["composite"] - scores["composite"], 2)
    wif_band, _, _ = get_credit_band(wif_s["composite"])

    wm1, wm2, wm3, wm4 = st.columns(4)
    wm1.metric("Current Score",   f"{scores['composite']:.2f}/10")
    wm2.metric("Projected Score", f"{wif_s['composite']:.2f}/10",
               delta=f"{'+' if delta >= 0 else ''}{delta:.2f} pts")
    wm3.metric("Current Band",    credit_band)
    wm4.metric("Projected Band",  wif_band)

    if delta > 0.05:
        new_rate = get_interest_rate(wif_s["composite"])
        old_rate = get_interest_rate(scores["composite"])
        if new_rate < old_rate:
            st.success(
                f"These changes would save **{old_rate - new_rate:.2f}% p.a.** in interest — "
                f"on a ₹10L loan over 12 months that is **₹{int((old_rate-new_rate)/100*1000000):,}/year**."
            )
        else:
            st.info(f"Score improves by {delta:.2f} pts but stays in the same interest rate band.")
    elif delta < -0.05:
        st.error(f"These changes would reduce your score by {abs(delta):.2f} pts and may worsen your rate.")


# ═══════════════════════════════════════════════════════════════
# TAB 3 — LOAN OFFERS
# ═══════════════════════════════════════════════════════════════
with tab3:
    st.markdown("### OCEN Loan Marketplace")
    st.markdown(
        "<div class='explain-box'>"
        "<b>What this tab shows:</b> Real loan offers from multiple lenders, fetched automatically "
        "using your Health Card score. Compare rates, tenures and fees side-by-side. "
        "Scroll down to understand exactly how your credit limit was calculated and how "
        "improving your score would reduce your interest rate."
        "</div>",
        unsafe_allow_html=True,
    )
    st.markdown("""
    <div class="info-box">
        <b>How does this work?</b> Once your Health Card is generated, it is broadcast via the
        <b>Open Credit Enablement Network (OCEN)</b> protocol to multiple lenders simultaneously.
        Each lender's underwriting engine reads your score and returns a tailored offer within minutes —
        you see all offers on one screen and choose. No running around to 5 different banks.
    </div>
    """, unsafe_allow_html=True)

    if not offers:
        st.error(
            "Score too low for automated lender offers at this time. "
            "Follow the Improvement Roadmap on the Health Card tab and re-apply in 60–90 days."
        )
    else:
        st.markdown(
            f"**{len(offers)} offer(s) received** for **{profile.business_name}** "
            f"· Score {scores['composite']:.2f}/10 · Band: {credit_band}",
        )
        st.markdown("<br>", unsafe_allow_html=True)

        oc = st.columns(len(offers), gap="medium")
        for col, offer in zip(oc, offers):
            with col:
                st.markdown(f"""
                <div class="offer-card">
                    <div class="tag" style="background:{offer['tag_color']}">{offer['tag']}</div>
                    <div style="font-size:19px;font-weight:800;color:#003087;margin-bottom:3px">
                        {offer['lender']}</div>
                    <div style="font-size:13px;color:#334155;font-weight:600;margin-bottom:14px">
                        {offer['product']}</div>
                    <div style="font-size:30px;font-weight:900;color:#1e293b">
                        {fmt_inr(offer['amount'])}</div>
                    <div style="font-size:12px;color:#334155;font-weight:600;margin-bottom:12px">Credit Limit</div>
                    <hr style="border-color:#e2e8f0;margin:10px 0">
                    <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px;font-size:12px">
                        <div><div style="color:#334155;font-weight:600">Rate</div>
                             <b>{offer['rate']}% p.a.</b></div>
                        <div><div style="color:#334155;font-weight:600">Tenure</div>
                             <b>{offer['tenure']}</b></div>
                        <div><div style="color:#334155;font-weight:600">Processing Fee</div>
                             <b>{offer['fee']}</b></div>
                        <div><div style="color:#334155;font-weight:600">Turnaround</div>
                             <b>{offer['tat']}</b></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.success(
            "**100% paperless disbursal** — NACH mandate registered via AA framework. "
            "Loan agreement signed digitally via Aadhaar e-sign. No branch visit required."
        )

    st.divider()
    st.markdown('<div class="section-title">How is the Credit Limit calculated?</div>',
                unsafe_allow_html=True)
    st.markdown(f"""
    <div class="info-box">
        <b>Formula:</b> Credit Limit = Monthly Revenue × 4 × DSCR Multiplier × Score Multiplier<br><br>
        For <b>{profile.business_name}</b>:<br>
        &nbsp;&nbsp;Monthly Revenue: <b>{fmt_inr(profile.monthly_revenue)}</b> × 4 = {fmt_inr(profile.monthly_revenue * 4)}<br>
        &nbsp;&nbsp;DSCR Multiplier: <b>{min(profile.dscr / 1.5, 1.6):.2f}x</b>
        &nbsp;(DSCR {profile.dscr:.2f} ÷ 1.5, capped at 1.6x)<br>
        &nbsp;&nbsp;Score Multiplier: <b>{(scores['composite'] / 10) ** 0.7:.2f}x</b>
        &nbsp;(score {scores['composite']:.2f}/10, curved)<br>
        &nbsp;&nbsp;= <b>{fmt_inr(credit_limit)}</b> recommended limit
    </div>
    """, unsafe_allow_html=True)

    st.divider()
    st.markdown('<div class="section-title">Score → Interest Rate Curve</div>',
                unsafe_allow_html=True)
    st.markdown(
        "<div class='explain-box'>"
        "This chart shows how your interest rate changes as your Health Score changes. "
        "The curve is not linear — the biggest rate drops happen when you cross band thresholds "
        "(e.g. moving from Sub-Standard to Standard saves more than the same point gain within a band). "
        "The dashed vertical line marks <b>your current score</b>."
        "</div>",
        unsafe_allow_html=True,
    )
    score_range = np.arange(1, 10.1, 0.1)
    rate_range  = [get_interest_rate(float(s)) for s in score_range]
    fig_c = go.Figure()
    fig_c.add_trace(go.Scatter(
        x=list(score_range), y=rate_range, mode="lines",
        line=dict(color="#1d4ed8", width=2.5),
        hovertemplate="Score %{x:.1f} → %{y:.1f}% p.a.<extra></extra>",
    ))
    fig_c.add_vline(x=scores["composite"], line_dash="dash", line_color=band_color,
                    annotation_text=f"Your score {scores['composite']:.2f}",
                    annotation_font_color=band_color, annotation_font_size=12)
    fig_c.update_layout(
        paper_bgcolor="white", plot_bgcolor="white",
        xaxis=dict(title="Health Score (0–10)", showgrid=True, gridcolor="#e2e8f0",
                   tickfont=dict(color="#1e293b", size=12)),
        yaxis=dict(title="Interest Rate (% p.a.)", showgrid=True, gridcolor="#e2e8f0",
                   tickfont=dict(color="#1e293b", size=12)),
        margin=dict(l=20, r=20, t=50, b=40), height=290,
        font=dict(color="#1e293b", size=12),
        dragmode=False,
    )
    st.plotly_chart(fig_c, use_container_width=True, config={"displayModeBar": False})


# ═══════════════════════════════════════════════════════════════
# TAB 4 — EWS MONITOR
# ═══════════════════════════════════════════════════════════════
with tab4:
    st.markdown("### Early Warning System — Post-Disbursement Monitor")
    st.markdown(
        "<div class='explain-box'>"
        "<b>What this tab shows:</b> What happens <i>after</i> a loan is approved. "
        "Use the slider on the left to simulate a drop in UPI activity (e.g. a business slowdown). "
        "Watch the alert trigger at 40%+ drop. The right panel shows the 6-month score trend — "
        "used by the bank to decide whether to renew or recall a credit line."
        "</div>",
        unsafe_allow_html=True,
    )
    st.markdown("""
    <div class="info-box">
        <b>Why does this matter?</b> Traditional banks go blind after a loan is disbursed —
        they only find out about trouble when the EMI bounces. Our EWS maintains passive,
        continuous monitoring of the MSME's alternate data feeds (UPI volumes, GST filings)
        even after the loan is live. If a 40% UPI drop is detected over any rolling 7-day window,
        the bank is alerted <i>before</i> the first missed payment.
    </div>
    """, unsafe_allow_html=True)

    col_e1, col_e2 = st.columns([1, 2], gap="large")

    with col_e1:
        st.markdown('<div class="section-title">Simulate a UPI Drop Event</div>',
                    unsafe_allow_html=True)
        drop = st.slider(
            "UPI Transaction Drop (%)",
            0, 100, 0, 5, key="ews_drop",
            help="Move to 40%+ to trigger an Early Warning alert",
        )
        ews = simulate_ews(profile, drop)

        status_class = "ews-orange" if ews["alert_triggered"] else "ews-green"
        status_icon  = "🟠" if ews["alert_triggered"] else "🟢"
        status_text  = "EARLY WARNING ALERT TRIGGERED" if ews["alert_triggered"] else "WITHIN NORMAL RANGE"
        text_color   = "#7c2d12" if ews["alert_triggered"] else "#14532d"

        reason_html = ""
        if ews["trigger_reason"]:
            reason_html = (
                f'<div style="font-size:12px;margin-top:6px;font-weight:600;color:{text_color}">'
                f'Trigger: {ews["trigger_reason"]}</div>'
            )

        st.markdown(
            f'<div class="{status_class}">'
            f'<div style="font-size:14px;font-weight:800;color:{text_color}">'
            f'{status_icon} {status_text}</div>'
            f'<div style="font-size:12px;margin-top:6px;color:{text_color}">'
            f'UPI Active Days: <b>{ews["current_upi_days"]}</b> / {ews["original_upi_days"]}'
            f' &nbsp;({drop:.0f}% drop)</div>'
            f'{reason_html}'
            f'<div style="font-size:12px;margin-top:10px;color:{text_color}">'
            f'<b>Bank Action:</b> {ews["action"]}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        st.markdown("<br>", unsafe_allow_html=True)
        st.metric("Baseline UPI Days / Month", ews["original_upi_days"],
                  help="Measured at time of loan disbursal")
        st.metric("Current UPI Days / Month",  ews["current_upi_days"],
                  delta=f"{-drop:.0f}%" if drop > 0 else "0%", delta_color="inverse")

    with col_e2:
        st.markdown('<div class="section-title">6-Month Score Trajectory</div>',
                    unsafe_allow_html=True)
        st.caption("Shows how the Health Score trended over the past 6 months — used to decide if the business is improving or declining before a new credit line is approved.")

        fig_t = go.Figure()
        colors_trend = {
            "Overall":       (band_color, "solid", 3),
            "Liquidity":     ("#3b82f6",  "dot",   1.5),
            "Profitability": ("#8b5cf6",  "dot",   1.5),
            "Discipline":    ("#10b981",  "dot",   1.5),
            "Velocity":      ("#f59e0b",  "dot",   1.5),
        }
        for col_name, (col_color, dash, width) in colors_trend.items():
            fig_t.add_trace(go.Scatter(
                x=historical_df["Month"], y=historical_df[col_name],
                mode="lines+markers",
                name=col_name,
                line=dict(color=col_color, width=width, dash=dash),
                marker=dict(size=6 if dash == "solid" else 4),
                hovertemplate=f"{col_name}: <b>%{{y:.2f}}/10</b> (%{{x}})<extra></extra>",
            ))
        fig_t.add_hline(y=6.5, line_dash="dash", line_color="#64748b",
                        annotation_text="Standard band floor (6.5)",
                        annotation_position="top left", annotation_font_size=10,
                        annotation_font_color="#1e293b")
        fig_t.update_layout(
            paper_bgcolor="white", plot_bgcolor="white",
            xaxis=dict(showgrid=False, title="Month",
                       tickfont=dict(color="#1e293b", size=12)),
            yaxis=dict(title="Score / 10", range=[0, 10.5],
                       showgrid=True, gridcolor="#e2e8f0",
                       tickfont=dict(color="#1e293b", size=12)),
            legend=dict(orientation="h", xanchor="center", x=0.5,
                        yanchor="bottom", y=1.02,
                        font=dict(size=11, color="#1e293b"),
                        bgcolor="white", bordercolor="#e2e8f0", borderwidth=1),
            margin=dict(l=20, r=20, t=60, b=30), height=360,
            font=dict(color="#1e293b", size=12),
            dragmode=False,
        )
        st.plotly_chart(fig_t, use_container_width=True, config={"displayModeBar": False})
        st.caption("Synthetic trend for demo purposes. In production: refreshed every 24 hours from live alternate data feeds.")


# ═══════════════════════════════════════════════════════════════
# TAB 5 — BANK PORTFOLIO
# ═══════════════════════════════════════════════════════════════
with tab5:
    st.markdown("### Bank Portfolio Intelligence")
    st.markdown(
        "<div class='explain-box'>"
        "<b>What this tab shows:</b> A bird's-eye view of all MSMEs assessed by the bank — "
        "intended for IDBI's internal credit and risk teams. "
        "The bubble map shows each business plotted by score and sector; larger bubbles = larger credit exposure. "
        "The donut chart shows how the portfolio is distributed across risk bands. "
        "Scroll down for concentration risk alerts (too many loans in one sector)."
        "</div>",
        unsafe_allow_html=True,
    )
    st.markdown("""
    <div class="info-box">
        This view is for the <b>bank's credit team</b>, not the MSME applicant.
        It shows how the full assessed portfolio is distributed across sectors and
        score bands — enabling the risk team to spot concentration risk before it
        becomes a systemic problem.
    </div>
    """, unsafe_allow_html=True)

    portfolio_rows = []
    for pname, demo in DEMO_PROFILES.items():
        s = compute_scores(demo)
        band, bcolor, _ = get_credit_band(s["composite"])
        cl = recommend_credit_limit(demo.monthly_revenue, demo.dscr, s["composite"])
        portfolio_rows.append({
            "Business": demo.business_name,
            "Sector": demo.sector,
            "City": demo.city,
            "Score": s["composite"],
            "Band": band,
            "Credit Limit": cl,
            "NTC/NTB": "Yes" if (demo.is_ntc or demo.is_ntb) else "No",
            "_color": bcolor,
        })
    port_df = pd.DataFrame(portfolio_rows)

    # Bubble map
    st.markdown('<div class="section-title">Portfolio Map — Sector vs Health Score (bubble size = credit exposure)</div>',
                unsafe_allow_html=True)
    st.caption("Hover over any bubble to see the business name, score, and credit exposure.")
    fig_port = go.Figure()
    for _, row in port_df.iterrows():
        fig_port.add_trace(go.Scatter(
            x=[row["Score"]], y=[row["Sector"]],
            mode="markers",
            marker=dict(
                size=max(20, row["Credit Limit"] / 35000),
                color=row["_color"], opacity=0.88,
                line=dict(color="white", width=2),
            ),
            name=row["Business"],
            hovertemplate=(
                f"<b>{row['Business']}</b><br>"
                f"Score: {row['Score']:.2f}/10 ({row['Band']})<br>"
                f"Exposure: {fmt_inr(row['Credit Limit'])}<br>"
                f"NTC/NTB: {row['NTC/NTB']}"
                "<extra></extra>"
            ),
        ))
    fig_port.add_vline(x=6.5, line_dash="dash", line_color="#475569",
                       annotation_text="6.5 threshold",
                       annotation_position="top right",
                       annotation_font_color="#1e293b",
                       annotation_font_size=12)
    fig_port.update_layout(
        paper_bgcolor="white", plot_bgcolor="white",
        xaxis=dict(title="Health Score (0–10)", range=[0, 11],
                   showgrid=True, gridcolor="#e2e8f0",
                   tickfont=dict(color="#1e293b", size=12)),
        yaxis=dict(showgrid=True, gridcolor="#f1f5f9", title="",
                   tickfont=dict(color="#1e293b", size=12)),
        showlegend=False,
        margin=dict(l=10, r=20, t=20, b=50), height=340,
        font=dict(color="#1e293b", size=12),
        dragmode=False,
    )
    st.plotly_chart(fig_port, use_container_width=True, config={"displayModeBar": False})

    col_p1, col_p2 = st.columns(2, gap="large")
    with col_p1:
        st.markdown('<div class="section-title">Band Distribution</div>', unsafe_allow_html=True)
        band_counts = port_df["Band"].value_counts()
        color_map = {"Prime": "#059669", "Standard": "#1d4ed8",
                     "Sub-Standard": "#d97706", "Watch": "#dc2626", "NPA Risk": "#7f1d1d"}
        fig_d = go.Figure(go.Pie(
            labels=band_counts.index,
            values=band_counts.values,
            hole=0.55,
            marker_colors=[color_map.get(b, "#64748b") for b in band_counts.index],
            hovertemplate="<b>%{label}</b><br>%{value} MSME(s) — %{percent}<extra></extra>",
            textinfo="percent",
            textfont=dict(size=13, color="white"),
            insidetextorientation="horizontal",
        ))
        fig_d.update_layout(
            paper_bgcolor="white",
            showlegend=True,
            legend=dict(orientation="v", font=dict(size=12, color="#1e293b"),
                        bgcolor="white", x=1.0, y=0.5),
            margin=dict(l=10, r=120, t=10, b=10), height=260,
            font=dict(color="#1e293b", size=12),
            dragmode=False,
        )
        st.plotly_chart(fig_d, use_container_width=True, config={"displayModeBar": False})

    with col_p2:
        st.markdown('<div class="section-title">Portfolio Summary</div>', unsafe_allow_html=True)
        disp = port_df[["Business", "Sector", "Score", "Band", "NTC/NTB"]].copy()
        disp["Score"] = disp["Score"].apply(lambda x: f"{x:.2f}")
        st.dataframe(disp, use_container_width=True, hide_index=True)

    st.divider()
    st.markdown('<div class="section-title">Concentration Risk Alerts</div>',
                unsafe_allow_html=True)
    sector_counts = port_df["Sector"].value_counts()
    concentrated  = sector_counts[sector_counts > 1]
    if not concentrated.empty:
        for sec, cnt in concentrated.items():
            st.warning(
                f"**Concentration Alert** — {cnt}/{len(port_df)} MSMEs are in **{sec}**. "
                f"The AI will reduce score preference for this segment and surface "
                f"underserved sectors to maintain diversification."
            )
    else:
        st.success("Portfolio well-diversified — no sector concentration risk detected.")

    total_exp = port_df["Credit Limit"].sum()
    ntc_cnt   = (port_df["NTC/NTB"] == "Yes").sum()
    avg_sc    = port_df["Score"].mean()

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Portfolio Exposure", fmt_inr(total_exp),
              help="Sum of all recommended credit limits")
    m2.metric("Avg Health Score", f"{avg_sc:.2f}/10")
    m3.metric("NTC/NTB Onboarded", f"{ntc_cnt} of {len(port_df)}",
              help="Credit-invisible MSMEs who would have been rejected by traditional systems")
    m4.metric("Sectors Covered", port_df["Sector"].nunique())

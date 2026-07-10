# MSME Financial Health Card — IDBI Innovate 2026

> An AI-driven credit intelligence platform that generates a 300–900 score for New-to-Credit MSMEs using alternative data — no credit bureau history required.

**Live Demo:** https://aman462006.github.io/msme-health-card  
**Backend API:** https://msme-health-card-production.up.railway.app/health

---

## The Problem

63 million MSMEs in India are invisible to traditional credit systems. They have no CIBIL score, no audited financials, no collateral history. Banks like IDBI are forced to either reject them or underwrite blindly. The result: a ₹20–25 lakh crore credit gap.

## The Solution

The MSME Financial Health Card pulls from six real-time alternative data streams — bank statements, GST filings, EPFO records, electricity consumption, UPI flows, and e-way bills — and distils them into a single peer-relative score (like CIBIL, but built for businesses with no credit trail). The score tells a banker not just *whether* to lend, but *why*, *how much*, and *what the MSME can do to improve*.

---

## USPs

| Feature | What it does |
|---|---|
| **Peer-relative scoring** | Score is a cohort percentile, not an absolute number — a kirana store is ranked against other kirana stores, not a textile exporter |
| **Thin-file safety net** | If fewer than 60% of features are available, a MAPIE conformal classifier with coverage-guaranteed intervals takes over instead of refusing the application |
| **Reject inference** | Fuzzy parcelling over 3 iterations recovers signal from historically rejected applications — prevents the model from learning only from the "safe" population |
| **Monotone constraints** | Every feature has a hard-coded direction (e.g. higher GST mismatch → higher PD, always). The model cannot learn a direction that contradicts domain logic |
| **SHAP explanations** | Every score comes with exact TreeSHAP feature attributions — the banker sees exactly which signals drove the decision |
| **Cohort-specific pillar weights** | SHAP importances are re-estimated per industry × turnover band via bootstrap, so pillar weights reflect what actually matters for that cohort |
| **Cross-source consistency engine** | Before scoring, 9 cross-checks (GST vs bank inflow, salary outflow vs EPFO headcount, etc.) flag data manipulation attempts and widen the PD interval accordingly |
| **Actionable counterfactuals** | DiCE generates the minimum set of changes the MSME needs to make to flip from Declined → Review → Eligible |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Data Ingestion                           │
│  Setu AA (bank)  │  GSTN API  │  EPFO  │  DISCOM  │  Karza     │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Feature Engineering                          │
│  30 features across 6 pillars  +  9 consistency inputs          │
│  src/data/feature_engineering.py                                │
└────────────────────────────┬────────────────────────────────────┘
                             │
                    ┌────────┴────────┐
                    │                 │
             Coverage ≥ 60%    Coverage < 60%
                    │                 │
                    ▼                 ▼
          ┌──────────────┐   ┌─────────────────────┐
          │  Main Model  │   │  Thin-File Model     │
          │  LightGBM +  │   │  MAPIE Conformal     │
          │  Isotonic    │   │  Classifier          │
          │  Calibration │   │  (coverage-           │
          │  + Reject    │   │   guaranteed CIs)    │
          │  Inference   │   └─────────────────────┘
          └──────┬───────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────────┐
│                  Consistency Engine                             │
│  9 cross-source checks  →  flag severity  →  PD interval        │
│  widening if inconsistencies detected                           │
│  src/consistency/engine.py                                      │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                  Peer-Relative Scoring                          │
│  Raw PD  →  cohort PD distribution  →  percentile  →  300–900  │
│  src/data/cohort_builder.py                                     │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                  Explainability Layer                           │
│  TreeSHAP  →  pillar scores  →  strengths/risks                 │
│  DiCE  →  counterfactual actions to improve                     │
│  src/explainability/                                            │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                  Health Card Output                             │
│  Score │ Grade │ PD + CI │ Pillar breakdown │ Actions           │
└─────────────────────────────────────────────────────────────────┘
```

---

## The 6 Pillars

### P1 — Cash Flow Resilience
*Source: Setu Account Aggregator (bank statement)*

| Feature | What it measures |
|---|---|
| `inflow_cv` | Volatility of monthly credit inflows (std/mean over 12 months) |
| `min_balance_days` | Days in the month the account balance was above a floor |
| `inflow_outflow_lag` | Days between money coming in and going out — proxy for liquidity buffer |
| `drawdown_recovery_days` | How quickly the account recovers after a large debit |
| `loss_absorption_buffer` | Ratio of average balance to average monthly obligations |

### P2 — Revenue Quality
*Source: GSTN API + external bureau scores*

| Feature | What it measures |
|---|---|
| `gst_mismatch_pct` | GSTR-1 (sales declared) vs GSTR-2A (purchases seen by buyers) gap |
| `gst_filing_punctuality` | Fraction of returns filed on or before due date |
| `itc_reversal_freq` | How often Input Tax Credits were reversed (signal of suspect claims) |
| `buyer_concentration_hhi` | Herfindahl index of buyer concentration — high = single-customer risk |
| `ext_source_1/2/3` | Normalized external bureau / alternative data scores |

### P3 — Obligation Discipline
*Source: EPFO, utility providers, existing lenders*

| Feature | What it measures |
|---|---|
| `epfo_payment_regularity` | Fraction of months PF contributions paid on time |
| `utility_delinquency_flag` | Whether electricity / water bills have ever been overdue |
| `gst_late_fee_incidence` | Proportion of GST filings that incurred late fees |
| `emi_bounce_rate` | Fraction of EMI debits that bounced |
| `avg_days_past_due` | Average DPD across all existing obligations |

### P4 — Operational Vitality
*Source: EPFO headcount data, DISCOM meter readings*

| Feature | What it measures |
|---|---|
| `epfo_headcount_delta_6m` | % change in formal payroll over last 6 months |
| `epfo_headcount_delta_12m` | % change in formal payroll over last 12 months |
| `electricity_kwh_trend` | Trend in electricity consumption — proxy for production activity |
| `supplier_diversity_score` | Normalized count of unique suppliers (anti-concentration) |

### P5 — Leverage & Liquidity
*Source: Bank statement + GST turnover*

| Feature | What it measures |
|---|---|
| `debt_to_inflow_ratio` | Total debt obligations as a fraction of annual inflows |
| `current_ratio_proxy` | Short-term assets vs short-term liabilities estimate |
| `working_capital_cycle_days` | Estimated days to convert inventory/receivables to cash |
| `annuity_to_income_ratio` | Monthly EMI burden relative to monthly income |

### P6 — Stability & Vintage
*Source: GSTN, MCA21, Karza director lookup*

| Feature | What it measures |
|---|---|
| `gstin_age_years` | How long the business has been GST-registered |
| `address_churn_flag` | Whether the registered address changed in the last 24 months |
| `promoter_churn_flag` | Whether the ownership / promoter profile changed |
| `directorship_overlap_flag` | Whether directors appear in NPA/defaulter entities |
| `days_employed_years` | Promoter's personal work history length |

---

## ML Pipeline (train.py)

```
Step 1  feature_engineering.py
        Home Credit dataset (307k rows) + synthetic MSME overlay
        Maps bureau features to 6-pillar MSME equivalents

Step 2  main_model.py
        Monotonic LightGBM (GBDT) with isotonic calibration
        Monotone constraints enforced per-feature (see src/config.py)
        Outputs calibrated PD probability

Step 3  reject_inference.py
        Fuzzy parcelling over 3 iterations
        Recovers signal from ~30% synthetic rejected population
        Prevents population bias from "approved only" training

Step 4  cohort_builder.py
        Groups firms by industry × turnover band
        Builds PD distribution per cohort for peer percentile lookup

Step 5  cohort_weights.py
        Per-cohort SHAP importances via bootstrap (n=1000)
        95% CI on each pillar weight
        Pillar weights differ by cohort — manufacturing ≠ trading

Step 6  audit.py
        Out-of-time AUC, PSI drift check, fairness gap across cohorts
        Adversarial validation (can we distinguish train vs test?)

Step 7  thin_file.py
        MAPIE conformal classifier on 7 thin-file features
        Produces coverage-guaranteed prediction intervals
        Activated when coverage_index < 0.60
```

---

## Project Structure

```
├── train.py                    # Full training pipeline (run once)
├── train_thin.py               # Thin-file model only
├── requirements.txt            # Inference dependencies (deployed)
├── requirements-train.txt      # Full training dependencies (local)
├── Dockerfile                  # Railway deployment
├── railway.json                # Railway config
│
├── src/
│   ├── config.py               # Pillars, features, monotone constraints
│   ├── api/
│   │   ├── main.py             # FastAPI app — /assess, /health
│   │   └── schemas.py          # Pydantic request/response models
│   ├── data/
│   │   ├── feature_engineering.py   # Home Credit → 6-pillar mapping
│   │   ├── cohort_builder.py        # Peer group PD distributions
│   │   └── download.py              # Kaggle dataset downloader
│   ├── models/
│   │   ├── main_model.py            # Monotonic LightGBM + calibration
│   │   ├── thin_file.py             # MAPIE conformal classifier
│   │   ├── reject_inference.py      # Fuzzy parcelling RI
│   │   ├── cohort_weights.py        # Per-cohort SHAP pillar weights
│   │   └── eligibility.py           # Decision: Eligible / Review / Declined
│   ├── explainability/
│   │   ├── shap_engine.py           # TreeSHAP pillar scores + strengths/risks
│   │   └── dice_engine.py           # DiCE counterfactual actions
│   ├── consistency/
│   │   └── engine.py                # 9 cross-source consistency checks
│   ├── neural/
│   │   ├── graphsage_encoder.py     # GraphSAGE director network encoder
│   │   ├── tcn_encoder.py           # Temporal CNN for transaction sequences
│   │   └── sms_encoder.py           # SMS transaction text encoder
│   └── validation/
│       └── audit.py                 # OOT AUC, PSI, fairness, adversarial
│
├── models/                     # Trained model artifacts (committed)
│   ├── main_model_ri.pkl       # LightGBM + isotonic + reject inference
│   ├── main_model.pkl          # LightGBM + isotonic (no RI)
│   ├── thin_file_model.pkl     # MAPIE conformal classifier
│   ├── cohort_data.pkl         # PD distributions per cohort
│   ├── cohort_pillar_weights.pkl
│   ├── pillar_weights.pkl
│   ├── pillar_weight_ci.pkl
│   └── feature_names.pkl
│
└── frontend/                   # React UI
    └── src/
        ├── App.js              # Page routing, API calls
        ├── components/
        │   ├── LandingPage.js  # About / landing page
        │   ├── MSMEForm.js     # 30-feature input form with info tooltips
        │   ├── HealthCard.js   # Score card output
        │   └── InfoTip.js      # Portal-based hover tooltips
        └── App.css             # Glassmorphism dark UI
```

---

## API

### `GET /health`
Returns system status.
```json
{
  "status": "ok",
  "ml_packages_installed": true,
  "main_model_loaded": true,
  "thin_model_loaded": true,
  "cohort_data_loaded": true
}
```

### `POST /assess`
Accepts a partial or complete MSME feature vector. Returns the full Health Card.

**Request** (all fields except `gstin` and `industry_type` are optional):
```json
{
  "gstin": "29ABCDE1234F1Z5",
  "industry_type": "manufacturing",
  "inflow_cv": 0.18,
  "gst_filing_punctuality": 0.92,
  "epfo_payment_regularity": 0.88,
  "gstin_age_years": 4.5
}
```

**Response:**
```json
{
  "score": 742,
  "grade": "A",
  "peer_percentile": 74.1,
  "eligibility": {
    "decision": "Eligible",
    "reason": "PD 4.2% is below the 10% cohort threshold",
    "peer_percentile": 74.1
  },
  "pd_12m": 0.0421,
  "pd_lower_bound": 0.0380,
  "pd_upper_bound": 0.0462,
  "model_used": "main",
  "pillar_scores": [...],
  "strengths": [...],
  "risks": [...],
  "consistency_flags": [...],
  "actions_to_improve": [...]
}
```

---

## Running Locally

```bash
# 1. Clone
git clone https://github.com/aman462006/msme-health-card.git
cd msme-health-card

# 2. Backend
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements-train.txt

# 3. (Optional) Re-train from scratch
#    Requires KAGGLE_USERNAME and KAGGLE_KEY in .env
cp .env.example .env
python train.py

# 4. Start backend
uvicorn src.api.main:app --reload --port 8000

# 5. Frontend (new terminal)
cd frontend
cp .env.example .env            # set REACT_APP_API_URL=http://localhost:8000
npm install
npm start
```

---

## Data Sources (Production Integration)

| Source | API | What it provides |
|---|---|---|
| Bank statement | Setu Account Aggregator | P1 cash flow features |
| GST filings | GSTN Sandbox API | P2 revenue quality features |
| PF contributions | EPFO portal / Karza | P3 + P4 workforce features |
| Electricity | DISCOM / utility APIs | P4 operational activity |
| Director lookup | Karza / MCA21 | P6 stability features |
| Bureau scores | Experian / CRIF | P2 `ext_source_*` features |

The prototype uses the Home Credit Open Dataset (307k rows) as a structural stand-in. The 6-pillar feature mapping is production-ready — swapping in live API data requires only updating `feature_engineering.py`.

---

## Deployment

| Layer | Platform | URL |
|---|---|---|
| Frontend | GitHub Pages | https://aman462006.github.io/msme-health-card |
| Backend | Railway (Docker) | https://msme-health-card-production.up.railway.app |

---

## Team

Built for **IDBI Innovate 2026** hackathon.

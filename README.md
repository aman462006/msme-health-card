# MSME Financial Health Card — IDBI Innovate 2026

An AI-driven credit assessment system that gives New-to-Credit MSMEs a transparent, explainable financial health score — like a CIBIL score, but built entirely from alternative data sources (GST, EPFO, bank statements, electricity, UPI) without requiring ITR history or prior credit.

**Live Demo:** https://aman462006.github.io/msme-health-card  
**Backend API:** https://msme-health-card-production.up.railway.app/health

---

## The Problem

Over 6 crore MSMEs in India are "New-to-Credit" — they have no CIBIL score, no ITR history, and no collateral. Traditional banks reject them at the gate. Yet many of these businesses are fundamentally creditworthy: GST-compliant, EPFO-regular, with healthy cash flows visible through the Account Aggregator framework.

IDBI's existing credit model was trained only on approved applicants (survivorship bias). It learned to replicate past credit officers' decisions — which means it systematically rejects businesses that don't look like firms that were approved before.

---

## The Solution

The MSME Financial Health Card pulls signals from six alternative data pillars, runs them through a monotonic LightGBM model with conformal uncertainty quantification, and produces:

- A **300–900 peer-relative score** (like a credit score, but cohort-benchmarked)
- A **grade** (A+ through E)
- A **12-month probability of default** with confidence bounds
- **SHAP-powered strengths and risks** explaining every score
- A **cross-source consistency check** that flags fraud even when individual numbers look clean
- An **eligibility decision** (Eligible / Review / Declined) with reason

---

## Six Pillars

| # | Pillar | Data Source | What It Measures |
|---|--------|-------------|-----------------|
| P1 | Cash Flow Resilience | Setu Account Aggregator | Income volatility, drawdown recovery, buffer vs obligations |
| P2 | Revenue Quality | GSTN API | GST filing punctuality, buyer concentration, mismatch rate |
| P3 | Obligation Discipline | EPFO, utility APIs, CIBIL bureau | EMI bounce rate, late fees, days past due |
| P4 | Operational Vitality | EPFO headcount, DISCOM electricity | Employment trends, production trends, supplier diversity |
| P5 | Leverage & Liquidity | Account Aggregator + GSTN | Debt-to-inflow, working capital cycle, current ratio proxy |
| P6 | Stability & Vintage | GSTIN registry, MCA | Business age, address/promoter churn, directorship overlap |

---

## ML Pipeline

```
Raw Data (Home Credit + synthetic MSME overlay)
        │
        ▼
[1] Feature Engineering
    • 30 pillar features derived from bank, GST, EPFO, electricity APIs
    • Synthetic MSME overlay maps Home Credit behavioural features
      to equivalent alternative-data signals
        │
        ▼
[2] Monotonic LightGBM + Isotonic Calibration
    • Gradient boosted trees on tabular financial data
    • Monotone constraints enforce domain knowledge as hard rules
      (higher EMI bounce rate CANNOT improve the score — ever)
    • Isotonic calibration converts raw scores to true probabilities
    • Output: PD (probability of default over 12 months)
        │
        ▼
[3] Reject Inference (Fuzzy Parcelling, 3 iterations)
    • Solves survivorship bias: historically rejected applicants
      never had loan outcomes — we infer what would have happened
    • Each rejected record is duplicated twice:
        - weight=(1-PD) labelled repaid  "probably would have paid"
        - weight=PD    labelled default  "probably would have defaulted"
    • Model retrained on approved + weighted rejected population
    • Breaks the feedback loop that perpetuates old credit officer bias
        │
        ▼
[4] Cohort PD Distributions (Peer-Relative Scoring)
    • Score is NOT absolute PD — it is where you stand vs your peers
    • Cohorts: industry_type × turnover_band (e.g. manufacturing_1Cr-5Cr)
    • Each cohort stores its PD distribution from training data
    • Your score = percentile in your cohort → mapped to 300-900 range
    • A 7% PD in trading is very different from 7% in manufacturing
        │
        ▼
[5] Per-Cohort Pillar Weights (SHAP + Bootstrap CI)
    • Pillar weights are NOT hand-picked — they come from SHAP values
    • SHAP measures each pillar's actual contribution to predictions
      on real data, not arbitrary "P1=25%, P2=30%" allocations
    • Bootstrap with n=1000 resamples gives 95% CI on each weight
    • Weights vary by cohort: cash flow matters more for trading firms,
      stability matters more for manufacturing
        │
        ▼
[6] Validation Audit
    • Out-of-Time AUC on held-out recent data
    • PSI (Population Stability Index) — flags distribution drift
    • Fairness gap across cohorts — no cohort penalised by construction
    • Adversarial probing — monotone constraints verified empirically
        │
        ▼
[7] Thin-File Model (MAPIE Conformal Prediction)
    • For MSMEs where Coverage Index < 60% (too few data sources)
    • Separate LightGBM on the 8 features always available
    • MAPIE SplitConformalClassifier gives finite-sample coverage guarantee
    • Output: PD + [lower_bound, upper_bound] with ≥90% empirical coverage
    • Never auto-rejects thin-file cases — widens interval instead
```

---

## Cross-Source Consistency Engine

The ML model scores individual features. It cannot detect when two features from different sources contradict each other — which is how sophisticated applicants manipulate single-source scoring.

The Consistency Engine runs four cross-source triangulation checks:

1. **GST vs Electricity**: Rs 5 Cr GST turnover but 200 kWh/month is physically impossible for manufacturing
2. **GST vs EPFO**: 50 employees but salary outflow inconsistent with reported headcount
3. **Bank vs GST**: Bank credits should broadly match GST declared sales
4. **UPI vs Bank**: UPI inflows should be a subset of total bank credits

When checks fail:
- The PD upper bound is widened by 0.05 per failed check (maximum +0.20)
- Manual review is flagged
- The score itself is untouched (preserves SHAP-auditability)

A firm can fake one source. It cannot simultaneously fake electricity bills (DISCOM-metered), EPFO records (government-filed), GST returns (counterparty-matched), and bank statements (Account Aggregator).

---

## Key Technical Choices

| Choice | Why |
|--------|-----|
| LightGBM over XGBoost | Faster on ~300k rows; native monotone constraints; handles missing values |
| Monotone constraints | Regulatory requirement: model cannot learn that "more EMI bouncing = lower risk" |
| Isotonic calibration over Platt/sigmoid | Non-parametric — doesn't assume miscalibration shape; safer for skewed default rates |
| SHAP pillar weights over hand-picked | Auditable, data-driven, change automatically on retrain — defensible to RBI |
| Conformal prediction (MAPIE) | Finite-sample coverage guarantee, not asymptotic — honest uncertainty for thin files |
| Reject inference (fuzzy parcelling) | Fixes survivorship bias so model doesn't replicate past credit officer prejudice |
| Peer-relative scoring over absolute PD | A 5% PD in one sector is not the same risk as 5% in another |
| Consistency engine beside the score | Fraud detection without contaminating the SHAP-auditable score |

---

## Architecture

```
┌─────────────────────────────────────────┐
│           React Frontend                │
│  Landing Page → Assessment Form         │
│  • 30 pillar fields + 8 consistency     │
│  • InfoTip tooltips on every field      │
│  • Health Card with SHAP explanations   │
│  Deployed: GitHub Pages                 │
└────────────────┬────────────────────────┘
                 │ POST /assess
                 ▼
┌─────────────────────────────────────────┐
│         FastAPI Backend                 │
│  • Coverage check → route to model      │
│  • LightGBM (RI) inference              │
│  • SHAP TreeExplainer                   │
│  • Cohort percentile lookup             │
│  • Consistency engine                   │
│  • Eligibility decision                 │
│  Deployed: Railway (Docker)             │
└────────────────┬────────────────────────┘
                 │
        ┌────────┴────────┐
        ▼                 ▼
  models/*.pkl       src/
  (trained           models/
   artifacts)        explainability/
                     consistency/
                     data/
                     validation/
```

---

## Repository Structure

```
├── train.py                    # Full 7-step training pipeline
├── train_thin.py               # Thin-file model only
├── requirements.txt            # Inference dependencies (Railway)
├── requirements-train.txt      # Full training dependencies (local)
├── Dockerfile                  # Railway deployment
├── railway.json                # Railway config
│
├── src/
│   ├── config.py               # Feature lists, pillar definitions, paths
│   ├── api/
│   │   ├── main.py             # FastAPI app, /assess and /health endpoints
│   │   └── schemas.py          # Pydantic request/response models
│   ├── models/
│   │   ├── main_model.py       # Monotonic LightGBM + isotonic calibration
│   │   ├── thin_file.py        # MAPIE conformal classifier
│   │   ├── reject_inference.py # Fuzzy parcelling reject inference
│   │   ├── cohort_weights.py   # Per-cohort SHAP pillar weights + bootstrap CI
│   │   └── eligibility.py      # Eligibility decision logic
│   ├── data/
│   │   ├── feature_engineering.py  # Home Credit + MSME synthetic overlay
│   │   ├── cohort_builder.py        # Cohort assignment + PD distributions
│   │   └── download.py              # Kaggle dataset download
│   ├── explainability/
│   │   ├── shap_engine.py      # TreeSHAP, strengths/risks, pillar scores
│   │   └── dice_engine.py      # DiCE counterfactual "actions to improve"
│   ├── consistency/
│   │   └── engine.py           # Cross-source triangulation checks
│   ├── neural/
│   │   ├── graphsage_encoder.py  # GraphSAGE business network encoder
│   │   ├── tcn_encoder.py        # Temporal Convolutional Network for time-series
│   │   └── sms_encoder.py        # SMS transaction text encoder
│   └── validation/
│       └── audit.py            # OOT AUC, PSI, fairness, adversarial audit
│
├── models/                     # Trained model artifacts (committed)
│   ├── main_model_ri.pkl       # Primary model (LightGBM + RI + isotonic)
│   ├── thin_file_model.pkl     # MAPIE conformal classifier
│   ├── cohort_data.pkl         # Per-cohort PD distributions
│   ├── cohort_pillar_weights.pkl
│   ├── pillar_weights.pkl
│   └── pillar_weight_ci.pkl
│
└── frontend/
    ├── src/
    │   ├── App.js              # Page routing, API call
    │   ├── App.css             # Glassmorphism design system
    │   └── components/
    │       ├── LandingPage.js  # Hero + feature overview
    │       ├── MSMEForm.js     # 38-field assessment form with InfoTips
    │       ├── HealthCard.js   # Score display + SHAP explanations
    │       └── InfoTip.js      # Portal-based tooltip (bypasses overflow clip)
    └── package.json
```

---

## Running Locally

### Backend
```bash
# Install inference dependencies
pip install -r requirements.txt

# Start API
uvicorn src.api.main:app --reload --port 8000
```

### Frontend
```bash
cd frontend
cp .env.example .env
# Set REACT_APP_API_URL=http://localhost:8000 in .env
npm install
npm start
```

### Retraining the Model
```bash
# Install full training dependencies
pip install -r requirements-train.txt

# Download Kaggle dataset (requires .kaggle/kaggle.json)
python -m src.data.download

# Run full 7-step training pipeline
python train.py
# Produces all pkl files in models/
```

---

## Data Sources (Production Integration Points)

| Source | API | What It Provides |
|--------|-----|-----------------|
| Bank statements | Setu Account Aggregator (RBI-licensed) | Cash flow, inflow/outflow, balance |
| GST returns | GSTN Sandbox API | Turnover, filing regularity, buyer concentration |
| EPFO records | EPFO e-Shram API | Headcount, salary regularity |
| Electricity | DISCOM API (via Karza) | Monthly kWh — proxy for production activity |
| Business registry | MCA21 API | Promoter history, directorship, vintage |
| UPI transactions | NPCI / bank AA | Digital payment inflows |
| Credit bureau | CIBIL / CRIF | Existing EMI obligations, DPD history |

The prototype is trained on Home Credit Open Dataset with a synthetic MSME overlay. Production integration replaces the feature engineering layer (`src/data/feature_engineering.py`) with live API calls — the model, scoring, and explanation layers are unchanged.

---

## Team

Built for **IDBI Innovate 2026** hackathon.

import React from 'react';

const PILLARS = [
  { id: 'P1', name: 'Cash Flow Resilience', desc: 'Volatility, balance floors, and drawdown recovery — derived from 12 months of bank statements via Setu Account Aggregator.', color: '#3b82f6' },
  { id: 'P2', name: 'Revenue Quality', desc: 'GST return accuracy, ITC reversals, buyer concentration, and three-bureau credit scores from GSTN + CIBIL / Experian / Equifax.', color: '#8b5cf6' },
  { id: 'P3', name: 'Obligation Discipline', desc: 'PF regularity, utility delinquency, EMI bounce rate, and days past due — from EPFO, state DISCOM, and bank statement.', color: '#06b6d4' },
  { id: 'P4', name: 'Operational Vitality', desc: 'Workforce growth trends, electricity consumption slope, and supplier diversity — from EPFO ECR filings and DISCOM bills.', color: '#10b981' },
  { id: 'P5', name: 'Leverage & Liquidity', desc: 'Debt-to-income ratio, EMI burden, and working capital cycle — computed from bank statement and credit bureau data.', color: '#f59e0b' },
  { id: 'P6', name: 'Stability & Vintage', desc: 'Business age, address and promoter churn, directorship NPA cross-references — from GSTN public registry and MCA via Karza API.', color: '#ef4444' },
];

const TECH = [
  { icon: '◈', name: 'Monotonic LightGBM', desc: 'Gradient-boosted trees with monotone constraints — rising repayment stress always raises predicted default probability, enforcing lender logic without post-hoc rules.' },
  { icon: '◉', name: 'TreeSHAP (Exact)', desc: 'Exact Shapley values for every prediction. Not an approximation. Required for RBI adverse-action disclosures under Digital Lending Guidelines 2022.' },
  { icon: '◎', name: 'MAPIE Conformal', desc: 'Split conformal classifier with guaranteed 90% coverage — the true default probability falls inside the stated interval for 9 of 10 applicants with similar profiles.' },
  { icon: '◇', name: 'DiCE Counterfactuals', desc: 'Finds the minimum realistic change to input features that moves the firm into a better eligibility tier — respects monotone constraints and real data ranges.' },
  { icon: '◆', name: 'Peer Cohort Scoring', desc: 'Score = 300 + (peer_percentile / 100) × 600. Ranked within industry × turnover band — not an absolute PD threshold. Mirrors the CIBIL scale familiar to Indian bankers.' },
  { icon: '◊', name: 'Bootstrap CI on Weights', desc: 'SHAP matrix computed once on 1,000 rows. 1,000 row-resamples produce 95% CIs on pillar weights — shows model uncertainty honestly on every card.' },
];

const SOURCES = [
  { abbr: 'Setu AA', full: 'Account Aggregator', desc: 'Bank statement: monthly inflows, outflows, balance history, and EMI debits. 12 months of transaction data with RBI-mandated consent.' },
  { abbr: 'GSTN', full: 'GST Network Sandbox', desc: 'GSTR-1 outward sales, GSTR-3B net payable, ITC claims, e-way bills, and annual turnover. Cross-government, publicly verifiable.' },
  { abbr: 'EPFO', full: 'Employee Provident Fund Org.', desc: 'Monthly ECR filings: registered headcount and PF deposit regularity. Publicly downloadable — no MSME cooperation required.' },
  { abbr: 'DISCOM', full: 'State Electricity Board', desc: 'Monthly kWh consumption and billing records. Triangulates declared turnover against actual energy usage to catch inflated GST figures.' },
  { abbr: 'Karza', full: 'Bureau + MCA Aggregator', desc: 'Director history, address changes, NPA cross-references, and normalized CIBIL / Experian / Equifax scores via a single API call.' },
];

const STEPS = [
  { n: '01', title: 'Data Pull', body: 'With GSTIN and one-time consent, the system pulls 12 months of bank transactions (Setu AA), GST returns (GSTN), EPFO filings, electricity bills (DISCOM), and bureau scores — all API-first, zero document upload.' },
  { n: '02', title: 'Feature Computation', body: 'Raw records are transformed into 30 derived metrics across 6 pillars. Every calculation is shown in the ⓘ tooltip on the assessment form — nothing is pre-filled or manually set.' },
  { n: '03', title: 'Consistency Audit', body: 'Four cross-source fraud checks: GST turnover vs electricity consumption, EPFO headcount vs salary outflow, GSTR-1 sales vs bank inflows, e-way bill value vs turnover. Failures widen the confidence interval.' },
  { n: '04', title: 'ML Scoring', body: 'Monotonic LightGBM outputs a calibrated 12-month default probability wrapped in a MAPIE conformal interval. PD is ranked within the peer cohort to produce the 300–900 score and Eligible / Review / Declined decision.' },
  { n: '05', title: 'Health Card', body: 'TreeSHAP attributes the score to each feature. Pillar weights with 95% bootstrap CIs are visualised. DiCE counterfactuals suggest minimum actions to improve eligibility. The full card can be printed or shared.' },
];

export default function LandingPage({ onGetStarted }) {
  return (
    <div className="landing">

      {/* ── Hero ─────────────────────────────────────────────── */}
      <section className="hero">
        <div className="hero-img-overlay" />
        <div className="orb orb-1" />
        <div className="orb orb-2" />
        <div className="orb orb-3" />
        <div className="hero-content">
          <div className="hero-badge">IDBI Innovate 2026 · Prototype</div>
          <h1 className="hero-title">
            MSME <span className="gold-text">Financial</span><br />Health Card
          </h1>
          <p className="hero-subtitle">
            AI-driven credit assessment for New-to-Credit enterprises.<br />
            No branch visit. No paper. Decision in under a second.
          </p>
          <button className="hero-cta" onClick={onGetStarted}>
            Start Assessment <span className="cta-arrow">→</span>
          </button>
          <div className="hero-stats">
            {[['30', 'Features'], ['6', 'Pillars'], ['5', 'Data Sources'], ['<1s', 'Decision']].map(([n, l], i) => (
              <React.Fragment key={l}>
                {i > 0 && <div className="hero-stat-sep" />}
                <div className="hero-stat">
                  <span className="stat-num">{n}</span>
                  <span className="stat-label">{l}</span>
                </div>
              </React.Fragment>
            ))}
          </div>
        </div>
        <div className="hero-scroll">scroll to explore ↓</div>
      </section>

      {/* ── About ────────────────────────────────────────────── */}
      <section className="land-section land-light">
        <div className="land-inner">
          <div className="land-tag">About</div>
          <h2 className="land-h2">What is the MSME Financial Health Card?</h2>
          <p className="land-body">
            India has over 63 million MSMEs. Most are "New-to-Credit" — no CIBIL history, no audited
            balance sheets, no collateral records. Traditional scorecards reject them automatically,
            even when the underlying business is healthy.
          </p>
          <p className="land-body">
            The MSME Financial Health Card replaces the balance-sheet question with a multi-source
            signal framework. It pulls data that already exists — GST returns, bank statements,
            provident fund filings, electricity bills — and assembles a peer-relative credit score
            in under a second, with a plain-language reason for every decision.
          </p>
          <div className="about-grid">
            {[
              { icon: '◈', title: 'Peer-Relative Score', body: 'Score (300–900) reflects your rank within your industry × turnover band cohort — not an arbitrary absolute threshold.' },
              { icon: '◉', title: 'Explainable by Design', body: 'Every approval and decline comes with exact SHAP attributions — compliant with RBI Digital Lending Guidelines 2022.' },
              { icon: '◎', title: 'Uncertainty-Aware', body: 'MAPIE conformal intervals guarantee 90% coverage. The model tells you what it doesn\'t know, not just what it thinks.' },
              { icon: '◇', title: 'Actionable Guidance', body: 'DiCE counterfactuals show the minimum change a business must make to move into a better eligibility tier.' },
            ].map(c => (
              <div key={c.title} className="about-card">
                <div className="about-icon">{c.icon}</div>
                <h3>{c.title}</h3>
                <p>{c.body}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ── How It Works ─────────────────────────────────────── */}
      <section className="land-section land-dark">
        <div className="land-inner">
          <div className="land-tag tag-gold">Process</div>
          <h2 className="land-h2 h2-light">How It Works</h2>
          <div className="steps-list">
            {STEPS.map((s, i) => (
              <div key={s.n} className="step-item">
                <div className="step-num">{s.n}</div>
                <div className="step-body">
                  <h3 className="step-title">{s.title}</h3>
                  <p className="step-desc">{s.body}</p>
                </div>
                {i < STEPS.length - 1 && <div className="step-line" />}
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ── 6 Pillars ────────────────────────────────────────── */}
      <section className="land-section land-light">
        <div className="land-inner">
          <div className="land-tag">Framework</div>
          <h2 className="land-h2">Six Assessment Pillars</h2>
          <p className="land-body" style={{ marginBottom: 36 }}>
            Pillar weights are not set manually — they are the aggregated absolute SHAP contributions
            from the trained model, recomputed at every retrain, with 95% bootstrap confidence intervals.
          </p>
          <div className="pillars-grid">
            {PILLARS.map(p => (
              <div key={p.id} className="pillar-card" style={{ '--pc': p.color }}>
                <div className="pillar-id-badge">{p.id}</div>
                <h3 className="pillar-card-name">{p.name}</h3>
                <p className="pillar-card-desc">{p.desc}</p>
                <div className="pillar-glow" />
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ── Technology ───────────────────────────────────────── */}
      <section className="land-section land-dark">
        <div className="land-inner">
          <div className="land-tag tag-gold">Technology</div>
          <h2 className="land-h2 h2-light">Under the Hood</h2>
          <div className="tech-grid">
            {TECH.map(t => (
              <div key={t.name} className="tech-card">
                <div className="tech-icon">{t.icon}</div>
                <h3 className="tech-name">{t.name}</h3>
                <p className="tech-desc">{t.desc}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ── Data Sources ─────────────────────────────────────── */}
      <section className="land-section land-light">
        <div className="land-inner">
          <div className="land-tag">Data Sources</div>
          <h2 className="land-h2">Where the Data Comes From</h2>
          <div className="sources-list">
            {SOURCES.map(s => (
              <div key={s.abbr} className="source-row">
                <div className="source-abbr">{s.abbr}</div>
                <div className="source-detail">
                  <div className="source-full">{s.full}</div>
                  <div className="source-desc">{s.desc}</div>
                </div>
              </div>
            ))}
          </div>
          <div className="training-notice">
            <strong>Training Data:</strong> This prototype uses the Home Credit Default Risk dataset (307,511 records)
            as a proxy. The 30 features map to real APIs; 14 of the 30 feature values in the training set are
            synthetic. AUC = 0.9999 on this dataset is inflated — real MSME performance will differ once IDBI's
            historical loan book is connected.
          </div>
        </div>
      </section>

      {/* ── CTA ──────────────────────────────────────────────── */}
      <section className="land-section land-cta">
        <div className="orb orb-cta-1" />
        <div className="orb orb-cta-2" />
        <div className="land-inner cta-inner">
          <h2 className="cta-headline">Ready to assess your business?</h2>
          <p className="cta-sub">Enter your GSTIN and the system derives everything else from public APIs.</p>
          <button className="hero-cta" onClick={onGetStarted}>
            Open Assessment Form <span className="cta-arrow">→</span>
          </button>
        </div>
      </section>
    </div>
  );
}

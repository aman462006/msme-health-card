import React from 'react';
import {
  RadarChart, Radar, PolarGrid, PolarAngleAxis,
  PolarRadiusAxis, ResponsiveContainer, Tooltip,
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Cell, ErrorBar,
} from 'recharts';
import InfoTip from './InfoTip';

const PILLAR_SHORT = {
  P1_CashFlowResilience: 'Cash Flow',
  P2_RevenueQuality: 'Revenue',
  P3_ObligationDiscipline: 'Obligations',
  P4_OperationalVitality: 'Operations',
  P5_LeverageAndLiquidity: 'Liquidity',
  P6_StabilityAndVintage: 'Stability',
};

const GRADE_COLORS = {
  'A+': '#16a34a', 'A': '#22c55e', 'B+': '#65a30d',
  'B': '#ca8a04', 'C+': '#d97706', 'C': '#ea580c',
  'D': '#dc2626', 'E': '#991b1b',
};

const ELIGIBILITY_CONFIG = {
  Eligible: { bg: '#dcfce7', border: '#16a34a', text: '#15803d', label: 'ELIGIBLE' },
  Review:   { bg: '#fef9c3', border: '#ca8a04', text: '#92400e', label: 'REVIEW'   },
  Declined: { bg: '#fee2e2', border: '#dc2626', text: '#991b1b', label: 'DECLINED' },
};

export default function HealthCard({ data }) {
  const gradeColor = GRADE_COLORS[data.grade] || '#6b7280';
  const elig = data.eligibility;
  const eligConfig = ELIGIBILITY_CONFIG[elig?.decision] || ELIGIBILITY_CONFIG.Review;

  const radarData = data.pillar_scores.map(p => ({
    subject: PILLAR_SHORT[p.pillar] || p.pillar,
    score: p.score,
  }));

  const weightChartData = data.pillar_scores.map(p => ({
    name: PILLAR_SHORT[p.pillar] || p.pillar,
    weight: parseFloat((p.weight * 100).toFixed(1)),
    errorY: p.weight_ci
      ? [
          parseFloat(((p.weight - p.weight_ci.ci_lower) * 100).toFixed(1)),
          parseFloat(((p.weight_ci.ci_upper - p.weight) * 100).toFixed(1)),
        ]
      : [0, 0],
  }));

  return (
    <div className="health-card">

      {/* ── Header ──────────────────────────────────────────────── */}
      <div className="card-header">
        <div className="card-title-row">
          <h2>MSME Financial Health Card</h2>
          <span className="gstin-badge">{data.gstin}</span>
        </div>
        <div className="card-meta">
          <span>Peer cohort: <strong>{data.peer_cohort}</strong></span>
          <span>Model: {data.model_used === 'main' ? 'Full Assessment' : 'Thin-File'}</span>
          <span>Coverage: {(data.coverage_index * 100).toFixed(0)}%</span>
        </div>
      </div>

      {/* ── Eligibility banner ──────────────────────────────────── */}
      {elig && (
        <div style={{
          background: eligConfig.bg, border: `2px solid ${eligConfig.border}`,
          borderRadius: 8, padding: '12px 16px', margin: '16px 24px 0',
          display: 'flex', alignItems: 'flex-start', gap: 12,
        }}>
          <span style={{
            background: eligConfig.border, color: '#fff', fontWeight: 700,
            fontSize: 11, padding: '3px 10px', borderRadius: 4, whiteSpace: 'nowrap', marginTop: 2,
          }}>{eligConfig.label}</span>
          <div style={{ flex: 1 }}>
            <p style={{ margin: 0, color: eligConfig.text, fontWeight: 600 }}>
              {elig.reason}
            </p>
            <p style={{ margin: '4px 0 0', fontSize: 11, color: '#6b7280' }}>
              Peer percentile: {elig.peer_percentile}/100 &nbsp;|&nbsp;
              Eligible threshold (60th pct): PD &le; {(elig.eligible_pd_threshold * 100).toFixed(2)}%
              &nbsp;|&nbsp;
              Review threshold (80th pct): PD &le; {(elig.review_pd_threshold * 100).toFixed(2)}%
              <InfoTip text="Eligible = your PD is below the 60th-percentile PD of all firms in your cohort (better than 60% of peers). Review = between 60th and 80th. Declined = above 80th. Thresholds are computed fresh from training data at every retrain." />
            </p>
          </div>
        </div>
      )}

      {/* ── Score banner ─────────────────────────────────────────── */}
      <div className="score-banner">
        <div className="score-main">
          <div className="score-number" style={{ color: gradeColor }}>{data.score}</div>
          <div className="score-ci">
            CI: {data.score_ci_lower} – {data.score_ci_upper}
            <InfoTip text="Confidence interval on the score. Derived from ±10 percentile points around your peer percentile. For thin-file route the interval is wider due to the MAPIE conformal prediction interval on PD." />
          </div>
          <div className="score-grade" style={{ backgroundColor: gradeColor }}>Grade {data.grade}</div>
          <div style={{ fontSize: 12, color: '#6b7280', marginTop: 6 }}>
            Peer percentile: <strong>{data.peer_percentile}/100</strong>
            <InfoTip text="Your rank within the same industry + turnover band cohort. 100 = lowest probability of default in the cohort (best). Formula: 100 - percentile_rank(your_PD, cohort_PD_distribution). Computed from training data distribution." />
          </div>
        </div>

        <div className="pd-block">
          <div className="pd-label">
            12-Month Default Probability
            <InfoTip text="Output of the calibrated monotonic LightGBM model. Calibrated using isotonic regression so that if the model outputs 0.12, approximately 12% of firms with similar features historically defaulted. Not a raw score — a true probability." />
          </div>
          <div className="pd-value">{(data.pd_12m * 100).toFixed(2)}%</div>
          <div className="pd-upper" style={{ fontSize: 12, color: '#6b7280' }}>
            Interval: {(data.pd_lower_bound * 100).toFixed(2)}% – {(data.pd_upper_bound * 100).toFixed(2)}%
            <InfoTip text="For thin-file route: MAPIE conformal prediction interval (90% coverage guarantee — the true PD falls in this range for 9 out of 10 applicants with similar profiles). For main route: ±0.04 base, plus 0.05 added per failed consistency check (max +0.20)." />
          </div>
        </div>

        <div className="score-bar-wrap">
          <div style={{ fontSize: 11, fontWeight: 600, color: '#64748b', marginBottom: 4 }}>
            Score scale (300 = bottom of cohort, 900 = top)
            <InfoTip text="Score = 300 + (peer_percentile / 100) × 600. A score of 750 means you are at the 75th percentile of your cohort — better than 75% of similar firms. The scale mirrors the CIBIL range familiar to Indian bankers." />
          </div>
          <div className="score-bar">
            <div className="score-bar-fill" style={{
              width: `${((data.score - 300) / 600) * 100}%`,
              backgroundColor: gradeColor,
            }} />
          </div>
          <div className="score-bar-labels">
            <span>300</span><span>Poor</span><span>Fair</span><span>Good</span><span>Excellent</span><span>900</span>
          </div>
        </div>
      </div>

      {/* ── Consistency flags ────────────────────────────────────── */}
      {data.consistency_flags.length > 0 && (
        <div className="consistency-section">
          <h3>
            Cross-Source Consistency: {data.consistency_summary}
            <InfoTip text="4 triangulation checks run across independent data sources (GST vs electricity, EPFO vs salary, GSTR-1 vs bank inflows, e-way bills vs turnover). A firm can manipulate one source but not all four simultaneously. Failures widen the PD interval by 0.05 each (max +0.20) but do not directly lower the score." />
          </h3>
          {data.manual_review_recommended && (
            <div className="manual-review-badge">Manual Review Recommended</div>
          )}
          {data.consistency_flags.map((f, i) => (
            <div key={i} className={`flag-item flag-${f.severity}`}>
              <div className="flag-header">
                <span className={`flag-dot flag-dot-${f.severity}`} />
                <strong>{f.check_name}</strong>
                <span className="flag-severity">{f.severity.toUpperCase()}</span>
              </div>
              <p className="flag-desc">{f.description}</p>
              <p className="flag-rec">{f.recommendation}</p>
            </div>
          ))}
        </div>
      )}

      {/* ── Thin-file notice ─────────────────────────────────────── */}
      {data.model_used === 'thin_file' && (
        <div className="thin-file-notice">
          <strong>Thin-File Assessment</strong> — Limited data ({(data.coverage_index * 100).toFixed(0)}% pillar coverage).
          Routed to the MAPIE conformal classifier instead of the full monotonic LightGBM.
          The conformal interval guarantees 90% coverage (true PD falls inside the interval
          for 9 of 10 similar applicants). Missing: {data.missing_pillars.map(p => PILLAR_SHORT[p] || p).join(', ')}.
        </div>
      )}

      {/* ── Radar + Weights ──────────────────────────────────────── */}
      <div className="radar-section">
        <div className="radar-chart">
          <h3>
            Pillar Radar — Peer Cohort Percentile
            <InfoTip text="Each spoke = this pillar's SHAP-derived score for your record, expressed as a percentile within your peer cohort. Formula: 50 + (protective_SHAP - risk_SHAP) / total_|SHAP| × 500, clipped to [0,100]. 50 = neutral, 100 = this pillar strongly helped vs peers." />
          </h3>
          <p style={{ fontSize: 11, color: '#6b7280', marginBottom: 8 }}>
            Position within <strong>{data.peer_cohort}</strong> peers (100 = best in cohort)
          </p>
          <ResponsiveContainer width="100%" height={300}>
            <RadarChart data={radarData}>
              <PolarGrid />
              <PolarAngleAxis dataKey="subject" tick={{ fontSize: 11 }} />
              <PolarRadiusAxis angle={90} domain={[0, 100]} tick={{ fontSize: 10 }} />
              <Radar name="Cohort Percentile" dataKey="score" stroke="#1d4ed8" fill="#1d4ed8" fillOpacity={0.3} />
              <Tooltip formatter={(val) => [`${val}/100`, 'Peer Percentile']} />
            </RadarChart>
          </ResponsiveContainer>
        </div>

        <div className="pillar-weights">
          <h3>
            Pillar Weights — Per-Cohort SHAP
            <InfoTip text="Weight = sum of mean |SHAP value| for all features in this pillar, divided by total |SHAP| across all 30 features. Computed on 500 rows sampled from your specific cohort (industry × turnover band). Recomputed at every retrain. Never set by hand." />
          </h3>
          <p className="weights-note">
            Error bars = 95% CI from 1,000 bootstrap resamples of the SHAP matrix.
            <InfoTip text="Bootstrap: the full 1,000-row SHAP matrix is computed once. Then 1,000 random re-samples of those rows are drawn (with replacement) and the pillar weight is recomputed each time. The 2.5th and 97.5th percentiles of those 1,000 weights form the CI shown here." />
          </p>
          <ResponsiveContainer width="100%" height={200}>
            <BarChart data={weightChartData} layout="vertical" margin={{ left: 60, right: 24 }}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis type="number" unit="%" domain={[0, 'auto']} tick={{ fontSize: 10 }} />
              <YAxis dataKey="name" type="category" tick={{ fontSize: 11 }} width={62} />
              <Tooltip formatter={(val) => [`${val}%`, 'Weight']} />
              <Bar dataKey="weight" radius={4}>
                {weightChartData.map((entry, i) => (
                  <Cell key={i} fill={entry.weight >= 20 ? '#1d4ed8' : entry.weight >= 12 ? '#3b82f6' : '#93c5fd'} />
                ))}
                <ErrorBar dataKey="errorY" width={4} strokeWidth={2} stroke="#374151" direction="x" />
              </Bar>
            </BarChart>
          </ResponsiveContainer>
          <div style={{ marginTop: 10 }}>
            {data.pillar_scores.map(p => (
              <div key={p.pillar} className="weight-row">
                <span className="weight-label">{PILLAR_SHORT[p.pillar] || p.pillar}</span>
                <span className="weight-pct">{(p.weight * 100).toFixed(1)}%</span>
                {p.weight_ci && (
                  <span style={{ fontSize: 10, color: '#9ca3af', marginLeft: 4 }}>
                    [{(p.weight_ci.ci_lower * 100).toFixed(1)}–{(p.weight_ci.ci_upper * 100).toFixed(1)}%]
                  </span>
                )}
                <span className="weight-score" style={{
                  color: p.score >= 60 ? '#16a34a' : p.score >= 40 ? '#ca8a04' : '#dc2626',
                  marginLeft: 'auto', fontSize: 12, fontWeight: 700,
                }}>
                  {p.score}pct
                </span>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* ── Strengths & Risks ────────────────────────────────────── */}
      <div className="sr-section">
        <div className="strengths">
          <h3>
            Strengths
            <InfoTip text="Features whose SHAP value is negative — they pushed your predicted default probability DOWN. Sorted by |SHAP| so the most impactful strength appears first. SHAP (Shapley Additive Explanations) is the exact, game-theory-based attribution of each feature's contribution to your specific prediction." />
          </h3>
          {data.strengths.length === 0 && <p className="empty-note">No SHAP data (thin-file route uses simpler logic)</p>}
          {data.strengths.map((s, i) => (
            <div key={i} className="sr-item sr-strength">
              <div className="sr-label">{s.label}</div>
              <div className="sr-meta">
                <span>Your value: {s.feature_value != null ? s.feature_value.toFixed(3) : '—'}</span>
                <span className="shap-badge shap-good">
                  SHAP: {s.shap_value.toFixed(4)}
                  <InfoTip text={`SHAP = ${s.shap_value.toFixed(4)} means this feature reduced your default probability by ${Math.abs(s.shap_value * 100).toFixed(2)} percentage points compared to the average applicant.`} />
                </span>
              </div>
            </div>
          ))}
        </div>

        <div className="risks">
          <h3>
            Risk Factors
            <InfoTip text="Features whose SHAP value is positive — they pushed your predicted default probability UP. These are the principal reasons for adverse action under RBI Digital Lending Guidelines. TreeSHAP is used (exact, not approximate) so the same input always gives the same reason." />
          </h3>
          {data.risks.length === 0 && <p className="empty-note">No SHAP data (thin-file route)</p>}
          {data.risks.map((r, i) => (
            <div key={i} className="sr-item sr-risk">
              <div className="sr-label">{r.label}</div>
              <div className="sr-meta">
                <span>Your value: {r.feature_value != null ? r.feature_value.toFixed(3) : '—'}</span>
                <span className="shap-badge shap-bad">
                  SHAP: +{r.shap_value.toFixed(4)}
                  <InfoTip text={`SHAP = +${r.shap_value.toFixed(4)} means this feature increased your default probability by ${(r.shap_value * 100).toFixed(2)} percentage points compared to the average applicant.`} />
                </span>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* ── DiCE Actions ─────────────────────────────────────────── */}
      {data.actions_to_improve.length > 0 && (
        <div className="actions-section">
          <h3>
            Actions to Improve
            <InfoTip text="DiCE (Diverse Counterfactual Explanations) searches for the minimum realistic change to your input features that would move you into a better eligibility tier. These are not arbitrary suggestions — DiCE constrains changes to values seen in the training population and respects monotone constraints (e.g. it will never suggest increasing your EMI bounce rate)." />
          </h3>
          <p className="actions-note">Minimum changes projected to improve your eligibility tier (DiCE counterfactuals).</p>
          {data.actions_to_improve.map((action, i) => (
            <div key={i} className="action-card">
              <div className="action-num">Option {i + 1}</div>
              {action.changes.map((c, j) => (
                <div key={j} className="action-change">
                  <span className="action-icon">-&gt;</span>
                  <span className="action-label">{c.label}</span>
                  <span className="action-from">{c.current_value.toFixed(3)}</span>
                  <span className="action-arrow">-&gt;</span>
                  <span className="action-to">{c.target_value.toFixed(3)}</span>
                </div>
              ))}
            </div>
          ))}
        </div>
      )}

      {/* ── Missing pillars ──────────────────────────────────────── */}
      {data.missing_pillars.length > 0 && (
        <div className="missing-section">
          <h3>
            Data Gaps
            <InfoTip text="Pillars where fewer than 50% of features were provided. Missing pillars reduce the Coverage Index. If Coverage Index falls below 0.60, the system routes to the thin-file model with wider confidence intervals. Providing data for these pillars may move you to the full model and improve your score." />
          </h3>
          <div className="missing-list">
            {data.missing_pillars.map(p => (
              <span key={p} className="missing-tag">{PILLAR_SHORT[p] || p}</span>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

import React, { useState } from 'react';
import InfoTip from './InfoTip';

const DEMO_PROFILES = {
  healthy: {
    label: 'Healthy MSME (Textile Mfg, Gujarat)',
    data: {
      gstin: '24AABCU9603R1ZP', industry_type: 'manufacturing',
      turnover_band: '1Cr-5Cr', state: 'Gujarat',
      inflow_cv: 0.18, min_balance_days: 18, inflow_outflow_lag: 2,
      drawdown_recovery_days: 8, loss_absorption_buffer: 35,
      gst_mismatch_pct: 0.02, gst_filing_punctuality: 0.96,
      itc_reversal_freq: 0.02, buyer_concentration_hhi: 0.22,
      ext_source_1: 0.72, ext_source_2: 0.68, ext_source_3: 0.74,
      epfo_payment_regularity: 0.95, utility_delinquency_flag: 0,
      gst_late_fee_incidence: 0.03, emi_bounce_rate: 0.01, avg_days_past_due: 1,
      epfo_headcount_delta_6m: 0.08, epfo_headcount_delta_12m: 0.12,
      electricity_kwh_trend: 0.09, supplier_diversity_score: 0.72,
      debt_to_inflow_ratio: 2.1, current_ratio_proxy: 1.8,
      working_capital_cycle_days: 45, annuity_to_income_ratio: 0.18,
      gstin_age_years: 5.2, address_churn_flag: 0, promoter_churn_flag: 0,
      directorship_overlap_flag: 0, days_employed_years: 6.5,
      gst_annual_turnover_lakhs: 320, electricity_kwh_monthly: 4200,
      epfo_headcount: 38, salary_outflow_monthly_lakhs: 5.2,
      gstr1_sales_lakhs: 27, upi_credit_inflow_lakhs: 18,
      bank_credit_inflow_lakhs: 9, ewaybill_value_lakhs: 22,
    }
  },
  risky: {
    label: 'High-Risk MSME (Trading, Maharashtra)',
    data: {
      gstin: '27AABCU9603R1ZM', industry_type: 'trading',
      turnover_band: '25L-1Cr', state: 'Maharashtra',
      inflow_cv: 0.62, min_balance_days: 3, inflow_outflow_lag: 14,
      drawdown_recovery_days: 45, loss_absorption_buffer: 5,
      gst_mismatch_pct: 0.21, gst_filing_punctuality: 0.54,
      itc_reversal_freq: 0.18, buyer_concentration_hhi: 0.72,
      ext_source_1: 0.31, ext_source_2: 0.28, ext_source_3: 0.25,
      epfo_payment_regularity: 0.52, utility_delinquency_flag: 1,
      gst_late_fee_incidence: 0.38, emi_bounce_rate: 0.22, avg_days_past_due: 18,
      epfo_headcount_delta_6m: -0.15, epfo_headcount_delta_12m: -0.22,
      electricity_kwh_trend: -0.12, supplier_diversity_score: 0.21,
      debt_to_inflow_ratio: 8.4, current_ratio_proxy: 0.6,
      working_capital_cycle_days: 180, annuity_to_income_ratio: 0.58,
      gstin_age_years: 1.2, address_churn_flag: 1, promoter_churn_flag: 1,
      directorship_overlap_flag: 1, days_employed_years: 0.8,
      gst_annual_turnover_lakhs: 68, electricity_kwh_monthly: 180,
      epfo_headcount: 12, salary_outflow_monthly_lakhs: 0.4,
      gstr1_sales_lakhs: 6, upi_credit_inflow_lakhs: 14,
      bank_credit_inflow_lakhs: 3, ewaybill_value_lakhs: 2,
    }
  },
  thinfile: {
    label: 'Thin-File MSME (New Services, Delhi)',
    data: {
      gstin: '07AABCU9603R1ZD', industry_type: 'services',
      turnover_band: '<25L', state: 'Delhi',
      gstin_age_years: 0.8, gst_filing_punctuality: 0.82,
      electricity_kwh_trend: 0.04, epfo_payment_regularity: 0.88,
      epfo_headcount_delta_6m: 0.05, utility_delinquency_flag: 0,
      gst_late_fee_incidence: 0.08,
    }
  }
};

// How each field is calculated from raw API data
const FIELD_INFO = {
  // P1
  inflow_cv: 'Calculated from bank statement (Setu AA): std(monthly credit totals) / mean(monthly credit totals) over 12 months. E.g. if monthly inflows are Rs80k, Rs1.2L, Rs60k — CV = 0.31. Higher = more volatile = higher risk.',
  min_balance_days: 'Calculated from bank statement (Setu AA): count of days in the last 12 months where the running account balance fell below 10% of that month\'s average inflow. More days below minimum = less financial cushion.',
  inflow_outflow_lag: 'Calculated from bank statement (Setu AA): average number of days between a credit (money received) and the next debit (money spent). Shorter lag = firm spends immediately = less buffer. Measured in days.',
  drawdown_recovery_days: 'Calculated from bank statement (Setu AA): after the largest single debit in a month, count the days until the balance returned to its pre-debit level. Longer recovery = less resilient to cash shocks.',
  loss_absorption_buffer: 'Calculated from bank statement (Setu AA): (average monthly balance - average monthly expenses) / average daily expenses. Result = number of days the firm can pay its bills from reserves alone, without any new inflow.',
  // P2
  gst_mismatch_pct: 'Calculated from GSTN API: abs(GSTR-1 outward sales - GSTR-3B net taxable value) / GSTR-1 sales. GSTR-1 and 3B should match. Mismatch indicates either under-reporting or accounting errors. 0 = perfect match.',
  gst_filing_punctuality: 'Calculated from GSTN API: count(monthly returns filed on or before due date) / total returns filed, over the last 24 months. 1.0 = always on time. 0.5 = late half the time.',
  itc_reversal_freq: 'Calculated from GSTN API: count(ITC claims later reversed by GSTN) / count(total ITC claims) over 24 months. Reversals happen when input credit is disallowed — signals accounting errors or inflated claims.',
  buyer_concentration_hhi: 'Calculated from GSTR-1 e-way bills: sum of (each buyer\'s share of total sales)^2. Called HHI (Herfindahl-Hirschman Index). 1.0 = all sales to one buyer (very risky). 0.1 = 10 equal buyers (diversified).',
  ext_source_1: 'Normalised credit bureau score (0 to 1, higher = better). Source: CIBIL API (paid per-pull). Represents creditworthiness based on past repayment history across all lenders.',
  ext_source_2: 'Normalised credit bureau score from a second bureau. Source: Experian India API. Cross-checking two bureaus catches gaps in coverage (not all lenders report to both).',
  ext_source_3: 'Normalised credit bureau score from a third bureau. Source: Equifax India API. Three-bureau cross-check is the RBI-recommended practice for MSME credit.',
  // P3
  epfo_payment_regularity: 'Calculated from EPFO public ECR data: count(months where PF was deposited by the 15th of the following month) / total months over 24 months. 1.0 = never late. EPFO filings are mandatory and publicly verifiable.',
  utility_delinquency_flag: 'Binary flag from state DISCOM API: 1 if any electricity bill was paid after the due date in the last 12 months, 0 if all paid on time. Even one missed utility payment is flagged — it indicates cash stress.',
  gst_late_fee_incidence: 'Calculated from GSTN API: fraction of GSTR filings where a late fee (Rs 50/day) was charged. Directly observable in the GSTN portal. 0 = always on time. 0.5 = late half the time.',
  emi_bounce_rate: 'Calculated from bank statement (Setu AA): fraction of scheduled EMI/loan repayment debits where the amount paid was less than 95% of the amount due. Bounced EMI = insufficient funds on payment date.',
  avg_days_past_due: 'Calculated from installment/loan history: mean of max(0, actual_payment_date - scheduled_due_date) across all past installments. 0 = always paid on or before due date. 15 = paid 15 days late on average.',
  // P4
  epfo_headcount_delta_6m: 'Calculated from EPFO ECR (publicly available monthly filing): (employees this month - employees 6 months ago) / employees 6 months ago. Positive = workforce grew. Negative = layoffs or downsizing.',
  epfo_headcount_delta_12m: 'Same formula as 6-month delta but over 12 months. Captures long-term workforce trend that the 6-month window may miss if the decline started 7-8 months ago.',
  electricity_kwh_trend: 'Calculated from state DISCOM monthly bills: linear regression slope of monthly kWh consumption over 12 months, divided by mean monthly kWh (normalised). Positive = growing energy use = growing operations.',
  supplier_diversity_score: 'Calculated from GSTR-2 purchase data: 1 minus the HHI of supplier concentration. 1.0 = purchases spread across many suppliers (resilient supply chain). 0 = single supplier (one disruption kills operations).',
  // P5
  debt_to_inflow_ratio: 'Calculated from bank statement + credit bureau: total outstanding loan balance / annual income. Higher = more of the firm\'s future income is already committed to repaying existing debt.',
  current_ratio_proxy: 'Approximated as 1 / (annuity_to_income_ratio + 0.01). No balance sheet needed. Higher = more liquid (more income free after paying existing obligations). A real current ratio would need ITR data.',
  working_capital_cycle_days: 'Approximated as (total loan balance / monthly EMI) x 30 = days to clear current debt. Real version uses (inventory days + receivables days - payables days) from GSTR turnover and bank timing.',
  annuity_to_income_ratio: 'Calculated from bank statement (Setu AA): total monthly EMI and loan repayment outflows / average monthly bank inflows. Higher = more income already locked into debt repayment. 0.18 means 18% of income goes to EMIs.',
  // P6
  gstin_age_years: 'Calculated from GSTN public registry: (today\'s date - GSTIN registration date) in years. Free API, no authentication needed. Older GSTIN = more established business = lower fraud and failure risk.',
  address_churn_flag: '1 if the registered business address changed in the last 12 months on the GSTN portal or MCA filing. 0 = stable address. Address changes can indicate business instability or attempt to evade creditors. Source: Karza API.',
  promoter_churn_flag: '1 if a director or major shareholder changed in the last 12 months on MCA records. 0 = stable ownership. Ownership changes mid-credit cycle increase repayment risk. Source: MCA portal via Karza API.',
  directorship_overlap_flag: '1 if this promoter\'s DIN (Director Identification Number) is linked to any company classified as NPA, wilful defaulter, or struck-off on MCA. 0 = clean. Source: Karza API cross-references MCA + RBI defaulter list.',
  days_employed_years: 'Calculated from MCA/GSTN: years since the promoter first registered a business entity. Longer track record = more experienced operator = lower failure risk.',
  // Consistency inputs
  gst_annual_turnover_lakhs: 'Raw input from GSTN API: total taxable sales declared in GSTR-1 for the last 12 months, in Rs lakhs. Used only for consistency checks (vs electricity and e-way bills) — not directly in the ML score.',
  electricity_kwh_monthly: 'Raw input from state DISCOM API: average monthly electricity consumption in kWh over the last 6 months. Used to cross-check against declared GST turnover — you cannot manufacture Rs 5 crore of goods on 200 kWh/month.',
  epfo_headcount: 'Raw input from EPFO public data: number of employees registered under EPFO (PF) as of the most recent ECR filing. Used to cross-check against salary outflow — if 50 employees but only Rs 1L salary outflow, something is wrong.',
  salary_outflow_monthly_lakhs: 'Raw input from bank statement (Setu AA): total monthly salary-tagged credit debits in Rs lakhs. Used to cross-check against EPFO headcount. Divide by headcount to get implied salary per employee.',
  gstr1_sales_lakhs: 'Raw input from GSTN API: total outward supplies declared in GSTR-1 for the most recent month, in Rs lakhs. Cross-checked against bank + UPI inflows — if you declared Rs 30L sales but only Rs 5L came into your account, investigation needed.',
  upi_credit_inflow_lakhs: 'Raw input from Setu AA: total UPI credit transactions for the most recent month in Rs lakhs. Combined with bank inflows to compute the collection ratio vs GSTR-1 sales.',
  bank_credit_inflow_lakhs: 'Raw input from Setu AA: total non-UPI bank credit transactions (NEFT, RTGS, cheques) for the most recent month in Rs lakhs.',
  ewaybill_value_lakhs: 'Raw input from GSTN e-way bill portal: total value of e-way bills generated in the last 12 months in Rs lakhs. For manufacturing/trading firms, this should track reasonably close to GST turnover.',
};

export default function MSMEForm({ onSubmit, loading }) {
  const [formData, setFormData] = useState({ gstin: '', industry_type: 'manufacturing' });
  const [activeDemo, setActiveDemo] = useState(null);

  const handleDemo = (key) => {
    setFormData(DEMO_PROFILES[key].data);
    setActiveDemo(key);
  };

  const handleChange = (e) => {
    const { name, value, type } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: type === 'number' ? (value === '' ? undefined : parseFloat(value)) : value
    }));
    setActiveDemo(null);
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    onSubmit(formData);
  };

  return (
    <div className="form-card">
      <h2>Assess MSME</h2>

      <div className="demo-buttons">
        <span className="demo-label">Demo profiles:</span>
        {Object.entries(DEMO_PROFILES).map(([key, p]) => (
          <button key={key} className={`demo-btn ${activeDemo === key ? 'active' : ''}`}
            onClick={() => handleDemo(key)} type="button">
            {p.label}
          </button>
        ))}
      </div>

      <form onSubmit={handleSubmit}>
        <div className="form-section">
          <h3>Identity</h3>
          <div className="form-row">
            <div className="form-group">
              <label>GSTIN *</label>
              <input name="gstin" value={formData.gstin || ''} onChange={handleChange} required placeholder="e.g. 24AABCU9603R1ZP" />
            </div>
            <div className="form-group">
              <label>Industry Type</label>
              <select name="industry_type" value={formData.industry_type || 'manufacturing'} onChange={handleChange}>
                <option value="manufacturing">Manufacturing</option>
                <option value="services">Services</option>
                <option value="trading">Trading</option>
              </select>
            </div>
            <div className="form-group">
              <label>Turnover Band</label>
              <select name="turnover_band" value={formData.turnover_band || ''} onChange={handleChange}>
                <option value="">Select</option>
                <option value="<25L">Below Rs 25L</option>
                <option value="25L-1Cr">Rs 25L - Rs 1Cr</option>
                <option value="1Cr-5Cr">Rs 1Cr - Rs 5Cr</option>
                <option value=">5Cr">Above Rs 5Cr</option>
              </select>
            </div>
            <div className="form-group">
              <label>State</label>
              <input name="state" value={formData.state || ''} onChange={handleChange} placeholder="e.g. Gujarat" />
            </div>
          </div>
        </div>

        <PillarSection title="P1 — Cash Flow Resilience" subtitle="Source: Bank statement via Setu Account Aggregator API" fields={[
          { name: 'inflow_cv',              label: 'Cash Flow Volatility (CV)',        step: 0.01 },
          { name: 'min_balance_days',       label: 'Days Above Minimum Balance',       step: 1    },
          { name: 'inflow_outflow_lag',     label: 'Payment Collection Lag (days)',    step: 1    },
          { name: 'drawdown_recovery_days', label: 'Recovery Time from Dip (days)',    step: 1    },
          { name: 'loss_absorption_buffer', label: 'Cash Buffer (days of outflow)',    step: 1    },
        ]} formData={formData} onChange={handleChange} />

        <PillarSection title="P2 — Revenue Quality" subtitle="Source: GSTN Sandbox API + credit bureau APIs (CIBIL / Experian / Equifax)" fields={[
          { name: 'gst_mismatch_pct',         label: 'GSTR-1 vs GSTR-3B Mismatch',     step: 0.01 },
          { name: 'gst_filing_punctuality',   label: 'GST Filing Punctuality (0-1)',    step: 0.01 },
          { name: 'itc_reversal_freq',        label: 'ITC Reversal Frequency',          step: 0.01 },
          { name: 'buyer_concentration_hhi',  label: 'Buyer Concentration HHI (0-1)',   step: 0.01 },
          { name: 'ext_source_1',             label: 'Credit Bureau Score 1 (0-1)',     step: 0.01 },
          { name: 'ext_source_2',             label: 'Credit Bureau Score 2 (0-1)',     step: 0.01 },
          { name: 'ext_source_3',             label: 'Credit Bureau Score 3 (0-1)',     step: 0.01 },
        ]} formData={formData} onChange={handleChange} />

        <PillarSection title="P3 — Obligation Discipline" subtitle="Source: EPFO public data + state DISCOM API + GSTN + bank statement" fields={[
          { name: 'epfo_payment_regularity', label: 'EPFO Payment Regularity (0-1)',  step: 0.01 },
          { name: 'utility_delinquency_flag',label: 'Utility Delinquency (0=No, 1=Yes)', step: 1 },
          { name: 'gst_late_fee_incidence',  label: 'GST Late Filing Rate',           step: 0.01 },
          { name: 'emi_bounce_rate',         label: 'EMI Bounce Rate',                step: 0.01 },
          { name: 'avg_days_past_due',       label: 'Avg Days Past Due',              step: 1    },
        ]} formData={formData} onChange={handleChange} />

        <PillarSection title="P4 — Operational Vitality" subtitle="Source: EPFO public ECR filings + state DISCOM API + GSTN purchase data" fields={[
          { name: 'epfo_headcount_delta_6m',  label: 'Headcount Growth 6m (-1 to +1)',  step: 0.01 },
          { name: 'epfo_headcount_delta_12m', label: 'Headcount Growth 12m (-1 to +1)', step: 0.01 },
          { name: 'electricity_kwh_trend',    label: 'Electricity Trend (-1 to +1)',     step: 0.01 },
          { name: 'supplier_diversity_score', label: 'Supplier Diversity (0-1)',         step: 0.01 },
        ]} formData={formData} onChange={handleChange} />

        <PillarSection title="P5 — Leverage & Liquidity" subtitle="Source: Bank statement (Setu AA) + credit bureau" fields={[
          { name: 'debt_to_inflow_ratio',       label: 'Debt-to-Income Ratio',          step: 0.1  },
          { name: 'current_ratio_proxy',        label: 'Liquidity Ratio',               step: 0.1  },
          { name: 'working_capital_cycle_days', label: 'Working Capital Cycle (days)',  step: 1    },
          { name: 'annuity_to_income_ratio',    label: 'Loan Repayment Burden',         step: 0.01 },
        ]} formData={formData} onChange={handleChange} />

        <PillarSection title="P6 — Stability & Vintage" subtitle="Source: GSTN public registry + MCA portal via Karza API" fields={[
          { name: 'gstin_age_years',          label: 'Business Vintage (years)',           step: 0.1 },
          { name: 'address_churn_flag',       label: 'Address Changed (0=No, 1=Yes)',      step: 1   },
          { name: 'promoter_churn_flag',      label: 'Promoter Changed (0=No, 1=Yes)',     step: 1   },
          { name: 'directorship_overlap_flag',label: 'Directorship Risk (0=No, 1=Yes)',    step: 1   },
          { name: 'days_employed_years',      label: 'Employment Continuity (years)',      step: 0.1 },
        ]} formData={formData} onChange={handleChange} />

        <div className="form-section">
          <h3>Consistency Engine Inputs <span style={{ fontSize: 11, fontWeight: 400, color: '#94a3b8' }}>(optional — used for cross-source fraud detection only, not in the ML score)</span></h3>
          <div className="form-row">
            {[
              { name: 'gst_annual_turnover_lakhs',    label: 'Annual GST Turnover (Rs Lakhs)'    },
              { name: 'electricity_kwh_monthly',      label: 'Monthly Electricity (kWh)'         },
              { name: 'epfo_headcount',               label: 'EPFO Headcount'                    },
              { name: 'salary_outflow_monthly_lakhs', label: 'Monthly Salary Outflow (Rs Lakhs)' },
              { name: 'gstr1_sales_lakhs',            label: 'Monthly GSTR-1 Sales (Rs Lakhs)'   },
              { name: 'upi_credit_inflow_lakhs',      label: 'Monthly UPI Inflow (Rs Lakhs)'     },
              { name: 'bank_credit_inflow_lakhs',     label: 'Monthly Bank Credit (Rs Lakhs)'    },
              { name: 'ewaybill_value_lakhs',         label: 'E-Way Bill Value (Rs Lakhs)'       },
            ].map(f => (
              <div className="form-group" key={f.name}>
                <label>
                  {f.label}
                  {FIELD_INFO[f.name] && <InfoTip text={FIELD_INFO[f.name]} />}
                </label>
                <input type="number" name={f.name} value={formData[f.name] ?? ''} onChange={handleChange} step="0.1" placeholder="optional" />
              </div>
            ))}
          </div>
        </div>

        <button type="submit" className="submit-btn" disabled={loading}>
          {loading ? 'Assessing...' : 'Generate Financial Health Card'}
        </button>
      </form>
    </div>
  );
}

function PillarSection({ title, subtitle, fields, formData, onChange }) {
  return (
    <div className="form-section">
      <h3>
        {title}
        {subtitle && <span style={{ fontSize: 10, fontWeight: 400, color: '#94a3b8', marginLeft: 8 }}>{subtitle}</span>}
      </h3>
      <div className="form-row">
        {fields.map(f => (
          <div className="form-group" key={f.name}>
            <label>
              {f.label}
              {FIELD_INFO[f.name] && <InfoTip text={FIELD_INFO[f.name]} />}
            </label>
            <input
              type="number"
              name={f.name}
              value={formData[f.name] ?? ''}
              onChange={onChange}
              step={f.step || 0.01}
              placeholder="optional"
            />
          </div>
        ))}
      </div>
    </div>
  );
}

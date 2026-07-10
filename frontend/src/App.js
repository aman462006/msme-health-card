import React, { useState } from 'react';
import LandingPage from './components/LandingPage';
import MSMEForm from './components/MSMEForm';
import HealthCard from './components/HealthCard';
import './App.css';

export default function App() {
  const [page, setPage] = useState('landing');
  const [cardData, setCardData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleAssess = async (formData) => {
    setLoading(true);
    setError(null);
    setCardData(null);
    try {
      const API = process.env.REACT_APP_API_URL || 'http://localhost:8000';
      const res = await fetch(`${API}/assess`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(formData),
      });
      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || 'Assessment failed');
      }
      setCardData(await res.json());
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  const goAssess = () => { setPage('assess'); window.scrollTo(0, 0); };
  const goLanding = () => { setPage('landing'); window.scrollTo(0, 0); };

  return (
    <div className={`app${page === 'assess' ? ' assess-mode' : ''}`}>
      <header className="app-header">
        <div className="header-inner">
          <div className="header-logo">IDBI Bank</div>
          <nav className="header-nav">
            <button className={`nav-link ${page === 'landing' ? 'nav-active' : ''}`} onClick={goLanding}>About</button>
            <button className={`nav-link ${page === 'assess' ? 'nav-active' : ''}`} onClick={goAssess}>Assess</button>
          </nav>
          {page === 'landing' && (
            <button className="header-cta" onClick={goAssess}>Get Started →</button>
          )}
        </div>
      </header>

      {page === 'landing' ? (
        <LandingPage onGetStarted={goAssess} />
      ) : (
        <main className="app-main">
          <div className="data-notice">
            <span className="data-notice-icon">⚠</span>
            <div>
              <strong>About This Prototype's Training Data</strong>
              Scores, weights, and probabilities are currently based on the{' '}
              <strong>Home Credit Default Risk dataset</strong> (307,511 personal loan records from Kaggle),
              used as a proxy because real MSME loan outcome data is not yet connected.
              Every value below is a <strong>derived metric</strong> computed from bank statements (Setu AA),
              GST returns (GSTN API), and EPFO records. Hover the <strong>ⓘ</strong> icon next to each
              field to see exactly how it is calculated.
            </div>
          </div>
          <MSMEForm onSubmit={handleAssess} loading={loading} />
          {error && <div className="error-banner">{error}</div>}
          {cardData && <HealthCard data={cardData} />}
        </main>
      )}

      <footer className="app-footer">
        <p>Monotonic LightGBM · TreeSHAP · MAPIE Conformal Prediction · RBI Digital Lending Guidelines</p>
      </footer>
    </div>
  );
}

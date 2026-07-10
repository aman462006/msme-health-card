import React, { useState, useRef } from 'react';
import ReactDOM from 'react-dom';

export default function InfoTip({ text }) {
  const [visible, setVisible] = useState(false);
  const [pos, setPos] = useState({ top: 0, left: 0 });
  const iconRef = useRef(null);

  const show = () => {
    if (iconRef.current) {
      const r = iconRef.current.getBoundingClientRect();
      setPos({
        top: r.top + window.scrollY - 8,
        left: r.left + r.width / 2 + window.scrollX,
      });
    }
    setVisible(true);
  };

  const hide = () => setVisible(false);

  const tooltip = visible ? ReactDOM.createPortal(
    <div style={{
      position: 'absolute',
      top: pos.top,
      left: pos.left,
      transform: 'translate(-50%, -100%)',
      background: '#1e293b',
      color: '#f1f5f9',
      fontSize: 11,
      lineHeight: 1.5,
      padding: '8px 10px',
      borderRadius: 6,
      width: 240,
      zIndex: 99999,
      pointerEvents: 'none',
      boxShadow: '0 4px 12px rgba(0,0,0,0.35)',
      whiteSpace: 'normal',
    }}>
      {text}
      <div style={{
        position: 'absolute', top: '100%', left: '50%',
        transform: 'translateX(-50%)',
        borderWidth: 5, borderStyle: 'solid',
        borderColor: '#1e293b transparent transparent transparent',
      }} />
    </div>,
    document.body
  ) : null;

  return (
    <span style={{ display: 'inline-flex', alignItems: 'center', verticalAlign: 'middle' }}>
      <i
        ref={iconRef}
        onMouseEnter={show}
        onMouseLeave={hide}
        style={{
          display: 'inline-flex', alignItems: 'center', justifyContent: 'center',
          width: 15, height: 15, borderRadius: '50%',
          background: '#cbd5e1', color: '#475569',
          fontSize: 9, fontWeight: 800, fontStyle: 'normal',
          cursor: 'help', marginLeft: 5, flexShrink: 0,
          userSelect: 'none', lineHeight: 1,
        }}
      >i</i>
      {tooltip}
    </span>
  );
}

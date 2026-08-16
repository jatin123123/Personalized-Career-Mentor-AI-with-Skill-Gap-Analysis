import { motion } from 'framer-motion';
import { useState, useEffect } from 'react';
import { CheckCircle2, XCircle, PlusCircle, Percent, Award } from 'lucide-react';
import TiltCard from './TiltCard';

const ICONS = {
  'Matched Skills': { icon: <CheckCircle2 size={20} />, bg: 'rgba(16, 185, 129, 0.12)', color: '#059669' },
  'Missing Skills': { icon: <XCircle size={20} />,      bg: 'rgba(244, 63, 94, 0.12)',  color: '#e11d48' },
  'Extra Skills':   { icon: <PlusCircle size={20} />,   bg: 'rgba(245, 158, 11, 0.12)', color: '#d97706' },
  'Match Score':    { icon: <Percent size={20} />,      bg: 'rgba(79, 70, 229, 0.12)',  color: '#4f46e5' },
};

export default function MetricCard({ label, value, color, delay = 0 }) {
  const iconConfig = ICONS[label] || { icon: <Award size={20} />, bg: 'rgba(79, 70, 229, 0.12)', color: '#4f46e5' };

  return (
    <motion.div
      initial={{ opacity: 0, y: 24, scale: 0.96 }}
      animate={{ opacity: 1, y: 0, scale: 1 }}
      transition={{ delay, duration: 0.5, ease: [0.16, 1, 0.3, 1] }}
    >
      <TiltCard className="metric-card">
        <div
          className="metric-icon"
          style={{ background: iconConfig.bg, color: iconConfig.color }}
        >
          {iconConfig.icon}
        </div>
        <AnimatedValue value={value} color={color} />
        <div className="metric-label">{label}</div>
      </TiltCard>
    </motion.div>
  );
}

function AnimatedValue({ value, color }) {
  const isNumber = typeof value === 'number';
  const isPct = typeof value === 'string' && value.endsWith('%');
  const numericVal = isNumber ? value : isPct ? parseInt(value, 10) : null;
  const [display, setDisplay] = useState(numericVal !== null ? 0 : value);

  useEffect(() => {
    if (numericVal === null || isNaN(numericVal)) {
      setDisplay(value);
      return;
    }

    let start = 0;
    const end = numericVal;
    const duration = 1200;
    const startTime = performance.now();

    function step(now) {
      const progress = Math.min((now - startTime) / duration, 1);
      const eased = 1 - Math.pow(1 - progress, 3); // cubic ease-out
      const curr = Math.round(start + (end - start) * eased);
      setDisplay(isPct ? `${curr}%` : curr);
      if (progress < 1) requestAnimationFrame(step);
    }
    requestAnimationFrame(step);
  }, [value, numericVal, isPct]);

  return (
    <div className="metric-value" style={color ? { color } : undefined}>
      {display}
    </div>
  );
}

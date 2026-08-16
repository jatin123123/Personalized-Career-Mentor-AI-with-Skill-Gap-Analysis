import { motion } from 'framer-motion';
import { Check, X, Plus } from 'lucide-react';

function parseSkills(value) {
  if (!value) return [];
  if (Array.isArray(value)) return value.map(String).filter(Boolean);
  if (typeof value === 'string') {
    const trimmed = value.trim();
    if (trimmed.startsWith('[') && trimmed.endsWith(']')) {
      try {
        const parsed = JSON.parse(trimmed);
        if (Array.isArray(parsed)) return parsed.map(String).filter(Boolean);
      } catch (e) {}
    }
    return trimmed.split(',').map((s) => s.trim()).filter((s) => s.length > 1);
  }
  return [];
}

const TYPE_CONFIG = {
  matched: { icon: <Check size={13} strokeWidth={3} />, label: 'Matched' },
  missing: { icon: <X size={13} strokeWidth={3} />, label: 'Missing' },
  extra:   { icon: <Plus size={13} strokeWidth={3} />, label: 'Extra' },
};

export default function SkillChips({ skills, type = 'matched' }) {
  const list = parseSkills(skills);
  const config = TYPE_CONFIG[type] || TYPE_CONFIG.matched;

  if (!list.length) {
    return <p style={{ fontSize: '0.84rem', color: 'var(--text-muted)', marginTop: 8 }}>None identified</p>;
  }

  return (
    <div className="chips">
      {list.map((skill, i) => (
        <motion.span
          key={i}
          className={`chip chip-${type}`}
          initial={{ opacity: 0, scale: 0.7, y: 8 }}
          animate={{ opacity: 1, scale: 1, y: 0 }}
          whileHover={{ scale: 1.06, y: -2 }}
          transition={{
            delay: i * 0.035,
            duration: 0.35,
            ease: [0.34, 1.56, 0.64, 1], // spring
          }}
        >
          <span style={{ display: 'flex', alignItems: 'center' }}>
            {config.icon}
          </span>
          <span>{skill}</span>
        </motion.span>
      ))}
    </div>
  );
}

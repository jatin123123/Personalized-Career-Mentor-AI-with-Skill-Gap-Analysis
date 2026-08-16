import { motion } from 'framer-motion';
import { Sparkles, FileText, ArrowRight, Bot, Cpu } from 'lucide-react';

export default function Header({ showHero = true, onGoHome, onTryHere }) {
  return (
    <>
      <nav className="navbar">
        <div className="navbar-inner">
          <button className="navbar-brand" onClick={onGoHome} type="button">
            <div className="navbar-logo">
              <Sparkles size={20} />
            </div>
            <span>Resume Matcher <span style={{ color: 'var(--primary)', fontSize: '0.8rem', fontWeight: 800 }}>AI</span></span>
          </button>

          <div className="navbar-actions">
            <div className="creator-badge">
              <span>Built by <strong>Jatin Jangid</strong></span>
            </div>
            <button className="nav-link-btn" onClick={onTryHere} type="button">
              <span>Try Analyzer</span>
              <ArrowRight size={15} />
            </button>
            <div className="navbar-status">
              <span className="status-dot" />
              <Cpu size={13} style={{ color: 'var(--primary)' }} />
              <span>Groq Llama-3.3</span>
            </div>
          </div>
        </div>
      </nav>

      {showHero && (
        <header className="header">
          <motion.div
            initial={{ opacity: 0, y: -16 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, ease: [0.16, 1, 0.3, 1] }}
          >
            <div className="header-badge">
              <span className="header-badge-dot" />
              <span>Next-Gen Career Mentor AI</span>
            </div>
          </motion.div>

          <motion.h1
            initial={{ opacity: 0, y: 24 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.1, ease: [0.16, 1, 0.3, 1] }}
          >
            Match your resume to <span className="gradient-text">any dream job</span>
          </motion.h1>

          <motion.p
            className="header-sub"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.2 }}
          >
            AI-powered skill gap intelligence. Compare your qualifications with target roles in milliseconds with deep semantic analysis.
          </motion.p>
        </header>
      )}
    </>
  );
}

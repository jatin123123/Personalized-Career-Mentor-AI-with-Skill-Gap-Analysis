import { motion } from 'framer-motion';
import { useState } from 'react';
import { Download, Globe, Check, Search, FileText, Zap, Sparkles, ArrowRight } from 'lucide-react';
import TiltCard from './TiltCard';
import { API_URL } from '../lib/api';

const STEPS = [
  { label: 'Download extension zip & extract folder' },
  { label: 'Navigate to chrome://extensions in Chrome/Brave' },
  { label: 'Enable Developer mode & click "Load unpacked"' },
  { label: 'Open any LinkedIn job & click the floating extension' },
];

const FEATURES = [
  { icon: <Search size={18} />, title: 'Automatic Job Scraping', desc: 'Reads the active job description directly from LinkedIn with 1 click.' },
  { icon: <FileText size={18} />, title: 'Local Resume Vault', desc: 'Upload your resume once — saved locally in chrome storage.' },
  { icon: <Zap size={18} />, title: 'Instant Fit Radar', desc: 'Instant match score, missing keywords, and interview guidance.' },
];

export default function ExtensionBanner() {
  const [downloading, setDownloading] = useState(false);
  const [done, setDone] = useState(false);

  async function handleDownload() {
    setDownloading(true);
    try {
      const res = await fetch(`${API_URL}/api/download-extension`);
      if (!res.ok) throw new Error('Download failed');
      const blob = await res.blob();
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = 'resume-matcher-extension.zip';
      a.click();
      URL.revokeObjectURL(url);
      setDone(true);
      setTimeout(() => setDone(false), 3000);
    } catch {
      alert('Could not reach the backend server. Please try again in a moment.');
    } finally {
      setDownloading(false);
    }
  }

  return (
    <motion.section
      className="ext-section"
      initial={{ opacity: 0, y: 36 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true, margin: '-60px' }}
      transition={{ duration: 0.6, ease: [0.16, 1, 0.3, 1] }}
    >
      <div className="ext-eyebrow">
        <Globe size={15} />
        <span>Official Browser Extension</span>
      </div>

      <div className="ext-grid">
        {/* Left: Description & Features */}
        <div className="ext-copy">
          <h2 className="ext-title">
            Analyze LinkedIn jobs <span className="gradient-text">directly in your browser</span>
          </h2>
          <p className="ext-desc">
            Never copy and paste job descriptions manually again. Our lightweight Chrome extension extracts job specifications instantly and scores your resume in real time.
          </p>

          <div className="ext-features">
            {FEATURES.map((f, i) => (
              <motion.div
                key={i}
                className="ext-feature"
                initial={{ opacity: 0, x: -12 }}
                whileInView={{ opacity: 1, x: 0 }}
                viewport={{ once: true }}
                transition={{ delay: 0.1 + i * 0.08 }}
              >
                <div className="ext-feature-icon">{f.icon}</div>
                <div>
                  <div className="ext-feature-title">{f.title}</div>
                  <div className="ext-feature-desc">{f.desc}</div>
                </div>
              </motion.div>
            ))}
          </div>

          <motion.button
            className="ext-download-btn"
            onClick={handleDownload}
            disabled={downloading}
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.98 }}
          >
            {done ? (
              <>
                <Check size={18} />
                <span>Downloaded Successfully!</span>
              </>
            ) : downloading ? (
              <>
                <Sparkles size={18} className="animate-spin" />
                <span>Preparing Extension Zip...</span>
              </>
            ) : (
              <>
                <Download size={18} />
                <span>Download Extension (.zip)</span>
              </>
            )}
          </motion.button>
          <div className="ext-note">100% Free · Manifest V3 Compliant · Runs Locally</div>
        </div>

        {/* Right: Installation Card */}
        <div className="ext-steps-col">
          <TiltCard className="ext-steps-card">
            <div className="ext-steps-title">Quick 30-Second Setup</div>
            {STEPS.map((s, i) => (
              <motion.div
                key={i}
                className="ext-step"
                initial={{ opacity: 0, x: 14 }}
                whileInView={{ opacity: 1, x: 0 }}
                viewport={{ once: true }}
                transition={{ delay: 0.15 + i * 0.08 }}
              >
                <div className="ext-step-num">{i + 1}</div>
                <div style={{ fontSize: '0.88rem', fontWeight: 600, color: 'var(--text)' }}>
                  {s.label}
                </div>
              </motion.div>
            ))}

            {/* Simulated Chrome Browser Mockup */}
            <div className="ext-preview">
              <div className="ext-preview-bar">
                <div className="ext-preview-dot" />
                <div className="ext-preview-dot ext-preview-dot--2" />
                <div className="ext-preview-dot ext-preview-dot--3" />
                <div className="ext-preview-url">linkedin.com/jobs/view/39482019/</div>
              </div>
              <div className="ext-preview-badge">
                <span className="status-dot" />
                <span>LinkedIn Job Detected — 1 Click to Match</span>
              </div>
            </div>
          </TiltCard>
        </div>
      </div>
    </motion.section>
  );
}

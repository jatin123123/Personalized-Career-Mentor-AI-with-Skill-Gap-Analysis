import { useState } from 'react';
import { motion } from 'framer-motion';
import { 
  Sparkles, 
  Download, 
  ArrowRight, 
  CheckCircle2, 
  Layers, 
  Zap, 
  BarChart3, 
  Target, 
  Cpu, 
  Globe, 
  FileText,
  TrendingUp,
  ShieldCheck
} from 'lucide-react';
import TiltCard from './TiltCard';
import { API_URL } from '../lib/api';

const features = [
  {
    icon: <Target size={24} style={{ color: 'var(--primary)' }} />,
    title: 'Precision Skill Mapping',
    text: 'Identifies exact hard & soft skill matches, semantic overlaps, and crucial missing qualifications in real-time.',
  },
  {
    icon: <Zap size={24} style={{ color: 'var(--accent-cyan)' }} />,
    title: 'Instant Action Recommendations',
    text: 'Transforms gaps into prioritized portfolio additions, bullet point revisions, and tailored interview answers.',
  },
  {
    icon: <Cpu size={24} style={{ color: 'var(--accent-rose)' }} />,
    title: 'Multi-Model Intelligence',
    text: 'Powered by Groq Llama-3.3 70B & Qwen with customizable analysis depth from quick check to comprehensive audit.',
  },
];

const telemetry = [
  { value: '94%', label: 'Match Confidence' },
  { value: '< 2.1s', label: 'Inference Speed' },
  { value: '40+ pts', label: 'Average Gap Rank' },
  { value: '100% Free', label: 'Local & Secure' },
];

export default function HomeLanding({ onTryHere }) {
  const [downloading, setDownloading] = useState(false);
  const [downloaded, setDownloaded] = useState(false);

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
      setDownloaded(true);
      setTimeout(() => setDownloaded(false), 3000);
    } catch {
      alert('Could not reach backend. Please ensure the backend server is running on port 8001.');
    } finally {
      setDownloading(false);
    }
  }

  return (
    <motion.div
      className="home-shell"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      transition={{ duration: 0.35 }}
    >
      {/* Hero Section */}
      <section className="home-hero">
        <div className="hero-copy">
          <motion.div
            className="home-kicker"
            initial={{ opacity: 0, y: 14 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.45 }}
          >
            <span className="kicker-dot" />
            <span>AI Career Optimization Engine</span>
          </motion.div>

          <motion.h1
            initial={{ opacity: 0, y: 24 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.08 }}
          >
            Land your next role with <span className="gradient-text">predictive intelligence</span>
          </motion.h1>

          <motion.p
            className="home-lede"
            initial={{ opacity: 0, y: 18 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.16 }}
          >
            Upload your resume, compare against any target job description, and receive an instant deep-level skill gap roadmap with AI-generated interview prep.
          </motion.p>

          <motion.div
            className="home-actions"
            initial={{ opacity: 0, y: 16 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.45, delay: 0.24 }}
          >
            <button className="home-primary-btn" onClick={onTryHere}>
              <span>Start Free Analysis</span>
              <ArrowRight size={18} />
            </button>
            <button className="home-secondary-btn" onClick={handleDownload} disabled={downloading}>
              <Globe size={18} style={{ color: 'var(--primary)' }} />
              <span>{downloaded ? 'Extension Downloaded!' : downloading ? 'Generating Zip...' : 'Get Chrome Extension'}</span>
            </button>
          </motion.div>
        </div>

        {/* 3D Holographic AI Scanner Visual */}
        <motion.div
          className="hero-visual"
          initial={{ opacity: 0, scale: 0.92, rotateY: -10 }}
          animate={{ opacity: 1, scale: 1, rotateY: 0 }}
          transition={{ duration: 0.8, delay: 0.15, ease: [0.16, 1, 0.3, 1] }}
        >
          <div className="scan-frame">
            <div className="scan-line" />
            <div className="hud-node hud-node-1" />
            <div className="hud-node hud-node-2" />
            <div className="hud-node hud-node-3" />
            <div className="hud-orbital hud-orbital-1" />
            <div className="hud-orbital hud-orbital-2" />

            {/* Main Holographic Widget */}
            <div className="visual-card-main">
              <div className="visual-topline">
                <span style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                  <Sparkles size={14} style={{ color: '#38bdf8' }} />
                  Live Resume Telemetry
                </span>
                <strong>Active</strong>
              </div>

              <div className="score-ring">
                <span>94</span>
                <small>% Match</small>
              </div>

              <div className="signal-bars">
                <i />
                <i />
                <i />
                <i />
                <i />
              </div>
            </div>

            {/* Side Priority Card */}
            <div className="visual-card-side">
              <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 8, fontSize: '0.75rem', color: 'rgba(255,255,255,0.7)' }}>
                <TrendingUp size={13} style={{ color: '#f43f5e' }} />
                <span>Skill Priority</span>
              </div>
              <strong style={{ fontSize: '0.85rem', color: '#38bdf8' }}>3 Crucial Gaps</strong>
              <div style={{ display: 'flex', flexDirection: 'column', gap: 6, marginTop: 10 }}>
                <div style={{ height: 6, borderRadius: 99, background: 'linear-gradient(90deg, #38bdf8, transparent)', width: '90%' }} />
                <div style={{ height: 6, borderRadius: 99, background: 'linear-gradient(90deg, #f59e0b, transparent)', width: '70%' }} />
                <div style={{ height: 6, borderRadius: 99, background: 'linear-gradient(90deg, #f43f5e, transparent)', width: '82%' }} />
              </div>
            </div>
          </div>
        </motion.div>
      </section>

      {/* Choose How to Analyze Section */}
      <section className="choice-grid" aria-label="Choose how to use Resume Matcher">
        <TiltCard className="choice-panel">
          <div className="choice-icon">
            <Sparkles size={26} />
          </div>
          <h2>Interactive Web Studio</h2>
          <p>
            Upload PDF resumes, paste job descriptions, customize LLM models &amp; temperature, and download full structured analysis reports.
          </p>
          <button className="choice-link" onClick={onTryHere}>
            <span>Open Studio</span>
            <ArrowRight size={16} />
          </button>
        </TiltCard>

        <TiltCard className="choice-panel">
          <div className="choice-icon" style={{ background: 'rgba(6, 182, 212, 0.1)', color: 'var(--accent-cyan)' }}>
            <Globe size={26} />
          </div>
          <h2>Browser Extension</h2>
          <p>
            Analyze LinkedIn &amp; job board listings directly in your browser. Saves your resume locally with instant one-click analysis.
          </p>
          <button className="choice-link" onClick={handleDownload} disabled={downloading}>
            <span>{downloaded ? 'Downloaded' : 'Download Zip'}</span>
            <Download size={16} />
          </button>
        </TiltCard>
      </section>

      {/* Feature Showcase Grid */}
      <section className="feature-band">
        <div className="band-heading">
          <span>Enterprise AI</span>
          <h2>Everything you need to outsmart the ATS and impress recruiters.</h2>
        </div>

        <div className="feature-grid">
          {features.map((feature, idx) => (
            <motion.div
              key={feature.title}
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ delay: idx * 0.1, duration: 0.5 }}
            >
              <TiltCard className="feature-tile">
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                  <span>0{idx + 1}</span>
                  {feature.icon}
                </div>
                <h3>{feature.title}</h3>
                <p>{feature.text}</p>
              </TiltCard>
            </motion.div>
          ))}
        </div>
      </section>

      {/* Telemetry Numbers Strip */}
      <section className="telemetry-strip">
        {telemetry.map((item, idx) => (
          <div className="telemetry-item" key={item.label}>
            <strong>{item.value}</strong>
            <span>{item.label}</span>
          </div>
        ))}
      </section>
    </motion.div>
  );
}

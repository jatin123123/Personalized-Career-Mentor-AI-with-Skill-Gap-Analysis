import Head from 'next/head';
import { useState, useRef } from 'react';
import { AnimatePresence, motion } from 'framer-motion';
import Header from '../components/Header';
import HomeLanding from '../components/HomeLanding';
import InputPanel from '../components/InputPanel';
import ResultsPanel from '../components/ResultsPanel';
import LoadingView from '../components/LoadingView';
import ExtensionBanner from '../components/ExtensionBanner';
import InteractiveBackground from '../components/InteractiveBackground';
import CursorGlow from '../components/CursorGlow';
import { API_URL } from '../lib/api';

export default function Home() {
  // 'home' | 'input' | 'loading' | 'results'
  const [view, setView] = useState('home');
  const [result, setResult] = useState(null);
  const [error, setError] = useState('');
  const formRef = useRef({});

  async function handleAnalyze(payload) {
    setError('');
    formRef.current = payload;
    setView('loading');
    try {
      const res = await fetch(`${API_URL}/api/analyze`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      const data = await res.json();
      if (data.success) {
        setResult(data.result);
        setView('results');
      } else {
        setError(data.error || data.detail || 'Analysis failed. Please check your GROQ API key.');
        setView('input');
      }
    } catch {
      setError('Could not reach the backend server. Please try again in a moment.');
      setView('input');
    }
  }

  function handleBack() {
    setView('input');
    setResult(null);
  }

  function handleGoHome() {
    setView('home');
    setResult(null);
    setError('');
  }

  function handleTryHere() {
    setView('input');
    setError('');
  }

  return (
    <>
      <Head>
        <title>Resume Matcher AI — Next-Gen Skill Gap Intelligence</title>
        <meta
          name="description"
          content="Award-winning AI-powered resume and job description matching with instant skill gap analysis, ATS scoring, and interview preparation."
        />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
      </Head>

      {/* Interactive Constellation Canvas & Iridescent Cursor Glow */}
      <InteractiveBackground />
      <CursorGlow />

      <Header showHero={view !== 'home'} onGoHome={handleGoHome} onTryHere={handleTryHere} />

      <main className={view === 'home' ? 'page page-home' : 'page'}>
        <AnimatePresence mode="wait">
          {view === 'home' && (
            <motion.div
              key="home"
              initial={{ opacity: 0, y: 16 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -16 }}
              transition={{ duration: 0.35, ease: [0.16, 1, 0.3, 1] }}
            >
              <HomeLanding onTryHere={handleTryHere} />
            </motion.div>
          )}

          {view === 'input' && (
            <motion.div
              key="input"
              initial={{ opacity: 0, y: 24, filter: 'blur(6px)' }}
              animate={{ opacity: 1, y: 0, filter: 'blur(0px)' }}
              exit={{ opacity: 0, y: -20, filter: 'blur(4px)' }}
              transition={{ duration: 0.45, ease: [0.16, 1, 0.3, 1] }}
            >
              <InputPanel onAnalyze={handleAnalyze} error={error} />
            </motion.div>
          )}

          {view === 'loading' && (
            <motion.div
              key="loading"
              initial={{ opacity: 0, scale: 0.95, filter: 'blur(8px)' }}
              animate={{ opacity: 1, scale: 1, filter: 'blur(0px)' }}
              exit={{ opacity: 0, scale: 0.95, filter: 'blur(4px)' }}
              transition={{ duration: 0.4 }}
            >
              <LoadingView />
            </motion.div>
          )}

          {view === 'results' && (
            <motion.div
              key="results"
              initial={{ opacity: 0, y: 30, filter: 'blur(8px)' }}
              animate={{ opacity: 1, y: 0, filter: 'blur(0px)' }}
              exit={{ opacity: 0, y: -20, filter: 'blur(4px)' }}
              transition={{ duration: 0.5, ease: [0.16, 1, 0.3, 1] }}
            >
              <ResultsPanel result={result} onBack={handleBack} />
            </motion.div>
          )}
        </AnimatePresence>
      </main>

      <AnimatePresence>
        {view === 'input' && (
          <motion.div
            key="ext-banner"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
            transition={{ duration: 0.4 }}
          >
            <ExtensionBanner />
          </motion.div>
        )}
      </AnimatePresence>

      <footer className="footer">
        <div className="footer-inner">
          <div className="footer-creator-pill">
            <span>Built &amp; Engineered with ❤️ by <strong>Jatin Jangid</strong></span>
          </div>
          <div className="footer-details">
            <span style={{ fontWeight: 800, color: 'var(--text)' }}>Resume Matcher AI</span>
            <span className="footer-divider">•</span>
            <span>Ultra-Fast Inference powered by Groq &amp; Llama-3.3</span>
            <span className="footer-divider">•</span>
            <span>LangChain AI Pipeline</span>
          </div>
        </div>
      </footer>
    </>
  );
}

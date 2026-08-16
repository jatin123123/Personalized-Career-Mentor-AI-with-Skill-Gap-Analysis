import { motion } from 'framer-motion';
import { useState, useEffect } from 'react';
import { Check, Cpu, Sparkles, BrainCircuit } from 'lucide-react';

const STEPS = [
  { label: 'Parsing resume semantics & structure', duration: 1200 },
  { label: 'Extracting hard & soft competency signals', duration: 2600 },
  { label: 'Matching role requirements & experience depth', duration: 4400 },
  { label: 'Synthesizing recommendations & interview prompts', duration: 6200 },
];

export default function LoadingView() {
  const [activeStep, setActiveStep] = useState(0);

  useEffect(() => {
    const timers = STEPS.map((step, i) =>
      setTimeout(() => setActiveStep(i + 1), step.duration)
    );
    return () => timers.forEach(clearTimeout);
  }, []);

  return (
    <div className="loading-overlay">
      {/* 3-Ring Orbital AI Reactor Spinner */}
      <motion.div
        className="loading-spinner-container"
        initial={{ opacity: 0, scale: 0.8 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ duration: 0.5, ease: [0.16, 1, 0.3, 1] }}
      >
        <div className="loading-ring loading-ring--1" />
        <div className="loading-ring loading-ring--2" />
        <div className="loading-ring loading-ring--3" />
        <div className="loading-core" />
      </motion.div>

      <motion.div
        initial={{ opacity: 0, y: 12 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.15, duration: 0.4 }}
      >
        <div className="loading-text">
          <span>Synthesizing Skill Gap Intelligence</span>
        </div>
      </motion.div>

      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.3 }}
      >
        <div className="loading-sub">
          Groq AI is executing multi-dimensional semantic comparison against the target job requirements.
        </div>
      </motion.div>

      {/* Progress Steps Telemetry */}
      <motion.div
        className="loading-steps"
        initial={{ opacity: 0, y: 16 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.45, duration: 0.5 }}
      >
        {STEPS.map((step, i) => {
          const isDone = activeStep > i;
          const isActive = activeStep === i;
          const statusClass = isDone ? 'done' : isActive ? 'active' : 'pending';

          return (
            <motion.div
              key={i}
              className={`loading-step ${statusClass}`}
              initial={{ opacity: 0, x: -12 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.5 + i * 0.1 }}
            >
              <span className="loading-step-icon">
                {isDone ? (
                  <Check size={13} strokeWidth={3} />
                ) : isActive ? (
                  <Sparkles size={13} style={{ color: 'white' }} />
                ) : (
                  i + 1
                )}
              </span>
              <span>{step.label}</span>
            </motion.div>
          );
        })}
      </motion.div>

      {/* Smooth Shimmering Progress Bar */}
      <motion.div
        className="loading-bar"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.8 }}
      >
        <div className="loading-bar-inner" />
      </motion.div>
    </div>
  );
}

import { useState, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  FileText, 
  Upload, 
  Sparkles, 
  Sliders, 
  AlertCircle, 
  CheckCircle, 
  ArrowRight,
  Code2,
  BrainCircuit,
  Briefcase
} from 'lucide-react';
import TiltCard from './TiltCard';
import { API_URL } from '../lib/api';

const MODELS = [
  'openai/gpt-oss-120b',
  'meta-llama/llama-prompt-guard-2-86m',
  'qwen/qwen3.8-27b',
];
const DEPTHS = ['Quick', 'Standard', 'Deep', 'Comprehensive'];

const SAMPLE_PRESETS = [
  {
    name: 'Full Stack Engineer',
    icon: <Code2 size={14} />,
    jd: `Senior Full Stack Developer (React / Next.js / Python / FastAPI / AWS)
Requirements:
- 4+ years building responsive web apps with React, Next.js, and TypeScript.
- Strong proficiency in Python, FastAPI, REST APIs, and PostgreSQL.
- Experience with Docker, AWS (ECS, S3, CloudFront), and CI/CD pipelines.
- Deep understanding of web performance, glassmorphism UI design, and micro-frontend architecture.
- Bonus: LLM integration (LangChain, Groq, OpenAI), Redis caching, and TailwindCSS/Vanilla CSS tokens.`,
    resume: `Alex Rivera — Senior Full Stack Software Engineer
Summary:
Full-stack software engineer with 5 years of experience building high-scale cloud-native web applications. Passionate about developer tooling, real-time UI animation, and AI pipelines.

Core Skills:
- Frontend: JavaScript (ES6+), TypeScript, React, Next.js, HTML5, CSS3, Framer Motion, Redux.
- Backend: Python, FastAPI, Node.js, Express, PostgreSQL, MongoDB, Redis.
- Cloud & DevOps: Docker, AWS (S3, EC2, Lambda), Git, GitHub Actions, Linux.

Experience:
Senior Frontend & Full Stack Developer @ CloudPeak (2022 - Present)
- Architected and shipped 4 client-facing Next.js web applications serving 250k+ monthly active users.
- Built high-performance asynchronous Python FastAPI microservices with sub-50ms p99 latency.
- Implemented real-time dashboard analytics and modern responsive CSS design systems with Framer Motion.`,
  },
  {
    name: 'AI / ML Engineer',
    icon: <BrainCircuit size={14} />,
    jd: `Lead AI/ML Engineer — Generative AI & NLP
Requirements:
- Master's or Bachelor's in CS/AI or equivalent industry experience.
- Extensive experience with Python, PyTorch, LangChain, Llama models, and Vector DBs (Chroma, Pinecone).
- Proven track record deploying LLM RAG pipelines, fine-tuning, and prompt optimization.
- Familiarity with FastAPI backend services and async task queues.
- Strong knowledge of evaluation metrics, embeddings, and token cost optimization.`,
    resume: `Elena Chen — Machine Learning & GenAI Engineer
Summary:
AI Engineer specializing in Large Language Models (LLMs), RAG architectures, and scalable AI infrastructure.

Technical Skills:
- AI & LLM: PyTorch, Hugging Face, LangChain, Llama 3, OpenAI API, Vector DBs (ChromaDB, Weaviate), RAG.
- Backend & Systems: Python, FastAPI, Docker, Kubernetes, PostgreSQL.

Selected Projects & Experience:
AI Engineer @ NeuralFlow (2023 - Present)
- Developed production RAG pipeline using LangChain and Llama-3, reducing query response hallucination by 42%.
- Built vector search retrieval system indexing 1.2M technical documents.`,
  }
];

const fadeUp = {
  hidden: { opacity: 0, y: 18 },
  show: (i) => ({
    opacity: 1,
    y: 0,
    transition: { delay: i * 0.08, duration: 0.45, ease: [0.16, 1, 0.3, 1] },
  }),
};

export default function InputPanel({ onAnalyze, error }) {
  const [jd, setJd] = useState('');
  const [resumeText, setResumeText] = useState('');
  const [tab, setTab] = useState('paste');
  const [model, setModel] = useState(MODELS[0]);
  const [depth, setDepth] = useState('Standard');
  const [temp, setTemp] = useState(0.1);
  const [localError, setLocalError] = useState('');
  const [pdfInfo, setPdfInfo] = useState('');
  const [dragOver, setDragOver] = useState(false);
  const fileRef = useRef(null);

  const applyPreset = (preset) => {
    setJd(preset.jd);
    setResumeText(preset.resume);
    setLocalError('');
  };

  async function handleFile(file) {
    if (!file || !file.name.endsWith('.pdf')) {
      setLocalError('Please upload a valid PDF resume file.');
      return;
    }
    setLocalError('');
    setPdfInfo('Extracting text from PDF...');
    const form = new FormData();
    form.append('file', file);
    try {
      const res = await fetch(`${API_URL}/api/upload-resume`, { method: 'POST', body: form });
      const data = await res.json();
      if (data.success) {
        setResumeText(data.text);
        setPdfInfo(`Successfully extracted ${data.char_count.toLocaleString()} characters from "${file.name}"`);
      } else {
        setLocalError(data.error || 'Could not extract PDF text.');
        setPdfInfo('');
      }
    } catch {
      setLocalError('Upload failed. Please ensure the backend server is running on port 8001.');
      setPdfInfo('');
    }
  }

  function handleSubmit() {
    setLocalError('');
    if (!jd.trim() || !resumeText.trim()) {
      setLocalError('Both the job description and your resume are required to run analysis.');
      return;
    }
    onAnalyze({
      job_description: jd,
      resume_text: resumeText,
      model,
      depth,
      temperature: temp,
    });
  }

  const displayError = localError || error;

  return (
    <div>
      {/* 1-Click Sample Preset Bar */}
      <motion.div
        className="preset-bar mb-24"
        initial={{ opacity: 0, y: -10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.4 }}
      >
        <span style={{ fontSize: '0.8rem', fontWeight: 800, color: 'var(--text-3)' }}>
          Quick Load Sample:
        </span>
        {SAMPLE_PRESETS.map((preset) => (
          <button
            key={preset.name}
            type="button"
            className="preset-pill"
            onClick={() => applyPreset(preset)}
          >
            {preset.icon}
            <span>{preset.name}</span>
          </button>
        ))}
      </motion.div>

      {/* Main 2-Column Inputs */}
      <div className="grid-2 mb-24">
        {/* Step 1: Job Description */}
        <motion.div custom={0} variants={fadeUp} initial="hidden" animate="show">
          <TiltCard>
            <div className="section-label">
              <span className="step-num">1</span>
              <span>Target Job Description</span>
            </div>
            <div className="textarea-wrapper">
              <textarea
                rows={11}
                placeholder="Paste the target job description or requirements here..."
                value={jd}
                onChange={(e) => setJd(e.target.value)}
              />
              {jd.length > 0 && (
                <span className="char-count">{jd.length.toLocaleString()} chars</span>
              )}
            </div>
          </TiltCard>
        </motion.div>

        {/* Step 2: Resume */}
        <motion.div custom={1} variants={fadeUp} initial="hidden" animate="show">
          <TiltCard>
            <div className="section-label">
              <span className="step-num">2</span>
              <span>Your Resume</span>
            </div>

            <div className="tabs">
              <button
                className={`tab-btn ${tab === 'paste' ? 'active' : ''}`}
                onClick={() => setTab('paste')}
                type="button"
              >
                Paste Text
              </button>
              <button
                className={`tab-btn ${tab === 'upload' ? 'active' : ''}`}
                onClick={() => setTab('upload')}
                type="button"
              >
                Upload PDF
              </button>
            </div>

            <AnimatePresence mode="wait">
              {tab === 'paste' ? (
                <motion.div
                  key="paste"
                  initial={{ opacity: 0, y: 8 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -8 }}
                  transition={{ duration: 0.2 }}
                >
                  <div className="textarea-wrapper">
                    <textarea
                      rows={9}
                      placeholder="Paste your full resume text here..."
                      value={resumeText}
                      onChange={(e) => setResumeText(e.target.value)}
                    />
                    {resumeText.length > 0 && (
                      <span className="char-count">{resumeText.length.toLocaleString()} chars</span>
                    )}
                  </div>
                </motion.div>
              ) : (
                <motion.div
                  key="upload"
                  initial={{ opacity: 0, y: 8 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -8 }}
                  transition={{ duration: 0.2 }}
                >
                  <div
                    className={`drop-zone ${dragOver ? 'dragover' : ''}`}
                    onClick={() => fileRef.current?.click()}
                    onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
                    onDragLeave={() => setDragOver(false)}
                    onDrop={(e) => {
                      e.preventDefault();
                      setDragOver(false);
                      handleFile(e.dataTransfer.files[0]);
                    }}
                  >
                    <div className="drop-icon-box">
                      <Upload size={24} />
                    </div>
                    <div className="drop-zone-text">Click to browse or drag &amp; drop PDF</div>
                    <div className="drop-zone-hint">Supports standard PDF resumes up to 10 MB</div>
                    <input
                      ref={fileRef}
                      type="file"
                      accept=".pdf"
                      style={{ display: 'none' }}
                      onChange={(e) => handleFile(e.target.files[0])}
                    />
                  </div>

                  {pdfInfo && (
                    <motion.div initial={{ opacity: 0, y: 6 }} animate={{ opacity: 1, y: 0 }}>
                      <div className="alert-success mt-8">
                        <CheckCircle size={18} />
                        <span>{pdfInfo}</span>
                      </div>
                    </motion.div>
                  )}
                </motion.div>
              )}
            </AnimatePresence>
          </TiltCard>
        </motion.div>
      </div>

      {/* Step 3: Model & Settings */}
      <motion.div custom={2} variants={fadeUp} initial="hidden" animate="show">
        <TiltCard className="mb-20">
          <div className="section-label">
            <span className="step-num">3</span>
            <span>Inference Settings</span>
          </div>

          <div className="config-row">
            <div>
              <label htmlFor="model-select">AI Inference Model</label>
              <select id="model-select" value={model} onChange={(e) => setModel(e.target.value)}>
                {MODELS.map((m) => <option key={m} value={m}>{m}</option>)}
              </select>
            </div>

            <div>
              <label htmlFor="depth-select">Analysis Depth</label>
              <select id="depth-select" value={depth} onChange={(e) => setDepth(e.target.value)}>
                {DEPTHS.map((d) => <option key={d} value={d}>{d}</option>)}
              </select>
            </div>

            <div>
              <label>Creativity / Temperature</label>
              <input
                type="range"
                min={0}
                max={1}
                step={0.1}
                value={temp}
                onChange={(e) => setTemp(parseFloat(e.target.value))}
              />
              <div className="slider-value">{temp.toFixed(1)}</div>
            </div>
          </div>
        </TiltCard>
      </motion.div>

      {/* Error Feedback */}
      <AnimatePresence>
        {displayError && (
          <motion.div
            initial={{ opacity: 0, y: 8, scale: 0.98 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: -8, scale: 0.98 }}
          >
            <div className="alert-error">
              <AlertCircle size={18} />
              <span>{displayError}</span>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Main Submit Action */}
      <motion.div custom={3} variants={fadeUp} initial="hidden" animate="show">
        <motion.button
          id="analyze-btn"
          className="btn-primary"
          onClick={handleSubmit}
          whileHover={{ scale: 1.01 }}
          whileTap={{ scale: 0.98 }}
          type="button"
        >
          <Sparkles size={20} />
          <span>Generate Skill Gap Analysis</span>
          <ArrowRight size={20} />
        </motion.button>
      </motion.div>
    </div>
  );
}

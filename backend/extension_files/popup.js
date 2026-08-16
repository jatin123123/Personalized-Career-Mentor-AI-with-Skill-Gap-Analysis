const API_DEFAULT = 'https://career-mentor-api-jatin.azurewebsites.net';

// ── Helpers ────────────────────────────────────────────────────
const $ = (id) => document.getElementById(id);
const parseList = (v) => {
  if (!v) return [];
  if (Array.isArray(v)) return v.map(String).filter(Boolean);
  if (typeof v === 'string') {
    const trimmed = v.trim();
    if (trimmed.startsWith('[') && trimmed.endsWith(']')) {
      try {
        const parsed = JSON.parse(trimmed);
        if (Array.isArray(parsed)) return parsed.map(String).filter(Boolean);
      } catch (e) {}
    }
    return trimmed.split(',').map(s => s.trim()).filter(s => s.length > 1);
  }
  return [];
};

// ══════════════════════════════════════════════════════════════
// TAB SWITCHING with sliding indicator
// ══════════════════════════════════════════════════════════════
const tabBar = $('tabBar');
const tabIndicator = $('tabIndicator');
const allTabs = document.querySelectorAll('.tab');

function positionIndicator(tab) {
  if (!tab || !tabBar || !tabIndicator) return;
  const barRect = tabBar.getBoundingClientRect();
  const tabRect = tab.getBoundingClientRect();
  tabIndicator.style.left = (tabRect.left - barRect.left) + 'px';
  tabIndicator.style.width = tabRect.width + 'px';
}

function switchTab(tabKey) {
  allTabs.forEach(t => t.classList.remove('active'));
  document.querySelectorAll('.tab-content').forEach(p => p.classList.remove('active'));

  const targetTab = document.querySelector(`[data-tab="${tabKey}"]`);
  const targetPanel = $('panel' + tabKey.charAt(0).toUpperCase() + tabKey.slice(1));

  if (targetTab && targetPanel) {
    targetTab.classList.add('active');
    targetPanel.classList.add('active');
    positionIndicator(targetTab);
  }
}

allTabs.forEach(btn => {
  btn.addEventListener('click', () => {
    switchTab(btn.dataset.tab);
  });
});

// Init indicator position
window.addEventListener('load', () => {
  const activeTab = document.querySelector('.tab.active');
  if (activeTab) positionIndicator(activeTab);
});

// ══════════════════════════════════════════════════════════════
// RIPPLE EFFECT for primary buttons
// ══════════════════════════════════════════════════════════════
document.querySelectorAll('.btn-primary').forEach(btn => {
  btn.addEventListener('mousedown', (e) => {
    if (btn.disabled) return;
    const ripple = document.createElement('span');
    ripple.classList.add('ripple');
    const rect = btn.getBoundingClientRect();
    const size = Math.max(rect.width, rect.height);
    ripple.style.width = ripple.style.height = size + 'px';
    ripple.style.left = (e.clientX - rect.left - size / 2) + 'px';
    ripple.style.top = (e.clientY - rect.top - size / 2) + 'px';
    btn.appendChild(ripple);
    ripple.addEventListener('animationend', () => ripple.remove());
  });
});

// ══════════════════════════════════════════════════════════════
// ONBOARDING GUIDE
// ══════════════════════════════════════════════════════════════
const onboardingSteps = [
  {
    icon: '📄',
    title: 'Upload Your Resume',
    desc: 'Go to the "My Resume" tab and upload your PDF. Your resume is stored locally in Chrome — it never leaves your machine.',
  },
  {
    icon: '🔍',
    title: 'Browse LinkedIn Jobs',
    desc: 'Navigate to any LinkedIn job posting. The extension auto-detects job pages and scrapes the description for you.',
  },
  {
    icon: '✨',
    title: 'Click Analyze',
    desc: 'Hit the "Analyze This Job" button. Our AI compares your resume against the job requirements in seconds.',
  },
  {
    icon: '📊',
    title: 'Get Your Results',
    desc: 'See your match score, matched & missing skills, and personalized recommendations to close skill gaps.',
  },
];

let obStep = 0;

function renderOnboardingStep() {
  const step = onboardingSteps[obStep];
  $('obIcon').textContent = step.icon;
  $('obStepLabel').textContent = `Step ${obStep + 1} of ${onboardingSteps.length}`;
  $('obTitle').textContent = step.title;
  $('obDesc').textContent = step.desc;

  // Update dots
  const dots = $('obDots').children;
  for (let i = 0; i < dots.length; i++) {
    dots[i].classList.toggle('active', i === obStep);
  }

  // Update button text
  $('obNext').textContent = obStep === onboardingSteps.length - 1 ? 'Get Started ✓' : 'Next →';
  $('obSkip').textContent = obStep === onboardingSteps.length - 1 ? 'Close' : 'Skip';
}

function showOnboarding() {
  obStep = 0;
  renderOnboardingStep();
  const overlay = $('onboardingOverlay');
  overlay.hidden = false;
  overlay.classList.remove('hiding');
}

function hideOnboarding() {
  const overlay = $('onboardingOverlay');
  overlay.classList.add('hiding');
  setTimeout(() => {
    overlay.hidden = true;
    overlay.classList.remove('hiding');
  }, 250);
  chrome.storage.local.set({ onboardingSeen: true });
}

$('obNext').addEventListener('click', () => {
  if (obStep < onboardingSteps.length - 1) {
    obStep++;
    renderOnboardingStep();
  } else {
    hideOnboarding();
  }
});

$('obSkip').addEventListener('click', hideOnboarding);
$('obClose').addEventListener('click', hideOnboarding);

// "Guide" button in header opens the guide
$('btnHowTo').addEventListener('click', showOnboarding);

// ══════════════════════════════════════════════════════════════
// SETTINGS
// ══════════════════════════════════════════════════════════════
async function loadSettings() {
  try {
    const s = await chrome.storage.local.get(['apiUrl', 'model', 'depth']);
    $('inputApiUrl').value = s.apiUrl || API_DEFAULT;
    $('inputModel').value  = s.model  || 'llama-3.3-70b-versatile';
    $('inputDepth').value  = s.depth  || 'Standard';
  } catch (e) {
    console.error('Failed to load settings:', e);
  }
}

$('btnSaveSettings').addEventListener('click', async () => {
  await chrome.storage.local.set({
    apiUrl: $('inputApiUrl').value.trim() || API_DEFAULT,
    model:  $('inputModel').value,
    depth:  $('inputDepth').value,
  });
  showToast('Settings saved!', 'success');
  $('settingsSaved').hidden = false;
  setTimeout(() => { $('settingsSaved').hidden = true; }, 2000);
});

// ══════════════════════════════════════════════════════════════
// RESUME STORAGE
// ══════════════════════════════════════════════════════════════
async function loadResumeStatus() {
  try {
    const { resumeText, resumeName } = await chrome.storage.local.get(['resumeText', 'resumeName']);
    const status = $('resumeStatus');
    if (resumeText) {
      status.classList.add('has-resume');
      $('resumeStatusText').textContent = 'Saved: ' + (resumeName || 'Resume.pdf') + ' (' + resumeText.length.toLocaleString() + ' chars)';
      $('btnClearResume').hidden = false;
    } else {
      status.classList.remove('has-resume');
      $('resumeStatusText').textContent = 'No resume saved yet. Upload a PDF below.';
      $('btnClearResume').hidden = true;
    }
  } catch (e) {
    console.error('Failed to load resume status:', e);
  }
}

// Drop zone
const dz = $('dropZone');
dz.addEventListener('click', () => $('fileInput').click());
dz.addEventListener('dragover', e => { e.preventDefault(); dz.classList.add('dragover'); });
dz.addEventListener('dragleave', () => dz.classList.remove('dragover'));
dz.addEventListener('drop', e => { e.preventDefault(); dz.classList.remove('dragover'); handleFile(e.dataTransfer.files[0]); });
$('fileInput').addEventListener('change', e => handleFile(e.target.files[0]));

async function handleFile(file) {
  if (!file || !file.name.toLowerCase().endsWith('.pdf')) {
    showError('uploadError', 'Only PDF files are supported.');
    return;
  }
  $('uploadError').hidden = true;
  $('uploadSuccess').hidden = true;
  $('uploadProgress').hidden = false;
  $('progressFill').style.width = '10%';

  let pct = 10;
  const timer = setInterval(() => {
    pct = Math.min(pct + 15, 85);
    $('progressFill').style.width = pct + '%';
  }, 120);

  const { apiUrl } = await chrome.storage.local.get('apiUrl');
  const form = new FormData();
  form.append('file', file);
  try {
    const res = await fetch((apiUrl || API_DEFAULT) + '/api/upload-resume', { method: 'POST', body: form });
    const data = await res.json();
    clearInterval(timer);
    $('progressFill').style.width = '100%';

    if (data.success) {
      await chrome.storage.local.set({ resumeText: data.text, resumeName: file.name });
      $('uploadSuccess').textContent = 'Extracted ' + data.char_count.toLocaleString() + ' chars from "' + file.name + '"';
      $('uploadSuccess').hidden = false;
      showToast('Resume saved successfully!', 'success');
      setTimeout(() => { $('uploadProgress').hidden = true; }, 600);
      await loadResumeStatus();
    } else {
      showError('uploadError', data.error || 'Could not extract text from PDF.');
      $('uploadProgress').hidden = true;
    }
  } catch (err) {
    clearInterval(timer);
    showError('uploadError', 'Upload failed. Could not reach the backend server. Please try again.');
    $('uploadProgress').hidden = true;
  }
}

$('btnClearResume').addEventListener('click', async () => {
  await chrome.storage.local.remove(['resumeText', 'resumeName']);
  $('uploadSuccess').hidden = true;
  showToast('Resume removed', 'success');
  await loadResumeStatus();
});

// ══════════════════════════════════════════════════════════════
// JOB DETECTION (LinkedIn)
// ══════════════════════════════════════════════════════════════
let currentJobData = null;

async function detectJob() {
  let tab;
  try {
    const [t] = await chrome.tabs.query({ active: true, currentWindow: true });
    tab = t;
  } catch {
    setStatus('Ready', false);
    return;
  }

  if (!tab || !tab.url || !tab.url.includes('linkedin.com/jobs')) {
    setStatus('Not on LinkedIn', false);
    $('jobTitle').textContent = 'Not on LinkedIn Jobs';
    $('jobBanner').querySelector('.job-banner-sub').textContent = 'Open any LinkedIn job page to analyze';
    $('btnAnalyze').disabled = true;
    return;
  }

  // Method 1: Send message to content script
  try {
    const data = await chrome.tabs.sendMessage(tab.id, { type: 'GET_JOB_DATA' });
    if (data && (data.title || data.description)) {
      setJobDetected(data);
      return;
    }
  } catch (e) {
    console.log('sendMessage failed, trying executeScript fallback');
  }

  // Method 2: Inject script directly
  try {
    const [result] = await chrome.scripting.executeScript({
      target: { tabId: tab.id },
      func: () => {
        const query = (selectors) => {
          for (const sel of selectors) {
            const el = document.querySelector(sel);
            if (el && el.innerText.trim()) return el.innerText.trim();
          }
          return '';
        };
        const title = query([
          '.job-details-jobs-unified-top-card__job-title h1',
          '.job-details-jobs-unified-top-card__job-title',
          'h1.t-24',
          'h1.t-20',
          '.topcard__title',
          '.jobs-unified-top-card__job-title',
          'h1'
        ]);
        const company = query([
          '.job-details-jobs-unified-top-card__company-name a',
          '.job-details-jobs-unified-top-card__company-name',
          '.job-details-jobs-unified-top-card__primary-description-container a',
          '.topcard__org-name-link',
          '.jobs-unified-top-card__company-name',
          'a[data-tracking-control-name="public_jobs_topcard-org-name"]'
        ]);
        let description = query([
          '#job-details',
          '.jobs-description-content__text',
          '.jobs-description__content',
          '.jobs-box__html-content',
          '.jobs-description',
          'article[class*="description"]',
          '.jobs-description-content',
          '.description__text'
        ]);
        if (!description) {
          const rightPane = document.querySelector('.jobs-search__job-details, .job-view-layout');
          if (rightPane) description = rightPane.innerText.trim();
        }
        if (!title && !description) return null;
        return {
          title: title || 'Job Posting',
          company: company || 'LinkedIn',
          description: description || title
        };
      },
    });
    if (result?.result) {
      setJobDetected(result.result);
      return;
    }
  } catch (e) {
    console.log('executeScript failed:', e.message);
  }

  // Method 3: Title fallback
  if (tab.title && tab.title.includes('|') && tab.url.includes('linkedin.com/jobs')) {
    const parts = tab.title.split('|');
    setJobDetected({
      title: parts[0].trim(),
      company: 'LinkedIn',
      description: parts[0].trim(),
    });
    setStatus('Job detected', true);
    return;
  }

  setStatus('Refresh tab & reopen', false);
}

function setJobDetected(data) {
  currentJobData = data;
  const banner = $('jobBanner');
  banner.classList.add('detected');
  $('jobTitle').textContent = data.title || 'Job detected';
  banner.querySelector('.job-banner-sub').textContent = data.company || 'LinkedIn';
  $('btnAnalyze').disabled = false;
  setStatus('Job ready', true);
}

function setStatus(text, active) {
  const pill = $('statusPill');
  $('statusText').textContent = text;
  pill.className = 'status-pill' + (active ? ' active' : '');
}

// ══════════════════════════════════════════════════════════════
// ANALYSIS
// ══════════════════════════════════════════════════════════════
$('btnAnalyze').addEventListener('click', runAnalysis);
$('btnReanalyze').addEventListener('click', runAnalysis);

async function runAnalysis() {
  const { resumeText, apiUrl, model, depth } = await chrome.storage.local.get(['resumeText', 'apiUrl', 'model', 'depth']);

  if (!resumeText) {
    showError('errorBox', 'No resume found. Please upload your resume in the "My Resume" tab.');
    switchTab('resume');
    return;
  }
  if (!currentJobData) {
    showError('errorBox', 'No job detected. Navigate to a LinkedIn job page first.');
    return;
  }

  $('errorBox').hidden = true;
  $('results').hidden = true;
  $('loader').hidden = false;
  $('btnAnalyze').disabled = true;
  setStatus('Analyzing...', true);

  try {
    const res = await fetch((apiUrl || API_DEFAULT) + '/api/analyze', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        job_description: currentJobData.description || currentJobData.title,
        resume_text: resumeText,
        model: model || 'llama-3.3-70b-versatile',
        depth: depth || 'Standard',
        temperature: 0.1,
      }),
    });
    const data = await res.json();

    $('loader').hidden = true;
    $('btnAnalyze').disabled = false;

    if (data.success) {
      renderResults(data.result);
      setStatus('Analyzed', true);
    } else {
      showError('errorBox', data.error || 'Analysis failed. Please check your GROQ API key.');
      setStatus('Error', false);
    }
  } catch (err) {
    $('loader').hidden = true;
    $('btnAnalyze').disabled = false;
    showError('errorBox', 'Cannot reach backend server. Please try again in a moment.');
    setStatus('Offline', false);
  }
}

// ══════════════════════════════════════════════════════════════
// RESULTS RENDERING with animations
// ══════════════════════════════════════════════════════════════
function renderResults(r) {
  const pct     = Number(r.overall_match_percentage) || 0;
  const matched = parseList(r.skills_matched);
  const missing = parseList(r.skills_missing);
  const extra   = parseList(r.skills_extra);
  const recs    = parseList(r.specific_recommendations);

  // Animate score ring
  animateScoreRing(pct);

  // Animate stat numbers with stagger
  setTimeout(() => animateNum($('statMatched').querySelector('.stat-val'), matched.length), 200);
  setTimeout(() => animateNum($('statMissing').querySelector('.stat-val'), missing.length), 350);
  setTimeout(() => animateNum($('statExtra').querySelector('.stat-val'),   extra.length),   500);

  // Render staggered chips
  renderChips('chipsMatched', matched, 'matched');
  renderChips('chipsMissing', missing, 'missing');

  // Recommendations
  const recList = $('recList');
  recList.innerHTML = '';
  recs.slice(0, 4).forEach((r, i) => {
    const el = document.createElement('div');
    el.className = 'rec-item';
    el.style.animationDelay = (0.15 + i * 0.1) + 's';
    el.innerHTML = '<span class="rec-num">' + (i + 1) + '</span><span>' + r + '</span>';
    recList.appendChild(el);
  });
  $('sectionRec').style.display = recs.length ? 'block' : 'none';

  $('results').hidden = false;

  // Confetti for high scores!
  if (pct >= 80) {
    setTimeout(fireConfetti, 600);
  }
}

// ── Score Ring Animation ──
function animateScoreRing(pct) {
  const ring = $('scoreRingFill');
  const circumference = 2 * Math.PI * 50; // r=50
  const offset = circumference - (pct / 100) * circumference;

  // Set gradient color based on score
  if (pct >= 80) {
    ring.style.stroke = 'url(#scoreGradGreen)';
  } else if (pct >= 60) {
    ring.style.stroke = 'url(#scoreGradAmber)';
  } else {
    ring.style.stroke = 'url(#scoreGradRed)';
  }

  // Reset then animate
  ring.style.strokeDasharray = circumference;
  ring.style.strokeDashoffset = circumference;

  requestAnimationFrame(() => {
    requestAnimationFrame(() => {
      ring.style.strokeDashoffset = offset;
    });
  });

  // Countup the percentage number
  animateCountup($('scoreNum'), pct);
}

// ── Smooth countup ──
function animateCountup(el, end) {
  const duration = 1000;
  const start = performance.now();
  const easeOut = t => 1 - Math.pow(1 - t, 3);

  function tick(now) {
    const elapsed = now - start;
    const progress = Math.min(elapsed / duration, 1);
    const current = Math.round(easeOut(progress) * end);
    el.textContent = current + '%';
    if (progress < 1) requestAnimationFrame(tick);
  }
  requestAnimationFrame(tick);
}

// ── Staggered chip rendering ──
function renderChips(containerId, list, type) {
  const el = $(containerId);
  if (!list.length) {
    el.innerHTML = '<span style="font-size:.72rem;color:var(--text-3)">None detected</span>';
    return;
  }
  el.innerHTML = list.slice(0, 12).map((s, i) =>
    `<span class="chip ${type}" style="animation-delay:${(i * 0.04).toFixed(2)}s">${s}</span>`
  ).join('');
}

// ── Stat number animation ──
function animateNum(el, end) {
  if (end === 0) { el.textContent = '0'; return; }
  let n = 0;
  const step = () => {
    n = Math.min(n + Math.ceil(end / 10), end);
    el.textContent = n;
    if (n < end) requestAnimationFrame(step);
  };
  requestAnimationFrame(step);
}

// ══════════════════════════════════════════════════════════════
// CONFETTI BURST 🎉
// ══════════════════════════════════════════════════════════════
function fireConfetti() {
  const container = document.createElement('div');
  container.className = 'confetti-container';
  document.body.appendChild(container);

  const colors = ['#7c3aed', '#06b6d4', '#ec4899', '#34d399', '#fbbf24', '#f87171'];
  const count = 35;

  for (let i = 0; i < count; i++) {
    const piece = document.createElement('div');
    piece.className = 'confetti-piece';
    piece.style.left = Math.random() * 100 + '%';
    piece.style.background = colors[Math.floor(Math.random() * colors.length)];
    piece.style.setProperty('--fall-duration', (1.5 + Math.random() * 2) + 's');
    piece.style.setProperty('--fall-delay', (Math.random() * 0.5) + 's');
    piece.style.setProperty('--spin', (360 + Math.random() * 720) + 'deg');
    piece.style.width = (4 + Math.random() * 5) + 'px';
    piece.style.height = (4 + Math.random() * 5) + 'px';
    piece.style.borderRadius = Math.random() > 0.5 ? '50%' : '2px';
    container.appendChild(piece);
  }

  // Cleanup after animation
  setTimeout(() => container.remove(), 3500);
}

// ══════════════════════════════════════════════════════════════
// TOAST NOTIFICATIONS
// ══════════════════════════════════════════════════════════════
function showToast(message, type = 'success') {
  const existing = document.querySelector('.toast');
  if (existing) existing.remove();

  const toast = document.createElement('div');
  toast.className = `toast ${type}`;
  toast.textContent = message;
  document.body.appendChild(toast);

  requestAnimationFrame(() => {
    requestAnimationFrame(() => {
      toast.classList.add('show');
    });
  });

  setTimeout(() => {
    toast.classList.remove('show');
    setTimeout(() => toast.remove(), 400);
  }, 2500);
}

// ══════════════════════════════════════════════════════════════
// ERROR DISPLAY
// ══════════════════════════════════════════════════════════════
function showError(id, msg) {
  const el = $(id);
  el.textContent = msg;
  el.hidden = false;
}

// ══════════════════════════════════════════════════════════════
// INIT
// ══════════════════════════════════════════════════════════════
(async () => {
  // Ensure overlay and loader are hidden
  $('onboardingOverlay').hidden = true;
  $('loader').hidden = true;

  await loadSettings();
  await loadResumeStatus();
  await detectJob();
})();

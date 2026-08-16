# Resume Matcher — Chrome Extension

A Chrome extension that auto-detects LinkedIn job postings and instantly analyses them against your saved resume using your local AI backend.

## Features

- 🔍 **Auto-detects** LinkedIn job pages and scrapes the job description
- 📄 **Stores your resume** permanently in the extension (upload once, use forever)
- ⚡ **One-click analysis** — just click the extension icon on any LinkedIn job
- 📊 **Results in seconds** — match score, matched/missing skills, top recommendations
- ⚙️ **Configurable** — choose AI model, analysis depth, and backend URL
- 🔔 **Badge indicator** — purple dot on the icon when you're on a job page

---

## Setup

### Step 1 — Generate Icons

Open `generate-icons.html` in your browser (double-click it). It will auto-download:
- `icon16.png`
- `icon48.png`  
- `icon128.png`

Move all three files into the `extension/icons/` folder.

### Step 2 — Start the Backend

```powershell
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8001
```

Make sure your `backend/.env` has `GROQ_API_KEY=your_key_here`.

### Step 3 — Load the Extension in Chrome

1. Open Chrome and go to `chrome://extensions/`
2. Enable **Developer mode** (top-right toggle)
3. Click **"Load unpacked"**
4. Select the `extension/` folder from this project
5. The extension icon will appear in your toolbar

### Step 4 — Upload Your Resume

1. Click the extension icon
2. Go to the **"My Resume"** tab
3. Upload your PDF resume — it's stored permanently in the extension

### Step 5 — Analyze Jobs

1. Go to any LinkedIn job posting: `linkedin.com/jobs/view/...`
2. Click the extension icon
3. The job title auto-detects
4. Click **"Analyze This Job"**
5. Get instant results!

---

## Extension File Structure

```
extension/
├── manifest.json       # Chrome extension config
├── popup.html          # Popup UI (3 tabs: Analyze, My Resume, Settings)
├── popup.css           # Premium glassmorphic styles
├── popup.js            # All popup logic — tabs, upload, analysis, results
├── background.js       # Service worker — badge indicator on job pages
├── content.js          # Content script injected into LinkedIn pages
├── generate-icons.html # Open in browser to generate PNG icons
└── icons/
    ├── icon16.png      # (generate with generate-icons.html)
    ├── icon48.png
    └── icon128.png
```

---

## Settings

Inside the extension's **Settings** tab:

| Setting | Default | Description |
|---|---|---|
| Backend URL | `http://localhost:8001` | URL of your FastAPI backend |
| AI Model | `llama-3.3-70b-versatile` | Groq model to use |
| Analysis Depth | `Standard` | Quick / Standard / Deep / Comprehensive |

---

## How LinkedIn Scraping Works

When you're on a LinkedIn job page, the extension uses `chrome.scripting.executeScript` to read the job title, company name, and full job description from the DOM. It targets multiple CSS selectors to handle LinkedIn's different page layouts.

The scraped job description is sent to your local backend's `/api/analyze` endpoint along with your stored resume text.

---

## Privacy

- Your resume is stored **locally** in Chrome's extension storage (`chrome.storage.local`) — never uploaded anywhere except your own backend
- The backend is **local** — your data never leaves your machine (unless you change the backend URL to a remote server)

---

## Troubleshooting

| Problem | Fix |
|---|---|
| "Cannot reach backend" | Make sure `uvicorn` is running on port 8001 |
| "No job detected" | Make sure you're on a `linkedin.com/jobs/view/...` page, not search |
| "Could not extract PDF" | Try a text-based PDF (not a scanned image) |
| Icons missing | Run `generate-icons.html` in your browser and move files to `icons/` |

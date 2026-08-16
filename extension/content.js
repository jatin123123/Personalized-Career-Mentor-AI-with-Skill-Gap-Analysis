// content.js — Injected into ALL LinkedIn job pages
// Actively scrapes job data and responds to messages from popup

(function () {
  chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
    if (msg.type === 'GET_JOB_DATA') {
      const data = scrapeJob();
      sendResponse(data);
    }
    return true;
  });

  function scrapeJob() {
    const query = (selectors) => {
      for (const sel of selectors) {
        const el = document.querySelector(sel);
        if (el && el.innerText.trim()) {
          return el.innerText.trim();
        }
      }
      return '';
    };

    // 1. Job Title
    const title = query([
      '.job-details-jobs-unified-top-card__job-title h1',
      '.job-details-jobs-unified-top-card__job-title',
      'h1.t-24',
      'h1.t-20',
      '.topcard__title',
      '.jobs-unified-top-card__job-title',
      'h1'
    ]);

    // 2. Company Name
    const company = query([
      '.job-details-jobs-unified-top-card__company-name a',
      '.job-details-jobs-unified-top-card__company-name',
      '.job-details-jobs-unified-top-card__primary-description-container a',
      '.topcard__org-name-link',
      '.jobs-unified-top-card__company-name',
      'a[data-tracking-control-name="public_jobs_topcard-org-name"]'
    ]);

    // 3. Job Description Container (prioritize the main active article/details)
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

    // If description is empty, fallback to active right-pane search details
    if (!description) {
      const rightPane = document.querySelector('.jobs-search__job-details, .job-view-layout');
      if (rightPane) {
        description = rightPane.innerText.trim();
      }
    }

    if (!title && !description) return null;

    return {
      title: title || 'Job Posting',
      company: company || 'LinkedIn',
      description: description || title
    };
  }
})();

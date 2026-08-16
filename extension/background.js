// background.js — Service worker
// Badge indicator + inject content script on LinkedIn job pages

const JOB_PATTERN = /linkedin\.com\/jobs\//;

// Show badge when user is on LinkedIn jobs
chrome.tabs.onUpdated.addListener((tabId, changeInfo, tab) => {
  if (changeInfo.status !== 'complete' || !tab.url) return;

  if (JOB_PATTERN.test(tab.url)) {
    chrome.action.setBadgeText({ text: ' ', tabId });
    chrome.action.setBadgeBackgroundColor({ color: '#635bff', tabId });

    // Re-inject content script if needed (handles SPA navigation)
    chrome.scripting.executeScript({
      target: { tabId },
      files: ['content.js'],
    }).catch(() => {});
  } else {
    chrome.action.setBadgeText({ text: '', tabId });
  }
});

chrome.tabs.onRemoved.addListener((tabId) => {
  chrome.action.setBadgeText({ text: '', tabId }).catch(() => {});
});

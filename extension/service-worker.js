/**
 * Service Worker (Background Script) for Email Classifier extension.
 *
 * Handles:
 * - API communication with local Flask server
 * - Caching classifications in chrome.storage.local
 * - Message passing between content script and side panel
 */

const API_BASE = "http://localhost:5544/api";
const CACHE_TTL_MS = 30 * 60 * 1000; // 30 minutes
const CACHE_PREFIX = "cls_";

// Open side panel when extension icon is clicked
chrome.sidePanel.setPanelBehavior({ openPanelOnActionClick: true });

// Listen for messages from content script and side panel
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message.type === "CLASSIFY_EMAILS") {
    handleClassifyRequest(message.gmail_ids, sender.tab?.id);
  }

  if (message.type === "GET_LABELS") {
    fetchLabels().then(sendResponse);
    return true; // Keep channel open for async response
  }

  if (message.type === "GET_SUMMARY") {
    fetchSummary().then(sendResponse);
    return true;
  }

  if (message.type === "CHECK_HEALTH") {
    checkHealth().then(sendResponse);
    return true;
  }
});

/**
 * Handle a batch classification request from the content script.
 */
async function handleClassifyRequest(gmailIds, tabId) {
  if (!gmailIds || gmailIds.length === 0) return;

  // 1. Check cache for existing classifications
  const cached = await getCachedClassifications(gmailIds);
  const uncachedIds = gmailIds.filter((id) => !cached[id]);

  let freshResults = {};

  // 2. Fetch uncached IDs from API
  if (uncachedIds.length > 0) {
    try {
      const response = await fetch(`${API_BASE}/classify`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ gmail_ids: uncachedIds }),
      });

      if (response.ok) {
        const data = await response.json();
        freshResults = data.classifications || {};

        // 3. Cache fresh results
        await cacheClassifications(freshResults);
      } else {
        console.error("API error:", response.status, response.statusText);
      }
    } catch (err) {
      console.error("Failed to fetch from API:", err.message);
    }
  }

  // 4. Merge cached + fresh results
  const allResults = { ...cached, ...freshResults };

  // 5. Send to content script (for badge injection)
  if (tabId) {
    try {
      chrome.tabs.sendMessage(tabId, {
        type: "CLASSIFICATION_RESULTS",
        classifications: allResults,
      });
    } catch (e) {
      // Tab may have closed
    }
  }

  // 6. Send to side panel (for sidebar display)
  try {
    chrome.runtime.sendMessage({
      type: "SIDEBAR_UPDATE",
      classifications: allResults,
    });
  } catch (e) {
    // Side panel may not be open
  }
}

/**
 * Get cached classifications from chrome.storage.local.
 */
async function getCachedClassifications(gmailIds) {
  const keys = gmailIds.map((id) => CACHE_PREFIX + id);

  return new Promise((resolve) => {
    chrome.storage.local.get(keys, (result) => {
      const cached = {};
      const now = Date.now();

      for (const id of gmailIds) {
        const entry = result[CACHE_PREFIX + id];
        if (entry && now - entry.timestamp < CACHE_TTL_MS) {
          cached[id] = entry.data;
        }
      }

      resolve(cached);
    });
  });
}

/**
 * Cache classifications in chrome.storage.local with TTL.
 */
async function cacheClassifications(classifications) {
  const items = {};
  const now = Date.now();

  for (const [id, data] of Object.entries(classifications)) {
    if (data) {
      items[CACHE_PREFIX + id] = { data, timestamp: now };
    }
  }

  if (Object.keys(items).length > 0) {
    chrome.storage.local.set(items);
  }
}

/**
 * Fetch label taxonomy from API.
 */
async function fetchLabels() {
  try {
    const response = await fetch(`${API_BASE}/labels`);
    if (response.ok) {
      return await response.json();
    }
  } catch (e) {
    console.error("Failed to fetch labels:", e.message);
  }
  return null;
}

/**
 * Fetch summary from API.
 */
async function fetchSummary() {
  try {
    const response = await fetch(`${API_BASE}/summary`);
    if (response.ok) {
      return await response.json();
    }
  } catch (e) {
    console.error("Failed to fetch summary:", e.message);
  }
  return null;
}

/**
 * Check API server health.
 */
async function checkHealth() {
  try {
    const response = await fetch(`${API_BASE}/health`);
    if (response.ok) {
      return await response.json();
    }
  } catch (e) {
    // Server not running
  }
  return null;
}

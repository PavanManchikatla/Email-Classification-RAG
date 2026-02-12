/**
 * Content Script injected into Gmail.
 *
 * Responsibilities:
 * 1. Extract visible email Gmail IDs from the DOM
 * 2. Send them to service worker for API lookup
 * 3. Inject colored classification badges into Gmail's email list
 */

// Debounce timer for MutationObserver
let debounceTimer = null;
const DEBOUNCE_MS = 300;

// Track which emails already have badges to avoid duplicates
const processedEmails = new Set();

/**
 * Extract Gmail message IDs from visible email rows.
 * Gmail uses various data attributes depending on the view.
 */
function extractVisibleEmailIds() {
  const ids = new Set();

  // Method 1: data-legacy-message-id on table rows (inbox list view)
  document.querySelectorAll("tr[data-legacy-message-id]").forEach((row) => {
    const id = row.getAttribute("data-legacy-message-id");
    if (id) ids.add(id);
  });

  // Method 2: data-message-id on div elements (conversation view)
  document.querySelectorAll("div[data-message-id]").forEach((el) => {
    const id = el.getAttribute("data-message-id");
    if (id) ids.add(id);
  });

  // Method 3: Extract from URL hash (single message view)
  const hash = window.location.hash;
  const match = hash.match(/#(?:inbox|all|sent|search)\/([a-zA-Z0-9]+)/);
  if (match && match[1].length > 10) {
    ids.add(match[1]);
  }

  return [...ids].filter(Boolean);
}

/**
 * Request classifications from the service worker.
 */
function requestClassifications(gmailIds) {
  if (gmailIds.length === 0) return;

  // Filter out already processed IDs to reduce API calls
  const newIds = gmailIds.filter((id) => !processedEmails.has(id));
  if (newIds.length === 0) return;

  chrome.runtime.sendMessage({
    type: "CLASSIFY_EMAILS",
    gmail_ids: newIds,
  });
}

/**
 * Inject classification badges into Gmail's email list.
 */
function injectBadges(classifications) {
  for (const [gmailId, data] of Object.entries(classifications)) {
    if (!data) continue;

    // Find the email row
    const row = document.querySelector(
      `tr[data-legacy-message-id="${gmailId}"]`
    );
    if (!row) continue;

    // Skip if already has a badge
    if (row.querySelector(".ec-badge")) continue;

    // Create badge element
    const badge = document.createElement("span");
    badge.className = `ec-badge ec-badge-${data.group.toLowerCase()}`;
    badge.textContent = data.label.replace(/_/g, " ");
    badge.title = `${data.label} (${Math.round(data.confidence * 100)}% confidence)`;

    // Add uncertainty indicator
    if (data.confidence < 0.7) {
      badge.textContent += " ?";
      badge.classList.add("ec-badge-uncertain");
    }

    // Insert badge into the subject cell area
    // Gmail's subject is typically in a cell with class containing "xY" or "a4W"
    const subjectArea =
      row.querySelector(".xY .y6") ||
      row.querySelector(".a4W") ||
      row.querySelector("td:nth-child(5)") ||
      row.querySelector("td:nth-child(4)");

    if (subjectArea) {
      subjectArea.insertBefore(badge, subjectArea.firstChild);
    }

    processedEmails.add(gmailId);
  }
}

/**
 * Listen for classification results from service worker.
 */
chrome.runtime.onMessage.addListener((message) => {
  if (message.type === "CLASSIFICATION_RESULTS") {
    injectBadges(message.classifications);
  }
});

/**
 * MutationObserver: watch for Gmail DOM changes (new emails loaded,
 * navigation, scrolling) and request classifications for new emails.
 */
const observer = new MutationObserver(() => {
  if (debounceTimer) clearTimeout(debounceTimer);

  debounceTimer = setTimeout(() => {
    const ids = extractVisibleEmailIds();
    if (ids.length > 0) {
      requestClassifications(ids);
    }
  }, DEBOUNCE_MS);
});

// Start observing once Gmail has loaded
function startObserver() {
  observer.observe(document.body, {
    childList: true,
    subtree: true,
  });

  // Initial scan
  const ids = extractVisibleEmailIds();
  if (ids.length > 0) {
    requestClassifications(ids);
  }
}

// Wait for Gmail to fully load before starting
if (document.readyState === "complete") {
  startObserver();
} else {
  window.addEventListener("load", startObserver);
}

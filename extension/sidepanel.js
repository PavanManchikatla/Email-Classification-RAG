/**
 * Side Panel logic for Email Classifier extension.
 *
 * Displays classified emails grouped by ACTION/INFO/NOISE
 * with connection status and auto-refresh.
 */

// Current classifications being displayed
let currentClassifications = {};

// DOM elements
const statusIndicator = document.getElementById("status-indicator");
const actionCount = document.getElementById("action-count");
const infoCount = document.getElementById("info-count");
const noiseCount = document.getElementById("noise-count");
const modelInfo = document.getElementById("model-info");
const loading = document.getElementById("loading");
const actionSection = document.getElementById("action-section");
const infoSection = document.getElementById("info-section");
const noiseSection = document.getElementById("noise-section");
const actionList = document.getElementById("action-list");
const infoList = document.getElementById("info-list");
const noiseList = document.getElementById("noise-list");
const refreshBtn = document.getElementById("refresh-btn");
const lastUpdated = document.getElementById("last-updated");

/**
 * Check API health and update status indicator.
 */
async function checkHealth() {
  chrome.runtime.sendMessage({ type: "CHECK_HEALTH" }, (response) => {
    if (response && response.status === "ok") {
      statusIndicator.className = "status-dot status-online";
      statusIndicator.title = "Connected to API";

      // Show model info
      const version = response.model_version || "unknown";
      const accuracy = response.model_accuracy
        ? `${Math.round(response.model_accuracy * 100)}%`
        : "N/A";
      modelInfo.textContent = `Model: ${version} | Accuracy: ${accuracy} | ${response.total_labeled}/${response.total_emails} labeled`;
    } else {
      statusIndicator.className = "status-dot status-offline";
      statusIndicator.title = "API server not running";
      modelInfo.textContent = "Start API server: python api_server.py";
    }
  });
}

/**
 * Render classifications grouped by type.
 */
function renderClassifications(classifications) {
  const groups = { ACTION: [], INFO: [], NOISE: [], OTHER: [] };

  for (const [gmailId, data] of Object.entries(classifications)) {
    if (!data) continue;
    const group = data.group || "OTHER";
    groups[group] = groups[group] || [];
    groups[group].push({ gmailId, ...data });
  }

  // Sort each group by confidence (lowest first, so uncertain items are visible)
  for (const group of Object.values(groups)) {
    group.sort((a, b) => a.confidence - b.confidence);
  }

  // Update counts
  actionCount.textContent = `${groups.ACTION.length} ACTION`;
  infoCount.textContent = `${groups.INFO.length} INFO`;
  noiseCount.textContent = `${groups.NOISE.length} NOISE`;

  // Render each section
  renderSection(actionSection, actionList, groups.ACTION, "action");
  renderSection(infoSection, infoList, groups.INFO, "info");
  renderSection(noiseSection, noiseList, groups.NOISE, "noise");

  // Hide loading, show sections
  loading.style.display = "none";

  // Update timestamp
  lastUpdated.textContent = `Updated: ${new Date().toLocaleTimeString()}`;
}

/**
 * Render a single group section.
 */
function renderSection(section, list, items, groupClass) {
  if (items.length === 0) {
    section.style.display = "none";
    return;
  }

  section.style.display = "block";
  list.innerHTML = "";

  for (const item of items) {
    const el = document.createElement("div");
    el.className = "email-item";

    const labelName = item.label.replace(/_/g, " ");
    const confidence = Math.round(item.confidence * 100);
    const isUncertain = item.confidence < 0.7;

    el.innerHTML = `
      <span class="email-label email-label-${groupClass}">${labelName}</span>
      <span class="email-confidence ${isUncertain ? "email-uncertain" : ""}">${confidence}%${isUncertain ? " ?" : ""}</span>
      <div class="email-subject" title="${item.gmailId}">${item.gmailId}</div>
    `;

    list.appendChild(el);
  }
}

/**
 * Listen for updates from service worker.
 */
chrome.runtime.onMessage.addListener((message) => {
  if (message.type === "SIDEBAR_UPDATE") {
    // Merge with existing classifications
    currentClassifications = {
      ...currentClassifications,
      ...message.classifications,
    };
    renderClassifications(currentClassifications);
  }
});

/**
 * Refresh button handler.
 */
refreshBtn.addEventListener("click", () => {
  checkHealth();
  // Re-request classifications from service worker
  chrome.runtime.sendMessage({ type: "GET_SUMMARY" }, (response) => {
    if (response) {
      modelInfo.textContent = `Total: ${response.total_emails} | Labeled: ${response.total_labeled} | Groups: ${response.groups.ACTION} action, ${response.groups.INFO} info, ${response.groups.NOISE} noise`;
    }
  });
});

// Initial setup
checkHealth();

// Periodic health check (every 60 seconds)
setInterval(checkHealth, 60000);

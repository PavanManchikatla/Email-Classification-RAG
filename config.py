"""
Centralized configuration for the email classification pipeline.

All secrets are loaded from environment variables (via .env file).
Every setting has a sensible default so the tool works without a .env file
(except for API keys which must be set explicitly).
"""

import os
import logging
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
DB_PATH = DATA_DIR / "emails.db"
MODEL_DIR = DATA_DIR / "model"

CREDENTIALS_PATH = PROJECT_ROOT / os.getenv("GMAIL_CREDENTIALS_FILE", "credentials.json")
TOKEN_PATH = PROJECT_ROOT / os.getenv("GMAIL_TOKEN_FILE", "token.json")
TOKENS_DIR = PROJECT_ROOT / "tokens"

# ---------------------------------------------------------------------------
# Gmail ingestion
# ---------------------------------------------------------------------------
GMAIL_MAX_RESULTS_PER_PAGE = int(os.getenv("GMAIL_MAX_RESULTS_PER_PAGE", "100"))
GMAIL_MAX_PAGES = int(os.getenv("GMAIL_MAX_PAGES", "5"))
GMAIL_BULK_MAX_PAGES = int(os.getenv("GMAIL_BULK_MAX_PAGES", "200"))
GMAIL_TARGET_TOTAL_EMAILS = int(os.getenv("GMAIL_TARGET_TOTAL_EMAILS", "15000"))

# ---------------------------------------------------------------------------
# LLM (for generating training labels)
# ---------------------------------------------------------------------------
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "claude-haiku-4-5-20251001")
CLASSIFY_BATCH_SIZE = int(os.getenv("CLASSIFY_BATCH_SIZE", "10"))

# ---------------------------------------------------------------------------
# Classification labels — single source of truth
#
# Organized into three action groups:
#   ACTION (6): need user response or immediate attention
#   INFO   (6): read at leisure, no urgency
#   NOISE  (3): batch-archive or ignore
# ---------------------------------------------------------------------------
LABELS = [
    # --- ACTION ---
    "job_opportunity",
    "job_interview",
    "personal",
    "finance_alert",
    "security_auth",
    "events_calendar",
    # --- INFORMATIONAL ---
    "job_application_confirm",
    "travel",
    "shopping_orders",
    "finance_receipt",
    "newsletter_content",
    "education",
    # --- NOISE ---
    "social_notification",
    "marketing_promo",
    "account_service",
]

# Label descriptions — used to build the LLM classification prompt
LABEL_DESCRIPTIONS = {
    "job_opportunity": "Recruiter outreach, job recommendations, referral messages, 'we found your profile' emails",
    "job_interview": "Interview scheduling, coding challenges, take-home assignments, offer letters, rejection notices — active hiring process",
    "personal": "Direct emails from friends/family, genuine 1:1 personal conversations",
    "finance_alert": "Bank alerts, fraud warnings, bill due reminders, tax documents, large transaction notices — needs review",
    "security_auth": "Password resets, 2FA codes, login alerts ('new sign-in from...'), breach notifications, account lockout",
    "events_calendar": "Event invitations, RSVPs, calendar notifications, meetup/webinar invites",
    "job_application_confirm": "'We received your application' confirmations, application portal links, status acknowledgments",
    "travel": "Flight/hotel bookings, itineraries, boarding passes, check-in reminders, trip notifications",
    "shopping_orders": "Order confirmations, shipping/delivery tracking, return/refund confirmations",
    "finance_receipt": "Payment receipts, subscription renewals, monthly statements — just records, no action needed",
    "newsletter_content": "Substantive content newsletters (Substack, industry blogs, curated digests) the user subscribed to",
    "education": "Online course updates (Coursera, Udemy), certifications, learning platform activity, academic communications",
    "social_notification": "Social media notifications (LinkedIn, Instagram, Facebook, etc.), likes, comments, connection requests",
    "marketing_promo": "Sales announcements, discount codes, product launches, 'we miss you' emails, cold promotional outreach",
    "account_service": "Terms of service updates, privacy policy changes, product announcements, generic service emails, anything else",
}

# Priority order for digest views (most important first)
PRIORITY_ORDER = [
    # --- ACTION: Handle these first ---
    "job_interview",
    "security_auth",
    "job_opportunity",
    "personal",
    "finance_alert",
    "events_calendar",
    # --- INFORMATIONAL: Read when you have time ---
    "job_application_confirm",
    "travel",
    "shopping_orders",
    "finance_receipt",
    "newsletter_content",
    "education",
    # --- NOISE: Batch-process or ignore ---
    "social_notification",
    "marketing_promo",
    "account_service",
]

# Labels that represent actionable emails (need user response)
ACTION_LABELS = {
    "job_interview",
    "security_auth",
    "job_opportunity",
    "personal",
    "finance_alert",
    "events_calendar",
}

# ---------------------------------------------------------------------------
# Auto-evolve settings
# ---------------------------------------------------------------------------
EVOLVE_CONFIDENCE_THRESHOLD = float(os.getenv("EVOLVE_CONFIDENCE_THRESHOLD", "0.5"))
EVOLVE_UNCERTAINTY_MARGIN = float(os.getenv("EVOLVE_UNCERTAINTY_MARGIN", "0.15"))
EVOLVE_MIN_CLUSTER_SIZE = int(os.getenv("EVOLVE_MIN_CLUSTER_SIZE", "20"))
EVOLVE_SCHEDULE_HOURS = int(os.getenv("EVOLVE_SCHEDULE_HOURS", "6"))
EVOLVE_MIN_NEW_LABELS_FOR_RETRAIN = int(os.getenv("EVOLVE_MIN_NEW_LABELS_FOR_RETRAIN", "50"))

# ---------------------------------------------------------------------------
# API server
# ---------------------------------------------------------------------------
API_PORT = int(os.getenv("API_PORT", "5544"))
API_HOST = os.getenv("API_HOST", "127.0.0.1")

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

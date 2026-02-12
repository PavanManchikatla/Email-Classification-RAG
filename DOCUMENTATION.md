# Email Classification Pipeline — Documentation

A personal email classification system that ingests emails from multiple Gmail accounts, generates training labels using Claude Haiku, trains a local ML model (TF-IDF + Random Forest), and provides CLI views for managing your inbox by priority.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Architecture](#architecture)
3. [Setup Guide](#setup-guide)
4. [Multi-Account Gmail Configuration](#multi-account-gmail-configuration)
5. [The 15-Category Taxonomy](#the-15-category-taxonomy)
6. [Pipeline Usage](#pipeline-usage)
7. [How the ML Model Works](#how-the-ml-model-works)
8. [Retraining and Adding Categories](#retraining-and-adding-categories)
9. [Security Practices](#security-practices)
10. [Troubleshooting](#troubleshooting)

---

## Project Overview

### Problem

Email inboxes are noisy. Important messages (interview requests, bank alerts, personal notes) get buried under promotional spam, social notifications, and account updates. Gmail's built-in categories (Primary, Social, Promotions, Updates) are too coarse to be truly useful.

### Solution

This pipeline classifies emails into **15 actionable categories** grouped by urgency:

- **ACTION** — needs your response (job interviews, security alerts, personal messages)
- **INFORMATIONAL** — read when you have time (order tracking, newsletters, receipts)
- **NOISE** — batch-archive or ignore (marketing, social notifications, TOS updates)

### Design Decisions

| Decision | Rationale |
|----------|-----------|
| **ML model, not LLM** for classification | Free, offline, instant (~ms per email). LLM costs money per call. |
| **LLM for training labels only** | Claude Haiku labels the initial dataset at ~$0.00018/email, then the ML model takes over. |
| **TF-IDF + Random Forest** | Fast to train (seconds), interpretable, works well with text classification at this scale. No GPU required. |
| **15 categories** | More granular than Gmail's 4 tabs, but not so many that classification becomes unreliable. Grouped by actionability. |
| **SQLite** | Zero configuration, single-file database, perfect for personal use. |
| **Separate scripts** | Each step is independently runnable and debuggable. No monolithic orchestrator. |

---

## Architecture

### File Structure

```
email-classification/
├── config.py                  # Centralized configuration (paths, labels, settings)
├── requirements.txt           # Python dependencies
├── .env.example               # Template for environment variables
├── .gitignore                 # Git ignore rules
├── DOCUMENTATION.md           # This file
├── credentials.json           # Google OAuth client (NOT committed)
├── tokens/                    # Per-account OAuth tokens (NOT committed)
│   ├── token_user1_gmail_com.json
│   └── token_user2_gmail_com.json
├── data/                      # Runtime data (NOT committed)
│   ├── emails.db              # SQLite database
│   └── model/
│       └── email_classifier.joblib
└── src/
    ├── __init__.py
    ├── db.py                  # Shared database layer
    ├── gmail_ingest.py        # Multi-account Gmail ingestion
    ├── generate_labels.py     # LLM-based label generation (Claude Haiku)
    ├── label_emails.py        # Interactive manual labeling CLI
    ├── train_model.py         # ML model training
    ├── classify.py            # ML model inference
    └── digest.py              # CLI output views
```

### Data Flow

```
Gmail API  ──>  gmail_ingest.py  ──>  SQLite DB (emails table)
                                          │
                 generate_labels.py  ─────┤  (LLM labels → email_labels table)
                 label_emails.py   ───────┤  (manual labels → email_labels table)
                                          │
                 train_model.py  ─────────┤  (reads labeled emails → trains model → .joblib)
                                          │
                 classify.py  ────────────┤  (loads model → labels remaining emails)
                                          │
                 digest.py  ──────────────┘  (reads DB → CLI views)
```

### Database Schema

**`emails` table:**

| Column | Type | Description |
|--------|------|-------------|
| `id` | INTEGER PK | Auto-increment row ID |
| `gmail_id` | TEXT | Gmail message ID |
| `account_email` | TEXT | Which Gmail account this came from |
| `thread_id` | TEXT | Gmail thread ID |
| `internal_date` | INTEGER | Unix timestamp in milliseconds |
| `from_addr` | TEXT | Sender address |
| `to_addr` | TEXT | Recipient address |
| `subject` | TEXT | Email subject line |
| `snippet` | TEXT | Gmail-generated snippet |
| `body` | TEXT | Extracted plain text body |
| `label_ids` | TEXT | Comma-separated Gmail label IDs |
| `created_at` | TIMESTAMP | When the row was inserted |

Unique constraint: `UNIQUE(account_email, gmail_id)` — prevents duplicates across accounts.

**`email_labels` table:**

| Column | Type | Description |
|--------|------|-------------|
| `id` | INTEGER PK | Auto-increment row ID |
| `email_id` | INTEGER FK | References `emails.id` |
| `label` | TEXT | Classification label |
| `confidence` | REAL | Confidence score (0.0–1.0) |
| `source` | TEXT | Who assigned the label: `manual`, `llm`, or `model` |
| `created_at` | TIMESTAMP | When the label was assigned |

Unique constraint: `UNIQUE(email_id)` — one label per email.

---

## Setup Guide

### Prerequisites

- Python 3.10+
- A Google Cloud project with the Gmail API enabled
- An Anthropic API key (for training label generation)

### Step 1: Clone and Create Virtual Environment

```bash
cd email-classification
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Step 2: Set Up Google Cloud Credentials

1. Go to the [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project (or use an existing one)
3. Enable the **Gmail API** under APIs & Services
4. Create **OAuth 2.0 Client ID** credentials (Desktop application type)
5. Download the JSON file and save it as `credentials.json` in the project root

> **Important:** The same `credentials.json` works for all your Gmail accounts. You only need one.

### Step 3: Configure Environment Variables

```bash
cp .env.example .env
```

Edit `.env` and set your Anthropic API key:

```
ANTHROPIC_API_KEY=sk-ant-your-key-here
```

### Step 4: Verify Setup

```bash
python3 -c "import config; from src.db import init_db; init_db(); print('Setup OK')"
```

This creates the `data/` directory and initializes the SQLite database.

---

## Multi-Account Gmail Configuration

### How It Works

The system uses a **single OAuth client** (`credentials.json`) to authenticate with multiple Gmail accounts. Each account gets its own token file stored in the `tokens/` directory.

When you run `gmail_ingest.py`:

1. It scans `tokens/` for existing authenticated accounts and reconnects them
2. It asks: "Do you want to add another email account? (y/n)"
3. For each new account, it opens your browser for Google OAuth consent
4. After consent, the token is saved as `tokens/token_<email>.json`
5. You can add as many accounts as you want

### Token Storage

```
tokens/
├── token_personaluser_gmail_com.json
├── token_jobsearch_gmail_com.json
└── token_generaluse_gmail_com.json
```

Token filenames are derived from the email address with `@` and `.` replaced by `_`.

### Permissions

The system requests **read-only** access (`gmail.readonly` scope). It cannot send, delete, or modify any emails.

### Legacy Token Migration

If you previously used a single `token.json` in the project root, the system automatically migrates it to the new `tokens/` directory format on first run.

### Removing an Account

Delete the corresponding token file from `tokens/`. The emails already in the database are preserved.

---

## The 15-Category Taxonomy

Categories are organized into three urgency groups:

### ACTION — Needs Your Response (6 categories)

| Label | Description | Examples |
|-------|-------------|----------|
| `job_opportunity` | Recruiter outreach, job recommendations | LinkedIn InMail from recruiter, "we found your profile" |
| `job_interview` | Active hiring process communications | Interview scheduling, coding challenges, offer/rejection letters |
| `personal` | Direct 1:1 emails from people you know | Friend/family emails, genuine personal conversations |
| `finance_alert` | Financial items needing review | Bank fraud alerts, bill due reminders, tax documents |
| `security_auth` | Security and authentication | Password resets, 2FA codes, "new sign-in from..." alerts |
| `events_calendar` | Event-related communications | Event invitations, RSVPs, calendar notifications, meetup invites |

### INFORMATIONAL — Read When You Have Time (6 categories)

| Label | Description | Examples |
|-------|-------------|----------|
| `job_application_confirm` | Application acknowledgments | "We received your application", portal links |
| `travel` | Travel-related notifications | Flight bookings, hotel confirmations, boarding passes |
| `shopping_orders` | E-commerce order lifecycle | Order confirmations, shipping tracking, delivery notices |
| `finance_receipt` | Financial records (no action needed) | Payment receipts, subscription renewals, statements |
| `newsletter_content` | Substantive content you subscribed to | Substack posts, industry newsletters, curated digests |
| `education` | Learning platform activity | Coursera updates, certifications, academic communications |

### NOISE — Batch-Archive or Ignore (3 categories)

| Label | Description | Examples |
|-------|-------------|----------|
| `social_notification` | Social media notifications | LinkedIn likes/comments, Instagram follows, Facebook updates |
| `marketing_promo` | Promotional and sales emails | Discount codes, product launches, "we miss you" campaigns |
| `account_service` | Generic service communications | TOS updates, privacy policies, product announcements |

### Disambiguation Rules

The LLM prompt includes specific rules for ambiguous cases:

- LinkedIn recruiter InMail → `job_opportunity` (not `social_notification`)
- LinkedIn "X viewed your profile" → `social_notification`
- "Thank you for applying" → `job_application_confirm` (not `job_interview`)
- Bank fraud alert → `finance_alert` (not `security_auth`)
- Amazon order confirmation → `shopping_orders` (not `finance_receipt`)
- Stripe payment receipt → `finance_receipt`
- Coursera "assignment due" → `education`; Coursera "50% off" → `marketing_promo`
- Company blog newsletter → `newsletter_content`; Company "sale" email → `marketing_promo`

---

## Pipeline Usage

### Step 1: Ingest Emails

```bash
python -m src.gmail_ingest
```

**What happens:**
1. Loads existing authenticated accounts from `tokens/`
2. Prompts to add new accounts (opens browser for OAuth)
3. Fetches emails from all authenticated accounts
4. Stores them in `data/emails.db`
5. Uses incremental sync — only fetches emails newer than what's already in the DB
6. Fetches from ALL folders (not just Inbox) for training diversity
7. Stops when reaching the target email count (default: 15,000)

**Configuration:**

| Setting | Default | Description |
|---------|---------|-------------|
| `GMAIL_MAX_RESULTS_PER_PAGE` | 100 | Emails per API page |
| `GMAIL_BULK_MAX_PAGES` | 200 | Max pages per account (bulk mode) |
| `GMAIL_TARGET_TOTAL_EMAILS` | 15000 | Stop when this many emails are in the DB |

### Step 2: Generate Training Labels

```bash
# Label all unlabeled emails
python -m src.generate_labels

# Preview without saving
python -m src.generate_labels --dry-run

# Re-label everything (for taxonomy changes)
python -m src.generate_labels --clear-existing

# Custom batch size
python -m src.generate_labels --batch-size 5
```

**What happens:**
1. Fetches unlabeled emails in batches
2. Sends each batch to Claude Haiku with the 15-category prompt
3. Parses JSON response, validates labels against `config.LABELS`
4. Saves labels with `source='llm'` and confidence scores

**Cost estimate:** ~$0.00018 per email. 15,000 emails costs ~$2.70.

### Step 3: Train the ML Model

```bash
python -m src.train_model

# Custom test split
python -m src.train_model --test-size 0.3
```

**What happens:**
1. Loads all labeled emails from the database
2. Combines `from_addr + subject + body_preview` as features
3. Splits 80/20 for train/test
4. Trains TF-IDF vectorizer + Random Forest classifier
5. Prints classification report (precision, recall, F1 per category)
6. Saves model to `data/model/email_classifier.joblib`

### Step 4: Classify with ML Model

```bash
# Classify all unlabeled emails
python -m src.classify

# Preview without saving
python -m src.classify --dry-run
```

**What happens:**
1. Loads the trained `.joblib` model
2. Fetches unlabeled emails
3. Predicts labels with confidence scores
4. Saves labels with `source='model'`

### Step 5: View Results

```bash
# Label distribution summary with bar chart
python -m src.digest summary

# Priority-sorted inbox (last 7 days)
python -m src.digest priority

# Priority inbox for last 30 days
python -m src.digest priority --days 30

# Today's emails with action items highlighted
python -m src.digest daily
```

### Step 6: Manual Labeling (Optional)

```bash
python -m src.label_emails
```

Interactive CLI that shows unlabeled emails one at a time with auto-label suggestions. Useful for correcting misclassifications or labeling edge cases.

---

## How the ML Model Works

### Feature Extraction: TF-IDF

**TF-IDF** (Term Frequency–Inverse Document Frequency) converts email text into numerical vectors:

- **TF (Term Frequency):** How often a word appears in this email
- **IDF (Inverse Document Frequency):** How rare a word is across all emails
- **TF-IDF = TF × IDF:** Words that are frequent in one email but rare overall get high scores

Configuration:
- `max_features=5000` — vocabulary limited to top 5,000 terms
- `ngram_range=(1, 2)` — uses single words and word pairs ("job offer", "password reset")
- `stop_words="english"` — removes common words (the, is, at, etc.)
- `sublinear_tf=True` — applies logarithmic scaling to term frequency

### Classifier: Random Forest

A **Random Forest** is an ensemble of decision trees that each vote on the classification:

- `n_estimators=100` — 100 decision trees
- Each tree is trained on a random subset of the data and features
- Final prediction = majority vote across all trees
- Confidence = fraction of trees that agreed on the winning label

### Why This Works Well for Email

1. **Sender patterns:** `noreply@linkedin.com` strongly predicts `social_notification`
2. **Subject keywords:** "interview" predicts `job_interview`, "shipped" predicts `shopping_orders`
3. **Body content:** 2FA codes contain specific patterns, receipts have dollar amounts
4. **Bigrams catch phrases:** "new sign-in" → `security_auth`, "order confirmed" → `shopping_orders`

### Expected Performance

With ~15,000 labeled emails and 15 categories:
- **Overall accuracy:** 80–90%+ depending on class balance
- **Strong categories:** `marketing_promo`, `social_notification`, `security_auth` (very distinctive patterns)
- **Weaker categories:** `personal`, `education` (fewer samples, more varied language)
- **Training time:** 2–5 seconds

---

## Retraining and Adding Categories

### Adding a New Category

1. Add the label name to `config.LABELS` list
2. Add its description to `config.LABEL_DESCRIPTIONS`
3. Add it to `config.PRIORITY_ORDER` in the appropriate position
4. If it's actionable, add it to `config.ACTION_LABELS`
5. Re-label existing emails:
   ```bash
   python -m src.generate_labels --clear-existing
   ```
6. Retrain the model:
   ```bash
   python -m src.train_model
   ```

### Improving Accuracy

- **Add more training data:** Ingest more emails, re-label with LLM
- **Manual corrections:** Use `label_emails.py` to fix misclassified emails, then retrain
- **Adjust disambiguation rules:** Edit the LLM prompt in `generate_labels.py`
- **Check low-confidence predictions:** `digest.py priority` flags emails with `[?]` when confidence < 70%

### Full Re-labeling Workflow

When changing the taxonomy (adding/removing/renaming categories):

```bash
# 1. Edit config.py with new categories
# 2. Clear all existing labels and re-label
python -m src.generate_labels --clear-existing

# 3. Retrain the model
python -m src.train_model

# 4. Classify any remaining unlabeled emails
python -m src.classify

# 5. Check results
python -m src.digest summary
```

---

## Security Practices

### Secrets Management

| Secret | Storage | Committed? |
|--------|---------|-----------|
| Anthropic API key | `.env` file | No (`.gitignore`) |
| Google OAuth client | `credentials.json` | No (`.gitignore`) |
| Per-account tokens | `tokens/*.json` | No (`.gitignore`) |
| Email database | `data/emails.db` | No (`.gitignore`) |

All secrets are loaded via environment variables using `python-dotenv`. The `.env.example` file provides a template without real values.

### OAuth Scopes

The system requests only `gmail.readonly` scope — it **cannot**:
- Send emails
- Delete emails
- Modify labels or settings
- Access contacts or calendar

### Data Privacy

- All data is stored locally in SQLite (`data/emails.db`)
- Email bodies are sent to Anthropic's API only during the labeling step
- The trained ML model runs entirely offline — no API calls during classification
- Token files contain OAuth refresh tokens and should never be shared

### What's Safe to Commit

- `config.py`, `src/*.py` — code only, no secrets
- `requirements.txt` — dependency list
- `.env.example` — template with placeholder values
- `.gitignore` — ensures secrets aren't accidentally committed
- `DOCUMENTATION.md` — this file

### What's NOT Safe to Commit

- `.env` — contains your Anthropic API key
- `credentials.json` — Google OAuth client secret
- `tokens/` — OAuth refresh tokens for each Gmail account
- `data/` — contains your actual emails and the trained model

---

## Troubleshooting

### "ModuleNotFoundError: No module named 'bs4'"

Make sure you're in the virtual environment:
```bash
source .venv/bin/activate
pip install -r requirements.txt
```

### "FileNotFoundError: credentials.json"

Place your Google OAuth credentials file in the project root:
```bash
cp /path/to/downloaded/credentials.json ./credentials.json
```

### "ANTHROPIC_API_KEY not set"

Create a `.env` file from the example:
```bash
cp .env.example .env
# Edit .env and add your API key
```

### "No unlabeled emails to classify"

All emails already have labels. If you want to re-label:
```bash
python -m src.generate_labels --clear-existing
```

### Gmail API Rate Limiting

The Gmail API has quotas. If you hit rate limits:
- Reduce `GMAIL_MAX_RESULTS_PER_PAGE` in `.env`
- Reduce `GMAIL_BULK_MAX_PAGES`
- Wait and re-run — incremental sync picks up where it left off

### 429 Too Many Requests (Anthropic)

The Anthropic SDK has built-in retry logic with exponential backoff. If you see 429 errors in the logs, the system is automatically retrying. Reduce `CLASSIFY_BATCH_SIZE` if it persists.

### Low Model Accuracy

- Check label distribution: `python -m src.digest summary`
- Categories with very few samples will have poor accuracy
- Ingest more emails from accounts that produce underrepresented categories
- Use manual labeling to correct systematic misclassifications
- Consider increasing `max_features` in `train_model.py` for larger datasets

### Database Migration Errors

If the database is corrupted or migration fails:
```bash
# Back up existing data
cp data/emails.db data/emails.db.backup

# Delete and reinitialize
rm data/emails.db
python3 -c "from src.db import init_db; init_db()"

# Re-ingest and re-label
python -m src.gmail_ingest
python -m src.generate_labels
python -m src.train_model
```

### Adding a New Gmail Account Later

Just re-run the ingest script:
```bash
python -m src.gmail_ingest
```

It will find your existing accounts and ask if you want to add another one. The incremental sync ensures existing emails aren't re-fetched.

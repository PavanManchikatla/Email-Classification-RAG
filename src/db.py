"""
Shared database layer for the email classification pipeline.

All modules import from here instead of managing their own connections.
Handles schema creation, migrations, and common queries.
"""

import sqlite3
import logging
from contextlib import contextmanager
from datetime import datetime, timedelta

import config

logger = logging.getLogger(__name__)


@contextmanager
def get_connection():
    """Yield a SQLite connection with Row factory. Auto-commits and closes."""
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(config.DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")  # Allow concurrent reads during writes
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def _migrate_emails_table_v2(conn):
    """
    Migrate emails table to add account_email column and composite unique key.

    SQLite cannot ALTER TABLE to change constraints, so we rebuild the table:
    1. Rename old table
    2. Create new table with correct schema
    3. Copy data (existing rows get account_email='unknown')
    4. Drop old table
    5. Recreate indexes

    The id column is preserved so email_labels foreign keys remain valid.
    """
    logger.info("Migration: rebuilding emails table for multi-account support...")

    conn.execute("ALTER TABLE emails RENAME TO emails_old;")

    conn.execute(
        """
        CREATE TABLE emails (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            gmail_id TEXT,
            account_email TEXT DEFAULT 'unknown',
            thread_id TEXT,
            internal_date INTEGER,
            from_addr TEXT,
            to_addr TEXT,
            subject TEXT,
            snippet TEXT,
            body TEXT,
            label_ids TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(account_email, gmail_id)
        );
        """
    )

    conn.execute(
        """
        INSERT INTO emails (id, gmail_id, account_email, thread_id, internal_date,
                           from_addr, to_addr, subject, snippet, body, label_ids, created_at)
        SELECT id, gmail_id, 'unknown', thread_id, internal_date,
               from_addr, to_addr, subject, snippet, body, label_ids, created_at
        FROM emails_old;
        """
    )

    conn.execute("DROP TABLE emails_old;")

    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_emails_internal_date
        ON emails(internal_date);
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_emails_account
        ON emails(account_email);
        """
    )

    logger.info("Migration: emails table rebuilt with account_email column")


def init_db():
    """Create tables if they don't exist and run migrations."""
    with get_connection() as conn:
        cur = conn.cursor()

        # Emails table — new schema with account_email + composite unique key
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS emails (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                gmail_id TEXT,
                account_email TEXT DEFAULT 'unknown',
                thread_id TEXT,
                internal_date INTEGER,
                from_addr TEXT,
                to_addr TEXT,
                subject TEXT,
                snippet TEXT,
                body TEXT,
                label_ids TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(account_email, gmail_id)
            );
            """
        )

        # Email labels table
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS email_labels (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email_id INTEGER NOT NULL,
                label TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(email_id),
                FOREIGN KEY(email_id) REFERENCES emails(id)
            );
            """
        )

        # Model versions table — tracks each training run
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS model_versions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                version TEXT NOT NULL,
                model_path TEXT NOT NULL,
                num_samples INTEGER,
                num_categories INTEGER,
                accuracy REAL,
                macro_f1 REAL,
                report_json TEXT,
                trigger TEXT DEFAULT 'manual',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """
        )

        # Category proposals table — stores discovered category candidates
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS category_proposals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                proposed_name TEXT NOT NULL,
                cluster_size INTEGER,
                sample_email_ids TEXT,
                llm_reasoning TEXT,
                status TEXT DEFAULT 'pending',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """
        )

        # --- Migrations (run BEFORE creating indexes on new columns) ---

        # Check if emails table needs v2 migration (account_email column)
        email_cols = {
            row[1] for row in cur.execute("PRAGMA table_info(emails)").fetchall()
        }
        if "account_email" not in email_cols:
            _migrate_emails_table_v2(conn)

        # Add confidence and source columns to email_labels if missing
        label_cols = {
            row[1] for row in cur.execute("PRAGMA table_info(email_labels)").fetchall()
        }

        if "confidence" not in label_cols:
            cur.execute(
                "ALTER TABLE email_labels ADD COLUMN confidence REAL DEFAULT 1.0"
            )
            logger.info("Migration: added 'confidence' column to email_labels")

        if "source" not in label_cols:
            cur.execute(
                "ALTER TABLE email_labels ADD COLUMN source TEXT DEFAULT 'manual'"
            )
            logger.info("Migration: added 'source' column to email_labels")

        # Indexes (after migrations so columns exist)
        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_emails_internal_date
            ON emails(internal_date);
            """
        )
        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_emails_account
            ON emails(account_email);
            """
        )

    logger.info("Database initialized at %s", config.DB_PATH)


# ---------------------------------------------------------------------------
# Write operations
# ---------------------------------------------------------------------------


def save_email(email_row: dict) -> bool:
    """
    Insert a single email row. Returns True if inserted, False if duplicate.
    """
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT OR IGNORE INTO emails
            (gmail_id, account_email, thread_id, internal_date, from_addr,
             to_addr, subject, snippet, body, label_ids)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """,
            (
                email_row["gmail_id"],
                email_row.get("account_email", "unknown"),
                email_row["thread_id"],
                email_row["internal_date"],
                email_row["from_addr"],
                email_row["to_addr"],
                email_row["subject"],
                email_row["snippet"],
                email_row["body"],
                email_row["label_ids"],
            ),
        )
        return cur.rowcount > 0


def save_label(
    email_id: int,
    label: str,
    confidence: float = 1.0,
    source: str = "manual",
):
    """Insert or replace a classification label for an email."""
    with get_connection() as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO email_labels (email_id, label, confidence, source)
            VALUES (?, ?, ?, ?);
            """,
            (email_id, label, confidence, source),
        )


def clear_labels():
    """Delete all labels. Used when re-labeling with a new taxonomy."""
    with get_connection() as conn:
        result = conn.execute("DELETE FROM email_labels;")
        logger.info("Cleared %d labels from email_labels", result.rowcount)


# ---------------------------------------------------------------------------
# Read operations
# ---------------------------------------------------------------------------


def get_unlabeled_emails(batch_size: int = 20) -> list:
    """Fetch emails that have no label yet, most recent first."""
    with get_connection() as conn:
        return conn.execute(
            """
            SELECT e.id, e.from_addr, e.subject, e.body
            FROM emails e
            LEFT JOIN email_labels l ON e.id = l.email_id
            WHERE l.email_id IS NULL
            ORDER BY e.internal_date DESC
            LIMIT ?;
            """,
            (batch_size,),
        ).fetchall()


def get_unlabeled_emails_full(batch_size: int = 20) -> list:
    """Fetch full email data for unlabeled emails (used by classifiers)."""
    with get_connection() as conn:
        return conn.execute(
            """
            SELECT e.id, e.from_addr, e.to_addr, e.subject, e.body, e.snippet
            FROM emails e
            LEFT JOIN email_labels l ON e.id = l.email_id
            WHERE l.email_id IS NULL
            ORDER BY e.internal_date DESC
            LIMIT ?;
            """,
            (batch_size,),
        ).fetchall()


def get_latest_internal_date(account_email: str = None) -> int | None:
    """Return the most recent internal_date in the DB, optionally per account."""
    with get_connection() as conn:
        if account_email:
            row = conn.execute(
                "SELECT MAX(internal_date) FROM emails WHERE account_email = ?;",
                (account_email,),
            ).fetchone()
        else:
            row = conn.execute(
                "SELECT MAX(internal_date) FROM emails;"
            ).fetchone()
        return row[0] if row and row[0] else None


def get_labeled_emails() -> list:
    """Fetch all labeled emails (for training). Returns full email + label."""
    with get_connection() as conn:
        return conn.execute(
            """
            SELECT e.id, e.from_addr, e.subject, e.body, l.label, l.confidence, l.source
            FROM emails e
            INNER JOIN email_labels l ON e.id = l.email_id
            ORDER BY e.internal_date DESC;
            """
        ).fetchall()


def get_label_summary() -> dict:
    """Return {label: count} for all labeled emails."""
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT label, COUNT(*) as cnt
            FROM email_labels
            GROUP BY label
            ORDER BY cnt DESC;
            """
        ).fetchall()
        return {row["label"]: row["cnt"] for row in rows}


def get_unlabeled_count() -> int:
    """Return count of unlabeled emails."""
    with get_connection() as conn:
        row = conn.execute(
            """
            SELECT COUNT(*)
            FROM emails e
            LEFT JOIN email_labels l ON e.id = l.email_id
            WHERE l.email_id IS NULL;
            """
        ).fetchone()
        return row[0] if row else 0


def get_total_email_count() -> int:
    """Return total number of emails in the database."""
    with get_connection() as conn:
        row = conn.execute("SELECT COUNT(*) FROM emails;").fetchone()
        return row[0] if row else 0


def get_account_email_counts() -> dict:
    """Return {account_email: count} for all accounts."""
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT account_email, COUNT(*) as cnt
            FROM emails
            GROUP BY account_email
            ORDER BY cnt DESC;
            """
        ).fetchall()
        return {row["account_email"]: row["cnt"] for row in rows}


def get_recent_emails(days: int = 7) -> list:
    """Fetch emails from the last N days with their labels (if any)."""
    cutoff_ms = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)
    with get_connection() as conn:
        return conn.execute(
            """
            SELECT e.id, e.from_addr, e.subject, e.internal_date,
                   l.label, l.confidence, l.source
            FROM emails e
            LEFT JOIN email_labels l ON e.id = l.email_id
            WHERE e.internal_date >= ?
            ORDER BY e.internal_date DESC;
            """,
            (cutoff_ms,),
        ).fetchall()


def get_email_with_label(email_id: int):
    """Fetch a single email with its label (if any)."""
    with get_connection() as conn:
        return conn.execute(
            """
            SELECT e.id, e.from_addr, e.subject, e.body,
                   l.label, l.confidence, l.source
            FROM emails e
            LEFT JOIN email_labels l ON e.id = l.email_id
            WHERE e.id = ?;
            """,
            (email_id,),
        ).fetchone()


def get_emails_by_ids(email_ids: list) -> list:
    """Fetch full email rows for given internal IDs."""
    if not email_ids:
        return []
    with get_connection() as conn:
        placeholders = ",".join("?" for _ in email_ids)
        return conn.execute(
            f"""
            SELECT e.id, e.from_addr, e.subject, e.body, e.snippet,
                   l.label, l.confidence, l.source
            FROM emails e
            LEFT JOIN email_labels l ON e.id = l.email_id
            WHERE e.id IN ({placeholders});
            """,
            email_ids,
        ).fetchall()


def get_labels_by_gmail_ids(gmail_ids: list) -> dict:
    """
    Batch lookup for Chrome extension.

    Returns {gmail_id: {label, confidence, source}} for given Gmail IDs.
    """
    if not gmail_ids:
        return {}
    with get_connection() as conn:
        placeholders = ",".join("?" for _ in gmail_ids)
        rows = conn.execute(
            f"""
            SELECT e.gmail_id, l.label, l.confidence, l.source
            FROM emails e
            INNER JOIN email_labels l ON e.id = l.email_id
            WHERE e.gmail_id IN ({placeholders});
            """,
            gmail_ids,
        ).fetchall()
        return {
            row["gmail_id"]: {
                "label": row["label"],
                "confidence": row["confidence"],
                "source": row["source"],
            }
            for row in rows
        }


def get_low_confidence_emails(threshold: float = 0.6, limit: int = 500) -> list:
    """Fetch emails with confidence below threshold (for clustering)."""
    with get_connection() as conn:
        return conn.execute(
            """
            SELECT e.id, e.from_addr, e.subject, e.body, e.snippet,
                   l.label, l.confidence
            FROM emails e
            INNER JOIN email_labels l ON e.id = l.email_id
            WHERE l.confidence < ?
            ORDER BY l.confidence ASC
            LIMIT ?;
            """,
            (threshold, limit),
        ).fetchall()


def get_labeled_count() -> int:
    """Return count of labeled emails."""
    with get_connection() as conn:
        row = conn.execute("SELECT COUNT(*) FROM email_labels;").fetchone()
        return row[0] if row else 0


# ---------------------------------------------------------------------------
# Model version operations
# ---------------------------------------------------------------------------


def save_model_version(
    version: str,
    model_path: str,
    num_samples: int,
    num_categories: int,
    accuracy: float,
    macro_f1: float,
    report_json: str,
    trigger: str = "manual",
):
    """Record a model training run."""
    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO model_versions
            (version, model_path, num_samples, num_categories,
             accuracy, macro_f1, report_json, trigger)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?);
            """,
            (version, model_path, num_samples, num_categories,
             accuracy, macro_f1, report_json, trigger),
        )
    logger.info("Saved model version: %s (accuracy=%.3f, f1=%.3f)", version, accuracy, macro_f1)


def get_latest_model_version():
    """Return the most recent model_versions row, or None."""
    with get_connection() as conn:
        return conn.execute(
            "SELECT * FROM model_versions ORDER BY created_at DESC LIMIT 1;"
        ).fetchone()


def get_model_version_history(limit: int = 10) -> list:
    """Return recent model versions for trend monitoring."""
    with get_connection() as conn:
        return conn.execute(
            "SELECT * FROM model_versions ORDER BY created_at DESC LIMIT ?;",
            (limit,),
        ).fetchall()


def get_model_version_count() -> int:
    """Return count of model versions."""
    with get_connection() as conn:
        row = conn.execute("SELECT COUNT(*) FROM model_versions;").fetchone()
        return row[0] if row else 0


# ---------------------------------------------------------------------------
# Category proposal operations
# ---------------------------------------------------------------------------


def save_category_proposal(
    proposed_name: str,
    cluster_size: int,
    sample_email_ids: str,
    llm_reasoning: str,
):
    """Record a proposed new category from clustering."""
    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO category_proposals
            (proposed_name, cluster_size, sample_email_ids, llm_reasoning)
            VALUES (?, ?, ?, ?);
            """,
            (proposed_name, cluster_size, sample_email_ids, llm_reasoning),
        )
    logger.info("Saved category proposal: %s (cluster_size=%d)", proposed_name, cluster_size)


def get_pending_proposals() -> list:
    """Return pending category proposals."""
    with get_connection() as conn:
        return conn.execute(
            """
            SELECT * FROM category_proposals
            WHERE status = 'pending'
            ORDER BY created_at DESC;
            """
        ).fetchall()


def update_proposal_status(proposal_id: int, status: str):
    """Update a proposal status to 'accepted' or 'rejected'."""
    with get_connection() as conn:
        conn.execute(
            "UPDATE category_proposals SET status = ? WHERE id = ?;",
            (status, proposal_id),
        )

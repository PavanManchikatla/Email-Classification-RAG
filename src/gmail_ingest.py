"""
Gmail ingestion script with multi-account support.

Fetches emails from one or more Gmail accounts with pagination and
incremental sync, then stores them in the local SQLite database.

Usage:
    python -m src.gmail_ingest
"""

from __future__ import print_function

import base64
import logging
from datetime import datetime
from pathlib import Path

from bs4 import BeautifulSoup
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

import config
from src import db

logger = logging.getLogger(__name__)

# Read-only scope — no modification permissions
SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]


# ---------------------------------------------------------------------------
# Authentication
# ---------------------------------------------------------------------------


def get_gmail_service(token_path: Path) -> tuple:
    """
    Authenticate with Gmail and return (service, account_email).

    Uses the shared credentials.json (OAuth client) and a per-account
    token file. If no valid token exists, launches browser OAuth flow.
    """
    creds = None
    credentials_path = str(config.CREDENTIALS_PATH)

    if token_path.exists():
        creds = Credentials.from_authorized_user_file(str(token_path), SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                credentials_path, SCOPES
            )
            creds = flow.run_local_server(port=0)

        token_path.parent.mkdir(parents=True, exist_ok=True)
        with open(str(token_path), "w") as token_file:
            token_file.write(creds.to_json())

    service = build("gmail", "v1", credentials=creds)

    # Get the account email address from the profile
    profile = service.users().getProfile(userId="me").execute()
    account_email = profile.get("emailAddress", "unknown")

    return service, account_email


def _token_path_for_account(account_email: str) -> Path:
    """Derive a safe token filename from an email address."""
    safe_name = account_email.replace("@", "_").replace(".", "_")
    return config.TOKENS_DIR / f"token_{safe_name}.json"


def get_existing_account_tokens() -> list:
    """Discover existing token files in the tokens directory."""
    if not config.TOKENS_DIR.exists():
        return []
    return sorted(config.TOKENS_DIR.glob("token_*.json"))


def load_existing_accounts() -> list:
    """
    Non-interactive: load existing authenticated accounts from tokens/ directory.

    Unlike authenticate_accounts(), this never prompts for new accounts.
    Used by auto_evolve.py for headless/scheduled operation.

    Returns list of (service, account_email) tuples.
    """
    accounts = []

    existing_tokens = get_existing_account_tokens()
    for tp in existing_tokens:
        try:
            service, email = get_gmail_service(tp)
            accounts.append((service, email))
            logger.info("Loaded existing account: %s", email)
        except Exception as e:
            logger.warning("Failed to load token %s: %s", tp.name, e)

    # Also check legacy token.json
    if not accounts and config.TOKEN_PATH.exists():
        try:
            service, email = get_gmail_service(config.TOKEN_PATH)
            accounts.append((service, email))
            logger.info("Loaded legacy account: %s", email)
        except Exception as e:
            logger.warning("Failed to load legacy token.json: %s", e)

    logger.info("Loaded %d existing account(s)", len(accounts))
    return accounts


def authenticate_accounts() -> list:
    """
    Interactive flow to authenticate one or more Gmail accounts.

    Loads existing tokens first, then prompts to add more accounts.
    Each new account opens a browser for Google OAuth consent.

    Returns list of (service, account_email) tuples.
    """
    accounts = []

    # Load existing tokens
    existing_tokens = get_existing_account_tokens()
    if existing_tokens:
        print(f"\nFound {len(existing_tokens)} existing account(s):")
        for tp in existing_tokens:
            try:
                service, email = get_gmail_service(tp)
                accounts.append((service, email))
                print(f"  - {email}")
            except Exception as e:
                logger.warning("Failed to load token %s: %s", tp.name, e)

    # Also check legacy token.json for backward compatibility
    if config.TOKEN_PATH.exists() and not existing_tokens:
        try:
            service, email = get_gmail_service(config.TOKEN_PATH)
            # Move legacy token to new location
            new_path = _token_path_for_account(email)
            if not new_path.exists():
                config.TOKENS_DIR.mkdir(parents=True, exist_ok=True)
                config.TOKEN_PATH.rename(new_path)
                logger.info("Migrated legacy token.json to %s", new_path.name)
            accounts.append((service, email))
            print(f"\nMigrated existing account: {email}")
        except Exception as e:
            logger.warning("Failed to load legacy token.json: %s", e)

    # Prompt to add new accounts
    while True:
        if accounts:
            add_more = input(
                "\nDo you want to add another email account? (y/n): "
            ).strip().lower()
        else:
            add_more = input(
                "\nNo accounts found. Authenticate a Gmail account? (y/n): "
            ).strip().lower()

        if add_more != "y":
            break

        # Use a temporary token path; rename after we know the email
        config.TOKENS_DIR.mkdir(parents=True, exist_ok=True)
        temp_token = config.TOKENS_DIR / "token_new_account_temp.json"

        try:
            print("Opening browser for Google sign-in...")
            service, email = get_gmail_service(temp_token)

            # Check if already authenticated
            already_exists = any(e == email for _, e in accounts)
            if already_exists:
                print(f"  Account {email} is already connected. Skipping.")
                if temp_token.exists():
                    temp_token.unlink()
                continue

            # Rename token to account-specific name
            final_path = _token_path_for_account(email)
            if temp_token.exists():
                if final_path.exists():
                    final_path.unlink()
                temp_token.rename(final_path)

            accounts.append((service, email))
            print(f"  Successfully authenticated: {email}")

        except Exception as e:
            logger.error("Authentication failed: %s", e)
            if temp_token.exists():
                temp_token.unlink()

    return accounts


# ---------------------------------------------------------------------------
# Gmail payload helpers
# ---------------------------------------------------------------------------


def get_header(headers: list, name: str) -> str:
    """Extract a specific header value by name (case-insensitive)."""
    for h in headers:
        if h.get("name", "").lower() == name.lower():
            return h.get("value", "")
    return ""


def extract_body_from_payload(payload: dict) -> str:
    """
    Extract a reasonable text body from a Gmail API message payload.

    Preference:
      1. text/plain part
      2. text/html (converted to text via BeautifulSoup)
      3. fallback to empty string
    """

    def walk_parts(part):
        mime_type = part.get("mimeType", "")
        body = part.get("body", {})
        data = body.get("data")

        if data:
            decoded_bytes = base64.urlsafe_b64decode(data + "===")
            text = decoded_bytes.decode("utf-8", errors="ignore")

            if mime_type == "text/plain":
                return text
            elif mime_type == "text/html":
                soup = BeautifulSoup(text, "html.parser")
                return soup.get_text(separator="\n")

        for sub in part.get("parts", []) or []:
            result = walk_parts(sub)
            if result:
                return result
        return None

    text = walk_parts(payload)
    if text:
        return text.strip()

    # Fallback: payload itself has data without parts
    body = payload.get("body", {})
    data = body.get("data")
    if data:
        decoded_bytes = base64.urlsafe_b64decode(data + "===")
        text = decoded_bytes.decode("utf-8", errors="ignore")
        soup = BeautifulSoup(text, "html.parser")
        return soup.get_text(separator="\n").strip()

    return ""


# ---------------------------------------------------------------------------
# Main ingest logic
# ---------------------------------------------------------------------------


def _fetch_single_email(service, msg_id: str, account_email: str) -> dict:
    """Fetch a single email by ID and return a dict ready for DB insertion."""
    msg_detail = service.users().messages().get(
        userId="me",
        id=msg_id,
        format="full",
    ).execute()

    payload = msg_detail.get("payload", {})
    headers = payload.get("headers", [])
    label_ids = msg_detail.get("labelIds", [])

    return {
        "gmail_id": msg_id,
        "account_email": account_email,
        "thread_id": msg_detail.get("threadId", ""),
        "internal_date": int(msg_detail.get("internalDate", 0)),
        "from_addr": get_header(headers, "From"),
        "to_addr": get_header(headers, "To"),
        "subject": get_header(headers, "Subject") or "(no subject)",
        "snippet": msg_detail.get("snippet", ""),
        "body": extract_body_from_payload(payload),
        "label_ids": ",".join(label_ids),
    }


def fetch_and_store_emails(
    service,
    account_email: str,
    max_per_page: int | None = None,
    max_pages: int | None = None,
    fetch_all_labels: bool = False,
) -> int:
    """
    Fetch emails from a single Gmail account with pagination and incremental sync.

    Args:
        service: authenticated Gmail API service
        account_email: the email address of this account
        max_per_page: results per API page (default from config)
        max_pages: max pages to fetch (default from config)
        fetch_all_labels: if True, fetch from ALL folders (not just INBOX)

    Returns the count of newly stored emails.
    """
    max_per_page = max_per_page or config.GMAIL_MAX_RESULTS_PER_PAGE
    max_pages = max_pages or config.GMAIL_MAX_PAGES

    # Incremental sync: only fetch emails newer than what we have for this account
    query = None
    latest_date = db.get_latest_internal_date(account_email=account_email)
    if latest_date:
        date_str = datetime.fromtimestamp(latest_date / 1000).strftime("%Y/%m/%d")
        query = f"after:{date_str}"
        logger.info(
            "[%s] Incremental sync: fetching emails after %s",
            account_email, date_str,
        )
    else:
        logger.info("[%s] First run: fetching all available emails", account_email)

    new_count = 0
    total_fetched = 0
    page_token = None

    for page_num in range(max_pages):
        logger.info(
            "[%s] Fetching page %d (max %d per page)...",
            account_email, page_num + 1, max_per_page,
        )

        request_kwargs = {
            "userId": "me",
            "maxResults": max_per_page,
        }
        if not fetch_all_labels:
            request_kwargs["labelIds"] = ["INBOX"]
        if page_token:
            request_kwargs["pageToken"] = page_token
        if query:
            request_kwargs["q"] = query

        results = service.users().messages().list(**request_kwargs).execute()

        messages = results.get("messages", [])
        if not messages:
            logger.info("[%s] No more messages found.", account_email)
            break

        logger.info(
            "[%s] Found %d messages on page %d",
            account_email, len(messages), page_num + 1,
        )

        for i, msg_stub in enumerate(messages, start=1):
            try:
                email_row = _fetch_single_email(
                    service, msg_stub["id"], account_email
                )
                was_new = db.save_email(email_row)
                if was_new:
                    new_count += 1
                total_fetched += 1
                if total_fetched % 50 == 0:
                    logger.info(
                        "[%s] Progress: %d fetched, %d new",
                        account_email, total_fetched, new_count,
                    )
            except Exception as e:
                logger.warning(
                    "[%s] Failed to fetch message %s: %s",
                    account_email, msg_stub["id"], e,
                )
                continue

        page_token = results.get("nextPageToken")
        if not page_token:
            logger.info("[%s] No more pages available.", account_email)
            break

    logger.info(
        "[%s] Done: %d fetched, %d new emails stored",
        account_email, total_fetched, new_count,
    )
    return new_count


def list_recent_emails_preview(limit: int = 10):
    """Quick preview from DB to confirm things are stored."""
    with db.get_connection() as conn:
        rows = conn.execute(
            """
            SELECT id, account_email, from_addr, subject,
                   substr(body, 1, 100) as body_preview
            FROM emails
            ORDER BY internal_date DESC
            LIMIT ?;
            """,
            (limit,),
        ).fetchall()

    logger.info("Recent emails from DB:")
    for row in rows:
        logger.info(
            "[%d] (%s) %s | %s",
            row["id"], row["account_email"], row["from_addr"], row["subject"],
        )


def run_multi_account_ingest():
    """Main entry point for multi-account email ingestion."""
    db.init_db()

    accounts = authenticate_accounts()
    if not accounts:
        print("No accounts authenticated. Exiting.")
        return

    target = config.GMAIL_TARGET_TOTAL_EMAILS
    print(f"\nFetching emails from {len(accounts)} account(s)...")
    print(f"Target: {target:,} emails total\n")

    total_new = 0
    for service, email in accounts:
        print(f"--- Fetching from {email} ---")
        new_count = fetch_and_store_emails(
            service=service,
            account_email=email,
            max_per_page=config.GMAIL_MAX_RESULTS_PER_PAGE,
            max_pages=config.GMAIL_BULK_MAX_PAGES,
            fetch_all_labels=True,
        )
        total_new += new_count

        current_total = db.get_total_email_count()
        print(f"  New from {email}: {new_count:,}")
        print(f"  Total in DB: {current_total:,}\n")

        if current_total >= target:
            print(f"Reached target of {target:,} emails.")
            break

    # Print summary
    counts = db.get_account_email_counts()
    print("=== Ingestion Summary ===")
    for acct, cnt in counts.items():
        print(f"  {acct}: {cnt:,} emails")
    print(f"  Total: {sum(counts.values()):,} emails")


if __name__ == "__main__":
    run_multi_account_ingest()

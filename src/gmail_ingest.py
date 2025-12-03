from __future__ import print_function
import os
import os.path
import base64
import sqlite3
from typing import Optional, Dict, Any

from bs4 import BeautifulSoup

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

# Read-only scope
SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]

DB_PATH = os.path.join("data", "emails.db")


def get_gmail_service():
    """
    Returns an authenticated Gmail API service object.
    """
    creds = None

    if os.path.exists("token.json"):
        creds = Credentials.from_authorized_user_file("token.json", SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file("credentials.json", SCOPES)
            creds = flow.run_local_server(port=0)

        with open("token.json", "w") as token:
            token.write(creds.to_json())

    service = build("gmail", "v1", credentials=creds)
    return service


# ---------- DB helpers ----------

def init_db():
    """
    Create the SQLite database and emails + labels tables if they don't exist.
    """
    os.makedirs("data", exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    # Emails table (unchanged)
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS emails (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            gmail_id TEXT UNIQUE,
            thread_id TEXT,
            internal_date INTEGER,
            from_addr TEXT,
            to_addr TEXT,
            subject TEXT,
            snippet TEXT,
            body TEXT,
            label_ids TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """
    )

    # Labels table (new)
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

    conn.commit()
    conn.close()



def save_email_to_db(email_row: Dict[str, Any]):
    """
    Insert a single email row into the DB. Ignore if gmail_id already exists.
    """
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute(
        """
        INSERT OR IGNORE INTO emails
        (gmail_id, thread_id, internal_date, from_addr, to_addr,
         subject, snippet, body, label_ids)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
        """,
        (
            email_row["gmail_id"],
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

    conn.commit()
    conn.close()


# ---------- Gmail payload helpers ----------

def get_header(headers, name: str) -> str:
    for h in headers:
        if h.get("name", "").lower() == name.lower():
            return h.get("value", "")
    return ""


def extract_body_from_payload(payload: Dict[str, Any]) -> str:
    """
    Extract a reasonable text body from a Gmail API message payload.

    Preference:
      1. text/plain part
      2. text/html (converted to text)
      3. fallback to empty string
    """
    # Recursive search for text/plain or text/html
    def walk_parts(part) -> Optional[str]:
        mime_type = part.get("mimeType", "")
        body = part.get("body", {})
        data = body.get("data")

        # If this part has data
        if data:
            decoded_bytes = base64.urlsafe_b64decode(data + "===")
            text = decoded_bytes.decode("utf-8", errors="ignore")

            if mime_type == "text/plain":
                return text
            elif mime_type == "text/html":
                # Strip HTML
                soup = BeautifulSoup(text, "html.parser")
                return soup.get_text(separator="\n")

        # If multipart, check its parts
        for sub in part.get("parts", []) or []:
            result = walk_parts(sub)
            if result:
                return result
        return None

    # If the payload itself has parts, walk them
    text = walk_parts(payload)
    if text:
        return text.strip()

    # Fallback: maybe the payload itself has data without parts
    body = payload.get("body", {})
    data = body.get("data")
    if data:
        decoded_bytes = base64.urlsafe_b64decode(data + "===")
        text = decoded_bytes.decode("utf-8", errors="ignore")
        soup = BeautifulSoup(text, "html.parser")
        return soup.get_text(separator="\n").strip()

    return ""


# ---------- Main ingest logic ----------

def fetch_and_store_emails(max_results: int = 200):
    """
    Fetch recent emails from Gmail and store them in SQLite.
    """
    init_db()
    service = get_gmail_service()

    print(f"Fetching up to {max_results} messages from INBOX...")

    results = service.users().messages().list(
        userId="me",
        labelIds=["INBOX"],
        maxResults=max_results,
    ).execute()

    messages = results.get("messages", [])
    print(f"Found {len(messages)} messages in this page.")

    for i, msg in enumerate(messages, start=1):
        msg_id = msg["id"]

        # Get full message with headers and body
        msg_detail = service.users().messages().get(
            userId="me",
            id=msg_id,
            format="full",
        ).execute()

        payload = msg_detail.get("payload", {})
        headers = payload.get("headers", [])
        label_ids = msg_detail.get("labelIds", [])
        snippet = msg_detail.get("snippet", "")
        internal_date = int(msg_detail.get("internalDate", 0))
        thread_id = msg_detail.get("threadId", "")

        from_addr = get_header(headers, "From")
        to_addr = get_header(headers, "To")
        subject = get_header(headers, "Subject") or "(no subject)"

        body_text = extract_body_from_payload(payload)

        email_row = {
            "gmail_id": msg_id,
            "thread_id": thread_id,
            "internal_date": internal_date,
            "from_addr": from_addr,
            "to_addr": to_addr,
            "subject": subject,
            "snippet": snippet,
            "body": body_text,
            "label_ids": ",".join(label_ids),
        }

        save_email_to_db(email_row)

        print(f"[{i}/{len(messages)}] Saved: {subject[:80]!r}")


def list_recent_emails_preview(limit: int = 10):
    """
    Quick preview from DB to confirm things are stored.
    """
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        """
        SELECT id, from_addr, subject, substr(body, 1, 100)
        FROM emails
        ORDER BY internal_date DESC
        LIMIT ?;
        """,
        (limit,),
    )
    rows = cur.fetchall()
    conn.close()

    print("Recent emails from DB:")
    for row in rows:
        eid, from_addr, subject, body_preview = row
        print(f"[{eid}] {from_addr} | {subject}")
        print(f"    {body_preview!r}")
        print()


if __name__ == "__main__":
    # Step 1: fetch and store latest N emails
    fetch_and_store_emails(max_results=200)

    # Step 2: preview some rows from DB
    print("\n--- DB Preview ---")
    list_recent_emails_preview(limit=5)

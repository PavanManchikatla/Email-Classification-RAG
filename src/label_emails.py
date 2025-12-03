import sqlite3
import os
from textwrap import shorten

DB_PATH = os.path.join("data", "emails.db")

# Allowed labels for v1
LABELS = [
    "job_application",
    "job_process",
    "newsletter_marketing",
    "finance",
    "social",
    "personal",
    "other",
]

def get_unlabeled_emails(batch_size: int = 20):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute(
        """
        SELECT e.id, e.from_addr, e.subject, e.body
        FROM emails e
        LEFT JOIN email_labels l ON e.id = l.email_id
        WHERE l.email_id IS NULL
        ORDER BY e.internal_date DESC
        LIMIT ?;
        """,
        (batch_size,),
    )

    rows = cur.fetchall()
    conn.close()
    return rows


def label_email(email_id: int, label: str):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute(
        """
        INSERT OR REPLACE INTO email_labels (email_id, label)
        VALUES (?, ?);
        """,
        (email_id, label),
    )

    conn.commit()
    conn.close()


def main():
    print("Email labeling tool")
    print("-------------------")
    print("Labels:", ", ".join(LABELS))
    print("Type label name to assign, 's' to skip, 'q' to quit.\n")

    while True:
        rows = get_unlabeled_emails(batch_size=5)
        if not rows:
            print("No more unlabeled emails found. You're done! 🎉")
            break

        for (email_id, from_addr, subject, body) in rows:
            print(f"\nID: {email_id}")
            print(f"From: {from_addr}")
            print(f"Subject: {subject}")
            preview = shorten(body.replace("\n", " "), width=200, placeholder="...")
            print(f"Body: {preview}")

            while True:
                user_input = input(f"Label ({'/'.join(LABELS)} / s=skip / q=quit): ").strip()
                if user_input == "q":
                    print("Quitting.")
                    return
                elif user_input == "s":
                    print("Skipped.")
                    break
                elif user_input in LABELS:
                    label_email(email_id, user_input)
                    print(f"Labeled as: {user_input}")
                    break
                else:
                    print("Invalid input. Try again.")

        print("\nBatch done. Fetching next batch...\n")


if __name__ == "__main__":
    main()

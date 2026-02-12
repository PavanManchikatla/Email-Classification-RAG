"""
Interactive email labeling tool.

Displays unlabeled emails and prompts the user to classify them manually.
If an auto-label already exists, it is shown as a suggestion.

Usage:
    python -m src.label_emails
"""

from textwrap import shorten

import config
from src import db


def _print_label_menu():
    """Display available labels grouped by type for easier selection."""
    action = [
        "job_opportunity", "job_interview", "personal",
        "finance_alert", "security_auth", "events_calendar",
    ]
    info = [
        "job_application_confirm", "travel", "shopping_orders",
        "finance_receipt", "newsletter_content", "education",
    ]
    noise = [
        "social_notification", "marketing_promo", "account_service",
    ]

    print("Available labels:")
    print(f"  ACTION:  {', '.join(action)}")
    print(f"  INFO:    {', '.join(info)}")
    print(f"  NOISE:   {', '.join(noise)}")


def main():
    db.init_db()

    print("Email labeling tool")
    print("-------------------")
    _print_label_menu()
    print("\nType label name to assign, 's' to skip, 'q' to quit.\n")

    while True:
        rows = db.get_unlabeled_emails(batch_size=5)
        if not rows:
            print("No more unlabeled emails found. You're done!")
            break

        for row in rows:
            email_id = row["id"]
            print(f"\nID: {email_id}")
            print(f"From: {row['from_addr']}")
            print(f"Subject: {row['subject']}")
            body = row["body"] or ""
            preview = shorten(body.replace("\n", " "), width=200, placeholder="...")
            print(f"Body: {preview}")

            # Show existing auto-label as a hint if available
            full = db.get_email_with_label(email_id)
            if full and full["label"]:
                conf = full["confidence"] or 1.0
                print(
                    f"  -> Auto-suggested: {full['label']} "
                    f"({conf:.0%}, source: {full['source']})"
                )

            while True:
                user_input = input(
                    f"Label ({'/'.join(config.LABELS)} / s=skip / q=quit): "
                ).strip()
                if user_input == "q":
                    print("Quitting.")
                    return
                elif user_input == "s":
                    print("Skipped.")
                    break
                elif user_input in config.LABELS:
                    db.save_label(email_id, user_input, confidence=1.0, source="manual")
                    print(f"Labeled as: {user_input}")
                    break
                else:
                    print("Invalid input. Try again.")

        print("\nBatch done. Fetching next batch...\n")


if __name__ == "__main__":
    main()

"""
CLI output views for the email classification pipeline.

Provides three views:
  - summary:  label distribution with bar chart
  - priority: priority-sorted inbox for recent emails
  - daily:    today's email digest with action items

Usage:
    python -m src.digest summary
    python -m src.digest priority [--days 7]
    python -m src.digest daily
"""

import argparse
import logging
from collections import defaultdict
from datetime import datetime

import config
from src import db

logger = logging.getLogger(__name__)

# Use the single source of truth from config
PRIORITY_ORDER = config.PRIORITY_ORDER
ACTION_LABELS = config.ACTION_LABELS


def _print_label_bar(label: str, count: int, max_count: int, bar_width: int = 30):
    """Print a single label row with bar chart."""
    bar_len = int((count / max_count) * bar_width) if max_count > 0 else 0
    bar = "\u2588" * bar_len
    print(f"  {label:24s} {bar} {count}")


def print_summary():
    """Print a label distribution summary with a simple bar chart, grouped by type."""
    db.init_db()
    summary = db.get_label_summary()
    unlabeled = db.get_unlabeled_count()

    if not summary and unlabeled == 0:
        print("No emails in the database.")
        return

    total_labeled = sum(summary.values())
    max_count = max(summary.values()) if summary else 0

    # Define label groups
    action_labels = [
        "job_opportunity", "job_interview", "personal",
        "finance_alert", "security_auth", "events_calendar",
    ]
    info_labels = [
        "job_application_confirm", "travel", "shopping_orders",
        "finance_receipt", "newsletter_content", "education",
    ]
    noise_labels = [
        "social_notification", "marketing_promo", "account_service",
    ]

    print("=== Email Classification Summary ===\n")

    # Print grouped sections
    groups = [
        ("ACTION (needs response)", action_labels),
        ("INFORMATIONAL (read later)", info_labels),
        ("NOISE (batch/archive)", noise_labels),
    ]

    for group_name, labels in groups:
        group_count = sum(summary.get(l, 0) for l in labels)
        if group_count == 0:
            continue
        print(f"  --- {group_name} ({group_count}) ---")
        for label in labels:
            count = summary.get(label, 0)
            if count == 0:
                continue
            _print_label_bar(label, count, max_count)
        print()

    # Show any labels not in the known groups (future-proofing)
    known = set(action_labels + info_labels + noise_labels)
    unknown = {l: c for l, c in summary.items() if l not in known}
    if unknown:
        print("  --- OTHER ---")
        for label, count in sorted(unknown.items(), key=lambda x: -x[1]):
            _print_label_bar(label, count, max_count)
        print()

    print(f"  Total labeled:   {total_labeled}")
    print(f"  Unlabeled:       {unlabeled}")
    print(f"  Total emails:    {total_labeled + unlabeled}")


def print_priority_inbox(days: int = 7):
    """
    Print a priority-sorted view of recent emails.

    Emails are grouped by label in priority order. Low-confidence
    auto-labels are flagged with [?] to invite manual review.
    """
    db.init_db()
    emails = db.get_recent_emails(days=days)

    if not emails:
        print(f"No emails from the last {days} days.")
        return

    print(f"=== Priority Inbox (last {days} days) ===\n")

    # Group by label
    grouped = defaultdict(list)
    for email in emails:
        label = email["label"] or "unlabeled"
        grouped[label].append(email)

    # Print in priority order
    display_order = PRIORITY_ORDER + ["unlabeled"]
    for label in display_order:
        if label not in grouped:
            continue

        emails_in_group = grouped[label]
        print(f"\n--- {label.upper().replace('_', ' ')} ({len(emails_in_group)}) ---")

        for e in emails_in_group[:15]:
            date_ms = e["internal_date"]
            if date_ms:
                date_str = datetime.fromtimestamp(date_ms / 1000).strftime("%m/%d %H:%M")
            else:
                date_str = "??/??"

            confidence = e["confidence"] or 1.0
            flag = " [?]" if confidence < 0.7 else ""

            from_addr = (e["from_addr"] or "")[:30]
            subject = (e["subject"] or "")[:50]

            print(f"  {date_str} | {from_addr:30s} | {subject}{flag}")

    # Show remaining labels not in priority order
    for label in sorted(grouped.keys()):
        if label in display_order:
            continue
        emails_in_group = grouped[label]
        print(f"\n--- {label.upper()} ({len(emails_in_group)}) ---")
        for e in emails_in_group[:10]:
            date_ms = e["internal_date"]
            date_str = datetime.fromtimestamp(date_ms / 1000).strftime("%m/%d %H:%M") if date_ms else "??/??"
            print(f"  {date_str} | {(e['from_addr'] or '')[:30]:30s} | {(e['subject'] or '')[:50]}")


def print_daily_digest():
    """
    Print a digest of today's emails grouped by category.
    Highlights action items (personal, job-related, finance).
    """
    db.init_db()
    emails = db.get_recent_emails(days=1)

    if not emails:
        print("No emails from today.")
        return

    print(f"=== Daily Digest: {datetime.now().strftime('%Y-%m-%d')} ===")
    print(f"Total emails: {len(emails)}\n")

    # Count by label
    counts = defaultdict(int)
    for e in emails:
        counts[e["label"] or "unlabeled"] += 1

    print("By category:")
    for label, count in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"  {label}: {count}")

    # Show action items
    action_emails = [e for e in emails if (e["label"] or "") in ACTION_LABELS]

    if action_emails:
        print(f"\n--- Action Items ({len(action_emails)}) ---")
        for e in action_emails:
            from_addr = (e["from_addr"] or "")[:30]
            subject = (e["subject"] or "")[:50]
            print(f"  [{e['label']}] {from_addr} - {subject}")
    else:
        print("\nNo action items today.")

    # Flag low-confidence labels for review
    low_confidence = [e for e in emails if (e["confidence"] or 1.0) < 0.7]
    if low_confidence:
        print(f"\n--- Needs Review ({len(low_confidence)}) ---")
        for e in low_confidence:
            print(
                f"  [{e['label'] or '?'}] {(e['from_addr'] or '')[:30]} "
                f"- {(e['subject'] or '')[:50]}"
            )


def main():
    parser = argparse.ArgumentParser(description="Email classification digest")
    parser.add_argument(
        "view",
        nargs="?",
        default="summary",
        choices=["summary", "priority", "daily"],
        help="Which view to display (default: summary)",
    )
    parser.add_argument(
        "--days", type=int, default=7,
        help="Days to look back for priority view (default: 7)",
    )
    args = parser.parse_args()

    if args.view == "summary":
        print_summary()
    elif args.view == "priority":
        print_priority_inbox(days=args.days)
    elif args.view == "daily":
        print_daily_digest()


if __name__ == "__main__":
    main()

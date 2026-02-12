"""
LLM-based label generation using Claude Haiku.

Classifies unlabeled emails in batches using 15 categories and saves
the results to the database with source='llm'. This bootstraps training
data for the ML classifier.

Usage:
    python -m src.generate_labels                    # label all unlabeled emails
    python -m src.generate_labels --dry-run           # preview without saving
    python -m src.generate_labels --clear-existing    # clear old labels first, then re-label
    python -m src.generate_labels --batch-size 5      # custom batch size
"""

import argparse
import json
import logging

import config
from src import db

logger = logging.getLogger(__name__)


def _build_system_prompt() -> str:
    """Build the classification prompt dynamically from config.LABELS."""
    categories_section = "\n".join(
        f"- {label}: {config.LABEL_DESCRIPTIONS[label]}"
        for label in config.LABELS
    )

    return f"""You are an email classifier for a personal inbox.
Classify each email into exactly one category.

Categories:
{categories_section}

DISAMBIGUATION RULES (use these when a classification is ambiguous):
- LinkedIn "new job match" or recruiter InMail -> job_opportunity (NOT social_notification)
- LinkedIn "X viewed your profile" or likes/comments -> social_notification
- "Thank you for applying" / application received -> job_application_confirm (NOT job_interview)
- Interview scheduling, offers, rejections -> job_interview
- Bank fraud alert -> finance_alert (NOT security_auth)
- "Your password was changed" or new sign-in alert -> security_auth
- Amazon/store order confirmation -> shopping_orders (NOT finance_receipt)
- Stripe/PayPal payment receipt -> finance_receipt
- Coursera "assignment due" -> education; Coursera "50% off" -> marketing_promo
- Company blog newsletter user subscribed to -> newsletter_content
- Company "sale" or "discount" email -> marketing_promo
- Eventbrite invitation -> events_calendar (NOT marketing_promo)

Respond with ONLY a JSON array. Each element must have these fields:
{{"id": <email_id>, "label": "<category>", "confidence": <float 0.0 to 1.0>}}

Example response:
[{{"id": 1, "label": "marketing_promo", "confidence": 0.95}}, {{"id": 2, "label": "personal", "confidence": 0.8}}]

Return ONLY the JSON array, no other text."""


SYSTEM_PROMPT = _build_system_prompt()


def _classify_batch_anthropic(emails: list[dict]) -> list[dict]:
    """Send a batch of emails to Anthropic API for classification."""
    try:
        import anthropic
    except ImportError:
        logger.error(
            "anthropic package not installed. Run: pip install anthropic"
        )
        return []

    if not config.ANTHROPIC_API_KEY:
        logger.error(
            "ANTHROPIC_API_KEY not set. Add it to your .env file."
        )
        return []

    # Build the user message with truncated bodies to control cost
    email_descriptions = []
    for e in emails:
        body_preview = (e["body"] or "")[:500]
        email_descriptions.append(
            f"Email ID: {e['id']}\n"
            f"From: {e['from_addr']}\n"
            f"Subject: {e['subject']}\n"
            f"Body preview: {body_preview}"
        )

    user_message = "Classify these emails:\n\n" + "\n\n---\n\n".join(
        email_descriptions
    )

    client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)

    response = client.messages.create(
        model=config.LLM_MODEL,
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    )

    return _parse_llm_response(response.content[0].text)


def _parse_llm_response(raw: str) -> list[dict]:
    """Parse the JSON response from the LLM, validate labels."""
    text = raw.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(
            line for line in lines
            if not line.strip().startswith("```")
        )

    try:
        results = json.loads(text)
        if isinstance(results, dict):
            for key in ("classifications", "results", "emails"):
                if key in results:
                    results = results[key]
                    break
    except json.JSONDecodeError:
        logger.error("Failed to parse LLM response: %s", raw[:300])
        return []

    if not isinstance(results, list):
        logger.error("Expected a JSON array, got: %s", type(results).__name__)
        return []

    validated = []
    for r in results:
        if not isinstance(r, dict) or "id" not in r:
            continue
        label = r.get("label", "account_service")
        if label not in config.LABELS:
            logger.warning(
                "LLM returned unknown label '%s' for email %s, defaulting to 'account_service'",
                label, r.get("id"),
            )
            label = "account_service"
        confidence = min(max(float(r.get("confidence", 0.5)), 0.0), 1.0)
        validated.append({
            "id": r["id"],
            "label": label,
            "confidence": confidence,
        })

    return validated


def generate_labels(
    batch_size: int | None = None,
    dry_run: bool = False,
    clear_existing: bool = False,
) -> int:
    """
    Fetch unlabeled emails, classify them with Claude Haiku, and save results.

    Args:
        batch_size: emails per LLM call (default from config)
        dry_run: if True, print classifications but don't save
        clear_existing: if True, delete all existing labels before re-labeling

    Returns:
        total number of emails classified
    """
    batch_size = batch_size or config.CLASSIFY_BATCH_SIZE
    db.init_db()

    if clear_existing and not dry_run:
        print("Clearing existing labels for re-classification...")
        db.clear_labels()

    total_classified = 0

    while True:
        unlabeled = db.get_unlabeled_emails_full(batch_size=batch_size)
        if not unlabeled:
            if total_classified == 0:
                logger.info("No unlabeled emails to classify.")
            break

        logger.info("Classifying batch of %d emails...", len(unlabeled))

        email_dicts = [
            {
                "id": row["id"],
                "from_addr": row["from_addr"],
                "subject": row["subject"],
                "body": row["body"],
            }
            for row in unlabeled
        ]

        results = _classify_batch_anthropic(email_dicts)

        if not results:
            logger.warning("No results from LLM for this batch. Stopping.")
            break

        for r in results:
            if dry_run:
                logger.info(
                    "  [DRY RUN] Email %d -> %s (%.0f%%)",
                    r["id"], r["label"], r["confidence"] * 100,
                )
            else:
                db.save_label(
                    r["id"], r["label"],
                    confidence=r["confidence"], source="llm",
                )
                total_classified += 1

        if dry_run:
            logger.info("Dry run complete. No labels saved.")
            return len(results)

        logger.info(
            "Batch done. Total classified so far: %d", total_classified
        )

    logger.info("Label generation complete: %d emails classified", total_classified)
    return total_classified


def main():
    parser = argparse.ArgumentParser(
        description="Generate email labels using Claude Haiku"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Preview classifications without saving",
    )
    parser.add_argument(
        "--clear-existing", action="store_true",
        help="Clear all existing labels before re-labeling (for taxonomy changes)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=None,
        help=f"Emails per LLM call (default: {config.CLASSIFY_BATCH_SIZE})",
    )
    args = parser.parse_args()

    count = generate_labels(
        batch_size=args.batch_size,
        dry_run=args.dry_run,
        clear_existing=args.clear_existing,
    )
    action = "Previewed" if args.dry_run else "Classified"
    print(f"\nDone. {action} {count} emails.")


if __name__ == "__main__":
    main()

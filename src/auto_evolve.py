"""
Auto-evolve orchestrator for the email classification pipeline.

Periodically ingests new emails, classifies them, detects uncertainty,
clusters uncertain emails for new category discovery, and retrains
the model with all accumulated labeled data.

Usage:
    python -m src.auto_evolve --once         # Single cycle
    python -m src.auto_evolve --schedule 6   # Run every 6 hours
    python -m src.auto_evolve --discover     # Only run category discovery
    python -m src.auto_evolve --review       # Review pending proposals
"""

import argparse
import logging
import time
from datetime import datetime

import config
from src import db

logger = logging.getLogger(__name__)


def run_evolution_cycle() -> dict:
    """
    Run one complete evolution cycle:

    1. Load existing Gmail accounts (non-interactive)
    2. Fetch new emails from all accounts
    3. Classify new emails with current model + flag uncertain ones
    4. If enough uncertain emails, cluster and propose new categories
    5. If enough new labels, retrain model with versioning
    6. Compare accuracy to previous version

    Returns a summary dict.
    """
    db.init_db()

    summary = {
        "timestamp": datetime.now().isoformat(),
        "new_emails": 0,
        "classified": 0,
        "uncertain": 0,
        "proposals": 0,
        "retrained": False,
        "accuracy": None,
        "previous_accuracy": None,
    }

    # --- Step 1: Ingest new emails ---
    try:
        from src.gmail_ingest import load_existing_accounts, fetch_and_store_emails

        accounts = load_existing_accounts()
        if not accounts:
            logger.warning("No authenticated accounts found. Skipping ingestion.")
        else:
            logger.info("Fetching new emails from %d account(s)...", len(accounts))
            for service, email in accounts:
                new_count = fetch_and_store_emails(
                    service=service,
                    account_email=email,
                    max_per_page=config.GMAIL_MAX_RESULTS_PER_PAGE,
                    max_pages=config.GMAIL_MAX_PAGES,  # Use normal pages, not bulk
                )
                summary["new_emails"] += new_count
                logger.info("[%s] Fetched %d new emails", email, new_count)
    except Exception as e:
        logger.error("Email ingestion failed: %s", e)

    # --- Step 2: Classify new emails + flag uncertain ---
    try:
        from src.classify import classify_and_flag

        num_classified, uncertain_ids = classify_and_flag()
        summary["classified"] = num_classified
        summary["uncertain"] = len(uncertain_ids)
        logger.info("Classified %d emails, %d uncertain", num_classified, len(uncertain_ids))
    except Exception as e:
        logger.error("Classification failed: %s", e)
        uncertain_ids = []

    # --- Step 3: Discover new categories (if enough uncertain emails) ---
    min_for_discovery = config.EVOLVE_MIN_CLUSTER_SIZE * 2  # Need at least 2x cluster size
    if len(uncertain_ids) >= min_for_discovery:
        try:
            from src.discover_categories import cluster_uncertain_emails, propose_category_names

            clusters = cluster_uncertain_emails(uncertain_ids)
            if clusters:
                proposals = propose_category_names(clusters)
                summary["proposals"] = len(proposals)
                logger.info("Proposed %d new categories", len(proposals))
        except Exception as e:
            logger.error("Category discovery failed: %s", e)
    else:
        logger.info(
            "Only %d uncertain emails (need %d). Skipping discovery.",
            len(uncertain_ids), min_for_discovery,
        )

    # --- Step 4: Retrain if enough new labels ---
    previous_version = db.get_latest_model_version()
    previous_samples = previous_version["num_samples"] if previous_version else 0
    current_labeled = db.get_labeled_count()
    new_labels_since_last = current_labeled - previous_samples

    if new_labels_since_last >= config.EVOLVE_MIN_NEW_LABELS_FOR_RETRAIN:
        try:
            from src.train_model import retrain_and_version

            logger.info(
                "Retraining: %d new labels since last version (threshold: %d)",
                new_labels_since_last, config.EVOLVE_MIN_NEW_LABELS_FOR_RETRAIN,
            )
            version, metrics = retrain_and_version(trigger="auto")
            if version:
                summary["retrained"] = True
                summary["accuracy"] = metrics.get("accuracy")

                # Compare with previous version
                if previous_version:
                    summary["previous_accuracy"] = previous_version["accuracy"]
                    diff = (summary["accuracy"] or 0) - (previous_version["accuracy"] or 0)
                    if diff < -0.05:
                        logger.warning(
                            "Accuracy DROPPED by %.1f%% (%.3f -> %.3f). "
                            "Check training data for issues.",
                            abs(diff) * 100,
                            previous_version["accuracy"],
                            summary["accuracy"],
                        )
                    else:
                        logger.info(
                            "Model %s: accuracy %.3f (change: %+.1f%%)",
                            version, summary["accuracy"], diff * 100,
                        )
        except Exception as e:
            logger.error("Retraining failed: %s", e)
    else:
        logger.info(
            "Not enough new labels for retrain (%d new, need %d).",
            new_labels_since_last, config.EVOLVE_MIN_NEW_LABELS_FOR_RETRAIN,
        )

    return summary


def print_summary(summary: dict):
    """Print a human-readable summary of the evolution cycle."""
    print(f"\n=== Evolution Cycle Summary ({summary['timestamp']}) ===")
    print(f"  New emails ingested:   {summary['new_emails']}")
    print(f"  Emails classified:     {summary['classified']}")
    print(f"  Uncertain predictions: {summary['uncertain']}")
    print(f"  Category proposals:    {summary['proposals']}")
    print(f"  Retrained:             {'Yes' if summary['retrained'] else 'No'}")

    if summary["retrained"]:
        acc = summary["accuracy"]
        prev = summary["previous_accuracy"]
        if acc is not None:
            print(f"  New accuracy:          {acc:.3f}")
        if prev is not None and acc is not None:
            diff = acc - prev
            print(f"  Previous accuracy:     {prev:.3f} ({'+' if diff >= 0 else ''}{diff:.3f})")

    # Check for pending proposals
    pending = db.get_pending_proposals()
    if pending:
        print(f"\n  {len(pending)} pending category proposal(s). "
              f"Run 'python -m src.auto_evolve --review' to review.")


def run_scheduled(interval_hours: int):
    """Run evolution cycles on a schedule."""
    interval_seconds = interval_hours * 3600

    print(f"Starting auto-evolve scheduler (every {interval_hours} hours)")
    print("Press Ctrl+C to stop.\n")

    cycle_num = 0
    while True:
        cycle_num += 1
        logger.info("=== Starting evolution cycle %d ===", cycle_num)

        try:
            summary = run_evolution_cycle()
            print_summary(summary)
        except Exception as e:
            logger.error("Evolution cycle %d failed: %s", cycle_num, e)

        next_run = datetime.now().timestamp() + interval_seconds
        next_run_str = datetime.fromtimestamp(next_run).strftime("%H:%M:%S")
        print(f"\nNext cycle at {next_run_str}. Sleeping {interval_hours}h...")

        try:
            time.sleep(interval_seconds)
        except KeyboardInterrupt:
            print("\nScheduler stopped by user.")
            break


def main():
    parser = argparse.ArgumentParser(
        description="Auto-evolve email classification model"
    )
    parser.add_argument(
        "--once", action="store_true",
        help="Run a single evolution cycle",
    )
    parser.add_argument(
        "--schedule", type=int, default=None,
        metavar="HOURS",
        help="Run evolution cycles every N hours",
    )
    parser.add_argument(
        "--discover", action="store_true",
        help="Only run category discovery (cluster + propose)",
    )
    parser.add_argument(
        "--review", action="store_true",
        help="Review pending category proposals",
    )
    args = parser.parse_args()

    if args.review:
        from src.discover_categories import review_proposals_cli
        review_proposals_cli()
        return

    if args.discover:
        from src.discover_categories import main as discover_main
        discover_main()
        return

    if args.schedule:
        run_scheduled(args.schedule)
    elif args.once:
        summary = run_evolution_cycle()
        print_summary(summary)
    else:
        # Default: run once
        summary = run_evolution_cycle()
        print_summary(summary)


if __name__ == "__main__":
    main()

"""
Classify unlabeled emails using the trained ML model.

Loads the TF-IDF + Random Forest model from disk, predicts labels
for unlabeled emails, and saves them with source='model'.

Usage:
    python -m src.classify              # classify all unlabeled emails
    python -m src.classify --dry-run    # preview without saving
"""

import argparse
import logging

import joblib
import numpy as np

import config
from src import db

logger = logging.getLogger(__name__)

MODEL_PATH = config.MODEL_DIR / "email_classifier.joblib"

# Uncertainty thresholds (from config, used by classify_and_flag)


def load_model():
    """Load the trained model from disk."""
    if not MODEL_PATH.exists():
        logger.error(
            "No trained model found at %s. Run 'python -m src.train_model' first.",
            MODEL_PATH,
        )
        return None

    model = joblib.load(str(MODEL_PATH))
    logger.info("Loaded model from %s", MODEL_PATH)
    return model


def compute_uncertainty(proba_row) -> dict:
    """
    Compute uncertainty metrics for a single prediction's probability vector.

    Returns:
        entropy: how spread out the prediction is (-sum(p * log(p)))
        margin: gap between top-2 probabilities (higher = more certain)
        max_prob: highest probability (the confidence score)
    """
    sorted_probs = sorted(proba_row, reverse=True)
    entropy = -sum(p * np.log(p + 1e-10) for p in proba_row)
    margin = sorted_probs[0] - sorted_probs[1] if len(sorted_probs) > 1 else 1.0
    return {
        "entropy": float(entropy),
        "margin": float(margin),
        "max_prob": float(sorted_probs[0]),
    }


def classify_emails(model, emails: list) -> list[dict]:
    """
    Classify emails using the trained model.

    Returns list of dicts with label, confidence, and uncertainty metrics.
    """
    texts = []
    ids = []
    for row in emails:
        body = (row["body"] or "")[:500]
        text = f"{row['from_addr']} {row['subject']} {body}"
        texts.append(text)
        ids.append(row["id"])

    predictions = model.predict(texts)
    probabilities = model.predict_proba(texts)

    results = []
    for i, (email_id, label) in enumerate(zip(ids, predictions)):
        confidence = float(np.max(probabilities[i]))
        uncertainty = compute_uncertainty(probabilities[i])
        results.append({
            "id": email_id,
            "label": label,
            "confidence": confidence,
            "uncertainty": uncertainty,
        })

    return results


def classify_unlabeled(dry_run: bool = False, batch_size: int = 100) -> int:
    """
    Main entry point: load model, fetch unlabeled emails, classify, save.

    Returns the number of emails classified.
    """
    db.init_db()
    model = load_model()
    if model is None:
        return 0

    total_classified = 0

    while True:
        unlabeled = db.get_unlabeled_emails_full(batch_size=batch_size)
        if not unlabeled:
            if total_classified == 0:
                logger.info("No unlabeled emails to classify.")
            break

        logger.info("Classifying batch of %d emails...", len(unlabeled))
        results = classify_emails(model, unlabeled)

        for r in results:
            if dry_run:
                logger.info(
                    "  [DRY RUN] Email %d -> %s (%.0f%%)",
                    r["id"], r["label"], r["confidence"] * 100,
                )
            else:
                db.save_label(
                    r["id"], r["label"],
                    confidence=r["confidence"], source="model",
                )
                total_classified += 1

        if dry_run:
            logger.info("Dry run complete. No labels saved.")
            return len(results)

        logger.info("Batch done. Total classified: %d", total_classified)

    logger.info("Classification complete: %d emails classified", total_classified)
    return total_classified


def classify_and_flag(batch_size: int = 100) -> tuple[int, list[int]]:
    """
    Classify unlabeled emails AND flag uncertain predictions.

    Like classify_unlabeled() but also collects email IDs where the model
    is uncertain (low margin or low max probability). These uncertain
    emails are candidates for clustering and new category discovery.

    Returns (num_classified, uncertain_email_ids).
    """
    db.init_db()
    model = load_model()
    if model is None:
        return 0, []

    total_classified = 0
    uncertain_ids = []

    margin_threshold = config.EVOLVE_UNCERTAINTY_MARGIN
    confidence_threshold = config.EVOLVE_CONFIDENCE_THRESHOLD

    while True:
        unlabeled = db.get_unlabeled_emails_full(batch_size=batch_size)
        if not unlabeled:
            break

        logger.info("Classifying batch of %d emails...", len(unlabeled))
        results = classify_emails(model, unlabeled)

        for r in results:
            db.save_label(
                r["id"], r["label"],
                confidence=r["confidence"], source="model",
            )
            total_classified += 1

            # Flag uncertain predictions
            u = r["uncertainty"]
            if u["margin"] < margin_threshold or u["max_prob"] < confidence_threshold:
                uncertain_ids.append(r["id"])

        logger.info(
            "Batch done. Total classified: %d, uncertain: %d",
            total_classified, len(uncertain_ids),
        )

    logger.info(
        "Classification complete: %d classified, %d uncertain",
        total_classified, len(uncertain_ids),
    )
    return total_classified, uncertain_ids


def main():
    parser = argparse.ArgumentParser(
        description="Classify emails using trained ML model"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Preview classifications without saving",
    )
    args = parser.parse_args()

    count = classify_unlabeled(dry_run=args.dry_run)
    print(f"\nDone. {'Previewed' if args.dry_run else 'Classified'} {count} emails.")


if __name__ == "__main__":
    main()

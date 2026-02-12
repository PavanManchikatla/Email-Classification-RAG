"""
Train a TF-IDF + Random Forest classifier on labeled email data.

Fetches labeled emails from the database, trains a scikit-learn pipeline,
evaluates with a train/test split, and saves the model to disk.

Usage:
    python -m src.train_model
"""

import argparse
import json
import logging
import os
from datetime import datetime

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

import config
from src import db

logger = logging.getLogger(__name__)

MODEL_PATH = config.MODEL_DIR / "email_classifier.joblib"


def load_training_data() -> tuple[list[str], list[str]]:
    """
    Load labeled emails from the database and return (texts, labels).

    Each text is a combination of from_addr, subject, and body preview
    to give the model sender + content signal.
    """
    rows = db.get_labeled_emails()

    if not rows:
        logger.error("No labeled emails found in the database.")
        return [], []

    texts = []
    labels = []
    for row in rows:
        # Combine sender, subject, and body (truncated) as features
        body = (row["body"] or "")[:500]
        text = f"{row['from_addr']} {row['subject']} {body}"
        texts.append(text)
        labels.append(row["label"])

    logger.info(
        "Loaded %d labeled emails across %d categories",
        len(texts), len(set(labels)),
    )
    return texts, labels


def build_pipeline() -> Pipeline:
    """Create the TF-IDF + Random Forest pipeline."""
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            stop_words="english",
            sublinear_tf=True,
        )),
        ("clf", RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1,
        )),
    ])


def train_and_evaluate(
    texts: list[str],
    labels: list[str],
    test_size: float = 0.2,
) -> tuple[Pipeline, dict]:
    """
    Train the model and return it along with evaluation metrics.

    If the dataset is too small for a meaningful split (< 10 samples),
    trains on the full dataset without evaluation.
    """
    pipeline = build_pipeline()

    if len(texts) < 10:
        logger.warning(
            "Only %d samples — training on full dataset without evaluation.",
            len(texts),
        )
        pipeline.fit(texts, labels)
        return pipeline, {"note": "too few samples for evaluation"}

    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=test_size, random_state=42, stratify=labels,
    )

    logger.info("Training on %d samples, testing on %d...", len(X_train), len(X_test))
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

    # Print human-readable report
    print("\n=== Classification Report ===")
    print(classification_report(y_test, y_pred, zero_division=0))

    return pipeline, report


def save_model(pipeline: Pipeline, report: dict = None, trigger: str = "manual"):
    """
    Save the trained model to disk with versioning.

    - Saves a versioned copy: email_classifier_v{N}.joblib
    - Also overwrites the 'latest' file so classify.py keeps working
    - Records metrics to model_versions DB table
    """
    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Generate version string
    version_num = db.get_model_version_count() + 1
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    version = f"v{version_num}_{timestamp}"

    # Save versioned file
    versioned_path = config.MODEL_DIR / f"email_classifier_{version}.joblib"
    joblib.dump(pipeline, str(versioned_path))
    logger.info("Versioned model saved to %s", versioned_path)

    # Also overwrite the 'latest' file (backward compat with classify.py)
    joblib.dump(pipeline, str(MODEL_PATH))
    logger.info("Latest model saved to %s", MODEL_PATH)

    # Extract metrics from report
    accuracy = 0.0
    macro_f1 = 0.0
    num_samples = 0
    num_categories = 0

    if report and "accuracy" in report:
        accuracy = report.get("accuracy", 0.0)
        macro_avg = report.get("macro avg", {})
        macro_f1 = macro_avg.get("f1-score", 0.0) if isinstance(macro_avg, dict) else 0.0
        # Count categories (exclude 'accuracy', 'macro avg', 'weighted avg')
        meta_keys = {"accuracy", "macro avg", "weighted avg"}
        num_categories = len([k for k in report if k not in meta_keys])
        # Get total support from weighted avg
        weighted_avg = report.get("weighted avg", {})
        num_samples = int(weighted_avg.get("support", 0)) if isinstance(weighted_avg, dict) else 0

    # Record to DB
    db.save_model_version(
        version=version,
        model_path=str(versioned_path),
        num_samples=num_samples,
        num_categories=num_categories,
        accuracy=accuracy,
        macro_f1=macro_f1,
        report_json=json.dumps(report) if report else "{}",
        trigger=trigger,
    )

    return version


def retrain_and_version(trigger: str = "auto") -> tuple:
    """
    Full retrain entry point for auto-evolve.

    Returns (version_string, metrics_dict) or (None, None) if no data.
    """
    db.init_db()

    texts, labels = load_training_data()
    if not texts:
        logger.warning("No training data for retrain.")
        return None, None

    pipeline, report = train_and_evaluate(texts, labels)
    version = save_model(pipeline, report=report, trigger=trigger)

    # Extract key metrics
    accuracy = report.get("accuracy", 0.0) if isinstance(report, dict) else 0.0
    macro_f1 = 0.0
    if isinstance(report, dict):
        macro_avg = report.get("macro avg", {})
        macro_f1 = macro_avg.get("f1-score", 0.0) if isinstance(macro_avg, dict) else 0.0

    metrics = {"accuracy": accuracy, "macro_f1": macro_f1, "num_samples": len(texts)}
    logger.info("Retrained model %s: accuracy=%.3f, macro_f1=%.3f", version, accuracy, macro_f1)
    return version, metrics


def main():
    parser = argparse.ArgumentParser(
        description="Train email classification model"
    )
    parser.add_argument(
        "--test-size", type=float, default=0.2,
        help="Fraction of data to use for testing (default: 0.2)",
    )
    args = parser.parse_args()

    db.init_db()

    texts, labels = load_training_data()
    if not texts:
        print("No training data available. Run generate_labels first.")
        return

    label_counts = {}
    for label in labels:
        label_counts[label] = label_counts.get(label, 0) + 1

    print(f"\nTraining data: {len(texts)} emails")
    print("Label distribution:")
    for label, count in sorted(label_counts.items(), key=lambda x: -x[1]):
        print(f"  {label}: {count}")

    pipeline, metrics = train_and_evaluate(texts, labels, test_size=args.test_size)
    version = save_model(pipeline, report=metrics, trigger="manual")

    print(f"\nModel {version} saved to {MODEL_PATH}")
    print("Run 'python -m src.classify' to classify new emails.")


if __name__ == "__main__":
    main()

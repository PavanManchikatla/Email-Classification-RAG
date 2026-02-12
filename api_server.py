"""
Local Flask API server for the email classification Chrome extension.

Serves classification results from the SQLite database to the
Chrome extension running in Gmail.

Usage:
    python api_server.py
    python api_server.py --port 5544

Endpoints:
    GET  /api/health   - Server status + model info
    POST /api/classify - Batch lookup by Gmail IDs
    GET  /api/labels   - Full label taxonomy
    GET  /api/summary  - Label distribution counts
"""

import argparse
import logging

from flask import Flask, request, jsonify
from flask_cors import CORS

import config
from src import db

logger = logging.getLogger(__name__)

app = Flask(__name__)

# Allow requests from Chrome extensions only
CORS(app, origins=["chrome-extension://*", "http://localhost:*"])

# Label group mapping
ACTION_LABELS = {
    "job_opportunity", "job_interview", "personal",
    "finance_alert", "security_auth", "events_calendar",
}
INFO_LABELS = {
    "job_application_confirm", "travel", "shopping_orders",
    "finance_receipt", "newsletter_content", "education",
}
NOISE_LABELS = {
    "social_notification", "marketing_promo", "account_service",
}


def _label_to_group(label: str) -> str:
    """Map a label to its group: ACTION, INFO, or NOISE."""
    if label in ACTION_LABELS:
        return "ACTION"
    elif label in INFO_LABELS:
        return "INFO"
    elif label in NOISE_LABELS:
        return "NOISE"
    return "OTHER"


@app.before_request
def ensure_db():
    """Initialize DB on first request."""
    if not hasattr(app, "_db_initialized"):
        db.init_db()
        app._db_initialized = True


@app.route("/api/health", methods=["GET"])
def health():
    """Server status with model info and email counts."""
    model_version = db.get_latest_model_version()
    total = db.get_total_email_count()
    labeled = db.get_labeled_count()

    return jsonify({
        "status": "ok",
        "model_version": model_version["version"] if model_version else None,
        "model_accuracy": model_version["accuracy"] if model_version else None,
        "total_emails": total,
        "total_labeled": labeled,
        "unlabeled": total - labeled,
    })


@app.route("/api/classify", methods=["POST"])
def classify():
    """
    Batch classification lookup by Gmail IDs.

    Request body: {"gmail_ids": ["id1", "id2", ...]}
    Response: {"classifications": {"id1": {"label": ..., "confidence": ..., "group": ...}, ...}}
    """
    data = request.get_json(silent=True)
    if not data or "gmail_ids" not in data:
        return jsonify({"error": "Request body must contain 'gmail_ids' array"}), 400

    gmail_ids = data["gmail_ids"]
    if not isinstance(gmail_ids, list):
        return jsonify({"error": "'gmail_ids' must be an array"}), 400

    # Cap at 200 IDs per request to prevent abuse
    gmail_ids = gmail_ids[:200]

    results = db.get_labels_by_gmail_ids(gmail_ids)

    # Add group mapping
    classifications = {}
    for gmail_id, label_data in results.items():
        classifications[gmail_id] = {
            "label": label_data["label"],
            "confidence": label_data["confidence"],
            "source": label_data["source"],
            "group": _label_to_group(label_data["label"]),
        }

    return jsonify({"classifications": classifications})


@app.route("/api/labels", methods=["GET"])
def labels():
    """Full label taxonomy with descriptions and groups."""
    groups = {
        "ACTION": [l for l in config.LABELS if l in ACTION_LABELS],
        "INFO": [l for l in config.LABELS if l in INFO_LABELS],
        "NOISE": [l for l in config.LABELS if l in NOISE_LABELS],
    }

    return jsonify({
        "labels": config.LABELS,
        "descriptions": config.LABEL_DESCRIPTIONS,
        "groups": groups,
    })


@app.route("/api/summary", methods=["GET"])
def summary():
    """Label distribution counts."""
    label_summary = db.get_label_summary()
    total = db.get_total_email_count()
    unlabeled = db.get_unlabeled_count()

    # Group counts
    group_counts = {"ACTION": 0, "INFO": 0, "NOISE": 0, "OTHER": 0}
    for label, count in label_summary.items():
        group = _label_to_group(label)
        group_counts[group] += count

    return jsonify({
        "labels": label_summary,
        "groups": group_counts,
        "total_emails": total,
        "total_labeled": sum(label_summary.values()),
        "unlabeled": unlabeled,
    })


def main():
    parser = argparse.ArgumentParser(description="Email classifier API server")
    parser.add_argument(
        "--port", type=int, default=config.API_PORT,
        help=f"Port to run on (default: {config.API_PORT})",
    )
    parser.add_argument(
        "--host", type=str, default=config.API_HOST,
        help=f"Host to bind to (default: {config.API_HOST})",
    )
    args = parser.parse_args()

    print(f"Starting Email Classifier API on http://{args.host}:{args.port}")
    print("Press Ctrl+C to stop.\n")

    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()

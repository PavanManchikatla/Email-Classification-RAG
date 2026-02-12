"""
Category discovery via clustering + LLM-based naming.

Analyzes uncertain/low-confidence email predictions to detect potential
new categories that the model hasn't seen before. Uses DBSCAN clustering
on TF-IDF vectors, then asks Claude Haiku to propose a category name.

Usage:
    python -m src.discover_categories          # cluster + propose
    python -m src.discover_categories --review  # review pending proposals
"""

import argparse
import json
import logging
from collections import Counter

import numpy as np
from sklearn.cluster import DBSCAN, KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_distances

import config
from src import db

logger = logging.getLogger(__name__)


def _build_texts_from_emails(emails: list) -> list[str]:
    """Build TF-IDF-ready text strings from email rows."""
    texts = []
    for row in emails:
        body = (row["body"] or "")[:500]
        text = f"{row['from_addr']} {row['subject']} {body}"
        texts.append(text)
    return texts


def cluster_uncertain_emails(email_ids: list) -> list[dict]:
    """
    Cluster uncertain emails to detect potential new categories.

    Args:
        email_ids: list of internal email IDs flagged as uncertain

    Returns:
        List of cluster dicts with top terms, sample IDs, and current labels.
    """
    if not email_ids:
        return []

    emails = db.get_emails_by_ids(email_ids)
    if len(emails) < config.EVOLVE_MIN_CLUSTER_SIZE:
        logger.info(
            "Only %d uncertain emails (min %d). Skipping clustering.",
            len(emails), config.EVOLVE_MIN_CLUSTER_SIZE,
        )
        return []

    logger.info("Clustering %d uncertain emails...", len(emails))

    texts = _build_texts_from_emails(emails)
    ids = [row["id"] for row in emails]
    current_labels = [row["label"] for row in emails]

    # Fit TF-IDF
    vectorizer = TfidfVectorizer(max_features=3000, stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(texts)
    feature_names = vectorizer.get_feature_names_out()

    # Try DBSCAN first (auto-detects cluster count)
    distance_matrix = cosine_distances(tfidf_matrix)
    clustering = DBSCAN(eps=0.5, min_samples=10, metric="precomputed")
    cluster_labels = clustering.fit_predict(distance_matrix)

    unique_labels = set(cluster_labels)
    unique_labels.discard(-1)  # Remove noise label

    # Fallback to KMeans if DBSCAN found nothing
    if not unique_labels:
        n_clusters = min(5, len(texts) // 20)
        if n_clusters < 2:
            logger.info("Not enough emails for KMeans clustering.")
            return []

        logger.info("DBSCAN found no clusters. Falling back to KMeans(n=%d)", n_clusters)
        clustering = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = clustering.fit_predict(tfidf_matrix)
        unique_labels = set(cluster_labels)

    clusters = []
    for cluster_id in sorted(unique_labels):
        mask = cluster_labels == cluster_id
        indices = np.where(mask)[0]

        if len(indices) < config.EVOLVE_MIN_CLUSTER_SIZE:
            continue

        # Top TF-IDF terms for this cluster
        cluster_tfidf = tfidf_matrix[indices].mean(axis=0)
        cluster_tfidf = np.asarray(cluster_tfidf).flatten()
        top_term_indices = cluster_tfidf.argsort()[-10:][::-1]
        top_terms = [feature_names[i] for i in top_term_indices]

        # Sample email IDs (closest to centroid)
        centroid = tfidf_matrix[indices].mean(axis=0)
        centroid = np.asarray(centroid).flatten()
        distances_to_centroid = cosine_distances(
            tfidf_matrix[indices], centroid.reshape(1, -1)
        ).flatten()
        closest_indices = distances_to_centroid.argsort()[:3]
        sample_ids = [ids[indices[i]] for i in closest_indices]

        # Current label distribution
        cluster_current_labels = [current_labels[i] for i in indices]
        label_counts = Counter(cluster_current_labels)

        clusters.append({
            "cluster_id": int(cluster_id),
            "size": len(indices),
            "top_terms": top_terms,
            "sample_ids": sample_ids,
            "current_labels": dict(label_counts),
        })

    logger.info("Found %d clusters with >= %d emails", len(clusters), config.EVOLVE_MIN_CLUSTER_SIZE)
    return clusters


def propose_category_names(clusters: list) -> list[dict]:
    """
    Use Claude Haiku to propose category names for discovered clusters.

    Skips clusters where >80% of emails already share one label
    (those are just existing categories with low confidence).
    """
    if not clusters:
        return []

    # Filter out homogeneous clusters
    novel_clusters = []
    for cluster in clusters:
        total = cluster["size"]
        label_counts = cluster["current_labels"]
        max_label_count = max(label_counts.values()) if label_counts else 0

        if max_label_count / total > 0.8:
            dominant = max(label_counts, key=label_counts.get)
            logger.info(
                "Cluster %d: %.0f%% are '%s'. Skipping (not novel).",
                cluster["cluster_id"], (max_label_count / total) * 100, dominant,
            )
            continue
        novel_clusters.append(cluster)

    if not novel_clusters:
        logger.info("No novel clusters found. All are existing categories.")
        return []

    logger.info("Proposing names for %d novel clusters...", len(novel_clusters))

    try:
        import anthropic
    except ImportError:
        logger.error("anthropic package not installed. Cannot propose category names.")
        return []

    if not config.ANTHROPIC_API_KEY:
        logger.error("ANTHROPIC_API_KEY not set. Cannot propose category names.")
        return []

    client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
    proposals = []

    for cluster in novel_clusters:
        # Get sample email details
        sample_emails = db.get_emails_by_ids(cluster["sample_ids"])
        sample_texts = []
        for e in sample_emails:
            sample_texts.append(
                f"  From: {e['from_addr']}\n"
                f"  Subject: {e['subject']}\n"
                f"  Body preview: {(e['body'] or '')[:200]}"
            )

        prompt = (
            f"I have a cluster of {cluster['size']} emails that don't fit well into "
            f"my existing categories: {', '.join(config.LABELS)}.\n\n"
            f"Top terms in this cluster: {', '.join(cluster['top_terms'])}\n"
            f"Current label distribution: {cluster['current_labels']}\n\n"
            f"Sample emails:\n" + "\n---\n".join(sample_texts) + "\n\n"
            f"Based on these emails, should I create a new category?\n"
            f"If yes, respond with JSON: {{\"new_category\": \"category_name\", "
            f"\"description\": \"short description\", \"reasoning\": \"why this is distinct\"}}\n"
            f"If no (they belong in existing categories), respond with: "
            f"{{\"new_category\": \"no_new_category\", \"reasoning\": \"why\"}}"
        )

        try:
            response = client.messages.create(
                model=config.LLM_MODEL,
                max_tokens=500,
                messages=[{"role": "user", "content": prompt}],
            )

            text = response.content[0].text.strip()
            # Parse JSON from response (handle markdown code blocks)
            if "```" in text:
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
                text = text.strip()

            result = json.loads(text)
            proposed_name = result.get("new_category", "no_new_category")

            if proposed_name != "no_new_category":
                proposal = {
                    "proposed_name": proposed_name,
                    "cluster_size": cluster["size"],
                    "sample_email_ids": json.dumps(cluster["sample_ids"]),
                    "llm_reasoning": result.get("reasoning", ""),
                    "description": result.get("description", ""),
                }

                db.save_category_proposal(
                    proposed_name=proposal["proposed_name"],
                    cluster_size=proposal["cluster_size"],
                    sample_email_ids=proposal["sample_email_ids"],
                    llm_reasoning=proposal["llm_reasoning"],
                )

                proposals.append(proposal)
                logger.info("Proposed new category: %s", proposed_name)
            else:
                logger.info(
                    "Cluster %d: LLM says no new category needed. %s",
                    cluster["cluster_id"], result.get("reasoning", ""),
                )

        except Exception as e:
            logger.warning("Failed to get LLM proposal for cluster %d: %s", cluster["cluster_id"], e)

    return proposals


def review_proposals_cli():
    """Interactive CLI to review and accept/reject pending category proposals."""
    db.init_db()
    proposals = db.get_pending_proposals()

    if not proposals:
        print("No pending category proposals.")
        return

    print(f"\n=== Pending Category Proposals ({len(proposals)}) ===\n")

    for p in proposals:
        print(f"ID: {p['id']}")
        print(f"  Proposed name: {p['proposed_name']}")
        print(f"  Cluster size:  {p['cluster_size']} emails")
        print(f"  Reasoning:     {p['llm_reasoning']}")

        # Show sample emails
        try:
            sample_ids = json.loads(p["sample_email_ids"])
            sample_emails = db.get_emails_by_ids(sample_ids)
            if sample_emails:
                print("  Sample emails:")
                for e in sample_emails:
                    print(f"    - [{e['from_addr']}] {e['subject']}")
        except (json.JSONDecodeError, TypeError):
            pass

        while True:
            choice = input("  Accept (a), Reject (r), Skip (s): ").strip().lower()
            if choice == "a":
                db.update_proposal_status(p["id"], "accepted")
                print(f"  Accepted! Add '{p['proposed_name']}' to config.LABELS and retrain.")
                break
            elif choice == "r":
                db.update_proposal_status(p["id"], "rejected")
                print("  Rejected.")
                break
            elif choice == "s":
                print("  Skipped.")
                break
            else:
                print("  Invalid input. Try a/r/s.")

        print()

    print("Review complete.")
    accepted = db.get_pending_proposals()  # remaining
    print(f"Remaining pending: {len(accepted)}")


def main():
    parser = argparse.ArgumentParser(description="Discover new email categories")
    parser.add_argument(
        "--review", action="store_true",
        help="Review pending category proposals",
    )
    parser.add_argument(
        "--threshold", type=float, default=None,
        help="Confidence threshold for selecting uncertain emails",
    )
    args = parser.parse_args()

    db.init_db()

    if args.review:
        review_proposals_cli()
        return

    # Discover mode: cluster low-confidence emails
    threshold = args.threshold or config.EVOLVE_CONFIDENCE_THRESHOLD
    print(f"Fetching emails with confidence < {threshold}...")

    low_conf = db.get_low_confidence_emails(threshold=threshold, limit=500)
    if not low_conf:
        print("No low-confidence emails found.")
        return

    email_ids = [row["id"] for row in low_conf]
    print(f"Found {len(email_ids)} low-confidence emails. Clustering...")

    clusters = cluster_uncertain_emails(email_ids)
    if not clusters:
        print("No meaningful clusters found.")
        return

    print(f"\nFound {len(clusters)} cluster(s):")
    for c in clusters:
        print(f"  Cluster {c['cluster_id']}: {c['size']} emails")
        print(f"    Top terms: {', '.join(c['top_terms'][:5])}")
        print(f"    Current labels: {c['current_labels']}")

    print("\nProposing category names via LLM...")
    proposals = propose_category_names(clusters)

    if proposals:
        print(f"\n{len(proposals)} new category proposal(s) saved.")
        print("Run 'python -m src.discover_categories --review' to review them.")
    else:
        print("No new categories proposed.")


if __name__ == "__main__":
    main()

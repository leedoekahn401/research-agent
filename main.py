"""Research Agent — Scheduled trend-detection pipeline.

Orchestrates: ingestion → clustering → analysis → output on a 24-hour loop.
"""

from datetime import datetime, timezone

from apscheduler.schedulers.blocking import BlockingScheduler

from config import logger, ROLLING_WINDOW_HOURS
from ingestion import ingest
from clustering import fetch_rolling_window, run_clustering, save_cluster_labels
from analysis import (
    build_keyword_timeline,
    find_center_articles,
    summarise_clusters,
    format_trends_markdown,
)

# Track the last successful run (in-memory; swap for a file/DB if needed)
_last_successful_run: datetime | None = None


def run_pipeline() -> None:
    """Single execution of the full trend-detection pipeline."""
    global _last_successful_run

    logger.info("=" * 60)
    logger.info("Pipeline run started")
    logger.info("=" * 60)

    # ── 1. Ingest new articles ──────────────────────────────────
    num_new = ingest(since=_last_successful_run)
    logger.info("Ingested %d new article(s).", num_new)

    # ── 2. Retrieve rolling window from ChromaDB ────────────────
    res = fetch_rolling_window()
    if res is None:
        logger.info("Pipeline completed (skipped clustering — sparse data).")
        _last_successful_run = datetime.now(timezone.utc)
        return

    embeddings = res["embeddings"]
    documents = res["documents"]
    metadatas = res["metadatas"]
    ids = res["ids"]

    # ── 3. Cluster with dynamic parameters ──────────────────────
    cluster_labels = run_clustering(embeddings)
    save_cluster_labels(ids, cluster_labels)

    # ── 4. Build keyword timeline (c-TF-IDF) ────────────────────
    docs_per_class = build_keyword_timeline(documents, metadatas, cluster_labels)

    # ── 5. Identify centre articles & summarise via LLM ─────────
    cluster_summaries = find_center_articles(
        embeddings, documents, cluster_labels, docs_per_class
    )
    trends = summarise_clusters(cluster_summaries)

    # ── 6. Output ───────────────────────────────────────────────
    output = format_trends_markdown(trends)
    logger.info("Trend summaries:\n%s", output)
    print(output)

    _last_successful_run = datetime.now(timezone.utc)
    logger.info("Pipeline run completed successfully.")


# ─────────────────────────────────────────────
# Scheduler
# ─────────────────────────────────────────────
if __name__ == "__main__":
    # Run once immediately on startup
    run_pipeline()

    # Then schedule every 24 hours
    scheduler = BlockingScheduler()
    scheduler.add_job(run_pipeline, "interval", hours=ROLLING_WINDOW_HOURS)
    logger.info(
        "Scheduler started — next run in %d hours. Press Ctrl+C to exit.",
        ROLLING_WINDOW_HOURS,
    )

    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        logger.info("Scheduler shut down.")
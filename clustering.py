"""Clustering: rolling-window retrieval, UMAP dimensionality reduction, and HDBSCAN."""

from datetime import datetime, timedelta, timezone

import hdbscan
import numpy as np
import umap

from config import (
    logger,
    chromadb_collection,
    MIN_DOCS_FOR_CLUSTERING,
    ROLLING_WINDOW_HOURS,
)


def fetch_rolling_window() -> dict | None:
    """Retrieve embeddings + documents from ChromaDB for the last rolling window.

    Returns the ChromaDB result dict, or ``None`` if fewer than
    ``MIN_DOCS_FOR_CLUSTERING`` documents are available.
    """
    cutoff_ts = (
        datetime.now(timezone.utc) - timedelta(hours=ROLLING_WINDOW_HOURS)
    ).timestamp()

    res = chromadb_collection.get(
        where={"date": {"$gte": cutoff_ts}},
        include=["embeddings", "documents", "metadatas"],
    )

    num_docs = len(res["ids"])
    logger.info("Rolling window returned %d document(s).", num_docs)

    if num_docs < MIN_DOCS_FOR_CLUSTERING:
        logger.warning(
            "Only %d document(s) in the window (minimum %d required). "
            "Skipping clustering for this cycle.",
            num_docs,
            MIN_DOCS_FOR_CLUSTERING,
        )
        return None

    return res


def reduce_dimensions(embeddings_raw: list) -> np.ndarray:
    """Apply UMAP with dynamically scaled parameters.

    ``n_neighbors`` and ``n_components`` are clamped so they never exceed
    ``num_points - 1``, preventing the common ``ValueError``.
    """
    num_points = len(embeddings_raw)

    n_neighbors = min(5, num_points - 1)
    n_components = min(5, num_points - 1)
    logger.info(
        "UMAP params: n_neighbors=%d, n_components=%d (num_points=%d)",
        n_neighbors, n_components, num_points,
    )

    umap_model = umap.UMAP(
        n_neighbors=n_neighbors,
        n_components=n_components,
        metric="cosine",
        random_state=42,
    )
    return umap_model.fit_transform(embeddings_raw)


def cluster_embeddings(reduced: np.ndarray) -> np.ndarray:
    """Run HDBSCAN on the reduced embedding space.

    ``min_cluster_size`` and ``min_samples`` are dynamically clamped to the
    number of data points to avoid runtime errors on sparse data.
    """
    num_points = len(reduced)

    min_cluster_size = min(3, num_points)
    min_samples = min(3, num_points)
    logger.info(
        "HDBSCAN params: min_cluster_size=%d, min_samples=%d",
        min_cluster_size, min_samples,
    )

    model = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
        cluster_selection_method="leaf",
        prediction_data=True,
    )
    labels = model.fit_predict(reduced)
    logger.info("Clusters found: %s", set(labels))
    return labels


def run_clustering(embeddings_raw: list) -> np.ndarray:
    """Convenience: UMAP → HDBSCAN in one call."""
    reduced = reduce_dimensions(embeddings_raw)
    return cluster_embeddings(reduced)


def save_cluster_labels(ids: list[str], cluster_labels: np.ndarray) -> None:
    """Persist cluster assignments back into ChromaDB metadata."""
    metadatas = [{"cluster": int(c)} for c in cluster_labels]
    chromadb_collection.update(ids=ids, metadatas=metadatas)
    logger.info("Saved cluster labels for %d document(s).", len(ids))

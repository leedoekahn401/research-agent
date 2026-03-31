"""Ingestion: fetch new articles from MongoDB, embed, and upsert into ChromaDB."""

from datetime import datetime, timedelta, timezone

from google.genai import types

from config import (
    logger,
    llm_client,
    mongo_collection,
    chromadb_collection,
    date_str_to_timestamp,
    EMBEDDING_MODEL,
    EMBEDDING_DIMENSIONALITY,
    BATCH_SIZE,
    ROLLING_WINDOW_HOURS,
)
from models import NewsArticle


def fetch_new_articles(since: datetime | None = None) -> list[dict]:
    """Query MongoDB for articles published after *since*.

    Validates every article against the ``NewsArticle`` schema.
    Returns the raw list of Mongo documents.
    """
    now = datetime.now(timezone.utc)
    since = since or (now - timedelta(hours=ROLLING_WINDOW_HOURS))
    since_str = since.strftime("%Y-%m-%dT%H:%M:%S.000Z")

    logger.info("Ingestion window: %s → now", since_str)

    query_cond = {"error": None, "date": {"$gte": since_str}}
    projection = {"_id": 1, "content": 1, "title": 1, "date": 1, "source": 1}

    docs = []
    cursor = mongo_collection.find(query_cond, projection)

    for item in cursor:
        try:
            # Validate input data against NewsArticle schema
            NewsArticle(**item)
            docs.append(item)
        except Exception as e:
            item_id = str(item.get("_id", "unknown"))
            logger.warning("Skipping invalid article ID %s: %s", item_id, e)

    logger.info("Validated & fetched %d article(s) from MongoDB.", len(docs))
    return docs


def embed_documents(contents: list[str]) -> list[list[float]]:
    """Embed a list of text documents via Gemini in batches.

    Returns a flat list of embedding vectors.
    """
    all_vectors: list[list[float]] = []

    for i in range(0, len(contents), BATCH_SIZE):
        batch = contents[i : i + BATCH_SIZE]
        result = llm_client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=batch,
            config=types.EmbedContentConfig(
                task_type="CLUSTERING",
                output_dimensionality=EMBEDDING_DIMENSIONALITY,
            ),
        )
        all_vectors.extend([e.values for e in result.embeddings])

    return all_vectors


def upsert_to_chromadb(
    ids: list[str],
    embeddings: list[list[float]],
    documents: list[str],
    metadatas: list[dict],
) -> None:
    """Upsert documents + embeddings into ChromaDB (idempotent by id)."""
    for i in range(0, len(ids), BATCH_SIZE):
        chromadb_collection.upsert(
            ids=ids[i : i + BATCH_SIZE],
            embeddings=embeddings[i : i + BATCH_SIZE],
            documents=documents[i : i + BATCH_SIZE],
            metadatas=metadatas[i : i + BATCH_SIZE],
        )
    logger.info("Upserted %d document(s) into ChromaDB.", len(ids))


def ingest(since: datetime | None = None) -> int:
    """End-to-end ingestion: fetch → embed → upsert.

    Returns the number of newly ingested documents.
    """
    docs = fetch_new_articles(since)
    if not docs:
        return 0

    ids = [str(d["_id"]) for d in docs]
    contents = [d["title"] + "\n" + d["content"] for d in docs]
    metadatas = [
        {
            "date": date_str_to_timestamp(d.get("date", "1970-01-01T00:00:00.000Z")),
            "date_str": d.get("date", "1970-01-01T00:00:00.000Z"),
        }
        for d in docs
    ]

    vectors = embed_documents(contents)
    upsert_to_chromadb(ids, vectors, contents, metadatas)
    return len(ids)

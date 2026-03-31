"""Analysis: c-TF-IDF keyword extraction and LLM-based trend summarisation."""

import json

import numpy as np
import pandas as pd
from google.genai import types
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from config import logger, llm_client, LLM_MODEL

# ─────────────────────────────────────────────
# LLM Prompt
# ─────────────────────────────────────────────
SYSTEM_PROMPT_SUMMARIZER = """\
You are a trend evolution assistant. You will be given a representative center \
article from a cluster of semantically similar news articles, as well as a timeline \
of key terms (extracted via c-TF-IDF) representing how the topic evolved over \
distinct days.

Your task is to:
1. Identify the core topic this cluster represents based on the center article.
2. Analyze the timeline of keywords to describe how the narrative or focus of the topic shifted over the days.
3. Write a concise 3-4 sentence summary capturing both the core theme and its evolution.
4. Suggest a short label (3-5 words) for this trend cluster.

Respond ONLY in JSON with the keys: "cluster_id", "label", "summary".
"""


# ═════════════════════════════════════════════
#  c-TF-IDF keyword timeline
# ═════════════════════════════════════════════
def build_keyword_timeline(
    documents: list[str],
    metadatas: list[dict],
    cluster_labels: np.ndarray,
    top_n: int = 10,
) -> pd.DataFrame:
    """Compute c-TF-IDF keywords per (cluster, day) and return a DataFrame.

    Returns an empty DataFrame when no valid dates are found.
    """
    df = pd.DataFrame({"Document": documents})
    df["cluster_id"] = [int(c) for c in cluster_labels]
    df["date"] = [m.get("date_str", "") for m in metadatas]

    def _parse_date(date_str: str) -> str:
        try:
            return date_str[:10] if len(date_str) >= 10 else "Unknown"
        except Exception:
            return "Unknown"

    df["time_bin"] = df["date"].apply(_parse_date)
    df = df[df["time_bin"] != "Unknown"]

    if df.empty:
        return pd.DataFrame()

    docs_per_class = (
        df.groupby(["cluster_id", "time_bin"], as_index=False)
        .agg({"Document": " ".join})
    )

    vectorizer = CountVectorizer(stop_words="english")
    X = vectorizer.fit_transform(docs_per_class["Document"])

    words_per_class = np.array(X.sum(axis=1)).flatten()
    tf = X.multiply(1.0 / np.where(words_per_class == 0, 1, words_per_class)[:, None])

    avg_nr_samples = int(X.shape[0])
    frequency = np.array((X > 0).sum(axis=0)).flatten()
    idf = np.log(1 + (avg_nr_samples / np.where(frequency == 0, 1, frequency)))

    c_tf_idf = tf.multiply(idf)
    vocab = vectorizer.get_feature_names_out()

    keywords_list = []
    for i in range(c_tf_idf.shape[0]):
        row = c_tf_idf.getrow(i).toarray().flatten()
        top_indices = row.argsort()[-top_n:][::-1]
        top_words = [vocab[idx] for idx in top_indices if row[idx] > 0]
        keywords_list.append(", ".join(top_words))

    docs_per_class["keywords"] = keywords_list
    return docs_per_class


# ═════════════════════════════════════════════
#  Centre-article extraction
# ═════════════════════════════════════════════
def find_center_articles(
    embeddings: list,
    documents: list[str],
    cluster_labels: np.ndarray,
    docs_per_class: pd.DataFrame,
    num_center: int = 3,
) -> list[dict]:
    """For each non-noise cluster, find the top *num_center* articles closest to
    the centroid and attach their keyword timeline.

    Returns a list of dicts with keys: ``cluster_id``, ``center_article``, ``timeline``.
    """
    unique_clusters = sorted(set(int(c) for c in cluster_labels if c != -1))
    summaries: list[dict] = []

    for cluster_id in unique_clusters:
        indices = [i for i, c in enumerate(cluster_labels) if int(c) == cluster_id]
        if not indices:
            continue

        emb_np = np.array([embeddings[i] for i in indices])
        docs = [documents[i] for i in indices]

        centroid = np.mean(emb_np, axis=0).reshape(1, -1)
        sims = cosine_similarity(emb_np, centroid).flatten()
        top_k = min(num_center, len(docs))
        center_indices = np.argsort(sims)[::-1][:top_k]
        center_text = "\n\n---\n\n".join(docs[idx] for idx in center_indices)

        # Keyword timeline for this cluster
        cluster_kw = docs_per_class[docs_per_class["cluster_id"] == cluster_id].sort_values("time_bin")
        timeline = ""
        for _, row in cluster_kw.iterrows():
            timeline += f"- {row['time_bin']}: {row['keywords']}\n"

        summaries.append({
            "cluster_id": cluster_id,
            "center_article": center_text,
            "timeline": timeline,
        })

    return summaries


# ═════════════════════════════════════════════
#  LLM trend summarisation
# ═════════════════════════════════════════════
def summarise_clusters(cluster_summaries: list[dict]) -> list[dict]:
    """Call the LLM for each cluster summary and return parsed JSON results."""
    trends: list[dict] = []

    for summary_data in cluster_summaries:
        c_id = summary_data["cluster_id"]
        user_content = (
            f"Cluster ID: {c_id}\n\n"
            f"Center Article Text:\n{summary_data['center_article']}\n\n"
            f"Topic Evolution Timeline:\n{summary_data['timeline']}"
        )

        try:
            response = llm_client.models.generate_content(
                model=LLM_MODEL,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT_SUMMARIZER,
                    response_mime_type="application/json",
                ),
                contents=user_content,
            )
            raw_json = response.text.strip()
            for prefix in ("```json", "```"):
                if raw_json.startswith(prefix):
                    raw_json = raw_json[len(prefix):]
            if raw_json.endswith("```"):
                raw_json = raw_json[:-3]

            parsed = json.loads(raw_json.strip())
            parsed.setdefault("cluster_id", c_id)
            trends.append(parsed)

        except Exception as e:
            logger.warning("Failed to summarise cluster %d: %s", c_id, e)

    return trends


def format_trends_markdown(trends: list[dict]) -> str:
    """Convert a list of trend dicts into a Markdown report string."""
    if not trends:
        return "_No trends detected this cycle._"

    parts: list[str] = []
    for t in trends:
        label = t.get("label", "Unknown Trend")
        summary = t.get("summary", "No summary provided.")
        c_id = t.get("cluster_id", "?")
        parts.append(f"## {label}\n\n{summary}\n\n<sub>Cluster ID: {c_id}</sub>\n\n---")

    return "\n\n".join(parts) + "\n"

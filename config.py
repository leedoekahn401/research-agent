"""Shared configuration, client initialization, and constants."""

import os
import logging
from datetime import datetime, timezone

import chromadb
from google import genai
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from dotenv import load_dotenv

# ─────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("research_agent")

# ─────────────────────────────────────────────
# Environment
# ─────────────────────────────────────────────
load_dotenv(override=True)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise EnvironmentError("GEMINI_API_KEY not found in environment.")

# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────
LLM_MODEL = "gemini-2.5-flash"
EMBEDDING_MODEL = "gemini-embedding-001"
EMBEDDING_DIMENSIONALITY = 768
BATCH_SIZE = 100
MIN_DOCS_FOR_CLUSTERING = 5
ROLLING_WINDOW_HOURS = 24

MONGO_URI = (
    "mongodb+srv://ducanh4012006_db_user:5zEVVC3o7Sjnl2le"
    "@cluster0.dwzpibi.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
)

# ─────────────────────────────────────────────
# Clients  (initialised once at import time)
# ─────────────────────────────────────────────
mongo_client = MongoClient(MONGO_URI, server_api=ServerApi("1"))
mongo_db = mongo_client["test"]
mongo_collection = mongo_db["news"]

llm_client = genai.Client(api_key=GEMINI_API_KEY)

chromadb_client = chromadb.HttpClient(host="localhost", port=8000)
chromadb_collection = chromadb_client.get_or_create_collection(
    name="test1",
    metadata={"hnsw:space": "cosine"},
)


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────
def date_str_to_timestamp(date_str: str) -> float:
    """Convert an ISO-8601 date string to a Unix timestamp (float).

    Handles formats like ``2026-03-31T12:00:00.000Z`` and ``2026-03-31T12:00:00``.
    Returns ``0.0`` on parse failure.
    """
    for fmt in ("%Y-%m-%dT%H:%M:%S.%fZ", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%S%z"):
        try:
            dt = datetime.strptime(date_str, fmt)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.timestamp()
        except ValueError:
            continue
    return 0.0


def timestamp_to_date_str(ts: float) -> str:
    """Convert a Unix timestamp back to an ISO-8601 date string (UTC)."""
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z")

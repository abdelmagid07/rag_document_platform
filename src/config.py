import os
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:  # optional during minimal environments/tests
    def load_dotenv() -> None:
        return None


load_dotenv()


class Config:
    """Centralized runtime configuration loaded from environment variables."""

    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    USE_LOCAL_EMBEDDINGS = os.getenv("USE_LOCAL_EMBEDDINGS", "false").lower() == "true"

    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))

    VECTOR_DIM = int(os.getenv("VECTOR_DIM", "1536"))
    INDEX_PATH = Path(os.getenv("INDEX_PATH", "data/indexes/faiss.index"))

    DEFAULT_TOP_K = int(os.getenv("DEFAULT_TOP_K", "5"))

    LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
    LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.2"))
    LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "1000"))

    USE_OCR = os.getenv("USE_OCR", "true").lower() == "true"

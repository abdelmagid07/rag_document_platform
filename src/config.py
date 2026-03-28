import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parents[1]


class Config:
    # API Keys
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

    # Chunking Settings
    CHUNK_SIZE = 500
    CHUNK_OVERLAP = 100

    # Vector Store
    VECTOR_DIM = 384  # BGE-small-v1.5 
    INDEX_DIR = PROJECT_ROOT / "data" / "indexes"
    INDEX_PATH = INDEX_DIR / "faiss.index"
    METADATA_PATH = INDEX_DIR / "metadata.json"

    # Database Settings
    DB_URL = os.getenv("DATABASE_URL")

    # Redis Settings
    REDIS_URL = os.getenv("REDIS_URL") or "redis://localhost:6379/0"
    CACHE_TTL = 3600  # 1 hour

    # Retrieval Settings
    DEFAULT_TOP_K = 5

    # LLM Settings
    LLM_MODEL = "gemini-2.5-flash-lite"
    LLM_TEMPERATURE = 0.2
    LLM_MAX_TOKENS = 1000

    # OCR / PDF Reading
    USE_OCR = True

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parents[1]


class Config:
    # OpenAI API Key
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    USE_LOCAL_EMBEDDINGS = False

    # Chunking Settings
    CHUNK_SIZE = 500
    CHUNK_OVERLAP = 100

    # Vector Store
    VECTOR_DIM = 1536  # OpenAI text-embedding-3-small
    INDEX_DIR = PROJECT_ROOT / "data" / "indexes"
    INDEX_PATH = INDEX_DIR / "faiss.index"
    METADATA_PATH = INDEX_DIR / "metadata.json"

    # Retrieval Settings
    DEFAULT_TOP_K = 5

    # LLM Settings
    LLM_MODEL = "gpt-4o-mini"
    LLM_TEMPERATURE = 0.2
    LLM_MAX_TOKENS = 1000

    # OCR / PDF Reading
    USE_OCR = True

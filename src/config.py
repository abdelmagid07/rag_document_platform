import os
import pathlib
import dotenv

load_dotenv()

class Config:
    # OpenAI API Key
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    USE_LOCAL_EMBEDDINGS = False

    # Chunking Settings
    CHUNK_SIZE = 500
    CHUNK_OVERLAP = 100

    # Vector Store 
    VECTOR_DIM = 1536 # OpenAI text-embedding-3-small
    INDEX_PATH = Path("indexes/faiss.index")
    DOC_TYPE_INDEX_DIR = Path("indexes/doc_type")

    # Retrieval Settings
    DEFAULT_TOP_K = 5
    AUTO_ROUTE = True

    # LLM Settings
    LLM_MODEL = "gpt-4o-mini"
    LLM_TEMPERATURE = 0.2
    LLM_MAX_TOKENS = 1000

    # OCR / PDF Reading 
    USE_OCR = True
    

    
import numpy as np
import openai
from sentence_transformers import SentenceTransformer

from ..config import Config


class Embedder:
    """Abstraction for embedding generation."""

    def __init__(self):
        self.model = None
        if Config.USE_LOCAL_EMBEDDINGS:
            self.model = SentenceTransformer("all-MiniLM-L6-v2")
        else:
            openai.api_key = Config.OPENAI_API_KEY

    def get_text_embedding(self, text: str) -> np.ndarray:
        if Config.USE_LOCAL_EMBEDDINGS and self.model is not None:
            return np.asarray(self.model.encode(text), dtype=np.float32)

        resp = openai.Embedding.create(
            model="text-embedding-ada-002",
            input=text,
        )
        return np.asarray(resp["data"][0]["embedding"], dtype=np.float32)

    def get_query_embedding(self, query: str) -> np.ndarray:
        return self.get_text_embedding(query)

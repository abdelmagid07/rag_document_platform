import numpy as np
from .config import Config
from sentence_transformers import SentenceTransformer
import openai

class Embedder:
    """Abstraction for embedding generation."""
    def __init__(self):
        if Config.USE_LOCAL_EMBEDDINGS:
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
        else:
            openai.api_key = Config.OPENAI_API_KEY

    def get_text_embedding(self, text: str) -> np.ndarray:
        if Config.USE_LOCAL_EMBEDDINGS:
            return self.model.encode(text)
        else:
            resp = openai.Embedding.create(
                model="text-embedding-ada-002",
                input=text
            )
            return np.array(resp['data'][0]['embedding'])

    def get_query_embedding(self, query: str) -> np.ndarray:
        return self.get_text_embedding(query)
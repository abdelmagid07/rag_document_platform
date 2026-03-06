from typing import List, Optional, Tuple
import numpy as np
from .vector_store import VectorStore
from .chunking import ChunkMetadata
from .embeddings import Embedder

class Retriever:
    """Metadata-aware retrieval with optional doc type routing."""
    def __init__(self):
        self.vector_store = VectorStore()
        self.embedder = Embedder()

    
    def retrieve(self, query: str, k: int = 5) -> List[Tuple[ChunkMetadata, float]]:
        q_emb = np.array([self.embedder.get_query_embedding(query)])
        D, I = self.vector_store.index.search(q_emb, k)
        scores = [(1 - d/max(D[0])) for d in D[0]]
        results = [(self.vector_store.chunks[i], scores[idx]) for idx, i in enumerate(I[0])]
        return results
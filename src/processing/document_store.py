import faiss
import numpy as np
from typing import List
from .chunking import ChunkMetadata
from .config import Config
import os

class VectorStore:
    def __init__(self):
        self.dim = dim
        self.index = faiss.IndexFlate2(dim)
        self.chunks = List[ChunkMetadata] = []

    def build_index(self, chunks: List[ChunkMetadata]):
        self.chunks = chunks
        embeddings = np.array([c.embedding for c in chunks], dtype=np.float32)
        self.index.add(embeddings)

    def save_index(self, path: str = None):
        path = path or Config.INDEX_PATH
        os.makedirs(os.path.dirname(path), exist_ok=True)
        faiss.write_index(self.index, str(path))

    def load_index(self, path: str = None):
        path = path or Config.INDEX_PATH
        self.index = faiss.read_index(str(path))

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

import os
from typing import List, Tuple

import faiss
import numpy as np

from ..config import Config
from ..models.embeddings import Embedder
from .chunking import ChunkMetadata


class EnhancedDocumentStore:
    """FAISS-backed store for chunk embeddings and metadata."""

    def __init__(self, dim: int | None = None):
        self.dim = dim or Config.VECTOR_DIM
        self.index = faiss.IndexFlatL2(self.dim)
        self.chunks: List[ChunkMetadata] = []

    def build_index(self, chunks: List[ChunkMetadata]) -> None:
        self.chunks = chunks
        embeddings = np.array([c.embedding for c in chunks], dtype=np.float32)
        if len(embeddings) > 0:
            self.index.add(embeddings)

    def save_index(self, path: str | None = None) -> None:
        target = str(path or Config.INDEX_PATH)
        os.makedirs(os.path.dirname(target), exist_ok=True)
        faiss.write_index(self.index, target)

    def load_index(self, path: str | None = None) -> None:
        target = str(path or Config.INDEX_PATH)
        self.index = faiss.read_index(target)


class IntelligentRetriever:
    """Retrieval wrapper with embedding + top-k scoring."""

    def __init__(self, store: EnhancedDocumentStore | None = None):
        self.document_store = store or EnhancedDocumentStore()
        self.embedder = Embedder()

    def retrieve(self, query: str, k: int = 5) -> List[Tuple[ChunkMetadata, float]]:
        if not self.document_store.chunks:
            return []

        q_emb = np.array([self.embedder.get_query_embedding(query)], dtype=np.float32)
        distances, indices = self.document_store.index.search(q_emb, k)

        max_distance = float(max(distances[0])) if len(distances[0]) else 1.0
        if max_distance == 0:
            max_distance = 1.0

        results: List[Tuple[ChunkMetadata, float]] = []
        for rank, chunk_idx in enumerate(indices[0]):
            if chunk_idx < 0 or chunk_idx >= len(self.document_store.chunks):
                continue
            score = 1 - (float(distances[0][rank]) / max_distance)
            results.append((self.document_store.chunks[chunk_idx], score))
        return results

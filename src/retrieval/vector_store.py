import json
import numpy as np
from pathlib import Path
from typing import Optional
import os
from dotenv import load_dotenv

load_dotenv()

from ..config import Config

class VectorStore:
    """
    Pure NumPy Vector Store. 
    Bypasses FAISS to prevent Windows startup hangs.
    """

    def __init__(self, dim: int = Config.VECTOR_DIM):
        self.dim = dim
        self._embeddings: list[np.ndarray] = []
        self._metadata: list[dict] = []

    @property
    def count(self) -> int:
        return len(self._metadata)

    def insert(self, embeddings: list[list[float]], metadata: list[dict]):
        if len(embeddings) != len(metadata):
            raise ValueError("embeddings and metadata must have the same length")
        
        for emb in embeddings:
            self._embeddings.append(np.array(emb, dtype=np.float32))
        self._metadata.extend(metadata)

    def search(self, query_embedding: list[float], top_k: int = 5) -> list[dict]:
        if not self._embeddings:
            return []

        query = np.array(query_embedding, dtype=np.float32)
        
        # Calculate L2 distances using NumPy
        # (Very fast for <100k vectors)
        embeddings_matrix = np.stack(self._embeddings)
        distances = np.linalg.norm(embeddings_matrix - query, axis=1)
        
        indices = np.argsort(distances)[:top_k]
        
        results = []
        for idx in indices:
            meta = self._metadata[idx].copy()
            dist = float(distances[idx])
            meta["score"] = float(1.0 / (1.0 + dist))
            results.append(meta)
        return results

    def save(self, index_path: Path = None, metadata_path: Path = None):
        index_path = index_path or Config.INDEX_PATH
        metadata_path = metadata_path or Config.METADATA_PATH
        
        index_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save embeddings as .npy and metadata as .json
        if self._embeddings:
            np.save(index_path.with_suffix('.npy'), np.stack(self._embeddings))
        
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(self._metadata, f, ensure_ascii=False)

    def load(self, index_path: Path = None, metadata_path: Path = None):
        index_path = index_path or Config.INDEX_PATH
        metadata_path = metadata_path or Config.METADATA_PATH

        npy_path = index_path.with_suffix('.npy')
        if not npy_path.exists() or not metadata_path.exists():
            return

        embs_array = np.load(npy_path)
        self._embeddings = [embs_array[i] for i in range(embs_array.shape[0])]
        
        with open(metadata_path, "r", encoding="utf-8") as f:
            self._metadata = json.load(f)

# Singleton management
_store: Optional[VectorStore] = None

def get_vector_store() -> VectorStore:
    global _store
    if _store is None:
        _store = VectorStore()
        _store.load()
    return _store
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

 
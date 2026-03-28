import json
from typing import List, Dict, Optional
import numpy as np
import pgvector.asyncpg
from .vector_store import VectorStore
from ..services.database import Database
from ..services.logger import logger

class PgVectorStore:
    """
    PostgreSQL + pgvector Store for persistent RAG.
    """

    @classmethod
    async def insert(cls, embeddings: np.ndarray, metadata: List[Dict], document_id: str, user_id: str):
        """
        Insert chunks and their embeddings into Postgres.
        """
        pool = await Database.get_pool()
        async with pool.acquire() as conn:
            async with conn.transaction():
                # Batch insert chunks using executemany
                chunk_data = []
                for i, (emb, meta) in enumerate(zip(embeddings, metadata)):
                    chunk_data.append((
                        document_id,
                        user_id,
                        i,
                        meta["text"],
                        emb
                    ))
                
                await conn.executemany(
                    "INSERT INTO chunks (document_id, user_id, chunk_index, content, embedding) VALUES ($1, $2, $3, $4, $5)",
                    chunk_data
                )
                logger.info(f"Inserted {len(chunk_data)} chunks for document {document_id} (user: {user_id})")

    @classmethod
    async def search(cls, query_embedding: np.ndarray, top_k: int = 5, user_id: str = None) -> List[Dict]:
        """
        Perform similarity search using pgvector (<=> operator for cosine distance).
        Filters by user_id and joins with documents table to get filenames.
        """
        pool = await Database.get_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT c.document_id, d.filename, c.content, 1 - (c.embedding <=> $1) as similarity 
                FROM chunks c
                JOIN documents d ON c.document_id = d.id
                WHERE c.user_id = $3
                ORDER BY c.embedding <=> $1 
                LIMIT $2
                """,
                query_embedding,
                top_k,
                user_id
            )
            
            results = []
            for row in rows:
                results.append({
                    "doc_id": str(row["document_id"]),
                    "filename": row["filename"],
                    "text": row["content"],
                    "score": float(row["similarity"])
                })
            return results

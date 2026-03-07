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
    async def insert(cls, embeddings: np.ndarray, metadata: List[Dict], document_id: str):
        """
        Insert chunks and their embeddings into Postgres.
        """
        pool = await Database.get_pool()
        async with pool.acquire() as conn:
            # Register pgvector for this connection
            await pgvector.asyncpg.register_vector(conn)
            
            async with conn.transaction():
                # Insert document metadata if needed (simplifying for now)
                # We assume the document_id is managed by the caller
                
                # Batch insert chunks using executemany
                chunk_data = []
                for i, (emb, meta) in enumerate(zip(embeddings, metadata)):
                    chunk_data.append((
                        document_id,
                        i,
                        meta["text"],
                        emb
                    ))
                
                await conn.executemany(
                    "INSERT INTO chunks (document_id, chunk_index, content, embedding) VALUES ($1, $2, $3, $4)",
                    chunk_data
                )
                logger.info(f"Inserted {len(chunk_data)} chunks for document {document_id}")

    @classmethod
    async def search(cls, query_embedding: np.ndarray, top_k: int = 5) -> List[Dict]:
        """
        Perform similarity search using pgvector (<=> operator for cosine distance).
        """
        pool = await Database.get_pool()
        async with pool.acquire() as conn:
            await pgvector.asyncpg.register_vector(conn)
            
            # Use cosine distance (<=>) or L2 distance (<->)
            rows = await conn.fetch(
                """
                SELECT document_id, content, 1 - (embedding <=> $1) as similarity 
                FROM chunks 
                ORDER BY embedding <=> $1 
                LIMIT $2
                """,
                query_embedding,
                top_k
            )
            
            results = []
            for row in rows:
                results.append({
                    "doc_id": str(row["document_id"]),
                    "text": row["content"],
                    "score": float(row["similarity"])
                })
            return results

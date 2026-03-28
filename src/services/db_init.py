import asyncio
from .database import Database
from .logger import logger


async def init_db():
    logger.info("Initializing database")

    # Enable pgvector 
    await Database.execute("""
        CREATE EXTENSION IF NOT EXISTS vector;
    """)

    # Drop existing tables to fix schema mismatch
    await Database.execute("DROP TABLE IF EXISTS chunks, documents CASCADE;")

    # Documents table
    await Database.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            id UUID PRIMARY KEY,
            user_id TEXT NOT NULL,
            filename TEXT,
            created_at TIMESTAMP DEFAULT NOW()
        );
    """)

    # Chunks table 
    await Database.execute("""
        CREATE TABLE IF NOT EXISTS chunks (
            id SERIAL PRIMARY KEY,
            document_id UUID REFERENCES documents(id) ON DELETE CASCADE,
            user_id TEXT NOT NULL,
            chunk_index INTEGER,
            content TEXT NOT NULL,
            embedding VECTOR(384)
        );
    """)

    # Vector Index (HNSW)
    await Database.execute("""
        CREATE INDEX IF NOT EXISTS idx_chunks_embedding 
        ON chunks USING hnsw (embedding vector_cosine_ops)
        WITH (m = 16, ef_construction = 64);
    """)

    # B-Tree index for foreign key efficiency
    await Database.execute("""
        CREATE INDEX IF NOT EXISTS idx_chunks_doc_id ON chunks (document_id);
    """)

    # B-Tree indexes for user_id filtering
    await Database.execute("""
        CREATE INDEX IF NOT EXISTS idx_documents_user_id ON documents (user_id);
    """)
    await Database.execute("""
        CREATE INDEX IF NOT EXISTS idx_chunks_user_id ON chunks (user_id);
    """)

    logger.info("Database initialized successfully.")


if __name__ == "__main__":
    asyncio.run(init_db())
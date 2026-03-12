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
            filename TEXT,
            created_at TIMESTAMP DEFAULT NOW()
        );
    """)

    # Chunks table 
    await Database.execute("""
        CREATE TABLE IF NOT EXISTS chunks (
            id SERIAL PRIMARY KEY,
            document_id UUID REFERENCES documents(id) ON DELETE CASCADE,
            chunk_index INTEGER,
            content TEXT NOT NULL,
            embedding VECTOR(384)
        );
    """)


    logger.info("Database initialized successfully.")


if __name__ == "__main__":
    asyncio.run(init_db())
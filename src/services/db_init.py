import asyncio
from .database import Database
from .logger import logger


async def init_db():
    logger.info("Initializing database")

    # Enable pgvector 
    await Database.execute("""
        CREATE EXTENSION IF NOT EXISTS vector;
    """)

    # Documents table
    await Database.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            id SERIAL PRIMARY KEY,
            title TEXT,
            source TEXT,
            created_at TIMESTAMP DEFAULT NOW()
        );
    """)

    # Chunks table 
    await Database.execute("""
        CREATE TABLE IF NOT EXISTS chunks (
            id SERIAL PRIMARY KEY,
            document_id INTEGER REFERENCES documents(id) ON DELETE CASCADE,
            content TEXT NOT NULL,
            embedding VECTOR(384)
        );
    """)


    logger.info("Database initialized successfully.")


if __name__ == "__main__":
    asyncio.run(init_db())
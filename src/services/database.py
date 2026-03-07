import asyncpg
from typing import Optional
from ..config import Config
from .logger import logger

class Database:
    _pool: Optional[asyncpg.Pool] = None

    @classmethod
    async def get_pool(cls) -> asyncpg.Pool:
        if cls._pool is None:
            try:
                cls._pool = await asyncpg.create_pool(
                    dsn=Config.DB_URL,
                    min_size=1,
                    max_size=10
                )
                logger.info("PostgreSQL connection pool created.")
            except Exception as e:
                logger.error(f"Failed to create database pool: {str(e)}")
                raise
        return cls._pool

    @classmethod
    async def close(cls):
        if cls._pool:
            await cls._pool.close()
            cls._pool = None
            logger.info("PostgreSQL connection pool closed.")

    @classmethod
    async def executemany(cls, query: str, args):
        pool = await cls.get_pool()
        async with pool.acquire() as conn:
            return await conn.executemany(query, args)

    @classmethod
    async def execute(cls, query: str, *args):
        pool = await cls.get_pool()
        async with pool.acquire() as conn:
            return await conn.execute(query, *args)

    @classmethod
    async def fetch(cls, query: str, *args):
        pool = await cls.get_pool()
        async with pool.acquire() as conn:
            return await conn.fetch(query, *args)

    @classmethod
    async def fetchrow(cls, query: str, *args):
        pool = await cls.get_pool()
        async with pool.acquire() as conn:
            return await conn.fetchrow(query, *args)

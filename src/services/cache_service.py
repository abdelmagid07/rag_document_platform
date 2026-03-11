import json
import logging
from typing import Optional, Any
import redis.asyncio as redis
from cachetools import LRUCache

from ..config import Config

logger = logging.getLogger(__name__)

class CacheService:
    """
    Multi-tier caching 
    Tier 1: In-memory LRU cache
    Tier 2: Redis distributed cache 
    """
    
    # L1: In-memory cache
    _l1_cache = LRUCache(maxsize=1000)
    
    # L2: Redis client
    _redis_client: Optional[redis.Redis] = None

    @classmethod
    async def get_redis(cls) -> redis.Redis:
        """Initialize or return the async Redis client."""
        if cls._redis_client is None:
            try:
                cls._redis_client = redis.from_url(
                    Config.REDIS_URL, 
                    decode_responses=True,
                    socket_timeout=2.0,
                    socket_connect_timeout=2.0
                )
                # Test connection
                await cls._redis_client.ping()
                logger.info("Connected to Redis for L2 caching.")
            except Exception as e:
                logger.error(f"Failed to connect to Redis: {e}")
                cls._redis_client = None
        return cls._redis_client

    @classmethod
    async def get(cls, key: str) -> Optional[Any]:
        """
        Check L1, then L2 (Redis).
        If hit in L2, promote to L1.
        """
        # Check L1
        if key in cls._l1_cache:
            logger.debug(f"L1 Cache Hit: {key[:50]}...")
            return cls._l1_cache[key]

        # Check L2 (Redis)
        redis_client = await cls.get_redis()
        if redis_client:
            try:
                cached_data = await redis_client.get(key)
                if cached_data:
                    logger.debug(f"L2 Cache Hit: {key[:50]}...")
                    data = json.loads(cached_data)
                    # Promote to L1
                    cls._l1_cache[key] = data
                    return data
            except Exception as e:
                logger.warning(f"Error reading from Redis: {e}")

        return None

    @classmethod
    async def set(cls, key: str, value: Any, ttl: int = Config.CACHE_TTL):
        """
        Set value in both L1 and L2.
        """
        # Set L1
        cls._l1_cache[key] = value

        # Set L2 
        redis_client = await cls.get_redis()
        if redis_client:
            try:
                serialized = json.dumps(value)
                await redis_client.set(key, serialized, ex=ttl)
            except Exception as e:
                logger.warning(f"Error writing to Redis: {e}")

    @classmethod
    def generate_key(cls, query: str) -> str:
        """
        Generate a cache key for a query. 
        Hash of the query text.
        """
        import hashlib
        return hashlib.sha256(query.strip().lower().encode()).hexdigest()




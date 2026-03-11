import asyncio
import time
from src.services.query_service import run_query
from src.services.cache_service import CacheService
from src.services.database import Database

async def test_cache_performance():
    query = "What is the main specialization of this platform?"
    top_k = 3
    
    print(f"\n--- Testing Caching Performance ---")
    print(f"Query: {query}")
    
    # Ensure Redis is clean for this specific test key
    cache_key = CacheService.generate_key(query)
    print(f"Debug: Generated Cache Key: {cache_key}")
    redis = await CacheService.get_redis()
    if redis:
        await redis.delete(cache_key)
        print("Debug: Deleted key from Redis")
    # Clear L1
    CacheService._l1_cache.clear()
    print("Debug: Cleared L1 Cache")

    # First Run (Cold Cache)
    print("Executing Run 1 (Cold)...")
    start_cold = time.time()
    answer1, sources1 = await run_query(query, top_k)
    cold_latency = (time.time() - start_cold) * 1000
    print(f"Cold Run Latency: {cold_latency:.2f}ms")
    
    # Second Run (L1 Cache Hit)
    print("Executing Run 2 (L1 Hit)...")
    start_l1 = time.time()
    answer2, sources2 = await run_query(query, top_k)
    l1_latency = (time.time() - start_l1) * 1000
    print(f"L1 Cache Hit Latency: {l1_latency:.2f}ms")
    
    # Verification L1
    assert answer1 == answer2, "L1 hit must return identical answer"
    assert l1_latency < 50, f"L1 latency should be <50ms, got {l1_latency:.2f}ms"

    # Third Run (L2 Cache Hit - simulating by clearing L1)
    print("Executing Run 3 (L2 Hit via Redis)...")
    CacheService._l1_cache.clear()
    start_l2 = time.time()
    answer3, sources3 = await run_query(query, top_k)
    l2_latency = (time.time() - start_l2) * 1000
    print(f"L2 Cache Hit Latency: {l2_latency:.2f}ms")

    # Verification L2
    assert answer1 == answer3, "L2 hit must return identical answer"
    assert l2_latency < 200, f"L2 latency should be <200ms (network hop), got {l2_latency:.2f}ms"
    
    print("\n--- Results Summary ---")
    improvement = ((cold_latency - l1_latency) / cold_latency) * 100
    print(f"L1 Speed Improvement: {improvement:.2f}%")
    print(f"Caching logic verified successfully!")

if __name__ == "__main__":
    asyncio.run(test_cache_performance())

import asyncio
import time
import random
import numpy as np
from unittest.mock import patch
from src.services.query_service import run_query
from src.services.cache_service import CacheService

def generate_zipf_workload(unique_queries, total_requests, s=1.1):
    """
    Generates a list of indices following a Zipfian distribution.
    Zipf's law states that many systems follow a power law where 
    a small number of items are requested very frequently.
    """
    ranks = np.arange(1, unique_queries + 1)
    weights = 1 / (ranks ** s)
    weights /= weights.sum()
    return np.random.choice(unique_queries, total_requests, p=weights)

async def run_load_test(total_requests=200, unique_count=50):
    # We mock the Gemini answer generation to test infrastructure speed and caching logic
    # without hitting API rate limits or incurring costs.
    with patch('src.services.query_service.generate_answer') as mock_gen:
        mock_gen.return_value = "This is a mocked RAG response for testing."
        
        print(f"\nInitiating Load Test")
        print(f"Scale: {total_requests} Requests | {unique_count} Unique Queries")
        
        unique_pool = [
            f"Query about specialization {i}" for i in range(unique_count)
        ]
        
        workload_indices = generate_zipf_workload(unique_count, total_requests)
        
        # Reset caches for a clean start
        CacheService._l1_cache.clear()
        redis = await CacheService.get_redis()
        if redis:
            await redis.flushdb()
        
        db_hits = 0
        cache_hits = 0
        latencies = []

        print("Running simulation...")
        
        start_sim = time.time()
        for idx in workload_indices:
            query = unique_pool[idx]
            
            # Check cache state before query
            key = CacheService.generate_key(query)
            is_cached = await CacheService.get(key) is not None
            
            start = time.time()
            # Standard query path (now with mocked LLM)
            await run_query(query, top_k=3)
            latency = (time.time() - start) * 1000
            latencies.append(latency)
            
            if is_cached:
                cache_hits += 1
            else:
                db_hits += 1
        
        sim_duration = time.time() - start_sim

        # Analysis
        avg_latency = sum(latencies) / len(latencies)
        hit_rate = (cache_hits / total_requests) * 100
        load_reduction = (cache_hits / (cache_hits + db_hits)) * 100 
        throughput = total_requests / sim_duration
        
        print("\n" + "="*40)
        print("LOAD TEST RESULTS")
        print("="*40)
        print(f"Total Requests:      {total_requests}")
        print(f"Throughput:         {throughput:.2f} req/s")
        print(f"Cache Hit Rate:     {hit_rate:.1f}%")
        print(f"DB Load Reduction:  {load_reduction:.1f}%")
        print(f"Avg Latency:        {avg_latency:.2f}ms")
        print("="*40)
        print(f"\nResume Data Point:")
        print(f"\"Achieved {hit_rate:.1f}% cache hit rate and {throughput:.1f} req/s throughput using a multi-tier caching strategy...\"")

if __name__ == "__main__":
    asyncio.run(run_load_test(total_requests=500, unique_count=200))

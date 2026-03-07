import time
from fastembed import TextEmbedding
from ..api.metrics import metrics_store
from .logger import logger

# Initialize FastEmbed model 
model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")


import numpy as np
import asyncio

async def get_embeddings(text_input: str | list[str]) -> np.ndarray:
    """
    Generate embeddings locally using FastEmbed.
    Returns a NumPy array for efficiency.
    """
    start = time.time()
    
    is_single_string = isinstance(text_input, str)
    texts = [text_input] if is_single_string else text_input
    
    def _embed():
        try:
            # model.embed returns a generator of numpy arrays
            embeddings_gen = model.embed(texts, batch_size=256)
            return np.array(list(embeddings_gen), dtype=np.float32)
        except Exception as e:
            logger.error(f"FastEmbed Error: {str(e)}")
            raise

    try:
        embeddings = await asyncio.to_thread(_embed)
    except Exception as e:
        logger.error(f"Embedding Thread Error: {str(e)}")
        raise
    
    latency = (time.time() - start) * 1000
    metrics_store.record("embedding", latency)
    
    return embeddings[0] if is_single_string else embeddings

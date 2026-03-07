import time
from fastembed import TextEmbedding
from ..api.metrics import metrics_store
from .logger import logger

# Initialize FastEmbed model 
model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")


async def get_embeddings(text_input: str | list[str]) -> list[float] | list[list[float]]:
    """
    Generate embeddings locally using FastEmbed.
    """
    start = time.time()
    
    # Expects a list of strings
    is_single_string = isinstance(text_input, str)
    texts = [text_input] if is_single_string else text_input
    
    # Returns a generator of numpy arrays
    embeddings_gen = model.embed(texts)
    embeddings = [list(emb) for emb in embeddings_gen]
    
    latency = (time.time() - start) * 1000
    metrics_store.record("embedding", latency)
    
    if is_single_string:
        return embeddings[0]
    
    return embeddings

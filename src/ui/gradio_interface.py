from typing import List, Dict, Tuple
from .chunking import ChunkMetadata
from .retriever import IntelligentRetriever
from .embeddings import Embedder
import openai
from .config import Config

def generate_answer(query: str, retrieved_chunks: List[Tuple[ChunkMetadata, float]]) -> Dict:
    context = "\n".join([c.text for c, _ in retrieved_chunks])
    prompt = f"Answer using only the context:\n{context}\nQuestion: {query}"
    if Config.USE_LOCAL_EMBEDDINGS:
        answer = "Local model stub"  # Replace with local LLM call
    else:
        resp = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role":"user","content":prompt}],
            temperature=0
        )
        answer = resp['choices'][0]['message']['content']
    return {"answer": answer, "sources": [c.chunk_id for c, _ in retrieved_chunks]}

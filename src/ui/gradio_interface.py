from typing import Dict, List, Tuple

import openai

from ..config import Config
from ..processing.chunking import ChunkMetadata


def generate_answer(query: str, retrieved_chunks: List[Tuple[ChunkMetadata, float]]) -> Dict:
    context = "\n".join([chunk.text for chunk, _ in retrieved_chunks])
    prompt = f"Answer using only the context:\n{context}\nQuestion: {query}"

    if Config.USE_LOCAL_EMBEDDINGS:
        answer = "Local model stub"
    else:
        resp = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        answer = resp["choices"][0]["message"]["content"]

    return {"answer": answer, "sources": [chunk.chunk_id for chunk, _ in retrieved_chunks]}

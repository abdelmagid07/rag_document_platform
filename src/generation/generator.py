from openai import AsyncOpenAI
import os
from dotenv import load_dotenv

load_dotenv()
# Using the Async client for streaming support
client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))


async def generate_answer_stream(query, documents):
    """
    Asynchronous generator that yields tokens from the LLM.
    """
    context = "\n\n".join([doc["text"] for doc in documents])

    prompt = f"""
Answer the question using the context below.

Context:
{context}

Question:
{query}
"""

    stream = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": prompt}
        ],
        stream=True
    )

    async for chunk in stream:
        if chunk.choices[0].delta.content is not None:
            yield chunk.choices[0].delta.content


async def generate_answer(query, documents):
    """
    Legacy sync-style wrapper (still async) that returns the full string.
    """
    full_response = ""
    async for token in generate_answer_stream(query, documents):
        full_response += token
    return full_response
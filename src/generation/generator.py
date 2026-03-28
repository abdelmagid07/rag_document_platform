from google import genai
import os
from dotenv import load_dotenv
from ..config import Config

load_dotenv()

# Initialize the new Google GenAI Client
client = genai.Client(api_key=Config.GEMINI_API_KEY)


async def generate_answer_stream(query, documents):
    """
    Asynchronous generator that yields tokens from Gemini using the local SDK (V1).
    """
    context = "\n\n".join([doc["text"] for doc in documents])

    prompt = f"""
Answer the question using the context below. 
Be concise and helpful.

Context:
{context}

Question:
{query}
"""

    # We use 'aio' for true async streaming, otherwise it blocks the event loop!
    try:
        response_stream = await client.aio.models.generate_content_stream(
            model=Config.LLM_MODEL,
            contents=prompt
        )

        async for chunk in response_stream:
            if chunk.text:
                yield chunk.text
    except Exception as e:
        from ..services.logger import logger
        logger.error(f"Gemini Streaming Error: {str(e)}")
        yield f"Error in generation: {str(e)}"


async def generate_answer(query, documents):
    """
    Returns the full string from Gemini using the new SDK.
    """
    context = "\n\n".join([doc["text"] for doc in documents])

    prompt = f"""
Answer the question using the context below. 
Be concise and helpful.

Context:
{context}

Question:
{query}
"""
    response = await client.aio.models.generate_content(
        model=Config.LLM_MODEL,
        contents=prompt
    )
    
    return response.text
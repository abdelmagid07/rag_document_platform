from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def get_embeddings(text_input: str | list[str]) -> list[float] | list[list[float]]:
    """
    Generate embeddings for a single string or a list of strings using OpenAI.
    """
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text_input
    )
    
    if isinstance(text_input, str):
        return response.data[0].embedding
    return [e.embedding for e in response.data]

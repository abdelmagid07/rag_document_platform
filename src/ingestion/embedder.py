from openai import OpenAI
import os

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def embed_chunks(chunks):

    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=chunks
    )

    embeddings = [e.embedding for e in response.data]

    return embeddings
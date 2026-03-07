import uuid
import tempfile
import os

from ..ingestion.document_loader import load_document
from ..ingestion.text_cleaner import clean_text
from ..ingestion.chunker import chunk_text
from .embedding_service import get_embeddings
from ..ingestion.vector_writer import write_vectors


async def ingest_document(upload_file) -> str:
    """
    Full ingestion pipeline:
    1. Save uploaded file to temp location
    2. Extract text using appropriate loader
    3. Clean the text
    4. Chunk into smaller pieces
    5. Generate embeddings for each chunk
    6. Store in vector store
    
    Returns the document_id.
    """
    document_id = str(uuid.uuid4())

    # Save uploaded file to temp location
    suffix = os.path.splitext(upload_file.filename)[1] if upload_file.filename else ".pdf"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await upload_file.read()
        tmp.write(content)
        path = tmp.name

    try:
        # Extract text
        text = load_document(path)

        # Clean
        text = clean_text(text)

        # Chunk
        chunks = chunk_text(text)

        # Generate embeddings
        embeddings = get_embeddings(chunks)

        # Store in vector store
        write_vectors(chunks, embeddings, document_id)

    finally:
        # Clean up temp file
        if os.path.exists(path):
            os.unlink(path)

    return document_id
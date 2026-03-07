import uuid
import tempfile

from ..ingestion.document_loader import load_pdf
from ..ingestion.text_cleaner import clean_text
from ..ingestion.chunker import chunk_text
from ..ingestion.embedder import embed_chunks
from ..ingestion.vector_writer import write_vectors


async def ingest_document(upload_file):

    document_id = str(uuid.uuid4())

    # save temp file
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(await upload_file.read())
        path = tmp.name

    # load text
    text = load_pdf(path)

    # clean
    text = clean_text(text)

    # chunk
    chunks = chunk_text(text)

    # embed
    embeddings = embed_chunks(chunks)

    # write to vector DB
    write_vectors(chunks, embeddings, document_id)

    return document_id
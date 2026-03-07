def write_vectors(chunks, embeddings, document_id):

    records = []

    for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):

        records.append({
            "doc_id": document_id,
            "chunk_id": i,
            "text": chunk,
            "embedding": emb
        })

    # insert into vector DB
    return records
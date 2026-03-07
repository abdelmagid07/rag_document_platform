class VectorStore:

    def search(self, embedding, top_k: int):
        raise NotImplementedError

    def insert(self, embeddings, metadata):
        raise NotImplementedError
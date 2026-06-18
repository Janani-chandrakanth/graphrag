import chromadb

client = chromadb.PersistentClient(
    path="./chroma_db"
)

collection = client.get_or_create_collection(
    name="requirements_chunks"
)


def store_chunk(chunk_id, chunk_text, embedding):
    """
    Store or update a chunk and its embedding in ChromaDB.
    """

    collection.upsert(
        ids=[chunk_id],
        documents=[chunk_text],
        embeddings=[embedding]
    )


def search_chunks(query_embedding, n_results=5):
    """
    Search similar chunks.
    """

    return collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results
    )


def get_collection_count():
    """
    Return number of vectors stored.
    """

    return collection.count()
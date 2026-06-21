"""
RAG retrieval module — finds relevant knowledge base chunks for a user query.
"""

import chromadb
from sentence_transformers import SentenceTransformer

CHROMA_DIR = "./chroma_db"
COLLECTION_NAME = "mental_health_docs"

_model = None
_collection = None


def _get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return _model


def _get_collection():
    global _collection
    if _collection is None:
        try:
            client = chromadb.PersistentClient(path=CHROMA_DIR)
            _collection = client.get_collection(COLLECTION_NAME)
        except Exception as e:
            print(f"[RAG] ChromaDB not ready: {e}")
            print("[RAG] Run 'python ingest.py' first.")
            _collection = None
    return _collection


def retrieve_context(query: str, n_results: int = 5) -> str:
    """
    Given a user query, returns relevant chunks from the knowledge base
    formatted as a string for injection into the Claude prompt.
    Returns empty string if DB is unavailable.
    """
    collection = _get_collection()
    if collection is None:
        return ""

    try:
        model = _get_model()
        query_embedding = model.encode(query).tolist()

        count = collection.count()
        if count == 0:
            return ""
        n = min(n_results, count)

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=n,
            include=["documents", "metadatas", "distances"],
        )

        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        if not docs:
            return ""

        parts = []
        for doc, meta, dist in zip(docs, metas, distances):
            if dist > 1.5:
                continue
            source = meta.get("source", "knowledge base")
            parts.append(f"[From: {source}]\n{doc.strip()}")

        return "\n\n---\n\n".join(parts)

    except Exception as e:
        print(f"[RAG] Retrieval error: {e}")
        return ""

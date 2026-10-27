"""
Module 7 Project — Semantic Search Engine
==========================================
search.py — query ChromaDB and return ranked results


"""

from pathlib import Path
import chromadb
from sentence_transformers import SentenceTransformer



# ── Configuration (must match ingest.py) ─────────────────────────────────────
CHROMA_PATH     = Path("chroma_data")
COLLECTION_NAME = "semantic_search"
MODEL_NAME      = "all-MiniLM-L6-v2"

client = chromadb.PersistentClient(path=str(CHROMA_PATH))
collection = client.get_collection(COLLECTION_NAME)
model = SentenceTransformer(MODEL_NAME)

def get_collection(chroma_path: Path, collection_name: str):
    """Create (or retrieve) a persistent ChromaDB collection."""
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))

    return client.get_or_create_collection(
    name=collection_name,
    embedding_function=None,
    metadata={"created_by": "semantic_search_project"},
    configuration={"hnsw": {"space": "cosine"}}
)
def semantic_search(query: str, n_results: int = 5, threshold: float = 0.3, distance_threshold=None):
    if not query.strip():
        return []

    # 1. Embed the query
    query_embedding = model.encode(query, convert_to_numpy=True)

    # 2. Load the collection
    collection = get_collection(CHROMA_PATH, COLLECTION_NAME)

    # 3. Query ChromaDB
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results
    )

    # 4. Extract results
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    scores = results["distances"][0]
    ids = results["ids"][0]

    # 5. Apply threshold filtering
    filtered = []
    for doc, meta, score, id_ in zip(docs, metas, scores, ids):
        if score <= threshold:   # lower distance = better match
            filtered.append({
                "id": id_,
                "text": doc,
                "metadata": meta,
                "score": score
            })

    return filtered


def search(
    query: str,
    n_results: int = 5,
    sources: list[str] = None,
    distance_threshold: float = None,
    threshold=None
) -> list[dict]:

    if not query.strip():
        return []

    # Embed query
    query_embedding = model.encode(query, convert_to_numpy=True)

    # Load collection
    collection = get_collection(CHROMA_PATH, COLLECTION_NAME)

    # Query ChromaDB
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results
    )

    docs = results["documents"][0]
    metas = results["metadatas"][0]
    scores = results["distances"][0]
    ids = results["ids"][0]

    output = []

    for doc, meta, score, id_ in zip(docs, metas, scores, ids):

        # Optional source filter
        if sources and meta.get("filename") not in sources:
            continue

        # Optional distance threshold
        if distance_threshold is not None and score > distance_threshold:
            continue

        output.append({
            "id": id_,
            "text": doc,
            "source": meta.get("filename"),
            "chunk_index": meta.get("chunk_index"),
            "distance": score,
            "score": 1 - score,  # convenience metric
        })

    # Sort best → worst
    output.sort(key=lambda x: x["distance"])

    return output

def get_collection_stats() -> dict:
    """
    Return basic stats about the indexed collection.

    Returns:
        {
            "total_chunks":   int,
            "unique_sources": int,
            "source_names":   list[str],
        }
    """
    collection = get_collection(CHROMA_PATH, COLLECTION_NAME)

    # Total number of chunks stored
    total_chunks = collection.count()

    # Extract metadata for all chunks
    results = collection.get(include=["metadatas"])

    metadatas = results["metadatas"]

    # Collect all filenames
    sources = set()
    for meta in metadatas:
        filename = meta.get("filename")
        if filename:
            sources.add(filename)

    return {
        "total_chunks": total_chunks,
        "unique_sources": len(sources),
        "source_names": sorted(list(sources)),
    }

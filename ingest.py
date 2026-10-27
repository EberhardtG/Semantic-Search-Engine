"""
Module 7 Project — Semantic Search Engine
==========================================
ingest.py — document loading, chunking, and ChromaDB storage

Run with:
    python ingest.py
    python ingest.py --chunk-size 200 --overlap 50
"""

import argparse
from pathlib import Path
import chromadb
from sentence_transformers import SentenceTransformer

# ── Configuration ─────────────────────────────────────────────────────────────
DOCS_DIR        = Path("docs")
CHROMA_PATH     = Path("chroma_data")
COLLECTION_NAME = "semantic_search"
MODEL_NAME      = "all-MiniLM-L6-v2"
DEFAULT_CHUNK_SIZE = 500
DEFAULT_OVERLAP    = 100


model = SentenceTransformer(MODEL_NAME)
# Create a persistent client (data survives script restarts)
client = chromadb.PersistentClient(path=str(CHROMA_PATH))

def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """
    Split text into fixed-size chunks with overlap.
    """
    chunks = []
    start = 0
    text = text.strip()

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()

        if chunk:  # avoid empty chunks
            chunks.append(chunk)

        # Move the window forward with overlap
        start = end - overlap

    return chunks


def load_documents(docs_dir: Path) -> list[dict]:
    """
    Read all .txt and .md files from docs_dir.

    Returns:
        List of dicts: {"filename": str, "text": str}
    """
    documents = []

    for file in docs_dir.iterdir():
        if file.suffix.lower() in [".txt", ".md"]:
            try:
                text = file.read_text(encoding="utf-8")
            except Exception as e:
                print(f"Failed to read {file}: {e}")
                continue

            documents.append({
                "filename": file.name,
                "text": text
            })
            print("Loading:", file.name)


    return documents

 


def get_collection(chroma_path: Path, collection_name: str):
    """Retrieve or create a persistent ChromaDB collection for storing embeddings."""
    return client.get_or_create_collection(
    name=collection_name,
    embedding_function=None,
    metadata={"created_by": "semantic_search_project"},
    configuration={"hnsw": {"space": "cosine"}}
)



def ingest(chunk_size: int = DEFAULT_CHUNK_SIZE, overlap: int = DEFAULT_OVERLAP):
    """
    Chunk all documents in docs/, embed them, and store them in ChromaDB.
    Supports configurable chunk size and overlap for chunking experiments.
    """
    #  Load all documents
    documents = load_documents(DOCS_DIR)

    #  Prepare master lists
    all_chunks = []
    all_metadatas = []
    all_ids = []

    # Chunk each document and store metadata.
    for doc in documents:
        chunks = chunk_text(doc["text"], chunk_size=chunk_size, overlap=overlap)
        for i, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            all_metadatas.append({
    "filename": doc["filename"],
    "chunk_index": i,
    "source": str(DOCS_DIR / doc["filename"]),
    "chunk_size": chunk_size,
    "overlap": overlap,
})

            all_ids.append(f"{doc['filename']}_{i}")

    if not all_chunks:
        print(f"No documents found in {DOCS_DIR}.")
        return

    #  Generate embeddings for all chunks.
    embeddings = model.encode(all_chunks, convert_to_numpy=True)

    # Add to ChromaDB collection.
    collection = get_collection(CHROMA_PATH, COLLECTION_NAME)
    collection.upsert(
        embeddings=list(embeddings),
        documents=all_chunks,
        metadatas=all_metadatas,
        ids=all_ids,
    )

    print(f"Indexed {len(all_chunks)} chunks from {len(documents)} docs.")

    for doc in documents:
     print("\n==============================")
     print("DOCUMENT:", doc["filename"])
     print("==============================")

     chunks = chunk_text(doc["text"], chunk_size=chunk_size, overlap=overlap)

     for i, chunk in enumerate(chunks):
        print(f"\n--- CHUNK {i} FROM {doc['filename']} ---")
        print(chunk[:200], "...")




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Index docs/ into ChromaDB")
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--overlap", type=int, default=DEFAULT_OVERLAP)
    args = parser.parse_args()
    ingest(chunk_size=args.chunk_size, overlap=args.overlap)

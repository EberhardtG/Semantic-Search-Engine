"""
Module 7 Project — Semantic Search Engine
==========================================
app.py — Streamlit search interface

Run with:
    streamlit run app.py

Make sure you've indexed documents first:
    python ingest.py
"""

import streamlit as st
from search import search, get_collection_stats,semantic_search
from ingest import DEFAULT_CHUNK_SIZE, DEFAULT_OVERLAP, ingest
from datetime import datetime

st.set_page_config(page_title="Sematic Search Engine", page_icon="✅", layout="wide")

#---Sidebar Controls---
with st.sidebar:
    n_results = st.slider(
        "select the number of results",
        min_value=1,
        max_value=20,
        value=5
          )

    st.divider()

    threshold = st.slider("Similarity threshold", 0.0, 1.0, 0.3)

    st.divider()

    st.subheader("Index Configuration")
    st.write(f"Chunk size: {DEFAULT_CHUNK_SIZE}")
    st.write(f"Overlap: {DEFAULT_OVERLAP}")

    stats = get_collection_stats()
    st.metric("Documents Indexed", stats["unique_sources"])
    st.write(f"Total documents: {stats['unique_sources']}")

    st.divider()

    sources = st.multiselect(
    "Filter by source file",
    stats["source_names"]
)

    st.divider()
    if st.button("Re-index documents"):
        ingest(chunk_size=DEFAULT_CHUNK_SIZE, overlap=DEFAULT_OVERLAP)
        st.success("Re-indexing complete!")

    st.divider()

    # This runs on EVERY re-run — so the timestamp changes each time
    st.write(f"Script ran at: **{datetime.now().strftime('%H:%M:%S')}**")



# --- Main Area ---
st.title("Semantic Search Engine")

# Query input
query = st.text_input("Enter your search query")

if st.button("Search"):

    # --- Basic validation ---
    if not query.strip():
        st.warning("Please enter a query before searching.")
        st.stop()

    # Non-word input (no letters)
    if not any(c.isalpha() for c in query):
        st.warning("Your query doesn't contain any letters — try typing real words.")
        st.stop()

    # Stopword-only queries
    STOPWORDS = {"the", "and", "or", "of", "to", "be", "a", "an"}
    if query.lower().strip() in STOPWORDS:
        st.info("Try a more descriptive query for better results.")
        st.stop()

    # Very short queries
    if len(query.strip()) < 3:
        st.warning("Try a longer query for better results.")
        st.stop()

    # --- Run search ---
    results = search(
        query,
        n_results=n_results,
        sources=sources,
        distance_threshold=threshold
    )

    # No results at all
    if not results:
        st.info("No results found.")
        st.stop()

    # Results exist, but none passed the threshold
    if all(r["distance"] > threshold for r in results):
        st.info("Results were found, but none met the similarity threshold.")
        st.stop()

    # Weak top match
    if results[0]["distance"] > 0.8:
        st.info("No strong semantic matches found. Try rephrasing your query.")

    # --- Display results ---
    st.subheader("Search Results")

    for r in results:
        source = r.get("source") or "Unknown source"
        chunk = r.get("chunk_index") or "N/A"

        st.write(
            f"**Source:** {source} | "
            f"**Chunk:** {chunk} | "
            f"**Distance:** {r['distance']:.3f}"
        )

        with st.expander("Show chunk text"):
            st.write(r["text"])

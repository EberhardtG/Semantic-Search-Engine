"""
Module 7 Project — Semantic Search Engine
==========================================
evaluate.py — precision and recall for your search system

Run with:
    python evaluate.py
    python evaluate.py --n-results 5 --threshold 0.4

Define your evaluation set in EVAL_SET, then run this script against
different index configurations (different chunk sizes) to compare results.
"""

import argparse
from search import search


# Define your evaluation set here.
# Each entry needs a query and the source filenames you expect to be relevant.
# Use at least 5 queries for the chunking experiment.
EVAL_SET = [
    {"query": "What does cosine similarity measure?",
     "relevant_ids": ["embeddings_and_vectors.txt_2"]},

    {"query": "When do background tasks run?",
     "relevant_ids": ["fastapi.txt_4"]},

    {"query": "What is a hallucination?",
     "relevant_ids": ["llms-and-ai.txt_2"]},

    {"query": "What is a decorator?",
     "relevant_ids": ["python-advanced.txt_1"]},

    {"query": "What is the base class for errors?",
     "relevant_ids": ["python-fundamentals.txt_5"]},

    {"query": "What does it mean to be stateless?",
     "relevant_ids": ["rest-apis.txt_0"]},

    {"query": "What does INNER JOIN accomplish?",
     "relevant_ids": ["sql-databases.txt_2"]},

    {"query": "What code supports chat interfaces?",
     "relevant_ids": ["streamlit.txt_5"]},
]



def precision_recall(
    retrieved_sources: list[str], relevant_sources: list[str]
) -> tuple[float, float]:
    """Compute precision and recall given retrieved and relevant chunk IDs."""
    # STEP 1 — Convert chunk IDs → source filenames
    retrieved_sources_set = {
        chunk_id.rsplit("_", 1)[0]
        for chunk_id in retrieved_sources
    }

    relevant_sources_set = {
        chunk_id.rsplit("_", 1)[0]
        for chunk_id in relevant_sources
    }

    # Intersection: sources that were retrieved AND relevant
    correct = retrieved_sources_set.intersection(relevant_sources_set)

    # Precision: of what we retrieved, how many were correct?
    precision = len(correct) / len(retrieved_sources_set) if retrieved_sources_set else 0.0

    # Recall: of all relevant sources, how many did we retrieve?
    recall = len(correct) / len(relevant_sources_set) if relevant_sources_set else 0.0

    return precision, recall


def evaluate(n_results=5, distance_threshold=None):
    """Run precision/recall evaluation over the EVAL_SET using the search engine."""
    precisions = []
    recalls = []

    for item in EVAL_SET:
        query = item["query"]
        relevant_ids = item["relevant_ids"]

        # Run your search engine with CLI parameters
        results = search(
             query,
            n_results=n_results,
            distance_threshold=distance_threshold
)


        

        # Extract retrieved chunk IDs
        retrieved_ids = [r["id"] for r in results]

        # Compute precision & recall
        precision, recall = precision_recall(retrieved_ids, relevant_ids)

        # Store results
        precisions.append(precision)
        recalls.append(recall)

        # Print per-query evaluation
        print("\nQuery:", query)
        print("Relevant:", relevant_ids)
        print("Retrieved:", retrieved_ids)
        print(f"Precision: {precision:.3f}")
        print(f"Recall:    {recall:.3f}")

    # Compute averages
    avg_precision = sum(precisions) / len(precisions)
    avg_recall = sum(recalls) / len(recalls)

    print("\n==============================")
    print("FINAL EVALUATION SUMMARY")
    print("==============================")
    print(f"Average Precision: {avg_precision:.3f}")
    print(f"Average Recall:    {avg_recall:.3f}")

    return avg_precision, avg_recall





if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate search quality")
    parser.add_argument("--n-results", type=int,   default=5)
    parser.add_argument("--threshold", type=float, default=None)
    args = parser.parse_args()

    evaluate(n_results=args.n_results, distance_threshold=args.threshold)


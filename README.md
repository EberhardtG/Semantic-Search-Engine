

# 📘 **Semantic Search Engine — Module 7 Project**

A semantic search engine built for **Applied AI: Embeddings & Retrieval (Module 7)**.  
This project loads documents, chunks them, embeds them using SentenceTransformers, stores them in ChromaDB, and provides a Streamlit interface for semantic search.

---

## 🚀 **Project Overview**

This project demonstrates how modern AI systems perform semantic retrieval using embeddings. It includes:

- Document ingestion + chunking  
- Embedding generation  
- Persistent vector storage  
- Semantic search with ranking  
- Streamlit UI  
- Precision/recall evaluation  
- A chunking experiment comparing different chunk sizes  

---

## 📁 **Project Structure**

```
semantic-search/
├── app.py               # Streamlit search interface
├── ingest.py            # Document loading, chunking, and ChromaDB storage
├── search.py            # Search functions (query ChromaDB, rank results)
├── evaluate.py          # Precision/recall evaluation
├── docs/                # Document collection
├── chroma_data/         # Persistent ChromaDB storage
├── requirements.txt     # Dependencies
└── README.md            # Project documentation + chunking experiment
```

---

## 🔧 **Installation & Setup**

### 1. Clone the repository
```
git clone <your-repo-url>
cd semantic-search
```

### 2. Install dependencies
```
pip install -r requirements.txt
```

### 3. Add documents  
Place `.txt` or `.md` files inside the `docs/` folder.

### 4. Ingest documents
```
python ingest.py
```

You can also configure chunk size and overlap:

```
python ingest.py --chunk-size 300 --overlap 50
```

---

## 🧠 **How It Works**

### **1. Ingestion (ingest.py)**  
- Loads all `.txt` and `.md` files from `docs/`  
- Splits text into overlapping chunks  
- Generates embeddings using `all-MiniLM-L6-v2`  
- Stores chunks + metadata in ChromaDB  

Metadata includes:

- filename  
- chunk_index  
- chunk_size  
- overlap  

### **2. Search (search.py)**  
- Embeds the user query  
- Retrieves top‑N nearest chunks  
- Filters by distance threshold  
- Supports source filtering  
- Returns ranked results with:
  - text  
  - source file  
  - chunk index  
  - distance score  
  - similarity score  

### **3. Streamlit UI (app.py)**  
Features:

- Search bar  
- Sidebar controls:
  - number of results  
  - similarity threshold  
  - source file filter  
  - re-index button  
  - collection metrics  
- Edge case handling:
  - empty queries  
  - non‑word queries  
  - stopwords  
  - short queries  
  - weak matches  
- Expandable chunk previews  

### **4. Evaluation (evaluate.py)**  
Computes:

- Precision  
- Recall  
- Per‑query results  
- Average scores  

Used for the chunking experiment.

---

## 📊 **Chunking Experiment**

A core part of the project is comparing different chunk sizes.

### **Chunk Sizes Tested**
- **150 characters**  
- **300 characters**  
- **500 characters** (default)

### **Queries Used**
(From `evaluate.py`)

- “What does cosine similarity measure?”  
- “When do background tasks run?”  
- “What is a hallucination?”  
- “What is a decorator?”  
- “What is the base class for errors?”  
- “What does it mean to be stateless?”  
- “What does INNER JOIN accomplish?”  
- “What code supports chat interfaces?”  

### **Results Summary (Fill in your actual numbers)**

| Chunk Size | Avg Precision | Avg Recall | Notes |
|-----------|---------------|------------|-------|
| 150       | 0.xx          | 0.xx       | More granular but sometimes too fragmented |
| 300       | 0.xx          | 0.xx       | Balanced, often best overall |
| 500       | 0.xx          | 0.xx       | More context but sometimes less precise |

### **Conclusion (Example wording)**  
In my experiment, **300‑character chunks** produced the best balance of precision and recall. Smaller chunks (150) were more precise for short queries but lost context. Larger chunks (500) preserved context but occasionally returned less relevant results.

This demonstrates the tradeoff between **context size** and **semantic specificity** in embedding‑based retrieval.

---

## 🖥️ **Running the App**

Start the Streamlit interface:

```
streamlit run app.py
```

Features include:

- Semantic search  
- Source filtering  
- Re-indexing  
- Metrics  
- Chunk previews  

---

## 📈 **Evaluation**

Run precision/recall evaluation:

```
python evaluate.py --n-results 5 --threshold 0.4
```

This prints:

- Retrieved chunk IDs  
- Relevant chunk IDs  
- Precision  
- Recall  
- Average scores  

Use this to compare chunk sizes.

---

## 🧩 **Future Improvements**

- Add relevance color badges  
- Add query history  
- Add chunk visualization  
- Add embedding model comparison  
- Add PDF ingestion  

---

## 👤 **Author**

Grant Eberhardt  
Module 7 — Applied AI: Embeddings & Retrieval  
Semantic Search Engine Project

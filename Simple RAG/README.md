# Simple Retrieval-Augmented Generation (RAG) Pipeline

This project implements a fully offline, modular **Retrieval-Augmented Generation (RAG)** pipeline using a custom architecture. The system ingests local multi-page PDF documents, converts them into semantic dense vector representations, indexes them via a fast similarity engine, and injects relevant context into a localized Large Language Model (LLM) to deliver factually grounded responses.

---

## ⚙️ Core Technical Pipeline

The system splits the retrieval and generation workload into decoupled scripts managed under a core source directory:

### 1. Document Ingestion & Embeddings (`src/ingestion.py`)

- **PDF Extraction:** Uses `pypdf.PdfReader` to parse and extract unstructured text from multi-page PDF documents sitting in your local data directory.
- **Sliding Window Chunking:** Breaks down the extracted corpus into precise semantic segments using a sliding window technique (`chunk_size=500` characters with an `overlap=50` character threshold) to maintain textual context across boundaries.
- **Vector Transformation:** Leverages `sentence-transformers` to load the efficient `all-MiniLM-L6-v2` model, transforming each text chunk into a dense **384-dimensional embedding vector**.

### 2. Semantic Indexing & Retrieval (`src/retrieval.py`)

- **Vector Database Setup:** Builds an in-memory `faiss.IndexFlatL2` index to measure exact Euclidean (L2) distances across the 384-dimensional space.
- **Top-K Search Optimization:** Translates incoming runtime user query strings into embeddings using the same encoder model, querying the FAISS index to extract the top-$K$ closest matching textual contexts (`k=2`).

### 3. Orchestration & Generative Prompting (`main.py`)

- **SSL Certificate Routing:** Configures pathing environments with `certifi` to ensure local system verification routes safely.
- **Prompt Augmentation:** Automatically wraps the recovered context passages and the original question into an explicit guardrailed instruction set:
  > _"Use the following context to answer the user's question. If you do not know the answer based on the context, say that you don't know."_
- **LLM Generation Backend:** Dispatches the augmented instruction to a local `ollama` deployment (running models like `llama3`) to synthesize an accurate response.

---

## 📂 Project Architecture

```text
Simple_RAG/
├── main.py                  # Orchestration script (Ingestion -> Indexing -> LLM Prompting)
├── requirements.txt         # Required system libraries
│
├── src/                     # Core production source module
│   ├── ingestion.py         # PDF text extraction and sentence-transformer vector generation
│   └── retrieval.py         # FAISS index initialization and semantic similarity search
│
└── data/                    # Storage directory for source documentation
    └── target_files.pdf     # Place your custom background PDFs here
```

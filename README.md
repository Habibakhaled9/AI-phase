# Session 2 — Basic RAG Implementation

This branch contains a basic RAG pipeline built using:

- OpenRouter embedding model
- PDF text extraction and chunking
- FAISS vector search
- Qwen 2.5 through Ollama for answer generation

## Pipeline

1. `create_embeddings.py` extracts and embeds PDF chunks.
2. `build_faiss_index.py` stores the vectors in FAISS.
3. `query_faiss.py` tests retrieval.
4. `normal_rag_query.py` generates answers using the retrieved context.

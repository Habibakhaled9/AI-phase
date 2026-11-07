"""
Vector Store module using ChromaDB and Ollama nomic-embed-text embeddings.

Persists chunks and semantic vectors into local storage.
Provides cosine similarity search and document collection management.
"""

import logging
from typing import Any, Dict, List, Optional
import chromadb
from chromadb.config import Settings as ChromaSettings

from graph_rag.config import settings
from graph_rag.ingestion.chunker import TextChunk
from graph_rag.llm.ollama_client import ollama_client

logger = logging.getLogger(__name__)

COLLECTION_NAME = "research_papers"


class VectorStore:
    """Manages persistent ChromaDB vector storage and semantic retrieval."""

    def __init__(self, persist_dir: Optional[str] = None) -> None:
        self.persist_dir = str(persist_dir or settings.CHROMA_PERSIST_DIR)
        self.client = chromadb.PersistentClient(path=self.persist_dir)
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    def add_chunks(self, chunks: List[TextChunk]) -> int:
        """
        Compute embeddings for chunks and store them in ChromaDB.
        Returns the number of chunks added.
        """
        if not chunks:
            return 0

        ids: List[str] = []
        documents: List[str] = []
        metadatas: List[Dict[str, Any]] = []

        for chunk in chunks:
            ids.append(chunk.chunk_id)
            documents.append(chunk.text)
            # Ensure metadata values are strings, ints, floats, or bools for ChromaDB
            clean_meta = {}
            for k, v in chunk.metadata.items():
                if isinstance(v, (str, int, float, bool)):
                    clean_meta[k] = v
                else:
                    clean_meta[k] = str(v)
            clean_meta["document_name"] = chunk.document_name
            clean_meta["document_title"] = chunk.document_title
            clean_meta["page"] = chunk.page_number
            clean_meta["chunk_index"] = chunk.chunk_index
            metadatas.append(clean_meta)

        # Batch embed using Ollama nomic-embed-text
        embeddings = ollama_client.get_embeddings_batch(documents)

        # Upsert into ChromaDB
        self.collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )
        return len(chunks)

    def search(
        self,
        query: str,
        n_results: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Perform semantic similarity search for a query text.
        Returns a ranked list of matched chunks with scores and metadata.
        """
        query_embedding = ollama_client.get_embedding(query)

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=filter_metadata,
            include=["documents", "metadatas", "distances"],
        )

        formatted_results: List[Dict[str, Any]] = []
        if results and results.get("ids") and results["ids"][0]:
            count = len(results["ids"][0])
            for i in range(count):
                # Cosine distance to similarity: similarity = 1 - distance
                distance = results["distances"][0][i] if results.get("distances") else 0.0
                similarity_score = max(0.0, 1.0 - distance)

                formatted_results.append(
                    {
                        "chunk_id": results["ids"][0][i],
                        "text": results["documents"][0][i],
                        "metadata": results["metadatas"][0][i],
                        "similarity_score": round(similarity_score, 4),
                        "distance": round(distance, 4),
                    }
                )

        return formatted_results

    def get_total_chunks(self) -> int:
        """Return total number of chunks currently stored in ChromaDB."""
        return self.collection.count()

    def delete_document_chunks(self, document_name: str) -> None:
        """Delete all chunks belonging to a specific document."""
        self.collection.delete(where={"document_name": document_name})


vector_store = VectorStore()

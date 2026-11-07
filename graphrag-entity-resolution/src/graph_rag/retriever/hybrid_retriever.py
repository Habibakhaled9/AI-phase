"""
Hybrid Retriever module.

Combines:
1. Dense Vector Retrieval (ChromaDB + nomic-embed-text)
2. Knowledge Graph Traversal (Neo4j entity relations)

Merges context into a unified, attributed context prompt for LLM answer generation.
"""

import re
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from graph_rag.embeddings.vector_store import vector_store
from graph_rag.graph.graph_queries import graph_queries

logger = logging.getLogger(__name__)


@dataclass
class RetrievedContext:
    """Consolidated context retrieved from vector store and knowledge graph."""
    chunks: List[Dict[str, Any]] = field(default_factory=list)
    graph_facts: List[Dict[str, Any]] = field(default_factory=list)
    assembled_text: str = ""
    sources: List[str] = field(default_factory=list)


class HybridRetriever:
    """Retrieves relevant passages and structured relations for a user query."""

    def __init__(self) -> None:
        self.vector_db = vector_store
        self.graph_q = graph_queries

    def retrieve(self, query: str, top_k: int = 5) -> RetrievedContext:
        """
        Execute hybrid search across ChromaDB and Neo4j.
        """
        # 1. Vector Search
        chunks = []
        try:
            chunks = self.vector_db.search(query=query, n_results=top_k)
        except Exception as e:
            logger.warning("Vector retrieval failed: %s", e)

        # 2. Graph Search
        graph_facts = []
        try:
            # Extract keywords (> 3 chars, alphanumeric)
            keywords = [w for w in re.findall(r"\b[A-Za-z0-9_-]{3,}\b", query) if not w.lower() in {
                "what", "when", "where", "which", "explain", "describe", "show", "tell", "paper", "research"
            }]

            for kw in keywords[:3]:
                entities = self.graph_q.search_entities(keyword=kw, limit=3)
                for ent in entities:
                    subgraph = self.graph_q.find_entity_subgraph(ent["name"])
                    graph_facts.extend(subgraph)
        except Exception as e:
            logger.warning("Graph retrieval failed: %s", e)

        # 3. Assemble unified context with source attribution
        context_parts = []
        sources = set()

        # Add graph facts if available
        if graph_facts:
            context_parts.append("### Structured Knowledge Graph Relationships:")
            for fact in graph_facts[:8]:
                src = fact.get("source", "")
                rel = fact.get("relation", "")
                tgt = fact.get("target", "")
                context_parts.append(f"- ({src}) --[{rel}]--> ({tgt})")
            context_parts.append("")

        # Add text chunks
        if chunks:
            context_parts.append("### Relevant Research Excerpts:")
            for idx, c in enumerate(chunks, start=1):
                doc_name = c.get("metadata", {}).get("document_name", "Unknown Paper")
                doc_title = c.get("metadata", {}).get("document_title", doc_name)
                page = c.get("metadata", {}).get("page", 1)
                text = c.get("text", "")
                sources.add(f"{doc_title} (Page {page})")
                context_parts.append(f"[{idx}] Source: {doc_title} (File: {doc_name}, Page {page}):\n{text}\n")
        else:
            if not graph_facts:
                context_parts.append("No specific documents matched the query in the knowledge base.")

        assembled_text = "\n".join(context_parts)

        return RetrievedContext(
            chunks=chunks,
            graph_facts=graph_facts,
            assembled_text=assembled_text,
            sources=sorted(list(sources)),
        )


hybrid_retriever = HybridRetriever()

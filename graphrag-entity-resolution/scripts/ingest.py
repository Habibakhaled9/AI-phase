"""
Batch Ingestion CLI script.

Indexes the documents in the `pdfs/` folder into ChromaDB and Neo4j.
Enforces deduplication: documents already indexed in `pdfs/metadata.csv` are skipped.

Usage:
    uv run python scripts/ingest.py [--limit N]
"""

import argparse
import sys
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from graph_rag.config import settings
from graph_rag.logging_config import configure_logging
from graph_rag.database.db import init_db
from graph_rag.ingestion.pipeline import pipeline
from graph_rag.ingestion.metadata_manager import metadata_manager
from graph_rag.embeddings.vector_store import vector_store
from graph_rag.graph.graph_builder import graph_builder


def main() -> None:
    configure_logging()
    parser = argparse.ArgumentParser(description="Graph RAG Ingestion CLI")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of PDFs to process")
    args = parser.parse_args()

    print("=" * 65)
    print("Graph RAG Ingestion Pipeline")
    print(f"PDF Directory:    {settings.PDF_DIR}")
    print(f"Metadata File:    {settings.METADATA_FILE}")
    print(f"ChromaDB Path:    {settings.CHROMA_PERSIST_DIR}")
    print(f"Neo4j URI:        {settings.NEO4J_URI}")
    print("=" * 65)

    # Initialize DB schemas
    init_db()

    # Pre-check Neo4j status
    neo4j_online = graph_builder.is_connected()
    print(f"Neo4j Status:     {'[ONLINE]' if neo4j_online else '[OFFLINE - will store in ChromaDB only]'}")

    def progress_callback(current: int, total: int, msg: str) -> None:
        print(f"[{current}/{total}] {msg}", flush=True)

    print("\nStarting batch ingestion...", flush=True)
    results = pipeline.ingest_batch_folder(progress_callback=progress_callback, limit=args.limit)

    completed = [r for r in results if r.status == "completed"]
    skipped = [r for r in results if r.status == "skipped_duplicate"]
    failed = [r for r in results if r.status == "failed"]

    print("\n" + "=" * 65)
    print("Ingestion Summary")
    print(f"  - Completed: {len(completed)}")
    print(f"  - Skipped (already indexed): {len(skipped)}")
    print(f"  - Failed:    {len(failed)}")
    print(f"  - Total chunks in ChromaDB: {vector_store.get_total_chunks()}")
    if neo4j_online:
        stats = graph_builder.get_graph_stats()
        print(f"  - Neo4j graph nodes: {stats.get('nodes', 0)}, edges: {stats.get('edges', 0)}")
    print("=" * 65)


if __name__ == "__main__":
    main()

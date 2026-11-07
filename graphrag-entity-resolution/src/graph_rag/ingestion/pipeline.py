"""
Ingestion Pipeline module.

Orchestrates full document ingestion with two core modes:
- Mode A (Batch / Startup): Indexes pdfs/ folder into ChromaDB and Neo4j.
- Mode B (Dynamic Upload): User uploads documents via Streamlit UI.
  Strictly enforces:
  - In-memory duplicate verification before saving to disk.
  - If duplicate: REJECT (do not save to disk, do not run backend chunking/embeddings/graph).
  - If new: save to pdfs/, run ingestion, append to metadata.csv, log in SQLite.
"""

import logging
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

from graph_rag.config import settings
from graph_rag.database.db import get_db_connection
from graph_rag.embeddings.vector_store import vector_store
from graph_rag.graph.entity_extractor import entity_extractor
from graph_rag.graph.graph_builder import graph_builder
from graph_rag.ingestion.archive_handler import archive_handler
from graph_rag.ingestion.chunker import chunker
from graph_rag.ingestion.dedup_checker import compute_hash, dedup_checker
from graph_rag.ingestion.file_handler import file_handler
from graph_rag.ingestion.metadata_manager import metadata_manager
from graph_rag.graph.entity_resolver import entity_resolver

logger = logging.getLogger(__name__)


@dataclass
class IngestionResult:
    """Detailed summary of an ingestion operation."""
    filename: str
    status: str  # 'completed', 'skipped_duplicate', 'failed'
    title: str = ""
    is_duplicate: bool = False
    match_reason: str = ""
    num_chunks: int = 0
    file_hash: str = ""
    error_message: str = ""


class IngestionPipeline:
    """Unified document ingestion pipeline."""

    def __init__(self) -> None:
        self.meta_mgr = metadata_manager
        self.dedup = dedup_checker
        self.vector_db = vector_store
        self.graph_db = graph_builder

    # ── Mode A: Batch Ingestion for pdfs/ folder ──
    def ingest_batch_folder(
        self,
        folder_path: Optional[Path] = None,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
        limit: Optional[int] = None,
    ) -> List[IngestionResult]:
        """
        Run initial batch ingestion over pdfs/ folder.
        Idempotent: Skips any paper already marked as ingested.
        """
        folder = folder_path or settings.PDF_DIR
        pdf_files = sorted(list(folder.glob("*.pdf")))
        if limit and limit > 0:
            pdf_files = pdf_files[:limit]
        total_files = len(pdf_files)
        results: List[IngestionResult] = []

        # Load existing metadata records to check ingestion status
        existing_records = {
            Path(r.get("local_pdf", "")).name.lower(): r
            for r in self.meta_mgr.get_all_records()
            if r.get("local_pdf")
        }

        # Ensure Neo4j constraints
        self.graph_db.init_constraints()

        for idx, pdf_file in enumerate(pdf_files, start=1):
            filename = pdf_file.name
            fname_lower = filename.lower()
            rec = existing_records.get(fname_lower, {})

            # Check if already processed
            already_ingested = bool(rec.get("ingested_at") and rec.get("file_hash"))
            if already_ingested:
                results.append(
                    IngestionResult(
                        filename=filename,
                        status="skipped_duplicate",
                        title=rec.get("title", filename),
                        is_duplicate=True,
                        match_reason="Already indexed in knowledge base",
                        file_hash=rec.get("file_hash", ""),
                    )
                )
                if progress_callback:
                    progress_callback(idx, total_files, f"Skipped (already indexed): {filename}")
                continue

            if progress_callback:
                progress_callback(idx, total_files, f"Ingesting: {filename}")

            try:
                # 1. Compute hash
                file_hash = compute_hash(pdf_file)

                # 2. Parse text
                parsed_doc = file_handler.parse(pdf_file, filename=filename)
                doc_title = rec.get("title") or parsed_doc.title
                if progress_callback:
                    progress_callback(idx, total_files, f"Parsed: {filename}")

                # 3. Chunk text
                chunks = chunker.chunk_document(parsed_doc)
                if progress_callback:
                    progress_callback(idx, total_files, f"Created {len(chunks)} chunks: {filename}")

                # 4. Store in ChromaDB
                self.vector_db.add_chunks(chunks)
                if progress_callback:
                    progress_callback(idx, total_files, f"Embedded and stored {len(chunks)} chunks: {filename}")

                # 5. Extract entities and populate Neo4j Graph
                self.graph_db.add_document_node(
                    document_name=filename,
                    title=doc_title,
                    metadata={"topic": rec.get("topic", "Research Paper"), "source": "initial"},
                )

                # Extract entities from one representative chunk to keep batch ingestion bounded.
                for chunk in chunks:
                    graph_result = entity_extractor.extract(
                        chunk.text,
                        document_title=doc_title,
                    )

                    self.graph_db.add_entities_and_relations(
                        filename,
                        graph_result,
                    )

                    entity_resolver.resolve_document_authors(
                        document_name=filename,
                        result=graph_result,
                    )
                if progress_callback:
                    progress_callback(idx, total_files, f"Updated Neo4j graph: {filename}")

                # 6. Update metadata.csv
                self.meta_mgr.mark_as_ingested(local_pdf=filename, file_hash=file_hash)

                results.append(
                    IngestionResult(
                        filename=filename,
                        status="completed",
                        title=doc_title,
                        is_duplicate=False,
                        num_chunks=len(chunks),
                        file_hash=file_hash,
                    )
                )
            except Exception as e:
                logger.error(f"Failed to ingest {filename}: {e}")
                results.append(
                    IngestionResult(
                        filename=filename,
                        status="failed",
                        title=rec.get("title", filename),
                        error_message=str(e),
                    )
                )

        return results

    # ── Mode B: Dynamic User Upload Ingestion ──
    def ingest_uploaded_file(
        self,
        file_bytes: bytes,
        filename: str,
        user_id: Optional[int] = None,
        source_archive: Optional[str] = None,
    ) -> IngestionResult:
        """
        Process a user-uploaded file.
        STRICT POLICY:
        - In-memory check against metadata.csv first.
        - If duplicate: DO NOT save to disk, DO NOT run backend processing.
        - If new: save to pdfs/, run pipeline, update metadata.csv & SQLite.
        """
        # Step 1: In-memory hash & deduplication check
        file_hash = compute_hash(file_bytes)
        dedup_res = self.dedup.check_bytes(file_bytes=file_bytes, filename=filename)

        if dedup_res.is_duplicate:
            logger.info(f"Duplicate upload rejected: {filename} ({dedup_res.message})")

            # Log rejection in SQLite for audit history
            try:
                with get_db_connection() as conn:
                    conn.execute(
                        """
                        INSERT INTO uploaded_documents
                        (user_id, filename, file_type, file_size_bytes, file_hash, source_archive, status, error_message)
                        VALUES (?, ?, ?, ?, ?, ?, 'duplicate_skipped', ?)
                        """,
                        (
                            user_id,
                            filename,
                            Path(filename).suffix.lower().lstrip("."),
                            len(file_bytes),
                            file_hash,
                            source_archive,
                            dedup_res.message,
                        ),
                    )
            except Exception as log_err:
                logger.warning(f"Could not log upload rejection: {log_err}")

            return IngestionResult(
                filename=filename,
                status="skipped_duplicate",
                title=dedup_res.matched_doc.get("title", filename) if dedup_res.matched_doc else filename,
                is_duplicate=True,
                match_reason=dedup_res.message,
                file_hash=file_hash,
            )

        # Step 2: Genuine New File -> Save into pdfs/ folder
        target_path = settings.PDF_DIR / filename
        # Ensure no name collision with existing non-identical file
        if target_path.exists():
            stem = Path(filename).stem
            suffix = Path(filename).suffix
            filename = f"{stem}_{file_hash[:6]}{suffix}"
            target_path = settings.PDF_DIR / filename

        try:
            with open(target_path, "wb") as f:
                f.write(file_bytes)

            # Step 3: Parse Document
            parsed_doc = file_handler.parse(target_path, filename=filename)
            doc_title = parsed_doc.title

            # Step 4: Chunk Text
            chunks = chunker.chunk_document(parsed_doc)

            # Step 5: Embed & Upsert into ChromaDB
            self.vector_db.add_chunks(chunks)

            # Step 6: Build Knowledge Graph in Neo4j
            self.graph_db.add_document_node(
                document_name=filename,
                title=doc_title,
                metadata={"topic": "User Upload", "source": "upload"},
            )
            for c in chunks[:3]:
                graph_result = entity_extractor.extract(
                    c.text,
                    document_title=doc_title,
                )
                self.graph_db.add_entities_and_relations(
                    filename,
                    graph_result,
                )

            # Step 7: Append to metadata.csv
            new_meta_row = {
                "topic": "User Upload",
                "title": doc_title,
                "authors": parsed_doc.metadata.get("author", "User Upload"),
                "published": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
                "arxiv_id": "",
                "abstract": parsed_doc.full_text[:300] + "..." if len(parsed_doc.full_text) > 300 else parsed_doc.full_text,
                "abstract_url": "",
                "pdf_url": "",
                "categories": "user-upload",
                "local_pdf": filename,
                "source": "upload",
                "file_hash": file_hash,
                "ingested_at": datetime.now(timezone.utc).isoformat(),
            }
            self.meta_mgr.append_document(new_meta_row)

            # Step 8: Log success in SQLite
            with get_db_connection() as conn:
                conn.execute(
                    """
                    INSERT INTO uploaded_documents
                    (user_id, filename, file_type, file_size_bytes, file_hash, source_archive, num_chunks, status)
                    VALUES (?, ?, ?, ?, ?, ?, ?, 'completed')
                    """,
                    (
                        user_id,
                        filename,
                        Path(filename).suffix.lower().lstrip("."),
                        len(file_bytes),
                        file_hash,
                        source_archive,
                        len(chunks),
                    ),
                )

            return IngestionResult(
                filename=filename,
                status="completed",
                title=doc_title,
                is_duplicate=False,
                num_chunks=len(chunks),
                file_hash=file_hash,
            )

        except Exception as e:
            logger.error(f"Failed to ingest uploaded file {filename}: {e}")
            # Clean up saved file if ingestion aborted
            if target_path.exists():
                try:
                    target_path.unlink()
                except Exception:
                    pass

            # Log failure in SQLite
            try:
                with get_db_connection() as conn:
                    conn.execute(
                        """
                        INSERT INTO uploaded_documents
                        (user_id, filename, file_type, file_size_bytes, file_hash, source_archive, status, error_message)
                        VALUES (?, ?, ?, ?, ?, ?, 'failed', ?)
                        """,
                        (
                            user_id,
                            filename,
                            Path(filename).suffix.lower().lstrip("."),
                            len(file_bytes),
                            file_hash,
                            source_archive,
                            str(e),
                        ),
                    )
            except Exception:
                pass

            return IngestionResult(
                filename=filename,
                status="failed",
                is_duplicate=False,
                error_message=str(e),
                file_hash=file_hash,
            )

    # ── Mode B Helper: Archive Upload Handler ──
    def ingest_uploaded_archive(
        self,
        archive_bytes: bytes,
        archive_name: str,
        user_id: Optional[int] = None,
    ) -> List[IngestionResult]:
        """
        Unpack an archive in a secure temporary sandbox and process all contained files.
        Cleans up sandbox after execution.
        """
        results: List[IngestionResult] = []
        try:
            extracted_files = archive_handler.extract(archive_bytes, archive_name=archive_name)
            for ef in extracted_files:
                with open(ef.path, "rb") as f:
                    file_bytes = f.read()
                res = self.ingest_uploaded_file(
                    file_bytes=file_bytes,
                    filename=ef.filename,
                    user_id=user_id,
                    source_archive=archive_name,
                )
                results.append(res)
        finally:
            # Clean up all temp directories in UPLOAD_TEMP_DIR
            for sandbox in settings.UPLOAD_TEMP_DIR.glob("sandbox_*"):
                archive_handler.cleanup_sandbox(sandbox)

        return results


pipeline = IngestionPipeline()

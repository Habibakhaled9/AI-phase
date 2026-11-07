"""
Deduplication Checker module.

Enforces strict duplicate detection against `pdfs/metadata.csv`.
If a document already exists:
- 🚫 DO NOT SAVE TO DISK
- 🚫 DO NOT APPLY BACKEND PROCESSING
"""

import hashlib
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, Optional, Union

from graph_rag.ingestion.metadata_manager import MetadataManager, metadata_manager


@dataclass
class DedupResult:
    """Result of deduplication check."""
    is_duplicate: bool
    match_type: Optional[str] = None  # 'hash', 'filename', 'title'
    matched_doc: Optional[Dict[str, Any]] = None
    message: str = ""
    file_hash: str = ""


def compute_hash(data: Union[bytes, Path, str]) -> str:
    """
    Compute SHA-256 hash of bytes or file contents.
    Works directly in-memory on bytes before writing to disk.
    """
    hasher = hashlib.sha256()
    if isinstance(data, bytes):
        hasher.update(data)
    else:
        path = Path(data)
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                hasher.update(chunk)
    return hasher.hexdigest().lower()


def normalize_title(title: str) -> str:
    """Clean and normalize a title for reliable comparison."""
    if not title:
        return ""
    # Lowercase, remove non-alphanumeric, collapse whitespace
    cleaned = re.sub(r"[^a-zA-Z0-9\s]", "", title.lower())
    return " ".join(cleaned.split())


class DedupChecker:
    """3-layer duplicate checker against metadata.csv."""

    def __init__(self, manager: Optional[MetadataManager] = None) -> None:
        self.manager = manager or metadata_manager

    def check_bytes(
        self,
        file_bytes: bytes,
        filename: str,
        extracted_title: Optional[str] = None,
    ) -> DedupResult:
        """
        Check in-memory file bytes before saving anything to disk.
        """
        file_hash = compute_hash(file_bytes)
        return self._check(file_hash=file_hash, filename=filename, extracted_title=extracted_title)

    def check_file(
        self,
        file_path: Union[str, Path],
        extracted_title: Optional[str] = None,
    ) -> DedupResult:
        """Check an existing file path."""
        path = Path(file_path)
        file_hash = compute_hash(path)
        return self._check(file_hash=file_hash, filename=path.name, extracted_title=extracted_title)

    def _check(
        self,
        file_hash: str,
        filename: str,
        extracted_title: Optional[str] = None,
    ) -> DedupResult:
        """Execute 3-layer check."""
        clean_filename = Path(filename).name.strip().lower()

        # ── Level 1: File Content Hash (Exact Match) ──
        matched = self.manager.find_by_hash(file_hash)
        if matched:
            title = matched.get("title") or matched.get("local_pdf", "Unknown")
            return DedupResult(
                is_duplicate=True,
                match_type="hash",
                matched_doc=matched,
                message=f"Duplicate content detected: matches existing paper '{title}' (Hash: {file_hash[:12]}...).",
                file_hash=file_hash,
            )

        # ── Level 2: Filename Match ──
        matched = self.manager.find_by_filename(clean_filename)
        if matched:
            title = matched.get("title") or matched.get("local_pdf", clean_filename)
            return DedupResult(
                is_duplicate=True,
                match_type="filename",
                matched_doc=matched,
                message=f"Duplicate filename detected: '{clean_filename}' is already registered as '{title}'.",
                file_hash=file_hash,
            )

        # ── Level 3: Title Similarity Match ──
        if extracted_title:
            norm_query_title = normalize_title(extracted_title)
            if len(norm_query_title) >= 10:  # Only check substantial titles
                for record in self.manager.get_all_records():
                    rec_title = record.get("title", "")
                    norm_rec_title = normalize_title(rec_title)
                    if not norm_rec_title:
                        continue

                    # Exact normalized match
                    if norm_query_title == norm_rec_title:
                        return DedupResult(
                            is_duplicate=True,
                            match_type="title",
                            matched_doc=record,
                            message=f"Duplicate paper title detected: matches '{rec_title}'.",
                            file_hash=file_hash,
                        )

                    # Fuzzy similarity ratio >= 0.90
                    ratio = SequenceMatcher(None, norm_query_title, norm_rec_title).ratio()
                    if ratio >= 0.90:
                        return DedupResult(
                            is_duplicate=True,
                            match_type="title",
                            matched_doc=record,
                            message=f"Duplicate paper detected: title matches '{rec_title}' ({int(ratio * 100)}% similarity).",
                            file_hash=file_hash,
                        )

        # File is genuinely new!
        return DedupResult(
            is_duplicate=False,
            message="Document is unique and ready for ingestion.",
            file_hash=file_hash,
        )


# Global singleton instance
dedup_checker = DedupChecker()

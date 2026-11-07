"""
Metadata Manager module.

Serves as the single source of truth for all documents in the system.
Maintains and updates `pdfs/metadata.csv` with tracking columns:
- topic, title, authors, published, arxiv_id, abstract, abstract_url, pdf_url, categories, local_pdf
- source: 'initial' | 'upload'
- file_hash: SHA-256 hash of the document content
- ingested_at: ISO timestamp of ingestion
"""

import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
import pandas as pd

from graph_rag.config import settings

# Re-entrant lock because update methods call the locked read helpers.
_csv_lock = threading.RLock()

EXPECTED_COLUMNS = [
    "topic",
    "title",
    "authors",
    "published",
    "arxiv_id",
    "abstract",
    "abstract_url",
    "pdf_url",
    "categories",
    "local_pdf",
    "source",
    "file_hash",
    "ingested_at",
]


class MetadataManager:
    """Manages reading, querying, and appending to pdfs/metadata.csv."""

    def __init__(self, metadata_path: Optional[Path] = None) -> None:
        self.metadata_path = metadata_path or settings.METADATA_FILE
        self._ensure_file_and_columns()

    def _ensure_file_and_columns(self) -> None:
        """Ensure metadata.csv exists and has all required schema columns."""
        with _csv_lock:
            if not self.metadata_path.exists():
                self.metadata_path.parent.mkdir(parents=True, exist_ok=True)
                df = pd.DataFrame(columns=EXPECTED_COLUMNS)
                df.to_csv(self.metadata_path, index=False)
                return

            try:
                df = pd.read_csv(self.metadata_path)
            except Exception:
                df = pd.DataFrame(columns=EXPECTED_COLUMNS)

            # Ensure tracking columns exist
            modified = False
            for col in EXPECTED_COLUMNS:
                if col not in df.columns:
                    df[col] = ""
                    modified = True

            # Populate default source for initial documents if empty
            if "source" in df.columns:
                mask = df["source"].isna() | (df["source"] == "")
                if mask.any():
                    df.loc[mask, "source"] = "initial"
                    modified = True

            if modified:
                df.to_csv(self.metadata_path, index=False)

    def load_metadata(self) -> pd.DataFrame:
        """Return the current metadata dataframe."""
        with _csv_lock:
            if not self.metadata_path.exists():
                return pd.DataFrame(columns=EXPECTED_COLUMNS)
            df = pd.read_csv(self.metadata_path)
            # Fill NaNs with empty string for string columns
            return df.fillna("")

    def get_known_hashes(self) -> Set[str]:
        """Return set of all non-empty SHA-256 hashes present in registry."""
        df = self.load_metadata()
        if "file_hash" not in df.columns:
            return set()
        hashes = set(df["file_hash"].dropna().astype(str).str.strip().str.lower())
        hashes.discard("")
        return hashes

    def get_known_filenames(self) -> Set[str]:
        """Return set of all registered local filenames (case-insensitive)."""
        df = self.load_metadata()
        if "local_pdf" not in df.columns:
            return set()
        names = set(df["local_pdf"].dropna().astype(str).str.strip().str.lower())
        names.discard("")
        return names

    def get_all_records(self) -> List[Dict[str, Any]]:
        """Return list of all registered document records as dictionaries."""
        df = self.load_metadata()
        return df.to_dict(orient="records")

    def find_by_hash(self, file_hash: str) -> Optional[Dict[str, Any]]:
        """Find a document by exact SHA-256 hash."""
        clean_hash = file_hash.strip().lower()
        if not clean_hash:
            return None
        df = self.load_metadata()
        match = df[df["file_hash"].astype(str).str.strip().str.lower() == clean_hash]
        if not match.empty:
            return match.iloc[0].to_dict()
        return None

    def find_by_filename(self, filename: str) -> Optional[Dict[str, Any]]:
        """Find a document by filename (local_pdf)."""
        clean_name = Path(filename).name.strip().lower()
        if not clean_name:
            return None
        df = self.load_metadata()
        match = df[df["local_pdf"].astype(str).str.strip().str.lower() == clean_name]
        if not match.empty:
            return match.iloc[0].to_dict()
        return None

    def append_document(self, doc_data: Dict[str, Any]) -> None:
        """Safely append a newly ingested document to metadata.csv."""
        with _csv_lock:
            df = self.load_metadata()
            new_row = {col: doc_data.get(col, "") for col in EXPECTED_COLUMNS}
            if not new_row.get("source"):
                new_row["source"] = "upload"
            if not new_row.get("ingested_at"):
                new_row["ingested_at"] = datetime.now(timezone.utc).isoformat()

            new_df = pd.DataFrame([new_row])
            updated_df = pd.concat([df, new_df], ignore_index=True)
            updated_df.to_csv(self.metadata_path, index=False)

    def mark_as_ingested(self, local_pdf: str, file_hash: str) -> None:
        """Update existing row with hash and ingested_at timestamp."""
        with _csv_lock:
            df = self.load_metadata()
            target_name = Path(local_pdf).name.strip().lower()
            mask = df["local_pdf"].astype(str).str.strip().str.lower() == target_name
            if mask.any():
                df.loc[mask, "file_hash"] = file_hash.strip().lower()
                df.loc[mask, "ingested_at"] = datetime.now(timezone.utc).isoformat()
                df.to_csv(self.metadata_path, index=False)

    def get_document_count(self) -> int:
        """Return total count of documents registered."""
        df = self.load_metadata()
        return len(df)


# Global singleton instance
metadata_manager = MetadataManager()

"""
Archive Handler module.

Safely detects and extracts compressed archives (ZIP, RAR, 7z, TAR/GZ)
into temporary sandbox directories and retrieves supported documents.
Prevents Zip Slip / path traversal vulnerabilities.
"""

import io
import shutil
import tarfile
import tempfile
import uuid
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Union

from graph_rag.config import settings

# Supported document extensions to extract from archives
SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".pptx", ".txt", ".md"}


@dataclass
class ExtractedFile:
    """Represents a document extracted from an archive."""
    path: Path
    filename: str
    original_archive: str
    relative_path: str


class ArchiveHandler:
    """Manages archive detection, extraction, and safe cleanup."""

    @staticmethod
    def is_archive(filename: str) -> bool:
        """Check if filename has a recognized archive extension."""
        name = filename.lower()
        return (
            name.endswith(".zip")
            or name.endswith(".rar")
            or name.endswith(".7z")
            or name.endswith(".tar")
            or name.endswith(".tar.gz")
            or name.endswith(".tgz")
        )

    def extract(
        self,
        source: Union[str, Path, bytes],
        archive_name: str = "archive.zip",
    ) -> List[ExtractedFile]:
        """
        Extract supported documents from an archive into a temporary sandbox.
        Returns a list of ExtractedFile objects.
        """
        sandbox_id = f"sandbox_{uuid.uuid4().hex[:10]}"
        sandbox_dir = settings.UPLOAD_TEMP_DIR / sandbox_id
        sandbox_dir.mkdir(parents=True, exist_ok=True)

        try:
            name_lower = archive_name.lower()
            if name_lower.endswith(".zip"):
                self._extract_zip(source, sandbox_dir)
            elif name_lower.endswith(".tar") or name_lower.endswith(".tar.gz") or name_lower.endswith(".tgz"):
                self._extract_tar(source, sandbox_dir)
            elif name_lower.endswith(".7z"):
                self._extract_7z(source, sandbox_dir)
            elif name_lower.endswith(".rar"):
                self._extract_rar(source, sandbox_dir)
            else:
                raise ValueError(f"Unsupported archive format: {archive_name}")

            # Collect supported documents
            extracted: List[ExtractedFile] = []
            for item in sandbox_dir.rglob("*"):
                if item.is_file() and item.suffix.lower() in SUPPORTED_EXTENSIONS:
                    # Ignore hidden or macOS metadata files (.__MACOSX)
                    if item.name.startswith(".") or "__MACOSX" in str(item):
                        continue
                    rel = str(item.relative_to(sandbox_dir))
                    extracted.append(
                        ExtractedFile(
                            path=item,
                            filename=item.name,
                            original_archive=archive_name,
                            relative_path=rel,
                        )
                    )
            return extracted
        except Exception:
            # Clean up on error
            shutil.rmtree(sandbox_dir, ignore_errors=True)
            raise

    def _extract_zip(self, source: Union[str, Path, bytes], dest: Path) -> None:
        """Safely extract ZIP archive avoiding directory traversal."""
        file_obj = io.BytesIO(source) if isinstance(source, bytes) else str(source)
        with zipfile.ZipFile(file_obj, "r") as z:
            dest_resolved = dest.resolve()
            for member in z.infolist():
                target = (dest / member.filename).resolve()
                if not str(target).startswith(str(dest_resolved)):
                    raise SecurityError(f"Directory traversal detected in ZIP: {member.filename}")
            z.extractall(dest)

    def _extract_tar(self, source: Union[str, Path, bytes], dest: Path) -> None:
        """Safely extract TAR/GZ archive."""
        if isinstance(source, bytes):
            file_obj = io.BytesIO(source)
            mode = "r:*"
            with tarfile.open(fileobj=file_obj, mode=mode) as t:
                self._safe_tar_extract(t, dest)
        else:
            with tarfile.open(str(source), mode="r:*") as t:
                self._safe_tar_extract(t, dest)

    def _safe_tar_extract(self, tar: tarfile.TarFile, dest: Path) -> None:
        dest_resolved = dest.resolve()
        for member in tar.getmembers():
            target = (dest / member.name).resolve()
            if not str(target).startswith(str(dest_resolved)):
                raise SecurityError(f"Directory traversal detected in TAR: {member.name}")
        tar.extractall(dest)

    def _extract_7z(self, source: Union[str, Path, bytes], dest: Path) -> None:
        """Extract 7z archive using py7zr."""
        import py7zr

        if isinstance(source, bytes):
            file_obj = io.BytesIO(source)
            with py7zr.SevenZipFile(file_obj, mode="r") as z:
                z.extractall(path=dest)
        else:
            with py7zr.SevenZipFile(str(source), mode="r") as z:
                z.extractall(path=dest)

    def _extract_rar(self, source: Union[str, Path, bytes], dest: Path) -> None:
        """Extract RAR archive using rarfile."""
        import rarfile

        if isinstance(source, bytes):
            file_obj = io.BytesIO(source)
            with rarfile.RarFile(file_obj) as rf:
                rf.extractall(path=dest)
        else:
            with rarfile.RarFile(str(source)) as rf:
                rf.extractall(path=dest)

    @staticmethod
    def cleanup_sandbox(sandbox_dir: Path) -> None:
        """Remove temporary archive sandbox directory."""
        if sandbox_dir.exists():
            shutil.rmtree(sandbox_dir, ignore_errors=True)


archive_handler = ArchiveHandler()

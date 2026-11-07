"""
PDF Parser module.

Extracts text, page boundaries, and document metadata from PDF files using PyMuPDF (fitz).
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union
import pymupdf as fitz  # PyMuPDF


@dataclass
class PageContent:
    """Extracted text and metadata for a single page."""
    page_number: int
    text: str


@dataclass
class ParsedDocument:
    """Standardized representation of a parsed document."""
    filename: str
    title: str
    total_pages: int
    full_text: str
    pages: List[PageContent] = field(default_factory=list)
    metadata: Dict[str, str] = field(default_factory=dict)


class PDFParser:
    """Extracts text and metadata from PDF files."""

    def parse(self, source: Union[str, Path, bytes], filename: str = "document.pdf") -> ParsedDocument:
        """Parse a PDF from a file path or in-memory bytes."""
        if isinstance(source, bytes):
            doc = fitz.open(stream=source, filetype="pdf")
        else:
            path = Path(source)
            filename = path.name
            doc = fitz.open(str(path))

        try:
            pages: List[PageContent] = []
            full_text_parts: List[str] = []

            for page_idx in range(len(doc)):
                page = doc[page_idx]
                page_text = page.get_text("text").strip()
                if page_text:
                    pages.append(PageContent(page_number=page_idx + 1, text=page_text))
                    full_text_parts.append(page_text)

            meta = doc.metadata or {}
            extracted_title = meta.get("title", "").strip()

            # If embedded title is absent or generic, attempt to extract from top lines of page 1
            if not extracted_title or extracted_title.lower().endswith(".pdf"):
                if pages:
                    first_lines = [
                        line.strip()
                        for line in pages[0].text.split("\n")
                        if len(line.strip()) > 5
                    ]
                    if first_lines:
                        extracted_title = first_lines[0]

            if not extracted_title:
                extracted_title = Path(filename).stem.replace("_", " ")

            return ParsedDocument(
                filename=filename,
                title=extracted_title,
                total_pages=len(doc),
                full_text="\n\n".join(full_text_parts),
                pages=pages,
                metadata={
                    "author": meta.get("author", ""),
                    "subject": meta.get("subject", ""),
                    "keywords": meta.get("keywords", ""),
                    "creator": meta.get("creator", ""),
                },
            )
        finally:
            doc.close()


# Global parser instance
pdf_parser = PDFParser()

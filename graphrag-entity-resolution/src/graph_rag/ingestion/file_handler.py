"""
File Handler module.

Detects document format and routes to the appropriate parser:
- PDF -> PyMuPDF
- DOCX -> python-docx
- PPTX -> python-pptx
- TXT / MD -> native text reader
"""

from pathlib import Path
from typing import Optional, Union

from graph_rag.ingestion.docx_parser import docx_parser
from graph_rag.ingestion.pdf_parser import PageContent, ParsedDocument, pdf_parser
from graph_rag.ingestion.pptx_parser import pptx_parser


class FileHandler:
    """Routes files to appropriate format-specific parsers."""

    @staticmethod
    def is_supported(filename: str) -> bool:
        """Check if file extension is supported for ingestion."""
        ext = Path(filename).suffix.lower()
        return ext in {".pdf", ".docx", ".pptx", ".txt", ".md"}

    def parse(
        self,
        source: Union[str, Path, bytes],
        filename: Optional[str] = None,
    ) -> ParsedDocument:
        """
        Parse document content into standardized ParsedDocument.
        Accepts either file path or raw bytes.
        """
        if not filename:
            if isinstance(source, (str, Path)):
                filename = Path(source).name
            else:
                filename = "uploaded_file.bin"

        ext = Path(filename).suffix.lower()

        if ext == ".pdf":
            return pdf_parser.parse(source, filename=filename)
        elif ext == ".docx":
            return docx_parser.parse(source, filename=filename)
        elif ext == ".pptx":
            return pptx_parser.parse(source, filename=filename)
        elif ext in {".txt", ".md"}:
            return self._parse_text(source, filename=filename)
        else:
            raise ValueError(f"Unsupported file format: '{ext}'. Supported: .pdf, .docx, .pptx, .txt, .md")

    def _parse_text(self, source: Union[str, Path, bytes], filename: str) -> ParsedDocument:
        """Parse plain text or markdown file."""
        if isinstance(source, bytes):
            text = source.decode("utf-8", errors="replace").strip()
        else:
            with open(source, "r", encoding="utf-8", errors="replace") as f:
                text = f.read().strip()

        lines = [line.strip() for line in text.split("\n") if line.strip()]
        title = lines[0][:120] if lines else Path(filename).stem.replace("_", " ")

        return ParsedDocument(
            filename=filename,
            title=title,
            total_pages=1,
            full_text=text,
            pages=[PageContent(page_number=1, text=text)] if text else [],
            metadata={"format": Path(filename).suffix.lower().lstrip(".")},
        )


file_handler = FileHandler()

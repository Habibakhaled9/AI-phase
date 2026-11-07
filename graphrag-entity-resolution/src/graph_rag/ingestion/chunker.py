"""
Text Chunker module.

Splits parsed documents into overlapping chunks with metadata preservation.
Maintains context window boundaries (~1000 tokens with ~150 token overlap).
"""

import re
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Dict, List
from graph_rag.config import settings
from graph_rag.ingestion.pdf_parser import ParsedDocument


@dataclass
class TextChunk:
    """A semantic chunk of text with document metadata."""
    chunk_id: str
    document_name: str
    document_title: str
    chunk_index: int
    page_number: int
    text: str
    token_count: int
    metadata: Dict[str, Any] = field(default_factory=dict)


class TextChunker:
    """Splits document text into overlapping chunks."""

    def __init__(self, chunk_size: int = None, chunk_overlap: int = None) -> None:
        self.chunk_size = chunk_size or settings.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or settings.CHUNK_OVERLAP

    def chunk_document(self, doc: ParsedDocument) -> List[TextChunk]:
        """
        Chunk a parsed document page by page or full text.
        Prefers page-aware chunking when pages are available.
        """
        chunks: List[TextChunk] = []
        global_chunk_idx = 0

        if doc.pages:
            for page in doc.pages:
                page_chunks = self._chunk_text(
                    text=page.text,
                    doc=doc,
                    page_number=page.page_number,
                    start_idx=global_chunk_idx,
                )
                chunks.extend(page_chunks)
                global_chunk_idx += len(page_chunks)
        else:
            chunks = self._chunk_text(
                text=doc.full_text,
                doc=doc,
                page_number=1,
                start_idx=0,
            )

        return chunks

    def _chunk_text(
        self,
        text: str,
        doc: ParsedDocument,
        page_number: int,
        start_idx: int,
    ) -> List[TextChunk]:
        """Split text by sentences/paragraphs into target chunk size."""
        # Simple whitespace token estimation (~1 word = 1.3 tokens)
        words = text.split()
        if not words:
            return []

        # Target words per chunk based on chunk_size tokens (~0.75 words per token)
        target_words = int(self.chunk_size * 0.75)
        overlap_words = int(self.chunk_overlap * 0.75)

        chunks: List[TextChunk] = []
        idx = 0
        current_idx = start_idx

        while idx < len(words):
            chunk_words = words[idx : idx + target_words]
            chunk_text = " ".join(chunk_words).strip()

            if chunk_text:
                chunk_id = f"{Path(doc.filename).stem}_p{page_number}_c{current_idx}"
                chunks.append(
                    TextChunk(
                        chunk_id=chunk_id,
                        document_name=doc.filename,
                        document_title=doc.title,
                        chunk_index=current_idx,
                        page_number=page_number,
                        text=chunk_text,
                        token_count=int(len(chunk_words) * 1.3),
                        metadata={
                            "document_name": doc.filename,
                            "document_title": doc.title,
                            "page": page_number,
                            "chunk_index": current_idx,
                            **doc.metadata,
                        },
                    )
                )
                current_idx += 1

            idx += max(1, target_words - overlap_words)

        return chunks



chunker = TextChunker()

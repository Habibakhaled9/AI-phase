"""
Ingestion module for Graph RAG.

Provides parsers, deduplication checking, metadata management, and pipeline orchestration.
"""

from graph_rag.ingestion.archive_handler import ArchiveHandler, archive_handler
from graph_rag.ingestion.chunker import TextChunk, TextChunker, chunker
from graph_rag.ingestion.dedup_checker import DedupChecker, DedupResult, dedup_checker
from graph_rag.ingestion.docx_parser import DocxParser, docx_parser
from graph_rag.ingestion.file_handler import FileHandler, file_handler
from graph_rag.ingestion.metadata_manager import MetadataManager, metadata_manager
from graph_rag.ingestion.pdf_parser import PageContent, ParsedDocument, PDFParser, pdf_parser

__all__ = [
    "metadata_manager",
    "MetadataManager",
    "dedup_checker",
    "DedupChecker",
    "DedupResult",
    "pdf_parser",
    "PDFParser",
    "docx_parser",
    "DocxParser",
    "pptx_parser",
    "archive_handler",
    "ArchiveHandler",
    "file_handler",
    "FileHandler",
    "chunker",
    "TextChunker",
    "TextChunk",
    "ParsedDocument",
    "PageContent",
]

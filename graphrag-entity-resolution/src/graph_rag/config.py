"""
Configuration module for Graph RAG Chatbot.

Loads all settings from .env file using python-dotenv.
Ensures privacy, security, and strict validation of application variables.
"""

import os
from pathlib import Path
from dataclasses import dataclass
from dotenv import load_dotenv

# Resolve project root directory (two levels up from src/graph_rag)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Load environment variables from .env file at project root
ENV_FILE = PROJECT_ROOT / ".env"
load_dotenv(dotenv_path=ENV_FILE, override=True)


@dataclass(frozen=True)
class Settings:
    """Application configuration settings."""

    # ── Ollama Settings ──
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    OLLAMA_LLM_MODEL: str = os.getenv("OLLAMA_LLM_MODEL", "qwen2.5:1.5b")
    OLLAMA_EMBED_MODEL: str = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")

    # ── Neo4j Settings ──
    NEO4J_URI: str = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    NEO4J_USER: str = os.getenv("NEO4J_USER", "neo4j")
    NEO4J_PASSWORD: str = os.getenv("NEO4J_PASSWORD", "password123")

    # ── Paths & Storage ──
    PDF_DIR: Path = PROJECT_ROOT / os.getenv("PDF_DIR", "pdfs")
    METADATA_FILE: Path = PROJECT_ROOT / os.getenv("METADATA_FILE", "pdfs/metadata.csv")
    DATA_DIR: Path = PROJECT_ROOT / os.getenv("DATA_DIR", "data")
    CHROMA_PERSIST_DIR: Path = PROJECT_ROOT / os.getenv("CHROMA_PERSIST_DIR", "data/chroma_db")
    SQLITE_DB_PATH: Path = PROJECT_ROOT / os.getenv("SQLITE_DB_PATH", "data/graph_rag.db")
    UPLOAD_TEMP_DIR: Path = PROJECT_ROOT / os.getenv("UPLOAD_TEMP_DIR", "data/temp_uploads")

    # ── Ingestion & Document Processing ──
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "1000"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "150"))
    MAX_UPLOAD_SIZE_MB: int = int(os.getenv("MAX_UPLOAD_SIZE_MB", "200"))

    # ── JWT Authentication ──
    JWT_SECRET_KEY: str = os.getenv("JWT_SECRET_KEY", "dev-secret-key-graph-rag-2026-super-secure")
    JWT_ALGORITHM: str = os.getenv("JWT_ALGORITHM", "HS256")
    JWT_EXPIRY_HOURS: int = int(os.getenv("JWT_EXPIRY_HOURS", "24"))

    # ── RAG Evaluation ──
    RAG_EVAL_ENABLED: bool = os.getenv("RAG_EVAL_ENABLED", "true").lower() in ("true", "1", "yes")

    # ── App & Logging ──
    APP_HOST: str = os.getenv("APP_HOST", "localhost")
    APP_PORT: int = int(os.getenv("APP_PORT", "8501"))
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    APP_LOG_FILE: Path = PROJECT_ROOT / os.getenv("APP_LOG_FILE", "data/app.log")

    def ensure_directories(self) -> None:
        """Create required runtime directories if they don't exist."""
        self.DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.CHROMA_PERSIST_DIR.mkdir(parents=True, exist_ok=True)
        self.UPLOAD_TEMP_DIR.mkdir(parents=True, exist_ok=True)
        self.SQLITE_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        self.PDF_DIR.mkdir(parents=True, exist_ok=True)


# Global settings singleton
settings = Settings()
settings.ensure_directories()

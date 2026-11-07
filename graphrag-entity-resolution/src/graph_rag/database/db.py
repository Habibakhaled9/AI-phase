"""
Database module for SQLite storage.

Manages connections, schema initialization, and transactional access for:
- Users & Authentication
- Chat Sessions & Messages
- Live Session Evaluation Logs
- Document Upload Registry
"""

import sqlite3
from contextlib import contextmanager
from typing import Generator
from graph_rag.config import settings

SCHEMA_SQL = """
PRAGMA foreign_keys = ON;
PRAGMA journal_mode = WAL;

-- ── Users Table ──
CREATE TABLE IF NOT EXISTS users (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    username        TEXT UNIQUE NOT NULL COLLATE NOCASE,
    email           TEXT UNIQUE NOT NULL COLLATE NOCASE,
    password_hash   TEXT NOT NULL,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ── Chat Sessions Table ──
CREATE TABLE IF NOT EXISTS chat_sessions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id         INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    title           TEXT NOT NULL,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ── Chat Messages Table ──
CREATE TABLE IF NOT EXISTS messages (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id      INTEGER NOT NULL REFERENCES chat_sessions(id) ON DELETE CASCADE,
    user_id         INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    role            TEXT NOT NULL CHECK(role IN ('user', 'assistant', 'system')),
    content         TEXT NOT NULL,
    sources         TEXT, -- JSON string of retrieved chunks/entities
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ── RAG Evaluation Logs Table ──
CREATE TABLE IF NOT EXISTS evaluation_logs (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    message_id          INTEGER REFERENCES messages(id) ON DELETE SET NULL,
    session_id          INTEGER REFERENCES chat_sessions(id) ON DELETE SET NULL,
    user_id             INTEGER REFERENCES users(id) ON DELETE SET NULL,
    question            TEXT NOT NULL,
    answer              TEXT NOT NULL,
    context_chunks      TEXT,    -- JSON array of retrieved context chunks
    faithfulness        REAL,    -- 0.0 to 1.0 (faithfulness to context)
    answer_relevance    REAL,    -- 0.0 to 1.0 (relevance to user query)
    context_relevance   REAL,    -- 0.0 to 1.0 (relevance of context to query)
    retrieval_ms        INTEGER, -- Retrieval latency in ms
    generation_ms       INTEGER, -- LLM generation latency in ms
    eval_ms             INTEGER, -- Evaluation latency in ms
    total_ms            INTEGER, -- Total turnaround latency in ms
    eval_reasoning      TEXT,    -- JSON string with judge commentary
    created_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ── Uploaded Documents Table ──
CREATE TABLE IF NOT EXISTS uploaded_documents (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id             INTEGER REFERENCES users(id) ON DELETE SET NULL,
    filename            TEXT NOT NULL,
    file_type           TEXT NOT NULL,
    file_size_bytes     INTEGER,
    file_hash           TEXT NOT NULL,
    source_archive      TEXT,
    num_chunks          INTEGER DEFAULT 0,
    status              TEXT DEFAULT 'pending', -- pending, processing, completed, failed, duplicate_skipped
    error_message       TEXT,
    created_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ── Indexes for Fast Lookups ──
CREATE INDEX IF NOT EXISTS idx_chat_sessions_user ON chat_sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id);
CREATE INDEX IF NOT EXISTS idx_eval_session ON evaluation_logs(session_id);
CREATE INDEX IF NOT EXISTS idx_eval_created ON evaluation_logs(created_at);
CREATE INDEX IF NOT EXISTS idx_uploads_hash ON uploaded_documents(file_hash);
"""


def get_raw_connection() -> sqlite3.Connection:
    """Return a raw SQLite connection with row factory configured."""
    settings.SQLITE_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(
        settings.SQLITE_DB_PATH,
        timeout=30.0,
    )
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.execute("PRAGMA journal_mode = WAL;")
    return conn


@contextmanager
def get_db_connection() -> Generator[sqlite3.Connection, None, None]:
    """Context manager for transactional database operations."""
    conn = get_raw_connection()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db() -> None:
    """Initialize database schemas and tables if they do not exist."""
    settings.ensure_directories()
    with get_db_connection() as conn:
        conn.executescript(SCHEMA_SQL)

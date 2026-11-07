"""Database package for SQLite connection and schema management."""

from graph_rag.database.db import get_db_connection, init_db

__all__ = ["get_db_connection", "init_db"]

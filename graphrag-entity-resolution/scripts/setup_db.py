"""
Database setup script for Graph RAG Chatbot.

Initializes SQLite database tables and creates required data directories.
Usage:
    uv run python scripts/setup_db.py
"""

import sys
from pathlib import Path

# Ensure src is in Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from graph_rag.config import settings
from graph_rag.database.db import init_db, get_db_connection


def main() -> None:
    print("=" * 60)
    print("Initializing Graph RAG Database")
    print(f"Database Path: {settings.SQLITE_DB_PATH}")
    print("=" * 60)

    try:
        init_db()
        # Verify tables created
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;")
            tables = [row["name"] for row in cursor.fetchall()]

        print("\n[SUCCESS] Database initialized successfully!")
        print("Created tables:")
        for t in tables:
            if not t.startswith("sqlite_"):
                print(f"  - {t}")
        print("=" * 60)
    except Exception as e:
        print(f"\n[ERROR] Failed to initialize database: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

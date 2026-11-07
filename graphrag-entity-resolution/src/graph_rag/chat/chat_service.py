"""
Chat Service module.

CRUD operations for chat sessions and conversational history in SQLite.
Supports multi-session navigation and persistent source citation storage.
"""

import json
from datetime import datetime, timezone
from typing import List, Optional

from graph_rag.chat.models import ChatMessage, ChatSession
from graph_rag.database.db import get_db_connection


class ChatService:
    """Manages chat sessions and conversation messages."""

    def create_session(self, user_id: int, title: str = "New Research Chat") -> ChatSession:
        """Create a new chat conversation thread."""
        now = datetime.now(timezone.utc).isoformat()
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO chat_sessions (user_id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
                (user_id, title, now, now),
            )
            session_id = cursor.lastrowid
            return ChatSession(id=session_id, user_id=user_id, title=title, created_at=now, updated_at=now)

    def list_user_sessions(self, user_id: int) -> List[ChatSession]:
        """List all chat sessions for a user, sorted newest first."""
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT id, user_id, title, created_at, updated_at FROM chat_sessions WHERE user_id = ? ORDER BY updated_at DESC",
                (user_id,),
            )
            rows = cursor.fetchall()
            return [
                ChatSession(
                    id=row["id"],
                    user_id=row["user_id"],
                    title=row["title"],
                    created_at=str(row["created_at"]),
                    updated_at=str(row["updated_at"]),
                )
                for row in rows
            ]

    def get_session(self, session_id: int) -> Optional[ChatSession]:
        """Fetch a specific session by ID."""
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT id, user_id, title, created_at, updated_at FROM chat_sessions WHERE id = ?",
                (session_id,),
            )
            row = cursor.fetchone()
            if row:
                return ChatSession(
                    id=row["id"],
                    user_id=row["user_id"],
                    title=row["title"],
                    created_at=str(row["created_at"]),
                    updated_at=str(row["updated_at"]),
                )
            return None

    def delete_session(self, session_id: int) -> bool:
        """Delete a chat session and all its messages."""
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM chat_sessions WHERE id = ?", (session_id,))
            return cursor.rowcount > 0

    def add_message(
        self,
        session_id: int,
        user_id: int,
        role: str,
        content: str,
        sources: Optional[List[str]] = None,
    ) -> ChatMessage:
        """Add a message to a chat session and update session timestamp."""
        sources_list = sources or []
        sources_json = json.dumps(sources_list)
        now = datetime.now(timezone.utc).isoformat()

        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO messages (session_id, user_id, role, content, sources, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (session_id, user_id, role, content, sources_json, now),
            )
            msg_id = cursor.lastrowid

            # Update session's updated_at and conditionally set title from first user message
            cursor.execute("SELECT count(*) as cnt FROM messages WHERE session_id = ?", (session_id,))
            msg_count = cursor.fetchone()["cnt"]

            if msg_count == 1 and role == "user":
                # First user message: generate a concise session title
                title_preview = content.strip().split("\n")[0][:45]
                cursor.execute(
                    "UPDATE chat_sessions SET title = ?, updated_at = ? WHERE id = ?",
                    (title_preview, now, session_id),
                )
            else:
                cursor.execute(
                    "UPDATE chat_sessions SET updated_at = ? WHERE id = ?",
                    (now, session_id),
                )

            return ChatMessage(
                id=msg_id,
                session_id=session_id,
                user_id=user_id,
                role=role,
                content=content,
                sources=sources_list,
                created_at=now,
            )

    def get_session_messages(self, session_id: int) -> List[ChatMessage]:
        """Fetch all messages for a session in chronological order."""
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT id, session_id, user_id, role, content, sources, created_at
                FROM messages
                WHERE session_id = ?
                ORDER BY created_at ASC
                """,
                (session_id,),
            )
            rows = cursor.fetchall()
            messages = []
            for r in rows:
                sources = []
                if r["sources"]:
                    try:
                        sources = json.loads(r["sources"])
                    except Exception:
                        sources = []
                messages.append(
                    ChatMessage(
                        id=r["id"],
                        session_id=r["session_id"],
                        user_id=r["user_id"],
                        role=r["role"],
                        content=r["content"],
                        sources=sources,
                        created_at=str(r["created_at"]),
                    )
                )
            return messages


chat_service = ChatService()

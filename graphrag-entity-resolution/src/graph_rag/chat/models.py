"""Chat history models."""

import json
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ChatSession:
    """A chat session conversation thread."""
    id: int
    user_id: int
    title: str
    created_at: str
    updated_at: str


@dataclass
class ChatMessage:
    """An individual message in a chat session."""
    id: int
    session_id: int
    user_id: int
    role: str  # 'user' | 'assistant'
    content: str
    sources: List[str] = field(default_factory=list)
    created_at: Optional[str] = None

    def get_sources_json(self) -> str:
        """Convert sources to JSON string for storage."""
        return json.dumps(self.sources)

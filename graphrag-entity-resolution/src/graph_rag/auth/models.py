"""Authentication models."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class User:
    """User account representation."""
    id: int
    username: str
    email: str
    password_hash: str
    created_at: Optional[str] = None

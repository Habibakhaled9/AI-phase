"""
JWT Token Handler.

Issues and verifies signed JSON Web Tokens for user sessions.
"""

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional
import jwt

from graph_rag.config import settings


def create_access_token(user_id: int, username: str) -> str:
    """Generate signed JWT token expiring after configured hours."""
    expire = datetime.now(timezone.utc) + timedelta(hours=settings.JWT_EXPIRY_HOURS)
    payload = {
        "sub": str(user_id),
        "username": username,
        "exp": expire,
        "iat": datetime.now(timezone.utc),
    }
    return jwt.encode(payload, settings.JWT_SECRET_KEY, algorithm=settings.JWT_ALGORITHM)


def decode_access_token(token: str) -> Optional[Dict[str, Any]]:
    """Decode and validate a JWT access token. Returns payload or None if invalid/expired."""
    try:
        payload = jwt.decode(
            token,
            settings.JWT_SECRET_KEY,
            algorithms=[settings.JWT_ALGORITHM],
        )
        return payload
    except (jwt.PyJWTError, Exception):
        return None

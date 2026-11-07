"""
Authentication Service module.

Handles:
- User registration with bcrypt password hashing
- User login credential verification
- Session token generation
- User lookup
"""

import sqlite3
from typing import Optional, Tuple
import bcrypt

from graph_rag.auth.jwt_handler import create_access_token, decode_access_token
from graph_rag.auth.models import User
from graph_rag.database.db import get_db_connection


class AuthService:
    """Manages user registration, credential verification, and sessions."""

    @staticmethod
    def hash_password(password: str) -> str:
        """Hash a plaintext password using bcrypt with salt."""
        salt = bcrypt.gensalt(rounds=12)
        hashed = bcrypt.hashpw(password.encode("utf-8"), salt)
        return hashed.decode("utf-8")

    @staticmethod
    def verify_password(plain_password: str, hashed_password: str) -> bool:
        """Verify plaintext password against bcrypt hash."""
        try:
            return bcrypt.checkpw(plain_password.encode("utf-8"), hashed_password.encode("utf-8"))
        except Exception:
            return False

    def register_user(
        self,
        username: str,
        email: str,
        password: str,
    ) -> Tuple[bool, str, Optional[User]]:
        """Register a new user account."""
        username = username.strip()
        email = email.strip().lower()

        if len(username) < 3:
            return False, "Username must be at least 3 characters long.", None
        if "@" not in email or "." not in email:
            return False, "Please provide a valid email address.", None
        if len(password) < 6:
            return False, "Password must be at least 6 characters long.", None

        pwd_hash = self.hash_password(password)

        try:
            with get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "INSERT INTO users (username, email, password_hash) VALUES (?, ?, ?)",
                    (username, email, pwd_hash),
                )
                user_id = cursor.lastrowid
                user = User(id=user_id, username=username, email=email, password_hash=pwd_hash)
                return True, "Registration successful!", user
        except sqlite3.IntegrityError:
            return False, "Username or email is already registered.", None
        except Exception as e:
            return False, f"Registration failed: {e}", None

    def login_user(
        self,
        username: str,
        password: str,
    ) -> Tuple[bool, str, Optional[str], Optional[User]]:
        """
        Authenticate user by username and password.
        Returns: (success, message, token, user)
        """
        username = username.strip()
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT id, username, email, password_hash, created_at FROM users WHERE username = ? COLLATE NOCASE",
                (username,),
            )
            row = cursor.fetchone()

        if not row:
            return False, "Invalid username or password.", None, None

        if not self.verify_password(password, row["password_hash"]):
            return False, "Invalid username or password.", None, None

        user = User(
            id=row["id"],
            username=row["username"],
            email=row["email"],
            password_hash=row["password_hash"],
            created_at=str(row["created_at"]),
        )
        token = create_access_token(user_id=user.id, username=user.username)
        return True, "Login successful!", token, user

    def get_user_by_id(self, user_id: int) -> Optional[User]:
        """Fetch user by ID."""
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT id, username, email, password_hash, created_at FROM users WHERE id = ?",
                (user_id,),
            )
            row = cursor.fetchone()

        if row:
            return User(
                id=row["id"],
                username=row["username"],
                email=row["email"],
                password_hash=row["password_hash"],
                created_at=str(row["created_at"]),
            )
        return None

    def validate_token(self, token: str) -> Optional[User]:
        """Verify JWT token and retrieve corresponding User object."""
        payload = decode_access_token(token)
        if not payload:
            return None
        user_id = int(payload.get("sub", 0))
        return self.get_user_by_id(user_id)


auth_service = AuthService()

"""User registration and opaque session-token authentication."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone

from backend.security.auth import (
    SESSION_TTL_SECONDS,
    digest_session_token,
    hash_password,
    issue_session_token,
    needs_rehash,
    normalize_email,
    validate_password,
    verify_password,
)
from backend.storage.database import initialize_database
from backend.storage import repository


def _public_user(row) -> dict:
    return {
        "id": int(row["id"]),
        "email": row["email"],
        "display_name": row["display_name"],
        "created_at": row["created_at"],
    }


def register_user(email: str, password: str, display_name: str) -> dict:
    initialize_database()
    normalized = normalize_email(email)
    validate_password(password)
    if not isinstance(display_name, str):
        raise ValueError("Display name must contain 1 to 80 characters.")
    name = display_name.strip()
    if not name or len(name) > 80:
        raise ValueError("Display name must contain 1 to 80 characters.")
    try:
        return repository.create_user(normalized, hash_password(password), name)
    except sqlite3.IntegrityError as exc:
        if "users.email" in str(exc).lower() or "unique" in str(exc).lower():
            raise ValueError("An account with that email already exists.") from exc
        raise


def login_user(email: str, password: str) -> dict:
    initialize_database()
    try:
        normalized = normalize_email(email)
    except ValueError as exc:
        raise ValueError("Invalid email or password.") from exc
    row = repository.get_user_by_email(normalized)
    if row is None or not verify_password(password, row["password_hash"]):
        raise ValueError("Invalid email or password.")
    if needs_rehash(row["password_hash"]):
        repository.update_password(int(row["id"]), hash_password(password))

    token = issue_session_token()
    expires_at = datetime.now(timezone.utc) + timedelta(seconds=SESSION_TTL_SECONDS)
    repository.create_auth_session(digest_session_token(token), int(row["id"]), expires_at)
    repository.touch_login(int(row["id"]))
    return {
        "access_token": token,
        "token_type": "bearer",
        "expires_in": SESSION_TTL_SECONDS,
        "user": _public_user(row),
    }


def current_user(token: str | None) -> dict | None:
    initialize_database()
    if not token:
        return None
    row = repository.get_user_for_session(digest_session_token(token))
    return _public_user(row) if row else None


def logout_user(token: str | None) -> None:
    if token:
        repository.revoke_auth_session(digest_session_token(token))

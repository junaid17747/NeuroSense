"""Authentication primitives shared by the API and the Streamlit dashboard."""

from __future__ import annotations

import hashlib
import hmac
import re
import secrets
from typing import Final

from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerificationError, VerifyMismatchError


PASSWORD_MIN_LENGTH: Final = 10
SESSION_TTL_SECONDS: Final = 60 * 60 * 12
_EMAIL_RE = re.compile(r"^[^\s@]+@[^\s@]+\.[^\s@]+$")
_PASSWORD_HASHER = PasswordHasher(time_cost=3, memory_cost=65536, parallelism=2)


def normalize_email(email: str) -> str:
    if not isinstance(email, str):
        raise ValueError("Email must be a string.")
    normalized = email.strip().casefold()
    if len(normalized) > 254 or not _EMAIL_RE.fullmatch(normalized):
        raise ValueError("Enter a valid email address.")
    return normalized


def validate_password(password: str) -> str:
    if not isinstance(password, str) or len(password) < PASSWORD_MIN_LENGTH:
        raise ValueError(f"Password must contain at least {PASSWORD_MIN_LENGTH} characters.")
    if len(password) > 256:
        raise ValueError("Password is too long.")
    if not any(char.islower() for char in password):
        raise ValueError("Password must contain a lowercase letter.")
    if not any(char.isupper() for char in password):
        raise ValueError("Password must contain an uppercase letter.")
    if not any(char.isdigit() for char in password):
        raise ValueError("Password must contain a number.")
    return password


def hash_password(password: str) -> str:
    return _PASSWORD_HASHER.hash(validate_password(password))


def verify_password(password: str, password_hash: str) -> bool:
    if not isinstance(password, str) or not isinstance(password_hash, str):
        return False
    try:
        return bool(_PASSWORD_HASHER.verify(password_hash, password))
    except (VerifyMismatchError, VerificationError, InvalidHashError):
        return False


def needs_rehash(password_hash: str) -> bool:
    try:
        return _PASSWORD_HASHER.check_needs_rehash(password_hash)
    except (TypeError, InvalidHashError):
        return True


def issue_session_token() -> str:
    return secrets.token_urlsafe(32)


def digest_session_token(token: str) -> str:
    if not isinstance(token, str) or not token:
        return ""
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def tokens_equal(left: str, right: str) -> bool:
    return hmac.compare_digest(left or "", right or "")

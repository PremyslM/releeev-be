"""
Authentication: bcrypt password hashing + JWT bearer tokens.

Phase A of multi-user: this provides register/login and a `get_current_user`
dependency. Data scoping (filtering every query by user_id) is Phase B.
"""
from __future__ import annotations

import time

import bcrypt
import jwt
from fastapi import Header, HTTPException

from src.config import settings

_ALGO = "HS256"


# ── Passwords ────────────────────────────────────────────────────────────────

def hash_password(password: str) -> str:
    # bcrypt caps input at 72 bytes; that's fine for normal passwords.
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, hashed: str) -> bool:
    try:
        return bcrypt.checkpw(password.encode("utf-8"), hashed.encode("utf-8"))
    except Exception:
        return False


# ── Tokens ───────────────────────────────────────────────────────────────────

def create_token(user_id: str) -> str:
    payload = {
        "sub": user_id,
        "iat": int(time.time()),
        "exp": int(time.time()) + settings.jwt_ttl_days * 86_400,
    }
    return jwt.encode(payload, settings.jwt_secret, algorithm=_ALGO)


def decode_token(token: str) -> str | None:
    try:
        payload = jwt.decode(token, settings.jwt_secret, algorithms=[_ALGO])
        return payload.get("sub")
    except Exception:
        return None


# ── FastAPI dependency ───────────────────────────────────────────────────────

async def get_current_user(authorization: str = Header(default="")) -> str:
    """Resolve the user id from an `Authorization: Bearer <token>` header."""
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token.strip():
        raise HTTPException(status_code=401, detail="Chybí přihlášení.")
    user_id = decode_token(token.strip())
    if not user_id:
        raise HTTPException(status_code=401, detail="Neplatný nebo vypršelý token.")
    return user_id

"""
API key and JWT authentication middleware.

WHY: Every production API needs authentication. We support two modes:
1. API key — Simple, stateless, good for service-to-service calls
2. JWT — Token-based, supports user-level permissions

ARCHITECTURE DECISION: API key as the default because it's the simplest
secure option for an internal enterprise API. JWT support is available
for user-facing features (per-user sessions, feedback attribution).
"""

from __future__ import annotations

from typing import Optional

from fastapi import HTTPException, Request, Security
from fastapi.security import APIKeyHeader, HTTPAuthorizationCredentials, HTTPBearer

from src.config.logging_config import get_logger
from src.config.settings import get_settings

logger = get_logger(__name__)

# API key header scheme
api_key_header = APIKeyHeader(name="x-api-key", auto_error=False)

# JWT bearer scheme (optional)
jwt_bearer = HTTPBearer(auto_error=False)


async def verify_api_key(
    api_key: Optional[str] = Security(api_key_header),
) -> str:
    """
    Verify the API key from the x-api-key header.

    Returns the validated API key on success.
    Raises 401 on missing key, 403 on invalid key.
    """
    if not api_key:
        raise HTTPException(
            status_code=401,
            detail="Missing API key. Include 'x-api-key' header.",
        )

    settings = get_settings()
    valid_keys = {settings.api_key.get_secret_value()}

    # Support secondary key for rotation
    if settings.api_key_secondary:
        valid_keys.add(settings.api_key_secondary.get_secret_value())

    if api_key not in valid_keys:
        logger.warning("Invalid API key attempt", key_prefix=api_key[:8] + "...")
        raise HTTPException(
            status_code=403,
            detail="Invalid API key",
        )

    return api_key


async def verify_jwt(
    credentials: Optional[HTTPAuthorizationCredentials] = Security(jwt_bearer),
) -> Optional[dict]:
    """
    Verify JWT token (optional authentication).

    Returns decoded token payload or None if no token provided.
    """
    if not credentials:
        return None

    try:
        from jose import JWTError, jwt as jose_jwt

        settings = get_settings()
        payload = jose_jwt.decode(
            credentials.credentials,
            settings.api_key.get_secret_value(),  # Using API key as JWT secret for simplicity
            algorithms=["HS256"],
        )
        return payload

    except ImportError:
        logger.warning("python-jose not installed, JWT verification disabled")
        return None
    except Exception as e:
        logger.warning("JWT verification failed", error=str(e))
        raise HTTPException(
            status_code=401,
            detail="Invalid authentication token",
        )


async def get_current_user(
    request: Request,
    api_key: str = Security(verify_api_key),
) -> dict:
    """
    Get the current authenticated user.

    Extracts user info from JWT if available, otherwise returns
    a default service user.
    """
    # Try JWT first for user-level auth
    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        jwt_payload = await verify_jwt(
            HTTPAuthorizationCredentials(
                scheme="Bearer",
                credentials=auth_header[7:],
            )
        )
        if jwt_payload:
            return {
                "user_id": jwt_payload.get("sub", "unknown"),
                "email": jwt_payload.get("email", ""),
                "roles": jwt_payload.get("roles", []),
            }

    # Default to API key user
    return {
        "user_id": "api_user",
        "email": "",
        "roles": ["api"],
    }

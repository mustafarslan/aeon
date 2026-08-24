"""
Aeon Memory OS — Authentication.

Fixes v4-plan.md Stage 0's core isolation gap: the server used to trust an
unauthenticated `X-User-ID` header verbatim as caller identity (any client
could claim to be any user). This module makes that identity verifiable.

Pluggable via AuthProvider, following the same pattern as LLMProvider
(llm.py) -- Aeon ships as open-source infrastructure for third-party
self-hosters (v4-plan.md), so it deliberately does not hardcode a specific
IdP. BearerJWTAuthProvider verifies any HS256-signed JWT (issued by the
adopter's own auth backend, or by an OIDC provider configured to sign with
a shared secret) -- this is "OIDC or equivalent" without Aeon needing to
speak to a specific vendor or fetch remote JWKS. An adopter wanting RS256 /
remote-JWKS OIDC can implement AuthProvider and pass it to get_auth_provider
via AEON_AUTH_MODE plus their own registration (see get_auth_provider()).

Fails closed: if no auth mode is configured, the server refuses to start
serving authenticated endpoints rather than silently trusting an
unverified header. Insecure passthrough (the old behavior) is available
ONLY via an explicit, loudly-logged opt-in -- for local development, never
the default.

Environment Variables:
    AEON_AUTH_MODE:    "jwt" (default if AEON_AUTH_SECRET is set) or
                       "insecure_dev_no_verify" (explicit opt-in required).
    AEON_AUTH_SECRET:  HMAC secret for JWT verification (AEON_AUTH_MODE=jwt).
    AEON_AUTH_ALGORITHM: JWT algorithm (default "HS256").
"""
import logging
import os
from abc import ABC, abstractmethod
from functools import lru_cache
from typing import Optional

import jwt

logger = logging.getLogger("aeon.auth")


class AuthError(Exception):
    """Raised when a request's credentials are missing or invalid."""


class AuthProvider(ABC):
    """Abstract base class for verifying a request's caller identity."""

    @abstractmethod
    async def verify(self, authorization_header: Optional[str]) -> str:
        """
        Verify the request's Authorization header and return the caller's
        verified user/tenant ID.

        Args:
            authorization_header: The raw `Authorization` header value
                (e.g. "Bearer <token>"), or None if absent.

        Returns:
            The verified user/tenant identity string.

        Raises:
            AuthError: If the header is missing, malformed, or the
                credentials don't verify.
        """
        raise NotImplementedError


class BearerJWTAuthProvider(AuthProvider):
    """
    Verifies a JWT bearer token's signature and expiry, returning its `sub`
    claim as the caller's identity. Works with any token issuer (a custom
    backend, or an OIDC provider configured for HS256) that shares
    AEON_AUTH_SECRET with this server -- Aeon does not fetch remote JWKS or
    assume a specific identity provider.
    """

    def __init__(self, secret: str, algorithm: str = "HS256"):
        if not secret:
            raise ValueError("BearerJWTAuthProvider requires a non-empty secret")
        self._secret = secret
        self._algorithm = algorithm

    async def verify(self, authorization_header: Optional[str]) -> str:
        if not authorization_header:
            raise AuthError("Missing Authorization header")

        scheme, _, token = authorization_header.partition(" ")
        if scheme.lower() != "bearer" or not token:
            raise AuthError("Authorization header must be 'Bearer <token>'")

        try:
            claims = jwt.decode(token, self._secret, algorithms=[self._algorithm])
        except jwt.ExpiredSignatureError as e:
            raise AuthError("Token has expired") from e
        except jwt.InvalidTokenError as e:
            raise AuthError(f"Invalid token: {e}") from e

        user_id = claims.get("sub")
        if not user_id or not isinstance(user_id, str):
            raise AuthError("Token is missing a valid 'sub' claim")
        return user_id


class InsecureDevAuthProvider(AuthProvider):
    """
    Trusts a caller-supplied identity with NO verification -- the exact
    behavior this module replaces. Exists only for local development
    against a server with no auth backend available yet. Never the
    default; requires AEON_AUTH_MODE=insecure_dev_no_verify explicitly,
    and logs a warning on every single request so it can't go unnoticed
    in a shared environment.
    """

    def __init__(self):
        logger.warning(
            "AEON_AUTH_MODE=insecure_dev_no_verify is active: caller identity "
            "is NOT verified. Every request trusts the X-User-ID header "
            "verbatim. Do not use this outside local development."
        )

    async def verify(self, authorization_header: Optional[str]) -> str:
        # Kept as X-User-ID (not Authorization) in this mode so it's
        # obviously a different, weaker code path than the real one --
        # never silently accepted as if it were a verified bearer token.
        raise AuthError(
            "InsecureDevAuthProvider.verify() should not be called directly; "
            "use verify_x_user_id_header() via the FastAPI dependency."
        )

    async def verify_x_user_id_header(self, x_user_id: Optional[str]) -> str:
        logger.warning("Insecure auth: trusting unverified X-User-ID=%r", x_user_id)
        if not x_user_id:
            raise AuthError("X-User-ID header is required (insecure dev mode)")
        return x_user_id


@lru_cache()
def get_auth_provider() -> AuthProvider:
    """
    Singleton AuthProvider, selected by AEON_AUTH_MODE / AEON_AUTH_SECRET.

    Fails closed: raises RuntimeError at first use (i.e. at server startup,
    via FastAPI dependency resolution) if no valid mode is configured,
    rather than falling back to trusting an unverified header.
    """
    mode = os.environ.get("AEON_AUTH_MODE")
    secret = os.environ.get("AEON_AUTH_SECRET")

    if mode == "insecure_dev_no_verify":
        return InsecureDevAuthProvider()

    if secret:
        algorithm = os.environ.get("AEON_AUTH_ALGORITHM", "HS256")
        return BearerJWTAuthProvider(secret, algorithm)

    raise RuntimeError(
        "No authentication configured. Set AEON_AUTH_SECRET to a strong "
        "shared secret to verify HS256 JWT bearer tokens, or explicitly set "
        "AEON_AUTH_MODE=insecure_dev_no_verify for local development only. "
        "Refusing to start with unauthenticated access (v4-plan.md Stage 0)."
    )

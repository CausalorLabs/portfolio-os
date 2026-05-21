"""
Deployment — Security & Access Layer.

Sprint 8: API security, rate limiting, and secrets management.
  - API authentication (token-based)
  - Rate limiting
  - CORS configuration
  - Secrets from environment
"""

from __future__ import annotations

import hashlib
import os
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from loguru import logger
from omegaconf import OmegaConf


def _load_config() -> dict:
    cfg_path = Path("configs/deployment.yaml")
    if cfg_path.exists():
        return OmegaConf.to_container(OmegaConf.load(cfg_path), resolve=True)
    return {}


class RateLimiter:
    """Simple in-memory rate limiter."""

    def __init__(self, max_rpm: int = 60):
        self._max_rpm = max_rpm
        self._requests: dict[str, list[float]] = defaultdict(list)

    def check(self, client_id: str = "default") -> bool:
        """Check if request is allowed within rate limit."""
        now = time.time()
        window_start = now - 60

        # Clean old requests
        self._requests[client_id] = [
            t for t in self._requests[client_id] if t > window_start
        ]

        if len(self._requests[client_id]) >= self._max_rpm:
            return False

        self._requests[client_id].append(now)
        return True

    def remaining(self, client_id: str = "default") -> int:
        """Get remaining requests in window."""
        now = time.time()
        window_start = now - 60
        recent = [t for t in self._requests.get(client_id, []) if t > window_start]
        return max(0, self._max_rpm - len(recent))


class SecurityLayer:
    """
    Security layer for the portfolio API.

    Provides token-based auth, rate limiting, and CORS management.
    """

    def __init__(self):
        cfg = _load_config().get("security", {})
        self._auth_enabled = cfg.get("api_auth", False)
        self._rate_limit_rpm = cfg.get("rate_limit_rpm", 60)
        self._cors_origins = cfg.get("cors_origins", ["http://localhost:8501"])
        self._secrets_from_env = cfg.get("secrets_from_env", True)

        self._rate_limiter = RateLimiter(self._rate_limit_rpm)
        self._api_tokens: set[str] = set()

        # Load tokens from environment if configured
        if self._secrets_from_env:
            token = os.environ.get("PORTFOLIO_API_TOKEN")
            if token:
                self._api_tokens.add(self._hash_token(token))

    def _hash_token(self, token: str) -> str:
        """Hash a token for secure storage."""
        return hashlib.sha256(token.encode()).hexdigest()

    def add_token(self, token: str) -> None:
        """Add an API token."""
        self._api_tokens.add(self._hash_token(token))

    def verify_token(self, token: str) -> bool:
        """Verify an API token."""
        if not self._auth_enabled:
            return True
        return self._hash_token(token) in self._api_tokens

    def check_rate_limit(self, client_id: str = "default") -> bool:
        """Check rate limit for a client."""
        return self._rate_limiter.check(client_id)

    def get_cors_origins(self) -> list[str]:
        """Get allowed CORS origins."""
        return list(self._cors_origins)

    def get_secret(self, name: str, default: str = "") -> str:
        """Get a secret from environment variables."""
        return os.environ.get(name, default)

    def summary(self) -> dict:
        return {
            "auth_enabled": self._auth_enabled,
            "rate_limit_rpm": self._rate_limit_rpm,
            "cors_origins": self._cors_origins,
            "tokens_configured": len(self._api_tokens),
        }

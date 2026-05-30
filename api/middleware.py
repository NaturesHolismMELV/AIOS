"""
middleware.py — AIOS Demo Deployment Middleware
================================================
Session 13 · v1.5.0 | Updated Session 15 · v1.7.0

Two middleware layers for the public hosted demo:

1. RateLimitMiddleware
   Simple in-memory sliding-window rate limiter per IP address.
   Defaults: 30 requests/minute for general API, 3 sandbox
   submissions/hour (enforced at the endpoint level via the
   SandboxRateGuard dependency).

2. APIKeyMiddleware
   Optional header-based API key check for protected endpoints.
   Set AIOS_API_KEY env var to enable. If unset, all requests pass
   (development mode). Demo key is printed at startup.

Design notes
------------
- No Redis / external dependency — in-memory only, resets on restart.
  Suitable for a single-instance demo deployment. For multi-instance
  production, replace with a shared store (e.g., Redis).
- IP extraction handles X-Forwarded-For (Railway/Render proxy headers).
- Sandbox submissions are rate-limited separately (heavier compute).
"""

import os
import time
import logging
from collections import defaultdict, deque
from typing import Optional

from fastapi import Request, Response, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

logger = logging.getLogger("aios.middleware")

# ── CONFIGURATION ──────────────────────────────────────────────────────────

# General API: requests per window
RATE_LIMIT_REQUESTS = int(os.environ.get("AIOS_RATE_LIMIT_REQUESTS", "60"))
RATE_LIMIT_WINDOW   = int(os.environ.get("AIOS_RATE_LIMIT_WINDOW",   "60"))   # seconds

# Sandbox submissions: separate stricter limit
SANDBOX_LIMIT_REQUESTS = int(os.environ.get("AIOS_SANDBOX_LIMIT", "5"))
SANDBOX_LIMIT_WINDOW   = int(os.environ.get("AIOS_SANDBOX_WINDOW", "3600"))   # 1 hour

# Startup grace period: bypass rate limiting for N seconds after server start
# Prevents dashboard polling burst from triggering 429s on first load
STARTUP_GRACE_SECONDS = int(os.environ.get("AIOS_STARTUP_GRACE", "15"))

# API key (optional — leave unset to run in open/dev mode)
AIOS_API_KEY: Optional[str] = os.environ.get("AIOS_API_KEY")

# Paths that bypass rate limiting entirely (health checks, static assets)
EXEMPT_PATHS = {"/", "/health", "/api/health", "/docs", "/openapi.json", "/favicon.ico"}

# Paths that require API key when AIOS_API_KEY is set
PROTECTED_PATHS_PREFIX = ("/sandbox/", "/melv/", "/api/beta", "/theorem/")

# Paths that are always public — exempt from API key even when AIOS_API_KEY is set
# /demo/ provides the public bifurcation demonstration (Session 30)
PUBLIC_PATHS_PREFIX = ("/demo/",)


# ── RATE LIMIT MIDDLEWARE ──────────────────────────────────────────────────

class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Sliding-window rate limiter per client IP.

    Tracks request timestamps in a deque per IP. On each request,
    prunes timestamps older than the window, then checks the count.
    """

    def __init__(self, app, requests: int = None, window: int = None,
                 grace_seconds: int = None):
        super().__init__(app)
        # Read at init time — allows pytest env var overrides before import
        self._requests = requests or int(os.environ.get("AIOS_RATE_LIMIT_REQUESTS", "60"))
        self._window   = window   or int(os.environ.get("AIOS_RATE_LIMIT_WINDOW",   "60"))
        self._grace    = grace_seconds if grace_seconds is not None else STARTUP_GRACE_SECONDS
        self._start_time = time.time()
        self._buckets: dict[str, deque] = defaultdict(deque)
        self._sandbox_buckets: dict[str, deque] = defaultdict(deque)

    def _client_ip(self, request: Request) -> str:
        """Extract real client IP, respecting proxy headers."""
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return forwarded.split(",")[0].strip()
        return request.client.host if request.client else "unknown"

    def _check_limit(self, buckets: dict, ip: str,
                     limit: int, window: int) -> tuple[bool, int]:
        """
        Check and update rate limit for an IP.
        Returns (allowed: bool, retry_after_seconds: int).
        """
        now = time.time()
        dq  = buckets[ip]

        # Prune old entries
        while dq and dq[0] < now - window:
            dq.popleft()

        if len(dq) >= limit:
            oldest = dq[0]
            retry_after = int(window - (now - oldest)) + 1
            return False, retry_after

        dq.append(now)
        return True, 0

    async def dispatch(self, request: Request, call_next):
        path = request.url.path

        # Exempt paths bypass all rate limiting
        # /demo/ has its own per-IP session rate limit in demo_router.py
        if path in EXEMPT_PATHS or path.startswith("/frontend/") or path.startswith("/demo/"):
            return await call_next(request)

        # Startup grace period — bypass rate limiting on first load
        if self._grace > 0 and (time.time() - self._start_time) < self._grace:
            return await call_next(request)

        ip = self._client_ip(request)

        # Sandbox submissions + assess + run endpoints: stricter limit
        _sandbox_heavy = (
            path == "/sandbox/submit"
            or path.startswith("/sandbox/assess/")
            or path.startswith("/sandbox/run/")
        )
        if _sandbox_heavy and request.method == "POST":
            allowed, retry = self._check_limit(
                self._sandbox_buckets, ip,
                SANDBOX_LIMIT_REQUESTS, SANDBOX_LIMIT_WINDOW
            )
            if not allowed:
                logger.warning("Sandbox rate limit hit: ip=%s", ip)
                return JSONResponse(
                    status_code=429,
                    content={
                        "error": "sandbox_rate_limit",
                        "detail": f"Sandbox submissions limited to {SANDBOX_LIMIT_REQUESTS} per hour.",
                        "retry_after_seconds": retry,
                    },
                    headers={"Retry-After": str(retry)},
                )

        # General rate limit
        allowed, retry = self._check_limit(
            self._buckets, ip, self._requests, self._window
        )
        if not allowed:
            logger.warning("Rate limit hit: ip=%s path=%s", ip, path)
            return JSONResponse(
                status_code=429,
                content={
                    "error": "rate_limit_exceeded",
                    "detail": f"Too many requests. Limit: {self._requests} per {self._window}s.",
                    "retry_after_seconds": retry,
                },
                headers={"Retry-After": str(retry)},
            )

        response = await call_next(request)
        # Inject rate limit headers
        response.headers["X-RateLimit-Limit"]     = str(self._requests)
        response.headers["X-RateLimit-Remaining"] = str(
            max(0, self._requests - len(self._buckets[ip]))
        )
        return response


# ── API KEY MIDDLEWARE ─────────────────────────────────────────────────────

class APIKeyMiddleware(BaseHTTPMiddleware):
    """
    Optional API key enforcement for protected endpoints.

    When AIOS_API_KEY env var is set, requests to protected paths must
    include the header:  X-API-Key: <key>

    Protected path prefixes: /sandbox/, /melv/, /api/beta
    All other paths (including the demo landing page and /api/health)
    are always accessible.

    In development mode (AIOS_API_KEY unset), this middleware is a
    transparent pass-through — no keys are checked.
    """

    def __init__(self, app):
        super().__init__(app)
        self._key = AIOS_API_KEY
        if self._key:
            logger.info("APIKeyMiddleware: key enforcement ENABLED")
        else:
            logger.warning(
                "APIKeyMiddleware: AIOS_API_KEY is NOT SET — running in open/dev mode. "
                "Set AIOS_API_KEY in deployment secrets before public launch."
            )

    async def dispatch(self, request: Request, call_next):
        # No key configured → pass through
        if not self._key:
            return await call_next(request)

        path = request.url.path

        # Public paths — always accessible regardless of API key (e.g. /demo/)
        if any(path.startswith(prefix) for prefix in PUBLIC_PATHS_PREFIX):
            return await call_next(request)

        # Check if this path requires a key
        requires_key = any(
            path.startswith(prefix) for prefix in PROTECTED_PATHS_PREFIX
        )
        if not requires_key:
            return await call_next(request)

        # Validate key from header
        provided = request.headers.get("X-API-Key", "")
        if provided != self._key:
            logger.warning("Invalid API key: ip=%s path=%s",
                           request.client.host if request.client else "?", path)
            return JSONResponse(
                status_code=401,
                content={
                    "error": "invalid_api_key",
                    "detail": "Valid X-API-Key header required for this endpoint.",
                    "docs":   "https://github.com/NaturesHolismMELV/AIOS#api-keys",
                },
            )

        return await call_next(request)


# ── STARTUP HELPER ─────────────────────────────────────────────────────────

def print_startup_banner(host: str = "0.0.0.0", port: int = 8000):
    """Print a clear startup summary for demo deployments."""
    key_info = f"  API Key:     {AIOS_API_KEY[:8]}..." if AIOS_API_KEY else \
               "  API Key:     ⚠  NOT SET — set AIOS_API_KEY before public launch"
    print(f"""
╔══════════════════════════════════════════════════════╗
║          AIOS / MELVcore v1.9.2  ·  Session 21.2     ║
║     Thermodynamic Agent Certification Platform       ║
╠══════════════════════════════════════════════════════╣
║  Server:      http://{host}:{port:<5}                   ║
║  Landing:     http://{host}:{port}/demo                 ║
║  Dashboard:   http://{host}:{port}/frontend/dashboard13 ║
║  MCP (HTTP):  http://{host}:{port}/mcp                  ║
║  MCP (SSE):   http://{host}:{port}/mcp/sse              ║
╠══════════════════════════════════════════════════════╣
{key_info}
║  Rate limit:  {RATE_LIMIT_REQUESTS} req/{RATE_LIMIT_WINDOW}s · Sandbox: {SANDBOX_LIMIT_REQUESTS}/hr             ║
║  Grace period: {STARTUP_GRACE_SECONDS}s after startup                        ║
╚══════════════════════════════════════════════════════╝
""")

"""
conftest helper — module-scoped TestClient for Session 21.2 tests.
Imported by test_session21_2.py via direct import to avoid pytest
conftest.py scope conflicts with other test modules.
"""
import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

# Raise rate limit so polling loops don't 429 during tests
os.environ.setdefault("AIOS_RATE_LIMIT_REQUESTS", "1000")
os.environ.setdefault("AIOS_RATE_LIMIT_WINDOW",   "60")
os.environ.setdefault("AIOS_SANDBOX_LIMIT",        "100")

_shared_client = None

def get_shared_client():
    """
    Return a module-level TestClient that has been started exactly once.
    Calling this multiple times is safe — the same client is returned.
    """
    global _shared_client
    if _shared_client is None:
        try:
            from fastapi.testclient import TestClient
            from api.server import app
            _shared_client = TestClient(app)
            _shared_client.__enter__()
        except Exception:
            _shared_client = None
    return _shared_client

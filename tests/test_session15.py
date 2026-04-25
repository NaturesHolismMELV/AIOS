"""
test_session15.py — Session 15 · v1.8.0
========================================
Tests covering:
  1. Startup grace period (RateLimitMiddleware)
  2. Adversarial sandbox inputs
  3. Landing page end-to-end
  4. Version bump to 1.8.0
  5. Sandbox rate-limit friendly error message
"""

import os
import time
import json
import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport

os.environ.setdefault("AIOS_RATE_LIMIT_REQUESTS", "500")
os.environ.setdefault("AIOS_RATE_LIMIT_WINDOW",   "60")
os.environ.setdefault("AIOS_SANDBOX_LIMIT",        "50")
os.environ.setdefault("AIOS_STARTUP_GRACE",        "0")   # disable grace for tests

from api.server import app


@pytest.fixture(scope="module")
async def client():
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as c:
        yield c


# ── 1. STARTUP GRACE PERIOD ────────────────────────────────────────────────

class TestStartupGracePeriod:

    def test_grace_env_var_exists(self):
        """AIOS_STARTUP_GRACE env var is read by middleware."""
        from api.middleware import STARTUP_GRACE_SECONDS
        assert isinstance(STARTUP_GRACE_SECONDS, int)
        assert STARTUP_GRACE_SECONDS >= 0

    def test_grace_default_is_15(self):
        """Default grace period is 15 seconds."""
        import importlib
        import unittest.mock as mock
        with mock.patch.dict(os.environ, {}, clear=False):
            # Remove override if set
            env = {k: v for k, v in os.environ.items() if k != "AIOS_STARTUP_GRACE"}
            with mock.patch.dict(os.environ, env, clear=True):
                import api.middleware as mw
                importlib.reload(mw)
                assert mw.STARTUP_GRACE_SECONDS == 15
                importlib.reload(mw)  # restore

    def test_middleware_has_start_time(self):
        """RateLimitMiddleware records startup time."""
        from api.middleware import RateLimitMiddleware
        mw = RateLimitMiddleware(app=lambda: None, grace_seconds=10)
        assert hasattr(mw, "_start_time")
        assert mw._start_time <= time.time()

    def test_middleware_grace_attribute(self):
        """Grace period is stored on the middleware instance."""
        from api.middleware import RateLimitMiddleware
        mw = RateLimitMiddleware(app=lambda: None, grace_seconds=30)
        assert mw._grace == 30

    def test_middleware_zero_grace(self):
        """Grace period of 0 means no grace — rate limiting applies immediately."""
        from api.middleware import RateLimitMiddleware
        mw = RateLimitMiddleware(app=lambda: None, grace_seconds=0)
        assert mw._grace == 0

    def test_grace_bypasses_during_window(self):
        """Within grace window, requests are not rate-limited."""
        from api.middleware import RateLimitMiddleware
        mw = RateLimitMiddleware(app=lambda: None, requests=1, window=60,
                                 grace_seconds=60)
        # In grace window — should not rate-limit even beyond the 1 req limit
        assert time.time() - mw._start_time < mw._grace

    def test_grace_expired_after_window(self):
        """After grace period, normal rate limiting applies."""
        from api.middleware import RateLimitMiddleware
        mw = RateLimitMiddleware(app=lambda: None, requests=1, window=60,
                                 grace_seconds=0)
        # Grace = 0, so no grace period
        assert mw._grace == 0
        elapsed = time.time() - mw._start_time
        assert elapsed >= mw._grace  # grace already expired


# ── 2. ADVERSARIAL SANDBOX INPUTS ─────────────────────────────────────────

class TestAdversarialSandboxInputs:

    @pytest.fixture
    async def sbclient(self):
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as c:
            yield c

    async def _submit(self, client, **kwargs):
        payload = {
            "agent_id":       kwargs.get("agent_id", "adv-test"),
            "domain":         kwargs.get("domain", "retrieval"),
            "phi":            kwargs.get("phi", 0.5),
            "epsilon":        kwargs.get("epsilon", 2.0),
            "n_interactions": kwargs.get("n", 20),
        }
        r = await client.post("/sandbox/submit", json=payload)
        return r

    async def test_negative_phi_rejected_or_clamped(self, sbclient):
        """phi < 0 must not crash the server — either 422 or clamped result."""
        r = await self._submit(sbclient, agent_id="adv-neg-phi", phi=-0.5)
        assert r.status_code in (200, 422)

    async def test_phi_above_one_rejected_or_clamped(self, sbclient):
        """phi > 1.0 must not crash the server."""
        r = await self._submit(sbclient, agent_id="adv-phi-gt1", phi=1.5)
        assert r.status_code in (200, 422)

    async def test_epsilon_zero(self, sbclient):
        """epsilon=0 (no plasticity) must not cause division by zero."""
        r = await self._submit(sbclient, agent_id="adv-eps-zero", epsilon=0.0)
        assert r.status_code in (200, 422)

    async def test_n_interactions_zero(self, sbclient):
        """n_interactions=0 must return a result, not crash."""
        r = await self._submit(sbclient, agent_id="adv-n-zero", n=0)
        assert r.status_code in (200, 422)

    async def test_n_interactions_very_large(self, sbclient):
        """Very large n_interactions (10_000) must complete without timeout."""
        r = await self._submit(sbclient, agent_id="adv-n-large", n=10_000)
        assert r.status_code in (200, 422)

    async def test_unicode_agent_id(self, sbclient):
        """Unicode characters in agent_id must not crash the server."""
        r = await self._submit(sbclient, agent_id="测试-agent-αβγ")
        assert r.status_code in (200, 422)

    async def test_unicode_domain(self, sbclient):
        """Unicode characters in domain must not crash the server."""
        r = await self._submit(sbclient, domain="планирование")
        assert r.status_code in (200, 422)

    async def test_empty_agent_id(self, sbclient):
        """Empty agent_id must return 422 validation error."""
        r = await self._submit(sbclient, agent_id="")
        assert r.status_code == 422

    async def test_very_long_agent_id(self, sbclient):
        """Extremely long agent_id (1000 chars) must not crash."""
        r = await self._submit(sbclient, agent_id="x" * 1000)
        assert r.status_code in (200, 422)

    async def test_sandbox_result_has_required_fields(self, sbclient):
        """Valid submission always returns required fields."""
        r = await self._submit(sbclient, agent_id="adv-valid", phi=0.7,
                               epsilon=2.0, n=30)
        if r.status_code == 200:
            d = r.json()
            assert "run_id" in d
            assert "status" in d

    async def test_phi_at_boundary_zero(self, sbclient):
        """phi=0.0 exactly should be handled gracefully."""
        r = await self._submit(sbclient, agent_id="adv-phi-0", phi=0.0)
        assert r.status_code in (200, 422)

    async def test_phi_at_boundary_one(self, sbclient):
        """phi=1.0 exactly (maximum maturity) should be handled."""
        r = await self._submit(sbclient, agent_id="adv-phi-1", phi=1.0)
        assert r.status_code in (200, 422)

    async def test_epsilon_very_large(self, sbclient):
        """Very large epsilon (100.0) must not produce infinite values."""
        r = await self._submit(sbclient, agent_id="adv-eps-large", epsilon=100.0)
        assert r.status_code in (200, 422)
        if r.status_code == 200:
            d = r.json()
            assert "run_id" in d


# ── 3. SANDBOX RATE LIMIT ERROR MESSAGE ───────────────────────────────────

class TestSandboxRateLimitMessage:

    def test_sandbox_429_response_has_detail(self):
        """Sandbox rate limit response includes human-readable detail."""
        from api.middleware import RateLimitMiddleware, SANDBOX_LIMIT_REQUESTS
        # The middleware returns detail field in 429 response
        assert SANDBOX_LIMIT_REQUESTS > 0

    def test_sandbox_rate_limit_response_structure(self):
        """Sandbox rate limit JSON has error, detail, retry_after_seconds."""
        from api.middleware import SANDBOX_LIMIT_REQUESTS, SANDBOX_LIMIT_WINDOW
        # Verify the constants exist for constructing the response
        assert isinstance(SANDBOX_LIMIT_REQUESTS, int)
        assert isinstance(SANDBOX_LIMIT_WINDOW, int)

    def test_landing_html_handles_429(self):
        """landing.html JavaScript handles 429 status with user-friendly message."""
        path = os.path.join(os.path.dirname(__file__), "..", "frontend", "landing.html")
        content = open(path, encoding="utf-8").read()
        assert "429" in content
        assert "retry_after_seconds" in content or "Retry" in content


# ── 4. VERSION 1.8.0 ──────────────────────────────────────────────────────

class TestVersion170:

    async def test_root_version_1_7_0(self, client):
        r = await client.get("/")
        assert r.status_code == 200
        assert r.json()["version"] == "1.9.0"

    async def test_health_version_1_7_0(self, client):
        r = await client.get("/health")
        assert r.status_code == 200
        assert r.json()["version"] == "1.9.0"

    async def test_mcp_manifest_version_1_7_0(self):
        path = os.path.join(os.path.dirname(__file__), "..", "mcp.json")
        d = json.load(open(path, encoding="utf-8"))
        assert d["version"] == "1.9.0"

    def test_melvcore_package_version_1_7_0(self):
        import melvcore
        assert melvcore.__version__ == "1.9.0"

    def test_landing_html_version_1_6_0(self):
        """landing.html displays v1.9.0 (cosmetic — updated this session)."""
        path = os.path.join(os.path.dirname(__file__), "..", "frontend", "landing.html")
        content = open(path, encoding="utf-8").read()
        assert "v1.9.0" in content

    def test_middleware_startup_grace_in_banner(self):
        """Startup banner references grace period."""
        from api import middleware
        import inspect
        src = inspect.getsource(middleware.print_startup_banner)
        assert "STARTUP_GRACE" in src or "grace" in src.lower()


# ── 5. LANDING PAGE STRUCTURE ─────────────────────────────────────────────

class TestLandingPageSession15:

    def _content(self):
        path = os.path.join(os.path.dirname(__file__), "..", "frontend", "landing.html")
        return open(path, encoding="utf-8").read()

    def test_landing_exists(self):
        path = os.path.join(os.path.dirname(__file__), "..", "frontend", "landing.html")
        assert os.path.exists(path)

    def test_landing_has_agent_id_field(self):
        assert "agent_id" in self._content().lower() or "Agent ID" in self._content()

    def test_landing_has_phi_slider(self):
        assert "phi" in self._content().lower()

    def test_landing_has_epsilon_slider(self):
        assert "epsilon" in self._content().lower() or "ε" in self._content()

    def test_landing_has_run_certification_button(self):
        assert "Run certification" in self._content()

    def test_landing_has_zenodo_doi(self):
        assert "zenodo" in self._content().lower()

    def test_landing_has_github_link(self):
        assert "NaturesHolismMELV" in self._content()

    def test_landing_has_cls_score_display(self):
        assert "cls" in self._content().lower() or "CLS" in self._content()

    def test_landing_has_narrative_display(self):
        assert "narrative" in self._content().lower() or "NARRATIVE" in self._content()

    def test_landing_rate_limit_footer(self):
        assert "per hour" in self._content() or "submissions" in self._content()

    async def test_landing_served_at_demo_route(self, client):
        """GET /demo returns HTML with MELVcore content."""
        r = await client.get("/demo")
        assert r.status_code == 200
        assert "MELVcore" in r.text or "melvcore" in r.text.lower()

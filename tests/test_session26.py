"""
Session 26 — ε Decomposition
==============================
Tests for compute_epsilon_profile(), ecosystem_epsilon_summary(),
diagnosis badges, STC metric, and POST /sandbox/assess/epsilon-profile.

ε_effective = ε_intrinsic + ε_environmental

Epistemic status:
  ε_intrinsic     ③ verified (from AgentProfile.epsilon)
  ε_environmental ② theoretical (tool friction weights principled, not calibrated)
  badges          ② theoretical (thresholds principled)
  STC             ② theoretical (reference time not yet empirically calibrated)

Blueprint for Harmony Ch. 5: the diagnostic question no other framework
asks — is the performance problem intrinsic to the agent, or a function
of the infrastructure it operates in?
"""
import pytest
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def kernel():
    from core.melv_engine import MELVKernel
    return MELVKernel()


@pytest.fixture
def kernel_with_agents(kernel):
    """Kernel with three agents covering all badge scenarios."""
    from core.melv_engine import AgentProfile
    # Normal agent
    kernel.register_agent(AgentProfile(
        agent_id="normal", name="Normal", domain="compute", phi=0.60, epsilon=3.0
    ))
    # High-ε agent → AGENT_VOLATILE
    kernel.register_agent(AgentProfile(
        agent_id="volatile", name="Volatile", domain="compute", phi=0.50, epsilon=7.0
    ))
    # Low-φ, high-ε → LEGACY_CANDIDATE
    kernel.register_agent(AgentProfile(
        agent_id="legacy", name="Legacy", domain="compute", phi=0.25, epsilon=5.5
    ))
    return kernel


@pytest.fixture
def client():
    from api.server import app
    return TestClient(app, headers={"X-API-Key": "test-key"})


# ---------------------------------------------------------------------------
# Class 1 — TestEpsilonDecomposition (4 tests)
# ---------------------------------------------------------------------------

class TestEpsilonDecomposition:

    def test_epsilon_effective_equals_intrinsic_plus_environmental(self, kernel_with_agents):
        """ε_effective must equal ε_intrinsic + ε_environmental (within float tolerance)."""
        k = kernel_with_agents
        ep = k.compute_epsilon_profile("normal")
        assert abs(ep.epsilon_effective - (ep.epsilon_intrinsic + ep.epsilon_environmental)) < 1e-3, (
            f"ε_effective={ep.epsilon_effective} ≠ "
            f"ε_intrinsic({ep.epsilon_intrinsic}) + ε_env({ep.epsilon_environmental})"
        )

    def test_epsilon_override_is_used_as_intrinsic(self, kernel_with_agents):
        """When epsilon_intrinsic is supplied explicitly, it should override agent.epsilon."""
        k  = kernel_with_agents
        ep = k.compute_epsilon_profile("normal", epsilon_intrinsic=6.5)
        assert ep.epsilon_intrinsic == pytest.approx(6.5, abs=1e-3), (
            f"Expected ε_intrinsic=6.5, got {ep.epsilon_intrinsic}"
        )

    def test_high_epsilon_healthy_env_no_volatile_badge(self, kernel_with_agents):
        """
        Session 30c: Agent with high ε (7.0) and moderate φ (0.50) in a default
        β environment (1.0) must NOT receive AGENT_VOLATILE. High ε is adaptive
        range — it is only a mismatch when φ AND β are both below their support
        thresholds simultaneously. This agent is in a monitoring zone, not a
        crisis zone.
        """
        k  = kernel_with_agents
        ep = k.compute_epsilon_profile("volatile")
        assert "AGENT_VOLATILE" not in ep.badges, (
            f"High-ε agent in default environment should not be AGENT_VOLATILE. "
            f"Got badges: {ep.badges}. phi={ep.phi:.2f}, beta={ep.beta_mean:.2f}"
        )

    def test_legacy_candidate_badge_low_phi_high_epsilon(self, kernel_with_agents):
        """Agent with φ ≤ 0.35 AND ε_effective ≥ 4.0 must receive LEGACY_CANDIDATE badge."""
        k  = kernel_with_agents
        ep = k.compute_epsilon_profile("legacy")
        assert "LEGACY_CANDIDATE" in ep.badges, (
            f"Expected LEGACY_CANDIDATE for φ={ep.phi}, ε_eff={ep.epsilon_effective}. "
            f"Got badges: {ep.badges}"
        )


# ---------------------------------------------------------------------------
# Class 2 — TestEnvironmentalBottleneck (2 tests)
# ---------------------------------------------------------------------------

class TestEnvironmentalBottleneck:

    def test_env_bottleneck_badge_on_low_beta(self, kernel):
        """
        Suppressing all β values should drive ε_environmental above the
        ENV_BOTTLENECK_THRESHOLD and trigger ENV_BOTTLENECKED badge.
        """
        from core.melv_engine import AgentProfile, ENV_BOTTLENECK_THRESHOLD

        kernel.register_agent(AgentProfile(
            agent_id="stressed", name="Stressed", domain="compute",
            phi=0.6, epsilon=3.0
        ))

        # Suppress all β resources to minimum — maximum friction
        for resource in ["compute", "api_quota", "vector_db",
                         "storage", "token_budget", "context_window"]:
            kernel.beta.set(resource, 0.1)

        ep = kernel.compute_epsilon_profile("stressed")
        assert ep.epsilon_environmental >= ENV_BOTTLENECK_THRESHOLD, (
            f"Expected ε_env ≥ {ENV_BOTTLENECK_THRESHOLD} with β=0.1 everywhere. "
            f"Got ε_env={ep.epsilon_environmental}"
        )
        assert "ENV_BOTTLENECKED" in ep.badges, (
            f"Expected ENV_BOTTLENECKED badge. Got: {ep.badges}"
        )

    def test_high_beta_reduces_environmental_component(self, kernel):
        """
        Raising all β values should reduce ε_environmental relative to low-β.
        """
        from core.melv_engine import AgentProfile

        kernel.register_agent(AgentProfile(
            agent_id="lush", name="Lush", domain="compute",
            phi=0.6, epsilon=3.0
        ))

        # Low-β baseline
        for resource in ["compute", "api_quota", "vector_db",
                         "storage", "token_budget", "context_window"]:
            kernel.beta.set(resource, 0.2)
        ep_low = kernel.compute_epsilon_profile("lush")

        # High-β environment
        for resource in ["compute", "api_quota", "vector_db",
                         "storage", "token_budget", "context_window"]:
            kernel.beta.set(resource, 2.5)
        ep_high = kernel.compute_epsilon_profile("lush")

        assert ep_low.epsilon_environmental > ep_high.epsilon_environmental, (
            f"Low-β ε_env ({ep_low.epsilon_environmental}) should exceed "
            f"high-β ε_env ({ep_high.epsilon_environmental})"
        )


# ---------------------------------------------------------------------------
# Class 3 — TestSTC (1 test)
# ---------------------------------------------------------------------------

class TestSTC:

    def test_stc_scales_with_epsilon_effective(self, kernel):
        """
        Higher ε_effective must produce a larger STC (slower path to cooperation).
        """
        from core.melv_engine import AgentProfile

        kernel.register_agent(AgentProfile(
            agent_id="fast", name="Fast", domain="compute", phi=0.7, epsilon=1.5
        ))
        kernel.register_agent(AgentProfile(
            agent_id="slow", name="Slow", domain="compute", phi=0.7, epsilon=7.0
        ))

        ep_fast = kernel.compute_epsilon_profile("fast")
        ep_slow = kernel.compute_epsilon_profile("slow")

        assert ep_slow.stc_seconds > ep_fast.stc_seconds, (
            f"High-ε agent STC ({ep_slow.stc_seconds}s) should exceed "
            f"low-ε agent STC ({ep_fast.stc_seconds}s)"
        )


# ---------------------------------------------------------------------------
# Class 4 — TestEpsilonProfileEndpoint (1 test)
# ---------------------------------------------------------------------------

class TestEpsilonProfileEndpoint:

    def test_epsilon_profile_endpoint_returns_expected_structure(self, client):
        """
        POST /sandbox/assess/epsilon-profile must return the required fields
        with correct types and valid ranges.

        Note: we do NOT use 'with client as c:' here because that triggers the
        full lifespan (including MCP StreamableHTTPSessionManager.run()), which
        raises RuntimeError on the second invocation within the same test session
        ("can only be called once per instance"). Instead we import the app
        directly and access kernel state through the module-level singleton,
        which is already initialised by earlier tests in the session.
        """
        from api.server import app
        from core.melv_engine import AgentProfile

        # Access the kernel directly — app is already initialised by the
        # module-level import; earlier tests have run the lifespan startup.
        # If kernel isn't on app.state yet (isolated test run), register via
        # the endpoint instead.
        try:
            kernel = app.state.kernel
            if "ep_test_agent" not in kernel.agents:
                kernel.register_agent(AgentProfile(
                    agent_id="ep_test_agent",
                    name="EP Test Agent",
                    domain="compute",
                    phi=0.55,
                    epsilon=4.0,
                ))
        except AttributeError:
            # app.state.kernel not yet set — register via Gateway API
            client.post("/melv/register", json={
                "agent_id": "ep_test_agent",
                "name": "EP Test Agent",
                "domain": "compute",
                "phi": 0.55,
                "epsilon": 4.0,
            })

        resp = client.post(
            "/sandbox/assess/epsilon-profile",
            json={"agent_ids": ["ep_test_agent"], "epsilon_overrides": {}}
        )
        assert resp.status_code == 200, f"Unexpected {resp.status_code}: {resp.text}"
        data = resp.json()

        # Top-level structure
        for field in ("session", "version", "agent_count", "profiles",
                      "badge_counts", "dominant_bottleneck", "epistemic_status"):
            assert field in data, f"Missing top-level field: '{field}'"

        assert data["session"]  == "26"
        assert data["version"]  == "2.2.0"
        assert data["agent_count"] == 1

        # Profile structure
        assert len(data["profiles"]) == 1
        p = data["profiles"][0]
        for field in ("agent_id", "epsilon_intrinsic", "epsilon_environmental",
                      "epsilon_effective", "phi", "beta_mean", "stc_seconds",
                      "badges", "resource_friction", "interpretation"):
            assert field in p, f"Missing profile field: '{field}'"

        # Semantic checks
        assert p["epsilon_effective"] == pytest.approx(
            p["epsilon_intrinsic"] + p["epsilon_environmental"], abs=1e-3
        )
        assert p["stc_seconds"] > 0
        assert isinstance(p["badges"], list)
        assert data["dominant_bottleneck"] in ("agent", "environment", "balanced")

        # Epistemic status present
        es = data["epistemic_status"]
        assert "epsilon_intrinsic"     in es
        assert "epsilon_environmental" in es

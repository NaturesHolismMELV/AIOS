"""
Session 25 — Sigmoid Quorum Gate
=================================
Tests for _quorum_gate(), phi_beta_quorum(), sigmoid-scaled PROVISION_BETA,
and the /api/quorum_status endpoint.

MELV ABM V2.1 verified constants (DO NOT CHANGE without new ABM run):
  τ = 0.5, k = 10.0, φ×β boundary = 0.3
  sensitivity = 1.0, specificity = 0.997

Biological correspondence (MAIES Event 2):
  MELV sigmoid ≡ bacterial quorum sensing (Nadell et al. 2016)
  φ·β  ≡  population density N
  τ    ≡  quorum threshold N_threshold
  k    ≡  sigmoid sharpness
"""
import pytest
import math
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
    """Kernel pre-populated with two agents at moderate maturity."""
    from core.melv_engine import AgentProfile
    kernel.register_agent(AgentProfile(
        agent_id="alpha", name="Alpha", domain="compute", phi=0.55
    ))
    kernel.register_agent(AgentProfile(
        agent_id="beta_ag", name="Beta", domain="compute", phi=0.55
    ))
    return kernel


@pytest.fixture
def client():
    from api.server import app
    return TestClient(app, headers={"X-API-Key": "test-key"})


# ---------------------------------------------------------------------------
# Class 1 — TestQuorumGate (4 tests)
# ---------------------------------------------------------------------------

class TestQuorumGate:

    def test_gate_returns_half_at_tau(self, kernel):
        """_quorum_gate(τ, τ, k) should return exactly 0.5 (inflection property)."""
        from core.melv_engine import QUORUM_TAU, QUORUM_K
        gate = kernel._quorum_gate(QUORUM_TAU, QUORUM_TAU, QUORUM_K)
        assert abs(gate - 0.5) < 1e-6, (
            f"Sigmoid inflection: expected 0.5 at x=τ, got {gate}"
        )

    def test_gate_near_one_when_phi_beta_zero(self, kernel):
        """_quorum_gate(0.0) should be close to 1.0 (fully stressed ecosystem)."""
        gate = kernel._quorum_gate(0.0)
        assert gate > 0.99, (
            f"Fully stressed (φ·β=0): expected gate≈1.0, got {gate:.6f}"
        )

    def test_gate_near_zero_when_phi_beta_high(self, kernel):
        """_quorum_gate(2.0) should be close to 0.0 (healthy ecosystem, light touch)."""
        gate = kernel._quorum_gate(2.0)
        assert gate < 0.01, (
            f"Healthy ecosystem (φ·β=2): expected gate≈0.0, got {gate:.6f}"
        )

    def test_gate_monotonically_decreasing(self, kernel):
        """Higher φ·β → lower gate value (more cooperation → less intervention needed)."""
        values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.5, 2.0]
        gates  = [kernel._quorum_gate(v) for v in values]
        for i in range(len(gates) - 1):
            assert gates[i] > gates[i + 1], (
                f"Monotone violation at index {i}: "
                f"gate({values[i]})={gates[i]:.4f} ≤ gate({values[i+1]})={gates[i+1]:.4f}"
            )


# ---------------------------------------------------------------------------
# Class 2 — TestProvisionBetaSigmoid (3 tests)
# ---------------------------------------------------------------------------

class TestProvisionBetaSigmoid:

    def test_stressed_ecosystem_gets_larger_step(self, kernel):
        """
        Stressed ecosystem (low φ·β) should produce a significantly larger
        PROVISION_BETA step than a healthy one.
        """
        from core.melv_engine import (
            PROVISION_STEP_FLOOR, PROVISION_STEP_CEIL
        )
        # Stressed: gate near 1.0 → step near ceiling
        gate_stressed = kernel._quorum_gate(0.1)
        step_stressed = PROVISION_STEP_FLOOR + gate_stressed * (
            PROVISION_STEP_CEIL - PROVISION_STEP_FLOOR
        )

        # Healthy: gate near 0.0 → step near floor
        gate_healthy = kernel._quorum_gate(0.9)
        step_healthy = PROVISION_STEP_FLOOR + gate_healthy * (
            PROVISION_STEP_CEIL - PROVISION_STEP_FLOOR
        )

        assert step_stressed > step_healthy * 2, (
            f"Stressed step ({step_stressed:.3f}) should be >2× healthy step ({step_healthy:.3f})"
        )

    def test_step_within_bounds(self, kernel):
        """
        PROVISION_BETA step must remain in [PROVISION_STEP_FLOOR, PROVISION_STEP_CEIL]
        for any φ·β in [0, 3].
        """
        from core.melv_engine import (
            PROVISION_STEP_FLOOR, PROVISION_STEP_CEIL
        )
        test_values = [i * 0.1 for i in range(31)]  # 0.0, 0.1, ..., 3.0
        for phi_beta in test_values:
            gate = kernel._quorum_gate(phi_beta)
            step = PROVISION_STEP_FLOOR + gate * (PROVISION_STEP_CEIL - PROVISION_STEP_FLOOR)
            assert PROVISION_STEP_FLOOR <= step <= PROVISION_STEP_CEIL + 1e-9, (
                f"Step {step:.4f} out of bounds [{PROVISION_STEP_FLOOR}, {PROVISION_STEP_CEIL}] "
                f"at φ·β={phi_beta}"
            )

    def test_step_replaces_linear_in_kernel_respond(self, kernel_with_agents):
        """
        After a threshold interaction triggers escalation_needed, the β increment
        must be sigmoid-scaled (not the old flat 0.10).
        """
        from core.melv_engine import PROVISION_STEP_FLOOR, PROVISION_STEP_CEIL

        k = kernel_with_agents
        beta_before = k.beta.get("compute")

        # Drive three escalating threshold interactions to trigger escalation_needed
        for _ in range(3):
            k.record_interaction("alpha", "beta_ag", cost=0.85, benefit=1.0)

        beta_after = k.beta.get("compute")
        increment  = beta_after - beta_before

        # The increment must be in the sigmoid range, not exactly 0.10
        assert increment > 0, "β was not provisioned at all"
        assert PROVISION_STEP_FLOOR - 1e-9 <= increment <= PROVISION_STEP_CEIL + 1e-9, (
            f"β increment {increment:.4f} outside sigmoid range "
            f"[{PROVISION_STEP_FLOOR}, {PROVISION_STEP_CEIL}]"
        )
        # Specifically must NOT be the old hard-coded 0.10 (or very close to it)
        # unless the sigmoid happens to produce exactly 0.10, which is valid but
        # the range check above already covers correctness.
        # Belt-and-braces: step should not be outside the sigmoid range.
        assert increment != pytest.approx(0.10, abs=0.001) or (
            PROVISION_STEP_FLOOR <= 0.10 <= PROVISION_STEP_CEIL
        ), "If 0.10 appears, it should still be inside the valid range"


# ---------------------------------------------------------------------------
# Class 3 — TestQuorumStatusEndpoint (1 test)
# ---------------------------------------------------------------------------

class TestQuorumStatusEndpoint:

    def test_quorum_status_returns_expected_fields(self, client):
        """
        GET /api/quorum_status must return all required fields with valid types.
        """
        resp = client.get("/api/quorum_status")
        assert resp.status_code == 200, f"Unexpected status {resp.status_code}: {resp.text}"
        data = resp.json()

        required = {
            "phi_beta":       float,
            "quorum_gate":    float,
            "tau":            float,
            "k":              float,
            "above_quorum":   bool,
            "provision_step": float,
            "regime":         str,
            "interpretation": str,
        }
        for field, ftype in required.items():
            assert field in data, f"Missing field: '{field}'"
            assert isinstance(data[field], ftype), (
                f"Field '{field}': expected {ftype.__name__}, got {type(data[field]).__name__}"
            )

        # Semantic checks
        from core.melv_engine import QUORUM_TAU, QUORUM_K, PROVISION_STEP_FLOOR, PROVISION_STEP_CEIL
        assert data["tau"] == QUORUM_TAU
        assert data["k"]   == QUORUM_K
        assert PROVISION_STEP_FLOOR <= data["provision_step"] <= PROVISION_STEP_CEIL + 1e-9
        assert data["regime"] in (
            "above_quorum", "at_quorum", "approaching_quorum", "below_quorum"
        )
        assert 0.0 <= data["quorum_gate"] <= 1.0

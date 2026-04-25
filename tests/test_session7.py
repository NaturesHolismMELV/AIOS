"""
test_session7.py — Nudge v2 test suite
=======================================
9 offline tests covering:
  1. retry_with_jitter produced at depth 1
  2. rephrase produced at depth 2
  3. yield produced at depth 3
  4. niche_diverge produced at depth 4+
  5. High-φ agent receives niche_diverge one depth earlier (depth 3)
  6. Low-φ agent retries longer (depth 1 stays retry_with_jitter)
  7. contention_depth escalation via MELVKernel.record_interaction
  8. oxpecker effect raises adjacent β by 0.05–0.10 (kernel applies, not agent)
  9. Gateway NudgeResponse is a dict with required fields (structure test)

All tests are offline (no API key required).

Run: python -m pytest tests/test_session7.py -v
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.nudge_engine import NudgeEngine, NudgeResponse
from core.melv_engine import MELVKernel, AgentProfile, BetaEnvironment


# ── FIXTURES ───────────────────────────────────────────────────────────────

@pytest.fixture
def engine():
    return NudgeEngine()


@pytest.fixture
def kernel():
    k = MELVKernel()
    k.register_agent(AgentProfile(
        agent_id="agent_a", name="AgentA", domain="compute", phi=0.7
    ))
    k.register_agent(AgentProfile(
        agent_id="agent_b", name="AgentB", domain="compute", phi=0.6
    ))
    return k


# ── TEST 1: depth 1 → retry_with_jitter ───────────────────────────────────

def test_depth1_produces_retry_with_jitter(engine):
    """Depth 1, normal φ → retry_with_jitter"""
    resp = engine.build_nudge_v2(
        action="nudge",
        beta_i=0.75,
        resource="token_budget",
        contention_depth=1,
        agent_phi=0.5,
    )
    assert isinstance(resp, NudgeResponse)
    assert resp.nudge_type == "retry_with_jitter"
    assert "delay_ms" in resp.params
    assert 200 <= resp.params["delay_ms"] <= 800
    assert resp.contention_depth == 1
    assert resp.resource == "token_budget"


# ── TEST 2: depth 2 → rephrase ────────────────────────────────────────────

def test_depth2_produces_rephrase(engine):
    """Depth 2, normal φ → rephrase with temperature_delta"""
    resp = engine.build_nudge_v2(
        action="nudge",
        beta_i=0.85,
        resource="api_quota",
        contention_depth=2,
        agent_phi=0.5,
    )
    assert resp.nudge_type == "rephrase"
    assert resp.params["temperature_delta"] == pytest.approx(0.2)
    assert resp.phi_delta > 0


# ── TEST 3: depth 3 → yield ───────────────────────────────────────────────

def test_depth3_produces_yield(engine):
    """Depth 3, normal φ → yield with duration_ms"""
    resp = engine.build_nudge_v2(
        action="niche_divergence",
        beta_i=1.1,
        resource="compute",
        contention_depth=3,
        agent_phi=0.5,
    )
    assert resp.nudge_type == "yield"
    assert resp.params["duration_ms"] >= 1000
    assert resp.params["resource"] == "compute"
    assert resp.phi_delta > 0


# ── TEST 4: depth 4+ → niche_diverge ─────────────────────────────────────

def test_depth4_produces_niche_diverge(engine):
    """Depth 4+, normal φ → niche_diverge with suggested_domain"""
    resp = engine.build_nudge_v2(
        action="niche_divergence",
        beta_i=1.4,
        resource="token_budget",
        contention_depth=4,
        agent_phi=0.5,
    )
    assert resp.nudge_type == "niche_diverge"
    assert resp.niche_suggestion != ""
    assert resp.params["suggested_domain"] != resp.params["current_resource"]
    assert resp.phi_delta == pytest.approx(0.05)


# ── TEST 5: high-φ agent → early niche_diverge at depth 3 ────────────────

def test_high_phi_agent_niche_diverge_at_depth3(engine):
    """High-φ agent (≥ 0.75) advances one depth → niche_diverge at depth 3"""
    resp = engine.build_nudge_v2(
        action="niche_divergence",
        beta_i=1.1,
        resource="compute",
        contention_depth=3,
        agent_phi=0.80,   # high φ — should advance depth by 1 → eff_depth=4
    )
    assert resp.nudge_type == "niche_diverge", (
        f"Expected niche_diverge for high-φ agent at depth 3, got {resp.nudge_type}"
    )


# ── TEST 6: low-φ agent → retries longer ─────────────────────────────────

def test_low_phi_agent_retries_at_depth2(engine):
    """Low-φ agent (< 0.50) held back one depth → retry_with_jitter at depth 2"""
    resp = engine.build_nudge_v2(
        action="nudge",
        beta_i=0.85,
        resource="api_quota",
        contention_depth=2,
        agent_phi=0.30,   # low φ — depth reduced by 1 → eff_depth=1
    )
    assert resp.nudge_type == "retry_with_jitter", (
        f"Expected retry_with_jitter for low-φ agent at depth 2, got {resp.nudge_type}"
    )


# ── TEST 7: MELVKernel contention depth escalation ────────────────────────

def test_kernel_contention_depth_escalation(kernel):
    """
    Contention depth increments on threshold/conflict, resets on cooperative.
    """
    a, b = "agent_a", "agent_b"

    # Cooperative interaction: depth stays 0
    kernel.record_interaction(a, b, cost=0.3, benefit=2.0, resource_type="compute")
    assert kernel.get_contention_depth(a, b) == 0

    # Two threshold interactions: depth should be 2
    # β(compute)=1.0, i=cost/benefit. Need βi ∈ [0.70, 1.0) for threshold.
    # i = 0.8/1.0 = 0.8; βi = 1.0 * 0.8 = 0.8 → threshold
    kernel.record_interaction(a, b, cost=0.8, benefit=1.0, resource_type="compute")
    assert kernel.get_contention_depth(a, b) == 1

    kernel.record_interaction(a, b, cost=0.8, benefit=1.0, resource_type="compute")
    assert kernel.get_contention_depth(a, b) == 2

    # Cooperative interaction resets depth
    kernel.record_interaction(a, b, cost=0.1, benefit=2.0, resource_type="compute")
    assert kernel.get_contention_depth(a, b) == 0


# ── TEST 8: oxpecker effect raises adjacent β ─────────────────────────────

def test_oxpecker_effect_raises_adjacent_beta():
    """
    When niche_diverge fires: the vacated resource β increases by 0.05–0.10.
    Vacating agent's own β is unchanged.
    BetaEnvironment is modified by kernel, not by agent.
    """
    env = BetaEnvironment()
    engine = NudgeEngine()

    before_token = env.get("token_budget")
    before_compute = env.get("compute")

    result = engine.apply_oxpecker_effect(
        vacating_agent="agent_a",
        resource_type="token_budget",
        environment=env,
    )

    assert result["oxpecker_effect"] is True
    assert result["vacating_agent"] == "agent_a"

    after_token = env.get("token_budget")
    delta = after_token - before_token

    assert 0.05 <= delta <= 0.10, (
        f"Expected β lift of 0.05–0.10 for token_budget, got {delta:.3f}"
    )

    # Compute β unchanged (no adjacent types specified)
    assert env.get("compute") == before_compute

    # Report contains the adjustment details
    assert "token_budget" in result["beta_adjustments"]
    adj = result["beta_adjustments"]["token_budget"]
    assert adj["before"] == pytest.approx(before_token)
    assert adj["after"] == pytest.approx(after_token)


# ── TEST 9: NudgeResponse to_dict() has required gateway fields ────────────

def test_nudge_response_dict_structure(engine):
    """NudgeResponse.to_dict() contains all fields required by the gateway response."""
    resp = engine.build_nudge_v2(
        action="niche_divergence",
        beta_i=1.2,
        resource="compute",
        contention_depth=4,
        agent_phi=0.6,
    )
    d = resp.to_dict()

    required_fields = {
        "nudge_type", "contention_depth", "params",
        "rationale", "phi_delta", "niche_suggestion",
        "beta_i", "resource", "cost_threshold",
    }
    missing = required_fields - set(d.keys())
    assert not missing, f"Missing fields in NudgeResponse dict: {missing}"

    # cost_threshold should match CostCalculator cap (2.0)
    assert d["cost_threshold"] == pytest.approx(2.0)

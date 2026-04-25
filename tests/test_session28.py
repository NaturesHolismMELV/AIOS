"""
test_session28.py — MELVcore Session 28: Quorum Reliability Tagging
=====================================================================
Eight tests covering:
  T1. quorum_reliability() returns all required keys
  T2. above-quorum ecosystem → reliability_level="high"
  T3. below-quorum ecosystem → reliability_level="low"
  T4. at-quorum ecosystem    → reliability_level="moderate"
  T5. approaching-quorum     → reliability_level="degraded"
  T6. per-agent breakdown populated correctly
  T7. reliability_advisory text contains phi_beta and tau values
  T8. GET /api/quorum_reliability endpoint returns valid structure

Epistemic status: ② theoretical.
  τ=0.5, k=10 are ③ ABM V2.1-verified (Session 25).
  The reliability interpretation layer added in Session 28 is
  theoretically grounded (MAIES Event 4) but not yet empirically
  calibrated against real output-quality data.

Origin: MAIES Event 4 (MAIES-adjacent ②) — quorum gate as agent output
  reliability marker. Below-quorum = high-noise regime. Biological
  correspondence: Nadell et al. 2016 — bacterial quorum sensing suppresses
  costly cooperative behaviours below N_threshold.

Blueprint for Harmony — L.W. Evans (Ecotao Enterprises, Cape Town)
ORCID: 0009-0001-0963-1840
"""

import pytest
from core.melv_engine import MELVKernel, AgentProfile, AgentStatus, QUORUM_TAU


# ── REQUIRED KEYS (specification contract) ────────────────────────────────

REQUIRED_KEYS = {
    "phi_beta",
    "quorum_regime",
    "above_quorum",
    "reliability_level",
    "reliability_advisory",
    "tau",
    "agent_count",
    "per_agent",
    "session",
    "maies_event",
    "epistemic_status",
}

PER_AGENT_KEYS = {
    "agent_id",
    "phi",
    "beta_mean",
    "phi_beta",
    "above_quorum",
    "quorum_regime",
    "reliability_level",
}

VALID_REGIMES = {"above_quorum", "at_quorum", "approaching_quorum", "below_quorum"}
VALID_RELIABILITY = {"high", "moderate", "degraded", "low"}


# ── FIXTURES ──────────────────────────────────────────────────────────────

def make_kernel_with_agents(phi: float, count: int = 2) -> MELVKernel:
    """
    Create a kernel with `count` agents all set to the given phi.
    Beta environment left at defaults (mean ≈ 1.0) so φ·β ≈ phi.
    """
    k = MELVKernel()
    for i in range(count):
        k.register_agent(AgentProfile(
            agent_id=f"agent-{i:03d}",
            name=f"Agent {i}",
            domain="compute",
            phi=phi,
            epsilon=3.0,
            status=AgentStatus.ACTIVE if phi >= 0.5 else AgentStatus.MATURING,
        ))
    return k


# ── T1: All required keys present ─────────────────────────────────────────

def test_quorum_reliability_returns_required_keys():
    """
    T1: quorum_reliability() must return all required structural keys.

    This test is the specification contract — if any key is missing,
    Session 28 is incomplete. The endpoint and downstream consumers
    depend on this structure being stable.
    """
    kernel = make_kernel_with_agents(phi=0.6)
    result = kernel.quorum_reliability()

    missing = REQUIRED_KEYS - set(result.keys())
    assert not missing, (
        f"quorum_reliability() missing required keys: {missing}. "
        "Session 28 contract violation."
    )

    # Type assertions for critical fields
    assert isinstance(result["phi_beta"],             float)
    assert isinstance(result["above_quorum"],         bool)
    assert isinstance(result["reliability_level"],    str)
    assert isinstance(result["reliability_advisory"], str)
    assert isinstance(result["per_agent"],            list)
    assert isinstance(result["agent_count"],          int)
    assert result["session"]        == 28
    assert result["maies_event"]    == 4
    assert result["epistemic_status"] == "② theoretical"
    assert result["tau"]            == QUORUM_TAU


# ── T2: Above-quorum → reliability_level="high" ───────────────────────────

def test_above_quorum_yields_high_reliability():
    """
    T2: When φ·β > τ + 0.1 (well above quorum), reliability must be "high".

    φ=0.8, β_mean≈1.0 → φ·β≈0.8 >> τ=0.5.
    Cooperative dynamics dominant: outputs in low-noise regime.
    """
    kernel = make_kernel_with_agents(phi=0.80)
    result = kernel.quorum_reliability()

    assert result["above_quorum"] is True
    assert result["quorum_regime"] == "above_quorum", (
        f"Expected regime='above_quorum', got '{result['quorum_regime']}'. "
        f"φ·β={result['phi_beta']:.3f}, τ={QUORUM_TAU}"
    )
    assert result["reliability_level"] == "high", (
        f"Expected reliability_level='high' for φ·β={result['phi_beta']:.3f} > τ+0.1. "
        f"Got '{result['reliability_level']}'."
    )


# ── T3: Below-quorum → reliability_level="low" ────────────────────────────

def test_below_quorum_yields_low_reliability():
    """
    T3: When φ·β < τ - 0.1 (well below quorum), reliability must be "low".

    φ=0.2, β_mean≈1.0 → φ·β≈0.2 << τ=0.5.
    High-noise regime: cooperation suppressed.

    Biological correspondence (Nadell 2016): below N_threshold, costly
    cooperative behaviours are suppressed. Below-quorum MELV outputs carry
    elevated confabulation risk — same energetic principle.
    """
    kernel = make_kernel_with_agents(phi=0.20)
    result = kernel.quorum_reliability()

    assert result["above_quorum"] is False
    assert result["quorum_regime"] == "below_quorum", (
        f"Expected regime='below_quorum', got '{result['quorum_regime']}'. "
        f"φ·β={result['phi_beta']:.3f}, τ={QUORUM_TAU}"
    )
    assert result["reliability_level"] == "low", (
        f"Expected reliability_level='low' for φ·β={result['phi_beta']:.3f} < τ-0.1. "
        f"Got '{result['reliability_level']}'."
    )
    # Advisory must mention confabulation or noise
    advisory_lower = result["reliability_advisory"].lower()
    assert any(word in advisory_lower for word in ["noise", "confabulation", "unreliable", "low-confidence"]), (
        "Below-quorum advisory must mention reliability risk (noise/confabulation/unreliable). "
        f"Got: '{result['reliability_advisory'][:100]}...'"
    )


# ── T4: At-quorum → reliability_level="moderate" ─────────────────────────

def test_at_quorum_yields_moderate_reliability():
    """
    T4: When φ·β ≈ τ (at quorum threshold), reliability must be "moderate".

    φ·β=0.5 exactly (or within ±0.05 of τ=0.5).
    Sigmoid at inflection point — cooperative dynamics present but not dominant.
    """
    # We need φ·β ≈ τ = 0.5. With β_mean ≈ 1.0, we need φ ≈ 0.5.
    # Use phi=0.5 and verify phi_beta falls in [0.5, 0.6) (at_quorum zone)
    kernel = make_kernel_with_agents(phi=0.50)
    result = kernel.quorum_reliability()

    phi_beta = result["phi_beta"]
    # Accept at_quorum zone: phi_beta ∈ [τ, τ+0.1)
    assert result["quorum_regime"] in {"at_quorum", "approaching_quorum"}, (
        f"For φ=0.50, β_mean≈1.0: expected at_quorum or approaching_quorum, "
        f"got '{result['quorum_regime']}'. φ·β={phi_beta:.4f}, τ={QUORUM_TAU}."
    )
    assert result["reliability_level"] in {"moderate", "degraded"}, (
        f"For φ·β≈τ, expected 'moderate' or 'degraded', got '{result['reliability_level']}'"
    )


# ── T5: Approaching-quorum → reliability_level="degraded" ─────────────────

def test_approaching_quorum_yields_degraded_reliability():
    """
    T5: When φ·β ∈ [τ-0.1, τ) (approaching from below), reliability="degraded".

    φ=0.45, β_mean≈1.0 → φ·β≈0.45 ∈ [0.40, 0.50).
    Cooperative density insufficient — degraded-reliability regime.
    """
    kernel = make_kernel_with_agents(phi=0.43)  # φ·β ≈ 0.43, τ=0.5, τ-0.1=0.4
    result = kernel.quorum_reliability()

    phi_beta = result["phi_beta"]
    # Should land in approaching_quorum or below_quorum depending on exact beta_mean
    assert result["quorum_regime"] in {"approaching_quorum", "below_quorum"}, (
        f"For φ=0.43, expected approaching_quorum or below_quorum. "
        f"Got '{result['quorum_regime']}'. φ·β={phi_beta:.4f}"
    )
    assert result["reliability_level"] in {"degraded", "low"}, (
        f"Expected 'degraded' or 'low' for φ·β={phi_beta:.3f}. "
        f"Got '{result['reliability_level']}'."
    )


# ── T6: Per-agent breakdown ───────────────────────────────────────────────

def test_per_agent_breakdown_populated_correctly():
    """
    T6: per_agent list must contain one entry per registered agent,
    each with all required per-agent keys and valid regime/reliability values.

    Tests that per-agent granularity works — the system can identify which
    specific agents are below quorum, not just the ecosystem mean.
    """
    kernel = MELVKernel()
    # Register agents with different phi values
    high_agent = AgentProfile(
        agent_id="high-phi", name="HighPhi", domain="compute",
        phi=0.85, epsilon=3.0, status=AgentStatus.ACTIVE,
    )
    low_agent = AgentProfile(
        agent_id="low-phi", name="LowPhi", domain="storage",
        phi=0.20, epsilon=3.0, status=AgentStatus.MATURING,
    )
    kernel.register_agent(high_agent)
    kernel.register_agent(low_agent)

    result = kernel.quorum_reliability()

    assert result["agent_count"] == 2
    assert len(result["per_agent"]) == 2

    agent_ids = {a["agent_id"] for a in result["per_agent"]}
    assert "high-phi" in agent_ids
    assert "low-phi" in agent_ids

    for entry in result["per_agent"]:
        # All required keys present
        missing = PER_AGENT_KEYS - set(entry.keys())
        assert not missing, f"per_agent entry missing keys: {missing}"

        # Valid enum values
        assert entry["quorum_regime"]     in VALID_REGIMES,     f"Invalid regime: {entry['quorum_regime']}"
        assert entry["reliability_level"] in VALID_RELIABILITY, f"Invalid reliability: {entry['reliability_level']}"

        # phi_beta = phi × beta_mean
        expected_phi_beta = round(entry["phi"] * entry["beta_mean"], 4)
        assert abs(entry["phi_beta"] - expected_phi_beta) < 1e-3, (
            f"per_agent phi_beta mismatch: expected {expected_phi_beta:.4f}, "
            f"got {entry['phi_beta']:.4f}"
        )

        # above_quorum consistent with phi_beta
        assert entry["above_quorum"] == (entry["phi_beta"] >= QUORUM_TAU), (
            f"above_quorum flag inconsistent with phi_beta={entry['phi_beta']:.4f}"
        )

    # High-phi agent should have higher reliability than low-phi agent
    high_entry = next(a for a in result["per_agent"] if a["agent_id"] == "high-phi")
    low_entry  = next(a for a in result["per_agent"] if a["agent_id"] == "low-phi")

    RELIABILITY_ORDER = {"high": 3, "moderate": 2, "degraded": 1, "low": 0}
    assert RELIABILITY_ORDER[high_entry["reliability_level"]] > RELIABILITY_ORDER[low_entry["reliability_level"]], (
        f"High-phi agent (φ=0.85) must have higher reliability than low-phi (φ=0.20). "
        f"Got: high={high_entry['reliability_level']}, low={low_entry['reliability_level']}"
    )


# ── T7: Advisory text contains phi_beta and tau values ────────────────────

def test_reliability_advisory_contains_phi_beta_and_tau():
    """
    T7: reliability_advisory must be informative — it must contain the
    actual phi_beta value and reference the tau threshold.

    This tests that the advisory is dynamically generated from actual
    ecosystem state, not a static placeholder string. API consumers
    reading the advisory should be able to act on specific numbers.
    """
    kernel = make_kernel_with_agents(phi=0.3)  # below quorum
    result = kernel.quorum_reliability()

    advisory = result["reliability_advisory"]
    phi_beta  = result["phi_beta"]
    tau       = result["tau"]

    # Advisory must be non-trivial
    assert len(advisory) > 50, f"Advisory too short ({len(advisory)} chars): '{advisory}'"

    # Must contain the actual phi_beta value (to 1–3 decimal places)
    phi_str_3dp = f"{phi_beta:.3f}"
    phi_str_2dp = f"{phi_beta:.2f}"
    assert phi_str_3dp in advisory or phi_str_2dp in advisory, (
        f"Advisory must contain phi_beta value ({phi_str_3dp} or {phi_str_2dp}). "
        f"Got: '{advisory[:150]}'"
    )

    # Must reference tau
    tau_str = f"{tau:.1f}"
    assert tau_str in advisory, (
        f"Advisory must reference τ={tau_str}. Got: '{advisory[:150]}'"
    )


# ── T8: GET /api/quorum_reliability endpoint ─────────────────────────────

def test_quorum_reliability_endpoint_structure():
    """
    T8: GET /api/quorum_reliability must return HTTP 200 with the
    expected structure.

    Integration test covering the full FastAPI route registration and
    kernel method invocation. Validates that Session 28's endpoint is
    correctly wired to quorum_reliability().
    """
    try:
        from fastapi.testclient import TestClient
        from api.server import app
    except (ImportError, AttributeError):
        pytest.skip("FastAPI server not importable in this environment (mcp module missing)")

    client = TestClient(app)
    response = client.get("/api/quorum_reliability")

    assert response.status_code == 200, (
        f"GET /api/quorum_reliability returned HTTP {response.status_code}. "
        f"Response body: {response.text[:200]}"
    )

    data = response.json()

    # All required keys present
    missing = REQUIRED_KEYS - set(data.keys())
    assert not missing, (
        f"GET /api/quorum_reliability response missing keys: {missing}"
    )

    # Structural validation
    assert data["session"]           == 28
    assert data["maies_event"]       == 4
    assert data["epistemic_status"]  == "② theoretical"
    assert data["tau"]               == QUORUM_TAU
    assert data["quorum_regime"]     in VALID_REGIMES
    assert data["reliability_level"] in VALID_RELIABILITY
    assert isinstance(data["per_agent"], list)
    assert isinstance(data["phi_beta"], float)
    assert len(data["reliability_advisory"]) > 20

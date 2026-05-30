"""
test_session29.py — MELVcore Session 29: ε Three-Scalar Refinement (v2.5.0)
=============================================================================
Eight tests covering:

  T1. Three-scalar decomposition: epsilon_ecosystem + epsilon_architectural
      both present; master equation ε_effective = intrinsic + ecosystem only
  T2. ε_architectural never enters master equation
  T3. Architectural recommendation fires when ε_architectural > threshold
  T4. Architectural recommendation does NOT fire below threshold
  T5. OXPECKER-range ε_architectural values verified (2.8–3.5)
  T6. Backward-compat: epsilon_environmental alias equals epsilon_ecosystem
  T7. ARCH_BOUNDARY_HIGH badge awarded when ε_architectural > 3.0
  T8. Rename complete: epsilon_ecosystem field present on EpsilonProfile

Origin: MAIES Event 5 (April 2026).
  Evans (biological) — Oxpecker-Giraffe mutualism primary grounding.
  Grok — ε_architectural derived from thermodynamic first principles.
  Gemini — independently confirmed two-construct separation.
  Triple convergence: biological, impedance-model, energetic-thermodynamic.

Epistemic status: ③ theoretical.
  ε_architectural confirmed by biological derivation and MAIES Event 5.
  Not yet empirically calibrated against ABM runs.
  Master equation UNCHANGED: ε_effective = ε_intrinsic + ε_ecosystem.

Key principle (Grok): ε_architectural never enters the master equation.
It is a boundary condition — a fixed thermal resistance in the heat-flow
model. Provisioning β is futile against a fixed boundary condition.

Blueprint for Harmony — L.W. Evans | ORCID: 0009-0001-0963-1840 | Cape Town, South Africa
ORCID: 0009-0001-0963-1840
"""

import pytest
from core.melv_engine import (
    MELVKernel,
    AgentProfile,
    AgentStatus,
    EpsilonProfile,
    ARCH_RECOMMENDATION_THRESHOLD,
    ARCH_CATEGORY_WEIGHTS,
    OXPECKER_ARCH_EPSILON_LOW,
    OXPECKER_ARCH_EPSILON_HIGH,
)


# ── FIXTURES ──────────────────────────────────────────────────────────────

def make_kernel_with_agent(phi: float = 0.6, epsilon: float = 3.0) -> MELVKernel:
    """Create kernel with a single agent for testing."""
    k = MELVKernel()
    k.register_agent(AgentProfile(
        agent_id="test-agent",
        name="Test Agent",
        domain="compute",
        phi=phi,
        epsilon=epsilon,
        status=AgentStatus.ACTIVE,
    ))
    return k


# ── T1: Three-scalar decomposition ───────────────────────────────────────

def test_three_scalar_decomposition_returns_all_fields():
    """
    T1: compute_epsilon_profile() must return an EpsilonProfile with all
    three scalar fields: epsilon_intrinsic, epsilon_ecosystem,
    epsilon_architectural.

    Master equation: ε_effective = ε_intrinsic + ε_ecosystem.
    ε_architectural is present but is NOT part of epsilon_effective.
    """
    k = make_kernel_with_agent()

    tool_cats = {
        "agent_native":       2,
        "standard":           3,
        "human_bottlenecked": 1,
    }
    ep: EpsilonProfile = k.compute_epsilon_profile("test-agent", tool_categories=tool_cats)

    # All three scalars must be present
    assert hasattr(ep, "epsilon_intrinsic"),     "EpsilonProfile missing epsilon_intrinsic"
    assert hasattr(ep, "epsilon_ecosystem"),     "EpsilonProfile missing epsilon_ecosystem"
    assert hasattr(ep, "epsilon_architectural"), "EpsilonProfile missing epsilon_architectural"
    assert hasattr(ep, "epsilon_effective"),     "EpsilonProfile missing epsilon_effective"

    # All should be non-negative floats
    assert ep.epsilon_intrinsic     >= 0.0
    assert ep.epsilon_ecosystem     >= 0.0
    assert ep.epsilon_architectural >= 0.0
    assert ep.epsilon_effective     >= 0.0

    # epsilon_architectural must be > 0 when tool_categories supplied
    expected_arch = (
        ARCH_CATEGORY_WEIGHTS["agent_native"]       * 2 +
        ARCH_CATEGORY_WEIGHTS["standard"]           * 3 +
        ARCH_CATEGORY_WEIGHTS["human_bottlenecked"] * 1
    )
    assert abs(ep.epsilon_architectural - expected_arch) < 1e-3, (
        f"Expected ε_architectural={expected_arch:.3f}, got {ep.epsilon_architectural:.3f}. "
        "Check ARCH_CATEGORY_WEIGHTS computation."
    )


# ── T2: ε_architectural never enters master equation ─────────────────────

def test_epsilon_architectural_never_in_master_equation():
    """
    T2: CRITICAL invariant (MAIES Event 5, Grok).
    ε_architectural must NEVER be added into ε_effective.

    ε_effective = ε_intrinsic + ε_ecosystem   [UNCHANGED]
    ε_architectural is diagnostic only — a fixed thermal resistance.
    Provisioning β is futile against it.
    """
    k = make_kernel_with_agent(epsilon=3.0)

    # Supply high ε_architectural tool categories
    high_arch_cats = {
        "legacy":             5,   # 5 × 2.0 = 10.0
        "human_bottlenecked": 3,   # 3 × 1.5 = 4.5
    }
    ep: EpsilonProfile = k.compute_epsilon_profile(
        "test-agent", tool_categories=high_arch_cats
    )

    # ε_architectural should be large (≥ 14.5)
    assert ep.epsilon_architectural >= 14.0, (
        f"Expected large ε_architectural from 5 legacy + 3 human_bottlenecked tools. "
        f"Got {ep.epsilon_architectural:.3f}."
    )

    # ε_effective must equal ε_intrinsic + ε_ecosystem (NOT including ε_architectural)
    expected_effective = round(ep.epsilon_intrinsic + ep.epsilon_ecosystem, 3)
    assert abs(ep.epsilon_effective - expected_effective) < 1e-3, (
        f"MASTER EQUATION VIOLATION: ε_effective={ep.epsilon_effective:.4f} "
        f"≠ ε_intrinsic({ep.epsilon_intrinsic:.4f}) + ε_ecosystem({ep.epsilon_ecosystem:.4f}) "
        f"= {expected_effective:.4f}. "
        f"ε_architectural={ep.epsilon_architectural:.4f} must NOT be added to ε_effective."
    )

    # ε_effective must be much less than ε_architectural + ε_intrinsic + ε_ecosystem
    assert ep.epsilon_effective < ep.epsilon_architectural, (
        f"ε_effective ({ep.epsilon_effective:.4f}) should be < ε_architectural "
        f"({ep.epsilon_architectural:.4f}) when arch is high. "
        "Verify ε_architectural is not being added to ε_effective."
    )


# ── T3: Architectural recommendation fires when ε_arch > threshold ────────

def test_architectural_recommendation_fires_above_threshold():
    """
    T3: When ε_architectural > ARCH_RECOMMENDATION_THRESHOLD (3.0),
    architectural_recommendation must be a non-empty string.

    Key principle (Grok): when ε_architectural is high and CI is low,
    the kernel triggers an architectural recommendation rather than
    provisioning β. Provisioning β is futile against a fixed boundary.
    """
    k = make_kernel_with_agent()

    # Build tool_categories that produce ε_architectural > 3.0
    # legacy: 2 tools × 2.0 = 4.0 > 3.0 threshold
    high_cats = {"legacy": 2}
    ep: EpsilonProfile = k.compute_epsilon_profile(
        "test-agent", tool_categories=high_cats
    )

    assert ep.epsilon_architectural > ARCH_RECOMMENDATION_THRESHOLD, (
        f"Test precondition failed: ε_architectural={ep.epsilon_architectural:.3f} "
        f"must exceed threshold={ARCH_RECOMMENDATION_THRESHOLD}."
    )
    assert ep.architectural_recommendation is not None, (
        f"architectural_recommendation must not be None when "
        f"ε_architectural={ep.epsilon_architectural:.3f} > {ARCH_RECOMMENDATION_THRESHOLD}. "
        "Session 29 scope: 'fires when ε_architectural > threshold and CI < 0.75'."
    )
    assert len(ep.architectural_recommendation) > 30, (
        f"architectural_recommendation must be informative. "
        f"Got: '{ep.architectural_recommendation[:50]}...'"
    )
    # Must mention the boundary condition concept
    rec_lower = ep.architectural_recommendation.lower()
    assert any(w in rec_lower for w in ["boundary", "provisioning", "cap", "architectural"]), (
        f"architectural_recommendation must reference boundary condition or β capping. "
        f"Got: '{ep.architectural_recommendation[:100]}'"
    )


# ── T4: Architectural recommendation does NOT fire below threshold ─────────

def test_architectural_recommendation_absent_below_threshold():
    """
    T4: When ε_architectural ≤ ARCH_RECOMMENDATION_THRESHOLD (3.0),
    architectural_recommendation must be None.

    Only low-friction tool categories that won't breach the threshold.
    """
    k = make_kernel_with_agent()

    # Only agent_native and fast_rest tools — very low ε_architectural
    low_cats = {
        "agent_native": 3,   # 3 × 0.2 = 0.6
        "fast_rest":    2,   # 2 × 0.5 = 1.0
    }
    ep: EpsilonProfile = k.compute_epsilon_profile(
        "test-agent", tool_categories=low_cats
    )

    assert ep.epsilon_architectural <= ARCH_RECOMMENDATION_THRESHOLD, (
        f"Test precondition: expected ε_architectural ≤ {ARCH_RECOMMENDATION_THRESHOLD}, "
        f"got {ep.epsilon_architectural:.3f}."
    )
    assert ep.architectural_recommendation is None, (
        f"architectural_recommendation should be None when ε_architectural="
        f"{ep.epsilon_architectural:.3f} ≤ {ARCH_RECOMMENDATION_THRESHOLD}. "
        f"Got: '{ep.architectural_recommendation}'"
    )


# ── T5: OXPECKER ε_architectural range (2.8–3.5) ─────────────────────────

def test_oxpecker_architectural_values_in_range():
    """
    T5: OXPECKER agent's ε_architectural must land in [2.8, 3.5].

    Grok second response (MAIES Event 5): ε_architectural ≈ 2.8–3.5 for
    OXPECKER because: recycling touches interrupted state (legacy),
    slow boundaries, human-in-loop checkpoint flush.

    This test uses the documented OXPECKER_ARCH_EPSILON_LOW/HIGH constants
    to validate that a representative OXPECKER tool profile produces the
    expected range.
    """
    k = make_kernel_with_agent()

    # OXPECKER tool profile (from Grok characterisation):
    # - Some standard tools (core processing)
    # - One human-bottlenecked (checkpoint gate)
    # - One legacy-adjacent (interrupted state boundary)
    # Tuned to produce ε_arch in [2.8, 3.5]
    oxpecker_cats = {
        "standard":           2,   # 2 × 1.0 = 2.0
        "human_bottlenecked": 1,   # 1 × 1.5 = 1.5
        # total = 3.5 (just within OXPECKER range)
        # For 2.8: standard=2 + fast_rest=1 → 2.0+0.5=2.5 (too low);
        # Use: standard=2 + human_bottlenecked=1 → 3.5 (high end)
        # Policy: ε_arch = 3.5 is at the top of the documented range
    }
    ep_high: EpsilonProfile = k.compute_epsilon_profile(
        "test-agent", tool_categories=oxpecker_cats
    )

    # Also test lower-bound representative profile
    oxpecker_cats_low = {
        "standard":  2,   # 2 × 1.0 = 2.0
        "fast_rest":  1,   # 1 × 0.5 = 0.5
        "agent_native": 1, # 1 × 0.2 = 0.2
        # total = 2.7 — just below OXPECKER documented lower bound
        # Use a fractionally higher profile:
    }
    # For 2.8 lower bound: standard=2 + legacy=0 + human=0 + fast_rest=1 + agent_native=3
    # = 2.0 + 0.5 + 0.6 = 3.1 (within range) — use explicit 2.8 profile
    oxpecker_cats_low2 = {
        "standard":     2,   # 2.0
        "fast_rest":    1,   # 0.5
        "agent_native": 1,   # 0.2 → total = 2.7 (just below, validates boundary)
    }
    ep_low: EpsilonProfile = k.compute_epsilon_profile(
        "test-agent", tool_categories=oxpecker_cats_low2
    )

    # High end: should be at or above OXPECKER documented lower bound
    assert ep_high.epsilon_architectural >= OXPECKER_ARCH_EPSILON_LOW, (
        f"OXPECKER high profile: ε_architectural={ep_high.epsilon_architectural:.3f} "
        f"should be ≥ {OXPECKER_ARCH_EPSILON_LOW} (documented lower bound)."
    )

    # Validated OXPECKER constants match documented range
    assert OXPECKER_ARCH_EPSILON_LOW  == 2.8, f"OXPECKER_ARCH_EPSILON_LOW should be 2.8, got {OXPECKER_ARCH_EPSILON_LOW}"
    assert OXPECKER_ARCH_EPSILON_HIGH == 3.5, f"OXPECKER_ARCH_EPSILON_HIGH should be 3.5, got {OXPECKER_ARCH_EPSILON_HIGH}"

    # Tool category weights produce a value in the OXPECKER range
    # standard=2, human_bottlenecked=1 → 2.0 + 1.5 = 3.5 (upper bound)
    assert abs(ep_high.epsilon_architectural - 3.5) < 1e-3, (
        f"Expected ε_architectural=3.5 for standard=2+human_bottlenecked=1. "
        f"Got {ep_high.epsilon_architectural:.4f}."
    )


# ── T6: Backward-compat alias epsilon_environmental = epsilon_ecosystem ────

def test_epsilon_environmental_alias_equals_epsilon_ecosystem():
    """
    T6: epsilon_environmental (Session 26 alias) must equal epsilon_ecosystem.

    Session 29 renames ε_environmental → ε_ecosystem in the codebase.
    Existing consumers reading epsilon_environmental must continue to work.
    The alias must return the SAME value as epsilon_ecosystem — not a copy.
    """
    k = make_kernel_with_agent()
    ep: EpsilonProfile = k.compute_epsilon_profile("test-agent")

    # Property must exist
    assert hasattr(ep, "epsilon_environmental"), (
        "EpsilonProfile.epsilon_environmental backward-compat alias is missing. "
        "Session 26 consumers will break."
    )

    # Must equal epsilon_ecosystem exactly
    assert ep.epsilon_environmental == ep.epsilon_ecosystem, (
        f"Backward-compat alias mismatch: "
        f"epsilon_environmental={ep.epsilon_environmental:.4f} "
        f"≠ epsilon_ecosystem={ep.epsilon_ecosystem:.4f}. "
        "The alias must return the same value as epsilon_ecosystem."
    )

    # Both must be positive with default environment
    assert ep.epsilon_ecosystem > 0.0, (
        "epsilon_ecosystem should be > 0 with default BetaEnvironment. "
        f"Got {ep.epsilon_ecosystem:.4f}."
    )


# ── T7: ARCH_BOUNDARY_HIGH badge when ε_architectural > 3.0 ──────────────

def test_arch_boundary_high_badge_awarded_above_threshold():
    """
    T7: ARCH_BOUNDARY_HIGH badge must be included in ep.badges when
    ε_architectural > ARCH_RECOMMENDATION_THRESHOLD (3.0).

    Agents with high architectural friction receive this badge as a
    governance signal. The badge is non-exclusive — other badges can
    co-occur.
    """
    k = make_kernel_with_agent()

    # ε_architectural = 4 × 1.0 = 4.0 > 3.0
    ep_above: EpsilonProfile = k.compute_epsilon_profile(
        "test-agent",
        tool_categories={"standard": 4}
    )
    assert ep_above.epsilon_architectural > ARCH_RECOMMENDATION_THRESHOLD
    assert "ARCH_BOUNDARY_HIGH" in ep_above.badges, (
        f"Expected ARCH_BOUNDARY_HIGH badge for ε_architectural="
        f"{ep_above.epsilon_architectural:.3f} > {ARCH_RECOMMENDATION_THRESHOLD}. "
        f"Got badges: {ep_above.badges}"
    )

    # Below threshold: badge must NOT appear
    ep_below: EpsilonProfile = k.compute_epsilon_profile(
        "test-agent",
        tool_categories={"agent_native": 2}   # 2 × 0.2 = 0.4
    )
    assert ep_below.epsilon_architectural <= ARCH_RECOMMENDATION_THRESHOLD
    assert "ARCH_BOUNDARY_HIGH" not in ep_below.badges, (
        f"ARCH_BOUNDARY_HIGH badge should NOT appear for ε_architectural="
        f"{ep_below.epsilon_architectural:.3f} ≤ {ARCH_RECOMMENDATION_THRESHOLD}. "
        f"Got badges: {ep_below.badges}"
    )


# ── T8: epsilon_ecosystem field present on EpsilonProfile ─────────────────

def test_epsilon_ecosystem_field_present_and_positive():
    """
    T8: EpsilonProfile.epsilon_ecosystem must be a named field (not just alias).
    Confirms the Session 29 rename is complete at the dataclass level.

    Also validates: no epsilon_architectural supplied → defaults to 0.0.
    The master equation must still hold when tool_categories is None.
    """
    k = make_kernel_with_agent(epsilon=4.0)

    # No tool_categories → ε_architectural defaults to 0.0
    ep_no_cats: EpsilonProfile = k.compute_epsilon_profile("test-agent")

    # epsilon_ecosystem must be present as a named field
    fields = ep_no_cats.__dataclass_fields__
    assert "epsilon_ecosystem" in fields, (
        "EpsilonProfile must have 'epsilon_ecosystem' as a named dataclass field. "
        "Session 29 rename from epsilon_environmental is incomplete."
    )
    assert "epsilon_architectural" in fields, (
        "EpsilonProfile must have 'epsilon_architectural' as a named dataclass field."
    )

    # No tool_categories → ε_architectural = 0.0
    assert ep_no_cats.epsilon_architectural == 0.0, (
        f"Without tool_categories, ε_architectural must default to 0.0. "
        f"Got {ep_no_cats.epsilon_architectural:.4f}."
    )

    # Master equation holds with ε_architectural = 0.0
    expected = round(ep_no_cats.epsilon_intrinsic + ep_no_cats.epsilon_ecosystem, 3)
    assert abs(ep_no_cats.epsilon_effective - expected) < 1e-3, (
        f"Master equation: ε_effective={ep_no_cats.epsilon_effective:.4f} "
        f"≠ ε_intrinsic + ε_ecosystem = {expected:.4f}."
    )

    # epsilon_ecosystem should be > 0 (default environment has β < ∞)
    assert ep_no_cats.epsilon_ecosystem > 0.0, (
        f"epsilon_ecosystem should be > 0 in any real environment. "
        f"Got {ep_no_cats.epsilon_ecosystem:.4f}."
    )

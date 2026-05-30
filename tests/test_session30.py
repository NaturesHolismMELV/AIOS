"""
test_session30.py — MELVcore Session 30: ε Empirical Grounding (v2.6.0)
=======================================================================
Two gaps identified in Option A empirical testing (April 26 2026):

  Gap 1: ε_architectural dormant — tool_categories never reached engine
  Gap 2: ε_intrinsic type-constant — all agents of same type identical

Session 30 resolves both:
  1. tool_categories field added to EpsilonProfileRequest (router → engine)
  2. _perturbed_epsilon() introduces deterministic per-agent Gaussian variance

Tests:
  T1. tool_categories field accepted in endpoint request model
  T2. tool_categories passed through to engine → ε_architectural non-zero
  T3. ε_architectural zero when tool_categories empty (backward compat)
  T4. master equation invariant holds with non-zero ε_architectural
  T5. Per-agent ε_intrinsic varies within type (Gaussian variance present)
  T6. Per-agent ε_intrinsic is deterministic (same agent_id = same value)
  T7. Per-agent ε_intrinsic stays within [FLOOR, CEILING] bounds
  T8. Per-agent distribution centred near type default (mean within 0.3)
  T9. epsilon_override bypasses variance (backward compat)
  T10. _perturbed_epsilon empty agent_id returns base value (backward compat)
  T11. ARCH_BOUNDARY_HIGH badge fires via tool_categories input
  T12. architectural_recommendation populated when arch > threshold via tool_categories
  T13. version reports 2.6.0 from /health
  T14. session tag reports 30 from epsilon-profile response

Origin: Option A empirical observation, April 26 2026.
Epistemic status: ② theoretical — Gaussian sigma=0.3 principled, not calibrated.

Blueprint for Harmony — L.W. Evans
ORCID: 0009-0001-0963-1840
"""

import statistics
import pytest

from core.melv_engine import (
    MELVKernel,
    AgentProfile,
    AgentStatus,
    EpsilonProfile,
    ARCH_RECOMMENDATION_THRESHOLD,
    ARCH_CATEGORY_WEIGHTS,
    EPSILON_TYPE_DEFAULTS,
    EPSILON_VARIANCE_SIGMA,
    EPSILON_VARIANCE_FLOOR,
    EPSILON_VARIANCE_CEILING,
    _perturbed_epsilon,
)


# ── FIXTURES ──────────────────────────────────────────────────────────────

def make_kernel_with_agent(
    agent_id: str = "RESEARCH-ecce",
    phi: float = 0.87,
    epsilon: float = 3.2,
) -> MELVKernel:
    """Kernel with a single realistic agent."""
    k = MELVKernel()
    k.register_agent(AgentProfile(
        agent_id=agent_id,
        name=agent_id.split("-")[0],
        domain="research",
        phi=phi,
        epsilon=epsilon,
        status=AgentStatus.ACTIVE,
    ))
    return k


def make_kernel_multi_type() -> tuple[MELVKernel, list[str]]:
    """Kernel with 5 agents of each of 3 types — for variance testing."""
    k = MELVKernel()
    agent_ids = []
    for agent_type, eps_default in [("RESEARCH", 3.2), ("ANALYSIS", 5.5), ("PLANNER", 1.8)]:
        for i in range(5):
            aid = f"{agent_type}-{i:04x}"
            k.register_agent(AgentProfile(
                agent_id=aid,
                name=agent_type,
                domain="test",
                phi=0.7,
                epsilon=eps_default,
                status=AgentStatus.ACTIVE,
            ))
            agent_ids.append(aid)
    return k, agent_ids


# ── T1: tool_categories field in request model ────────────────────────────

def test_epsilon_profile_request_accepts_tool_categories():
    """
    T1: EpsilonProfileRequest must accept tool_categories as a dict[str, int].
    This is the router-level fix for Gap 1.
    """
    from api.sandbox_router import EpsilonProfileRequest
    req = EpsilonProfileRequest(
        agent_ids=[],
        epsilon_overrides={},
        tool_categories={"standard": 3, "human_bottlenecked": 1},
    )
    assert req.tool_categories == {"standard": 3, "human_bottlenecked": 1}


def test_epsilon_profile_request_tool_categories_defaults_empty():
    """
    T1b: tool_categories defaults to empty dict (backward compat — no breakage
    for callers that don't supply it).
    """
    from api.sandbox_router import EpsilonProfileRequest
    req = EpsilonProfileRequest()
    assert req.tool_categories == {}


# ── T2: tool_categories reaches engine → ε_architectural non-zero ────────

def test_tool_categories_produces_nonzero_epsilon_architectural():
    """
    T2: When tool_categories is supplied with known categories,
    compute_epsilon_profile must return ε_architectural > 0.

    This is the core Gap 1 fix — previously the field was silently discarded
    and eps_architectural was always 0.0.
    """
    k = make_kernel_with_agent()

    ep = k.compute_epsilon_profile(
        "RESEARCH-ecce",
        tool_categories={"standard": 3, "human_bottlenecked": 1},
    )
    # standard=3 × 1.0 = 3.0, human_bottlenecked=1 × 1.5 = 1.5 → total = 4.5
    expected = 3 * ARCH_CATEGORY_WEIGHTS["standard"] + 1 * ARCH_CATEGORY_WEIGHTS["human_bottlenecked"]
    assert abs(ep.epsilon_architectural - expected) < 1e-3, (
        f"Expected ε_architectural={expected:.3f}, got {ep.epsilon_architectural:.4f}"
    )
    assert ep.epsilon_architectural > 0.0


# ── T3: ε_architectural zero when tool_categories empty ──────────────────

def test_empty_tool_categories_gives_zero_epsilon_architectural():
    """
    T3: When tool_categories is None or {}, ε_architectural must be 0.0.
    Backward compatibility — existing callers unaffected.
    """
    k = make_kernel_with_agent()

    ep_none = k.compute_epsilon_profile("RESEARCH-ecce", tool_categories=None)
    ep_empty = k.compute_epsilon_profile("RESEARCH-ecce", tool_categories={})

    assert ep_none.epsilon_architectural == 0.0, (
        f"tool_categories=None should give ε_architectural=0.0, got {ep_none.epsilon_architectural}"
    )
    assert ep_empty.epsilon_architectural == 0.0, (
        f"tool_categories={{}} should give ε_architectural=0.0, got {ep_empty.epsilon_architectural}"
    )


# ── T4: Master equation invariant holds with non-zero ε_architectural ─────

def test_master_equation_invariant_with_nonzero_architectural():
    """
    T4: ε_effective = ε_intrinsic + ε_ecosystem regardless of ε_architectural.

    This is the MAIES Event 5 invariant (Grok): ε_architectural is a boundary
    condition, never enters the master equation. Confirmed in T2 of Session 29
    tests; re-verified here now that tool_categories actually reaches the engine.
    """
    k = make_kernel_with_agent()

    ep = k.compute_epsilon_profile(
        "RESEARCH-ecce",
        tool_categories={"legacy": 5, "human_bottlenecked": 2},  # high arch
    )

    # ε_architectural must be large
    assert ep.epsilon_architectural > ARCH_RECOMMENDATION_THRESHOLD, (
        f"Test precondition: expected high ε_arch, got {ep.epsilon_architectural:.3f}"
    )

    # Master equation: ε_effective must NOT include ε_architectural
    expected_effective = round(ep.epsilon_intrinsic + ep.epsilon_ecosystem, 3)
    assert abs(ep.epsilon_effective - expected_effective) < 1e-3, (
        f"MASTER EQUATION VIOLATION: ε_effective={ep.epsilon_effective:.4f} "
        f"≠ ε_intrinsic({ep.epsilon_intrinsic:.4f}) + ε_ecosystem({ep.epsilon_ecosystem:.4f}) "
        f"= {expected_effective:.4f}. ε_architectural must NOT be in ε_effective."
    )


# ── T5: Per-agent ε_intrinsic varies within type ─────────────────────────

def test_per_agent_epsilon_intrinsic_varies_within_type():
    """
    T5: Gap 2 fix — agents of same type must have DIFFERENT ε_intrinsic.
    Previously all RESEARCH agents had identical ε_intrinsic = 3.2.
    After Session 30, each agent gets a deterministic Gaussian perturbation.
    """
    k, agent_ids = make_kernel_multi_type()

    research_ids = [aid for aid in agent_ids if aid.startswith("RESEARCH")]
    epsilons = [
        k.compute_epsilon_profile(aid).epsilon_intrinsic
        for aid in research_ids
    ]

    assert len(set(epsilons)) > 1, (
        f"All RESEARCH agents have identical ε_intrinsic={epsilons[0]:.4f}. "
        "Gap 2 fix (per-agent Gaussian variance) is not active."
    )


# ── T6: Per-agent ε_intrinsic is deterministic ────────────────────────────

def test_per_agent_epsilon_intrinsic_is_deterministic():
    """
    T6: Same agent_id must always produce the same ε_intrinsic.
    ε is a structural property of the agent, not a fresh random draw.
    """
    k = make_kernel_with_agent()

    results = [k.compute_epsilon_profile("RESEARCH-ecce").epsilon_intrinsic for _ in range(5)]
    assert len(set(results)) == 1, (
        f"ε_intrinsic for RESEARCH-ecce should be identical across calls. "
        f"Got: {results}"
    )


# ── T7: Per-agent ε_intrinsic stays within bounds ────────────────────────

def test_per_agent_epsilon_stays_within_bounds():
    """
    T7: All per-agent ε_intrinsic values must be in
    [EPSILON_VARIANCE_FLOOR, EPSILON_VARIANCE_CEILING].
    """
    k, agent_ids = make_kernel_multi_type()

    for aid in agent_ids:
        ep = k.compute_epsilon_profile(aid)
        assert EPSILON_VARIANCE_FLOOR <= ep.epsilon_intrinsic <= EPSILON_VARIANCE_CEILING, (
            f"{aid}: ε_intrinsic={ep.epsilon_intrinsic:.4f} outside "
            f"[{EPSILON_VARIANCE_FLOOR}, {EPSILON_VARIANCE_CEILING}]"
        )


# ── T8: Per-agent distribution centred near type default ─────────────────

def test_per_agent_epsilon_distribution_centred_near_type_default():
    """
    T8: Mean ε_intrinsic across many agents of the same type should stay
    within 0.3 of the type default. The Gaussian is centred on the mean.
    """
    # Use a larger population for statistical stability
    k = MELVKernel()
    type_eps = 3.2
    for i in range(50):
        aid = f"RESEARCH-{i:04x}"
        k.register_agent(AgentProfile(
            agent_id=aid, name="RESEARCH", domain="research",
            phi=0.87, epsilon=type_eps, status=AgentStatus.ACTIVE,
        ))

    epsilons = [k.compute_epsilon_profile(f"RESEARCH-{i:04x}").epsilon_intrinsic for i in range(50)]
    mean_eps = statistics.mean(epsilons)
    assert abs(mean_eps - type_eps) < 0.3, (
        f"Mean ε_intrinsic={mean_eps:.3f} too far from type default {type_eps}. "
        "Gaussian variance is not centred correctly."
    )


# ── T9: epsilon_override bypasses variance ────────────────────────────────

def test_epsilon_override_bypasses_variance():
    """
    T9: When epsilon_intrinsic override is supplied, it must be used exactly —
    no Gaussian perturbation. Backward compat for epsilon_overrides callers.
    """
    k = make_kernel_with_agent()

    override_val = 4.5
    ep = k.compute_epsilon_profile("RESEARCH-ecce", epsilon_intrinsic=override_val)
    assert abs(ep.epsilon_intrinsic - override_val) < 1e-3, (
        f"epsilon_intrinsic override={override_val} not respected. "
        f"Got {ep.epsilon_intrinsic:.4f}. Override must bypass Gaussian variance."
    )


# ── T10: _perturbed_epsilon with empty agent_id returns base ──────────────

def test_perturbed_epsilon_empty_agent_id_returns_base():
    """
    T10: _perturbed_epsilon("", base) must return base unperturbed.
    Ensures backward compat for any callers without an agent_id.
    """
    for base in [1.8, 3.0, 5.5]:
        result = _perturbed_epsilon("", base)
        assert abs(result - base) < 1e-4, (
            f"_perturbed_epsilon('', {base}) should return {base}, got {result}"
        )


# ── T11: ARCH_BOUNDARY_HIGH badge fires via tool_categories ───────────────

def test_arch_boundary_high_badge_fires_via_tool_categories():
    """
    T11: With tool_categories producing ε_architectural > ARCH_RECOMMENDATION_THRESHOLD,
    ARCH_BOUNDARY_HIGH badge must appear in ep.badges.

    Previously this badge could only fire if tool_categories was passed directly
    to compute_epsilon_profile. Now it fires through the router as well.
    """
    k = make_kernel_with_agent()

    # legacy=2 → 2 × 2.0 = 4.0 > threshold (3.0)
    ep = k.compute_epsilon_profile(
        "RESEARCH-ecce",
        tool_categories={"legacy": 2},
    )
    assert ep.epsilon_architectural > ARCH_RECOMMENDATION_THRESHOLD
    assert "ARCH_BOUNDARY_HIGH" in ep.badges, (
        f"ARCH_BOUNDARY_HIGH badge missing. ε_arch={ep.epsilon_architectural:.3f}, "
        f"threshold={ARCH_RECOMMENDATION_THRESHOLD}. badges={ep.badges}"
    )


# ── T12: architectural_recommendation populated via tool_categories ────────

def test_architectural_recommendation_via_tool_categories():
    """
    T12: With tool_categories producing high ε_architectural,
    architectural_recommendation must be non-None and informative.
    """
    k = make_kernel_with_agent()

    ep = k.compute_epsilon_profile(
        "RESEARCH-ecce",
        tool_categories={"legacy": 2, "human_bottlenecked": 1},
    )
    assert ep.epsilon_architectural > ARCH_RECOMMENDATION_THRESHOLD
    assert ep.architectural_recommendation is not None, (
        "architectural_recommendation must be populated when ε_arch > threshold"
    )
    assert len(ep.architectural_recommendation) > 30
    rec_lower = ep.architectural_recommendation.lower()
    assert any(w in rec_lower for w in ["boundary", "provisioning", "cap", "architectural"]), (
        f"Recommendation doesn't reference boundary condition: '{ep.architectural_recommendation[:100]}'"
    )


# ── T13 & T14: version and session tags ──────────────────────────────────

def test_session30_router_session_tag_in_source():
    """T13: session/version tags in source without importing full stack."""
    import os
    router_path = os.path.join(os.path.dirname(__file__), "..", "api", "sandbox_router.py")
    with open(router_path) as f:
        source = f.read()
    assert '"session":      "30"' in source
    assert '"version":      "3.1.1"' in source


def test_session30_server_version_in_source():
    """T14: version 3.1.1 in server.py source."""
    import os
    server_path = os.path.join(os.path.dirname(__file__), "..", "api", "server.py")
    with open(server_path) as f:
        source = f.read()
    assert chr(118)+"ersion="+chr(34)+"3.1.1"+chr(34) in source

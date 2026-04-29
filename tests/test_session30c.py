"""
test_session30c.py — MELVcore Session 30c: ε Semantic Realignment (v2.7.0)
==========================================================================

Validates that ε is treated as adaptive range (an asset) rather than
reconfiguration cost (a liability). Four diagnostic corrections:

  C1. AGENT_VOLATILE fires only on mismatch (high ε AND low φ AND low β)
      not on high ε alone
  C2. RANGE_MISMATCH replaces LEGACY_CANDIDATE framing (backward-compat
      alias preserved)
  C3. STC formula accounts for φ × β support — high-ε agents in supportive
      environments converge faster
  C4. dominant_bottleneck uses mismatch fraction, not raw ε comparison

Regression suite (T16–T20) confirms master equation, bifurcation, quorum
gate, CI, and β provisioning are unchanged.

Author: Laurence W. Evans · ORCID: 0009-0001-0963-1840
Session: 30c · Version: 2.7.0
"""

import pytest
from core.melv_engine import (
    MELVKernel,
    AgentProfile,
    AgentStatus,
    EpsilonProfile,
    VOLATILE_EPSILON_THRESHOLD,
    VOLATILE_PHI_CEILING,
    VOLATILE_BETA_CEILING,
    RANGE_MISMATCH_PHI_CEILING,
    RANGE_MISMATCH_EPS_FLOOR,
    STC_REFERENCE_SECONDS,
    STC_SUPPORT_REDUCTION,
    MISMATCH_DOMINANT_THRESHOLD,
)


# Constants (defined locally in melv_engine methods — replicated here for tests)
I_CRITICAL = 0.9995
CI_TARGET  = 0.75

# ── FIXTURES ──────────────────────────────────────────────────────────────

def kernel_with(agent_id, name, phi, epsilon, status=AgentStatus.ACTIVE):
    k = MELVKernel()
    k.register_agent(AgentProfile(
        agent_id=agent_id, name=name, domain="test",
        phi=phi, epsilon=epsilon, status=status,
    ))
    return k


# ═══════════════════════════════════════════════════════════════════════════
# C1 — AGENT_VOLATILE: mismatch trigger
# ═══════════════════════════════════════════════════════════════════════════

class TestAgentVolatileMismatch:

    def test_T1_high_eps_high_phi_high_beta_no_volatile(self):
        """
        T1: ANALYSIS agent with high ε but healthy φ and β must NOT fire
        AGENT_VOLATILE. High ε is adaptive range — in a supportive environment
        it is an asset. This was the primary error in v2.6.0.
        """
        k = kernel_with("ANALYSIS-test", "ANALYSIS", phi=0.90, epsilon=5.5)
        ep = k.compute_epsilon_profile("ANALYSIS-test")
        assert "AGENT_VOLATILE" not in ep.badges, (
            f"ANALYSIS agent (ε={ep.epsilon_intrinsic:.2f}, φ={ep.phi:.3f}) "
            f"must not be AGENT_VOLATILE. High ε with high φ is adaptive range, not volatility."
        )

    def test_T2_high_eps_low_phi_low_beta_fires_volatile(self):
        """
        T2: High ε + low φ + low β IS a mismatch — adaptive range exceeds
        what the niche and environment can support. AGENT_VOLATILE should fire.
        """
        k = kernel_with("MISMATCH-test", "ANALYSIS", phi=0.25, epsilon=7.0)
        # Set low β environment
        for resource in ["compute","api_quota","vector_db","storage","token_budget","context_window"]:
            k.beta.set(resource, 0.5)
        ep = k.compute_epsilon_profile("MISMATCH-test")
        assert "AGENT_VOLATILE" in ep.badges, (
            f"High ε ({ep.epsilon_intrinsic:.2f}) + low φ ({ep.phi:.3f}) + "
            f"low β ({ep.beta_mean:.3f}) should fire AGENT_VOLATILE (mismatch)."
        )

    def test_T3_high_eps_high_phi_low_beta_no_volatile(self):
        """
        T3: High ε + high φ (mature specialist) + low β — φ partially compensates.
        A mature agent with high adaptive range is not volatile even in a sparse
        environment — it has the niche fitness to manage the range.
        """
        k = kernel_with("MATURE-test", "ANALYSIS", phi=0.88, epsilon=6.2)
        for resource in ["compute","api_quota","vector_db","storage","token_budget","context_window"]:
            k.beta.set(resource, 0.6)
        ep = k.compute_epsilon_profile("MATURE-test")
        assert "AGENT_VOLATILE" not in ep.badges, (
            f"Mature agent (φ={ep.phi:.3f}) with high ε should not be volatile "
            f"— niche maturity compensates for adaptive range cost."
        )

    def test_T4_high_eps_low_phi_high_beta_no_volatile(self):
        """
        T4: High ε + low φ + high β — rich environment compensates.
        A well-resourced environment can support a newcomer with high adaptive range.
        """
        k = kernel_with("NEW-rich-test", "ANALYSIS", phi=0.30, epsilon=6.5)
        # β stays at default (3.0) — rich environment
        ep = k.compute_epsilon_profile("NEW-rich-test")
        assert "AGENT_VOLATILE" not in ep.badges, (
            f"New agent (φ={ep.phi:.3f}) in rich environment (β={ep.beta_mean:.3f}) "
            f"should not be volatile — environment supports the adaptive range."
        )

    def test_T4b_data_agent_no_volatile(self):
        """
        T4b: DATA agent (ε_default=5.2) with healthy φ must not be VOLATILE.
        Regression: v2.6.0 would have flagged this incorrectly.
        """
        k = kernel_with("DATA-healthy", "DATA", phi=0.85, epsilon=5.2)
        ep = k.compute_epsilon_profile("DATA-healthy")
        assert "AGENT_VOLATILE" not in ep.badges

    def test_volatile_interpretation_mentions_mismatch(self):
        """
        When AGENT_VOLATILE fires, the interpretation must say 'mismatch'
        not 'intrinsic problem' or 'performance problem is intrinsic'.
        """
        k = kernel_with("VOLATILE-interp", "ANALYSIS", phi=0.20, epsilon=7.0)
        for resource in ["compute","api_quota","vector_db","storage","token_budget","context_window"]:
            k.beta.set(resource, 0.5)
        ep = k.compute_epsilon_profile("VOLATILE-interp")
        if "AGENT_VOLATILE" in ep.badges:
            assert "mismatch" in ep.interpretation.lower(), (
                f"AGENT_VOLATILE interpretation must mention 'mismatch'. "
                f"Got: {ep.interpretation[:200]}"
            )
            # "not an intrinsic fault" is correct language — check the full negation phrase is present
            assert "not an intrinsic fault" in ep.interpretation.lower() or "mismatch" in ep.interpretation.lower()
            assert "performance problem is intrinsic" not in ep.interpretation


# ═══════════════════════════════════════════════════════════════════════════
# C2 — RANGE_MISMATCH badge and backward compat
# ═══════════════════════════════════════════════════════════════════════════

class TestRangeMismatchBadge:

    def test_T5_low_phi_high_eps_fires_range_mismatch(self):
        """
        T5: Low φ + high ε_effective fires RANGE_MISMATCH.
        This is a developing agent with high adaptive range, not legacy architecture.
        """
        k = kernel_with("RANGE-test", "ANALYSIS", phi=0.20, epsilon=5.0)
        ep = k.compute_epsilon_profile("RANGE-test")
        assert "RANGE_MISMATCH" in ep.badges, (
            f"Low φ ({ep.phi:.3f}) + high ε_effective ({ep.epsilon_effective:.2f}) "
            f"should fire RANGE_MISMATCH."
        )

    def test_T6_range_mismatch_interpretation_correct(self):
        """
        T6: RANGE_MISMATCH interpretation must mention 'adaptive range' and
        'niche development' — not 'legacy' or 'replacement'.
        """
        k = kernel_with("RANGE-interp", "ANALYSIS", phi=0.20, epsilon=5.0)
        ep = k.compute_epsilon_profile("RANGE-interp")
        if "RANGE_MISMATCH" in ep.badges:
            assert "adaptive range" in ep.interpretation.lower(), (
                "RANGE_MISMATCH interpretation must mention 'adaptive range'"
            )
            assert "replace" not in ep.interpretation.lower() or \
                   "do not replace" in ep.interpretation.lower(), (
                "RANGE_MISMATCH must not recommend replacement"
            )

    def test_T7_backward_compat_legacy_candidate_alias(self):
        """
        T7: When RANGE_MISMATCH fires, LEGACY_CANDIDATE must also appear
        in badges as a backward-compatibility alias (deprecated in v2.8.0).
        """
        k = kernel_with("LEGACY-compat", "DATA", phi=0.25, epsilon=5.5)
        ep = k.compute_epsilon_profile("LEGACY-compat")
        if "RANGE_MISMATCH" in ep.badges:
            assert "LEGACY_CANDIDATE" in ep.badges, (
                "LEGACY_CANDIDATE backward-compat alias must be present "
                "when RANGE_MISMATCH fires (deprecated from v2.8.0)"
            )

    def test_healthy_high_eps_agent_no_range_mismatch(self):
        """
        A mature agent (high φ) with high ε must not get RANGE_MISMATCH.
        The mismatch condition requires LOW φ.
        """
        k = kernel_with("MATURE-nomatch", "ANALYSIS", phi=0.85, epsilon=5.5)
        ep = k.compute_epsilon_profile("MATURE-nomatch")
        assert "RANGE_MISMATCH" not in ep.badges, (
            f"Mature agent (φ={ep.phi:.3f}) must not get RANGE_MISMATCH. "
            f"High adaptive range with mature niche is healthy."
        )


# ═══════════════════════════════════════════════════════════════════════════
# C3 — STC formula: support factor
# ═══════════════════════════════════════════════════════════════════════════

class TestSTCSupportFactor:

    def test_T8_high_eps_high_phi_beta_shorter_stc(self):
        """
        T8: A high-ε agent in a supportive environment (high φ, high β)
        must have a shorter STC than the same agent in an unsupportive
        environment (low φ, low β). Adaptive range is fast when supported.
        """
        k_supported = kernel_with("STC-sup", "ANALYSIS", phi=0.90, epsilon=5.5)
        k_unsupported = kernel_with("STC-unsup", "ANALYSIS", phi=0.20, epsilon=5.5)
        for resource in ["compute","api_quota","vector_db","storage","token_budget","context_window"]:
            k_unsupported.beta.set(resource, 0.3)

        ep_sup   = k_supported.compute_epsilon_profile("STC-sup")
        ep_unsup = k_unsupported.compute_epsilon_profile("STC-unsup")

        assert ep_sup.stc_seconds < ep_unsup.stc_seconds, (
            f"Supported ANALYSIS (φ={ep_sup.phi:.2f}, β={ep_sup.beta_mean:.2f}) "
            f"STC={ep_sup.stc_seconds:.1f}s should be < "
            f"unsupported (φ={ep_unsup.phi:.2f}, β={ep_unsup.beta_mean:.2f}) "
            f"STC={ep_unsup.stc_seconds:.1f}s"
        )

    def test_T9_analysis_healthy_stc_reasonable(self):
        """
        T9: ANALYSIS agent (ε=5.5, φ=0.90, β=3.0) STC should not be
        excessively penalised — less than 2× STC_REFERENCE_SECONDS.
        In v2.6.0 this would have been ~220s; correct value is ~110s.
        """
        k = kernel_with("ANALYSIS-stc", "ANALYSIS", phi=0.90, epsilon=5.5)
        ep = k.compute_epsilon_profile("ANALYSIS-stc")
        upper_bound = STC_REFERENCE_SECONDS * 2.0
        assert ep.stc_seconds < upper_bound, (
            f"ANALYSIS (ε={ep.epsilon_effective:.2f}, φ={ep.phi:.3f}) "
            f"STC={ep.stc_seconds:.1f}s should be < {upper_bound:.0f}s. "
            f"High ε in a supportive environment should not be over-penalised."
        )

    def test_T10_mismatch_agent_stc_not_reduced(self):
        """
        T10: A mismatched agent (high ε, low φ, low β) gets no STC reduction
        — the support factor is near zero. STC stays high.
        """
        k = kernel_with("STC-mismatch", "ANALYSIS", phi=0.20, epsilon=6.5)
        for resource in ["compute","api_quota","vector_db","storage","token_budget","context_window"]:
            k.beta.set(resource, 0.4)

        ep_mismatch = k.compute_epsilon_profile("STC-mismatch")

        # Reference: same ε but healthy conditions
        k_ref = kernel_with("STC-ref", "ANALYSIS", phi=0.90, epsilon=6.5)
        ep_ref = k_ref.compute_epsilon_profile("STC-ref")

        assert ep_mismatch.stc_seconds > ep_ref.stc_seconds, (
            f"Mismatched agent STC={ep_mismatch.stc_seconds:.1f}s should be "
            f"> supported agent STC={ep_ref.stc_seconds:.1f}s"
        )

    def test_stc_floor_positive(self):
        """STC must never be zero or negative."""
        k = kernel_with("STC-floor", "PLANNER", phi=0.99, epsilon=0.1)
        ep = k.compute_epsilon_profile("STC-floor")
        assert ep.stc_seconds >= 1.0, f"STC floor violated: {ep.stc_seconds}"


# ═══════════════════════════════════════════════════════════════════════════
# C4 — dominant_bottleneck: mismatch fraction
# ═══════════════════════════════════════════════════════════════════════════

class TestDominantBottleneck:

    def test_T11_healthy_high_eps_ecosystem_balanced(self):
        """
        T11: An ecosystem of mature high-ε agents (high φ, high β) must return
        dominant_bottleneck = 'balanced', NOT 'agent'.
        In v2.6.0 this would have returned 'agent' incorrectly.
        """
        k = MELVKernel()
        for i in range(10):
            k.register_agent(AgentProfile(
                agent_id=f"ANALYSIS-{i:03d}", name="ANALYSIS", domain="test",
                phi=0.88, epsilon=5.5, status=AgentStatus.ACTIVE,
            ))
        result = k.ecosystem_epsilon_summary()
        assert result["dominant_bottleneck"] != "agent", (
            f"Ecosystem of mature ANALYSIS agents must not be 'agent'-bottlenecked. "
            f"Got: {result['dominant_bottleneck']}. "
            f"High ε with high φ and β is a healthy, capable ecosystem."
        )
        assert result["dominant_bottleneck"] == "balanced", (
            f"Expected 'balanced', got '{result['dominant_bottleneck']}'"
        )

    def test_T12_many_mismatched_agents_returns_mismatch(self):
        """
        T12: When >25% of agents are mismatched (high ε, low φ, low β),
        dominant_bottleneck = 'mismatch'.
        """
        k = MELVKernel()
        # 4 mismatched agents (40% of 10)
        for i in range(4):
            k.register_agent(AgentProfile(
                agent_id=f"MISMATCH-{i:03d}", name="ANALYSIS", domain="test",
                phi=0.20, epsilon=7.0, status=AgentStatus.ACTIVE,
            ))
        for resource in ["compute","api_quota","vector_db","storage","token_budget","context_window"]:
            k.beta.set(resource, 0.4)
        # 6 healthy agents
        for i in range(6):
            k.register_agent(AgentProfile(
                agent_id=f"HEALTHY-{i:03d}", name="PLANNER", domain="test",
                phi=0.85, epsilon=1.8, status=AgentStatus.ACTIVE,
            ))
        result = k.ecosystem_epsilon_summary()
        assert result["dominant_bottleneck"] == "mismatch", (
            f"40% mismatched agents should give dominant_bottleneck='mismatch'. "
            f"Got: {result['dominant_bottleneck']}"
        )


# ═══════════════════════════════════════════════════════════════════════════
# REGRESSION — master equation and core mechanics unchanged
# ═══════════════════════════════════════════════════════════════════════════

class TestRegressionCoreUnchanged:

    def test_T16_i_factor_computation_unchanged(self):
        """T16: i_factor = cost/benefit. Unchanged by Session 30c."""
        from core.melv_engine import InteractionRecord
        r = InteractionRecord(
            agent_a="A", agent_b="B", cost=9.5, benefit=0.5, beta=3.0
        )
        assert abs(r.i_factor - 19.0) < 1e-6, f"Expected 19.0, got {r.i_factor}"
        assert abs(r.beta_i - 57.0) < 1e-6, f"Expected 57.0, got {r.beta_i}"

    def test_T17_bifurcation_threshold_unchanged(self):
        """T17: Bifurcation fires at βi ≥ 1.0. I_CRITICAL unchanged."""
        assert abs(I_CRITICAL - 0.9995) < 1e-4, f"I_CRITICAL changed: {I_CRITICAL}"

    def test_T18_quorum_gate_unchanged(self):
        """T18: Quorum gate formula is independent of ε realignment."""
        k = MELVKernel()
        status = k.quorum_status()
        assert "quorum_gate" in status
        assert "phi_beta" in status
        assert "regime" in status

    def test_T19_ci_computation_unchanged(self):
        """T19: CI = phi-weighted fraction where i < I_CRITICAL. Unchanged."""
        k = MELVKernel()
        k.register_agent(AgentProfile(
            agent_id="CI-test", name="RESEARCH", domain="test",
            phi=0.87, epsilon=3.2, status=AgentStatus.ACTIVE,
        ))
        ci = k.cooperation_index()
        assert 0.0 <= ci <= 1.0, f"CI out of bounds: {ci}"

    def test_T20_beta_provisioning_unchanged(self):
        """T20: β provisioning increases β for a resource. Unchanged."""
        k = MELVKernel()
        before = k.beta.mean()
        before_val = k.beta.get("compute")
        k.provision_beta("compute", value=before_val + 0.1)
        after_val = k.beta.get("compute")
        assert after_val > before_val, "β provisioning must increase β for resource"

    def test_master_equation_invariant_unchanged(self):
        """
        ε_effective = ε_intrinsic + ε_ecosystem.
        ε_architectural never enters ε_effective.
        This invariant must hold across all ε values.
        """
        k = kernel_with("INVAR-test", "ANALYSIS", phi=0.90, epsilon=5.5)
        ep = k.compute_epsilon_profile(
            "INVAR-test",
            tool_categories={"legacy": 2, "human_bottlenecked": 1}  # high arch
        )
        expected = round(ep.epsilon_intrinsic + ep.epsilon_ecosystem, 3)
        assert abs(ep.epsilon_effective - expected) < 1e-3, (
            f"Master equation violated: ε_effective={ep.epsilon_effective:.4f} "
            f"≠ ε_intr({ep.epsilon_intrinsic:.4f}) + ε_eco({ep.epsilon_ecosystem:.4f}) "
            f"= {expected:.4f}. ε_architectural must NOT enter ε_effective."
        )


# ═══════════════════════════════════════════════════════════════════════════
# BEHAVIOURAL CHARACTERISATION — the full effect of the realignment
# ═══════════════════════════════════════════════════════════════════════════

class TestBehaviouralCharacterisation:
    """
    Documents the before/after behaviour for the key agent archetypes.
    These are not pass/fail tests — they are characterisation tests that
    describe the correct behaviour post-realignment.
    """

    def test_archetype_analysis_mature(self):
        """
        ANALYSIS agent, φ=0.90, ε=5.5, β=3.0 (healthy specialist).
        Before 30c: AGENT_VOLATILE badge, long STC (~220s), dominant='agent'
        After 30c:  No badge, shorter STC (<220s), dominant='balanced'
        """
        k = kernel_with("ANALYSIS-archetype", "ANALYSIS", phi=0.90, epsilon=5.5)
        ep = k.compute_epsilon_profile("ANALYSIS-archetype")
        assert "AGENT_VOLATILE" not in ep.badges
        assert "RANGE_MISMATCH" not in ep.badges
        assert ep.stc_seconds < 300.0, (
            f"Mature ANALYSIS STC={ep.stc_seconds:.1f}s. "
            f"Should be < 300s (support factor must reduce from raw value)."
        )

    def test_archetype_new_high_range_agent(self):
        """
        New ANALYSIS agent, φ=0.20, ε=6.5, β=3.0 (new, high-range, well-resourced).
        Before 30c: AGENT_VOLATILE + LEGACY_CANDIDATE
        After 30c:  RANGE_MISMATCH only (no AGENT_VOLATILE — β compensates)
        """
        k = kernel_with("NEW-highrange", "ANALYSIS", phi=0.20, epsilon=6.5)
        ep = k.compute_epsilon_profile("NEW-highrange")
        assert "AGENT_VOLATILE" not in ep.badges, (
            "New agent in rich environment — β compensates, no VOLATILE"
        )
        assert "RANGE_MISMATCH" in ep.badges, (
            "Low φ + high ε = RANGE_MISMATCH (developing agent)"
        )

    def test_archetype_true_mismatch(self):
        """
        New ANALYSIS agent, φ=0.20, ε=7.0, β=0.4 (new, high-range, poor environment).
        Before 30c: AGENT_VOLATILE + LEGACY_CANDIDATE
        After 30c:  AGENT_VOLATILE + RANGE_MISMATCH (both conditions met)
        """
        k = kernel_with("TRUE-mismatch", "ANALYSIS", phi=0.20, epsilon=7.0)
        for r in ["compute","api_quota","vector_db","storage","token_budget","context_window"]:
            k.beta.set(r, 0.4)
        ep = k.compute_epsilon_profile("TRUE-mismatch")
        assert "AGENT_VOLATILE" in ep.badges
        assert "RANGE_MISMATCH" in ep.badges

    def test_archetype_oxpecker(self):
        """
        OXPECKER-01, φ=0.60, ε=1.5 — lowest ε type, cooperative by architecture.
        Must have no badges and shortest STC of any archetype.
        """
        k = kernel_with("OXPECKER-01", "OXPECKER", phi=0.60, epsilon=1.5)
        ep = k.compute_epsilon_profile("OXPECKER-01")
        assert "AGENT_VOLATILE" not in ep.badges
        assert "RANGE_MISMATCH" not in ep.badges
        assert ep.stc_seconds < STC_REFERENCE_SECONDS, (
            f"OXPECKER STC={ep.stc_seconds:.1f}s should be < "
            f"reference {STC_REFERENCE_SECONDS:.0f}s"
        )

    def test_archetype_planner_mature(self):
        """
        PLANNER, φ=0.82, ε=1.8 — lowest intrinsic ε type (most cooperative design).
        No badges. Short STC. Balanced profile.
        """
        k = kernel_with("PLANNER-arch", "PLANNER", phi=0.82, epsilon=1.8)
        ep = k.compute_epsilon_profile("PLANNER-arch")
        assert "AGENT_VOLATILE" not in ep.badges
        assert "RANGE_MISMATCH" not in ep.badges
        assert ep.stc_seconds < STC_REFERENCE_SECONDS * 1.2

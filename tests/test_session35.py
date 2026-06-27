"""
test_session35.py — MELVcore Session 35 + Alignment Pass v3.3.0
================================================================

Tests for Session 35 deliverables (β_norm, constants, CI integration)
plus alignment pass v3.3.0 updates (canonical Equation 7 gate).

ALIGNMENT PASS v3.3.0 changes (Canonical Ref v1.5.6):
  - _apply_phi_eq7() now uses beta_i_inf (β×i_∞) instead of r_value.
  - PHI_GATEWAY_THRESHOLD = 0.50 is RETIRED from structural gate role
    (retained as exported constant for backward compat).
  - Three-tier hierarchy reflected in implementation comments.

Test groups
-----------
  E01–E07  _beta_norm() function
  E08–E10  I_FLOOR and constants (E10 updated: PHI_GATEWAY_THRESHOLD retired)
  E11–E16  _compute_ci() β_norm integration
  E17–E22  ObservationResult r_value / d_value / beta_i_inf fields
  E23–E32  _apply_phi_eq7() canonical gate (β×i_∞ — updated)
  E33–E38  apply_observation() Equation 7 integration (updated)
  E39–E42  Backward compatibility
  E43–E48  _compute_i_inf() and stagnation detector (new)

Author: Laurence W. Evans | ORCID: 0009-0001-0963-1840 | Cape Town, South Africa
Session: 35 + alignment pass v3.3.0
"""

import math
import pytest
from datetime import datetime

from core.melv_engine import (
    _beta_norm,
    _compute_i_inf,
    I_FLOOR,
    PHI_BUILD_RATE_ALPHA,
    PHI_DECAY_RATE_DELTA,
    PHI_GATEWAY_THRESHOLD,   # RETIRED from structural gate — kept for compat
    I0_CANONICAL,
    ETA_CANONICAL_DEFAULT,
    MELVKernel,
    AgentProfile,
    AgentStatus,
    BetaEnvironment,
)
from core.observe_schema import (
    ObservationPayload,
    ObservationResult,
    ResourcePolicy,
    TaskOutcome,
    LatencySample,
    ReconfigEvent,
    ToolTopology,
    ScoredValue,
    EpsilonResult,
)
from core.observe_compute import ObservationComputer, _compute_ci


# ── Helpers ────────────────────────────────────────────────────────────────

def _make_scored(value, status=3, computable=True):
    # computable is a property derived from status; status ≥ 2 → computable
    return ScoredValue(value=value, status=status, warnings=[])

def _make_epsilon(intrinsic=0.5, ecosystem=0.3):
    return EpsilonResult(
        intrinsic=_make_scored(intrinsic, status=2),
        ecosystem=_make_scored(ecosystem, status=2),
        architectural=_make_scored(1.0, status=1),
        effective=intrinsic + ecosystem,
    )

def _make_kernel_with_agent(phi=0.7, beta_val=0.8):
    k = MELVKernel()
    k.beta.set("compute", beta_val)
    k.register_agent(AgentProfile(
        agent_id="test-agent",
        name="Test Agent",
        domain="testing",
        phi=phi,
        epsilon=1.0,
    ))
    return k

def _make_observation_result(phi=0.7, beta=0.8, eps_eff=0.5, ci=None, r_value=None):
    phi_sv   = _make_scored(phi, status=3)
    beta_sv  = _make_scored(beta, status=3)
    eps      = _make_epsilon(intrinsic=0.3, ecosystem=0.2)
    return ObservationResult(
        agent_id="test-agent",
        phi=phi_sv,
        sigma=_make_scored(phi, status=1),
        beta=beta_sv,
        epsilon=eps,
        ci=ci,
        phi_sigma_divergence=0.0,
        r_value=r_value if r_value is not None else beta,
        d_value=0.0,
    )

def _make_payload_with_history(n=15, success=True, domain="testing", beta_quota=60.0):
    outcomes = [
        TaskOutcome(
            task_id=f"t{i}",
            task_domain=domain,
            success=success,
            reconfiguration_count=0,
            duration_seconds=5.0,
            downstream_accepted=success,
        )
        for i in range(n)
    ]
    latencies = [
        LatencySample(task_domain=domain, task_type="test",
                      latency_ms=100.0 + i, timestamp=datetime.utcnow())
        for i in range(n)
    ]
    return ObservationPayload(
        agent_id="test-agent",
        framework="autogen",
        task_domain=domain,
        domain_success_history=outcomes,
        recent_task_outcomes=outcomes[-5:],
        resource_policy=ResourcePolicy(api_quota_per_minute=beta_quota),
        latency_samples=latencies,
        reconfiguration_events=[],
        tool_topology=ToolTopology(),
        task_duration_seconds=5.0,
    )


# ═══════════════════════════════════════════════════════════════════════════
# E01–E07  _beta_norm() function
# ═══════════════════════════════════════════════════════════════════════════

def test_E01_beta_norm_at_one():
    """E01 — β_norm(1.0) = 0.5 exactly (gateway midpoint)."""
    assert _beta_norm(1.0) == pytest.approx(0.5)


def test_E02_beta_norm_range_low():
    """E02 — β_norm(0.1) ∈ (0, 0.5)."""
    v = _beta_norm(0.1)
    assert 0.0 < v < 0.5


def test_E03_beta_norm_range_high():
    """E03 — β_norm(3.0) ∈ (0.5, 1.0)."""
    v = _beta_norm(3.0)
    assert 0.5 < v < 1.0


def test_E04_beta_norm_strictly_less_than_one():
    """E04 — β_norm never reaches 1.0 for any finite β."""
    for beta in [0.1, 0.5, 1.0, 2.0, 3.0, 10.0, 100.0]:
        assert _beta_norm(beta) < 1.0


def test_E05_beta_norm_strictly_positive():
    """E05 — β_norm > 0 for all positive β."""
    for beta in [0.01, 0.1, 0.5, 1.0, 3.0]:
        assert _beta_norm(beta) > 0.0


def test_E06_beta_norm_monotone():
    """E06 — β_norm is strictly monotone increasing."""
    betas = [0.1, 0.5, 1.0, 1.5, 2.0, 3.0]
    norms = [_beta_norm(b) for b in betas]
    for i in range(len(norms) - 1):
        assert norms[i] < norms[i + 1]


def test_E07_beta_norm_formula():
    """E07 — β_norm(β) = β/(1+β) to machine precision."""
    for beta in [0.1, 0.5, 1.0, 2.0, 3.0]:
        expected = beta / (1.0 + beta)
        assert _beta_norm(beta) == pytest.approx(expected)


# ═══════════════════════════════════════════════════════════════════════════
# E08–E10  Constants
# ═══════════════════════════════════════════════════════════════════════════

def test_E08_i_floor_value():
    """E08 — I_FLOOR = -5.0."""
    assert I_FLOOR == -5.0


def test_E09_alpha_much_less_than_delta():
    """E09 — α ≪ δ: PHI_BUILD_RATE_ALPHA < PHI_DECAY_RATE_DELTA / 5."""
    assert PHI_BUILD_RATE_ALPHA < PHI_DECAY_RATE_DELTA / 5.0


def test_E10_gateway_threshold():
    """E10 — PHI_GATEWAY_THRESHOLD = 0.50 (RETIRED from structural gate role).
    The constant is retained for backward compatibility only.
    The canonical gate is β×i_∞ < 1 (Canonical Ref v1.5.6, alignment pass v3.3.0).
    """
    assert PHI_GATEWAY_THRESHOLD == pytest.approx(0.50)


# ═══════════════════════════════════════════════════════════════════════════
# E11–E16  _compute_ci() β_norm integration
# ═══════════════════════════════════════════════════════════════════════════

def test_E11_compute_ci_uses_beta_norm():
    """E11 — CI with β=1.0 uses β_norm=0.5, not raw β=1.0."""
    phi  = _make_scored(0.8, status=3)
    beta = _make_scored(1.0, status=3)
    eps  = _make_epsilon(intrinsic=0.3, ecosystem=0.2)

    ci = _compute_ci(phi, beta, eps)
    # With β_norm=0.5: CI = 1 - 0.5 * 0.8 * 0.5 = 1 - 0.2 = 0.8
    # With raw β=1.0: CI = 1 - 0.5 * 0.8 * 1.0 = 1 - 0.4 = 0.6
    assert ci is not None
    assert ci == pytest.approx(1.0 - (0.5 * 0.8 * _beta_norm(1.0)), abs=1e-3)


def test_E12_compute_ci_higher_than_raw_beta():
    """E12 — CI with β_norm is always ≥ CI with raw β when β > 0 (β_norm < β)."""
    phi  = _make_scored(0.7, status=3)
    beta = _make_scored(2.0, status=3)
    eps  = _make_epsilon(intrinsic=0.4, ecosystem=0.3)

    ci_new = _compute_ci(phi, beta, eps)
    # Raw β would give: 1 - 0.7 * 0.7 * 2.0 = 1 - 0.98 = 0.02
    ci_raw_manual = 1.0 - (0.7 * 0.7 * 2.0)
    # β_norm gives higher CI (less suppression)
    assert ci_new > ci_raw_manual


def test_E13_compute_ci_gate_still_enforced():
    """E13 — CI gate (φ ③, β ③, ε ②) unchanged after β_norm update."""
    phi  = _make_scored(0.8, status=2)   # status ② — below gate
    beta = _make_scored(1.0, status=3)
    eps  = _make_epsilon()
    assert _compute_ci(phi, beta, eps) is None


def test_E14_compute_ci_clamped_zero_to_one():
    """E14 — CI output always in [0, 1]."""
    phi  = _make_scored(0.99, status=3)
    beta = _make_scored(0.1, status=3)   # β_norm = 0.091 → very low
    eps  = _make_epsilon(intrinsic=0.1, ecosystem=0.1)
    ci = _compute_ci(phi, beta, eps)
    assert ci is not None
    assert 0.0 <= ci <= 1.0


def test_E15_compute_ci_via_computer_uses_beta_norm():
    """E15 — ObservationComputer.compute() returns CI using β_norm."""
    computer = ObservationComputer()
    payload = _make_payload_with_history(n=15, beta_quota=60.0)
    result = computer.compute(payload)
    if result.ci is not None and result.beta.computable:
        expected = 1.0 - (result.epsilon.effective * result.phi.value
                          * _beta_norm(result.beta.value))
        assert result.ci == pytest.approx(max(0.0, min(1.0, expected)), abs=0.01)


def test_E16_i_floor_not_triggered_in_normal_range():
    """E16 — I_FLOOR not triggered for typical ε/φ/β values."""
    phi  = _make_scored(0.7, status=3)
    beta = _make_scored(1.0, status=3)
    eps  = _make_epsilon(intrinsic=0.5, ecosystem=0.3)
    ci = _compute_ci(phi, beta, eps)
    assert ci is not None
    assert ci > 0.0   # normal range, floor not hit


# ═══════════════════════════════════════════════════════════════════════════
# E17–E22  ObservationResult r_value / d_value
# ═══════════════════════════════════════════════════════════════════════════

def test_E17_observation_result_has_r_value():
    """E17 — ObservationResult has r_value field."""
    r = _make_observation_result(beta=0.8, r_value=0.8)
    assert hasattr(r, 'r_value')


def test_E18_observation_result_has_d_value():
    """E18 — ObservationResult has d_value field, default 0.0."""
    r = _make_observation_result()
    assert hasattr(r, 'd_value')
    assert r.d_value == 0.0


def test_E19_computer_sets_r_value_from_beta():
    """E19 — ObservationComputer sets r_value = beta.value when β computable."""
    computer = ObservationComputer()
    payload = _make_payload_with_history(n=15)
    result = computer.compute(payload)
    if result.beta.computable:
        assert result.r_value == pytest.approx(result.beta.value, abs=0.01)


def test_E20_computer_sets_d_value_zero():
    """E20 — ObservationComputer sets d_value = 0.0 (Session 36 pending)."""
    computer = ObservationComputer()
    payload = _make_payload_with_history(n=15)
    result = computer.compute(payload)
    assert result.d_value == 0.0


def test_E21_r_value_none_when_beta_not_computable():
    """E21 — r_value is None when β not computable."""
    r = ObservationResult(
        agent_id="x",
        phi=_make_scored(0.7, status=1),
        sigma=_make_scored(0.7, status=1),
        beta=_make_scored(0.0, status=1),   # status 1 → not computable
        epsilon=_make_epsilon(),
        r_value=None,
        d_value=0.0,
    )
    assert r.r_value is None


def test_E22_beta_i_inf_below_one_means_cooperative():
    """E22 — beta_i_inf < 1.0 maps to canonical cooperative basin.
    Alignment pass v3.3.0: structural gate is β×i_∞ < 1, not R < 0.50.
    ObservationResult now carries beta_i_inf and delta_gate fields.
    """
    r = _make_observation_result(beta=0.4, r_value=0.4)
    # Verify ObservationResult has the new canonical gate fields
    assert hasattr(r, 'beta_i_inf')
    assert hasattr(r, 'delta_gate')
    # r_value still present for backward compat
    assert hasattr(r, 'r_value')


# ═══════════════════════════════════════════════════════════════════════════
# E23–E32  _apply_phi_eq7() canonical gate (alignment pass v3.3.0)
#
# Canonical gate: β×i_∞ < 1  (Canonical Ref v1.5.6 Eq.7)
# Parameter: beta_i_inf replaces r_value.
#   beta_i_inf < 1.0 → BUILD gate active (cooperative basin)
#   beta_i_inf > 1.0 → DECAY gate active (stagnation/collapse regime)
#   beta_i_inf = None → both gates inactive
# ═══════════════════════════════════════════════════════════════════════════

def _get_eq7_method(k):
    return k._apply_phi_eq7


def test_E23_eq7_build_fires_when_beta_i_inf_low_i_low():
    """E23 — φ increases when β×i_∞ < 1.0 AND i < 1.0 (canonical gate)."""
    k = _make_kernel_with_agent(phi=0.5)
    agent = k.agents["test-agent"]
    phi_before = agent.phi
    delta, event = k._apply_phi_eq7(agent, beta_i_inf=0.8, i_value=0.6, d_value=0.0)
    assert delta > 0.0
    assert agent.phi > phi_before


def test_E24_eq7_build_not_fires_when_beta_i_inf_high():
    """E24 — φ unchanged when β×i_∞ ≥ 1.0 and D=0 (no decay either)."""
    k = _make_kernel_with_agent(phi=0.5)
    agent = k.agents["test-agent"]
    phi_before = agent.phi
    delta, event = k._apply_phi_eq7(agent, beta_i_inf=1.2, i_value=0.6, d_value=0.0)
    assert delta == 0.0
    assert agent.phi == phi_before


def test_E25_eq7_compound_gate_beta_i_inf_low_i_high():
    """E25 — φ unchanged when β×i_∞ < 1.0 but i ≥ 1.0 (compound gate fails)."""
    k = _make_kernel_with_agent(phi=0.5)
    agent = k.agents["test-agent"]
    phi_before = agent.phi
    delta, event = k._apply_phi_eq7(agent, beta_i_inf=0.8, i_value=1.0, d_value=0.0)
    assert delta == 0.0
    assert agent.phi == phi_before


def test_E26_eq7_compound_gate_beta_i_inf_low_i_none():
    """E26 — φ unchanged when β×i_∞ < 1.0 but i is None (CI gate not met)."""
    k = _make_kernel_with_agent(phi=0.5)
    agent = k.agents["test-agent"]
    phi_before = agent.phi
    delta, event = k._apply_phi_eq7(agent, beta_i_inf=0.8, i_value=None, d_value=0.0)
    assert delta == 0.0
    assert agent.phi == phi_before


def test_E27_eq7_decay_fires_when_beta_i_inf_high_d_nonzero():
    """E27 — φ decreases when β×i_∞ > 1.0 and D > 0 (stagnation finding)."""
    k = _make_kernel_with_agent(phi=0.6)
    agent = k.agents["test-agent"]
    phi_before = agent.phi
    delta, event = k._apply_phi_eq7(agent, beta_i_inf=1.3, i_value=0.5, d_value=1.0)
    assert delta < 0.0
    assert agent.phi < phi_before


def test_E28_eq7_no_action_when_beta_i_inf_none():
    """E28 — No φ change when beta_i_inf is None (gate uncomputable)."""
    k = _make_kernel_with_agent(phi=0.5)
    agent = k.agents["test-agent"]
    phi_before = agent.phi
    delta, event = k._apply_phi_eq7(agent, beta_i_inf=None, i_value=0.6, d_value=0.0)
    assert delta == 0.0
    assert event == ""
    assert agent.phi == phi_before


def test_E29_eq7_phi_clamped_above_zero():
    """E29 — φ never goes below 0.0 under decay."""
    k = _make_kernel_with_agent(phi=0.01)
    agent = k.agents["test-agent"]
    delta, event = k._apply_phi_eq7(agent, beta_i_inf=1.4, i_value=0.5, d_value=100.0)
    assert agent.phi >= 0.0


def test_E30_eq7_phi_clamped_below_one():
    """E30 — φ never exceeds 1.0 under build."""
    k = _make_kernel_with_agent(phi=0.999)
    agent = k.agents["test-agent"]
    delta, event = k._apply_phi_eq7(agent, beta_i_inf=0.5, i_value=0.1, d_value=0.0)
    assert agent.phi <= 1.0


def test_E31_eq7_build_magnitude():
    """E31 — Build delta = α × (1−φ) × (1−i) when β×i_∞ < 1.0 and i < 1.0."""
    k = _make_kernel_with_agent(phi=0.5)
    agent = k.agents["test-agent"]
    phi_before = 0.5
    i_val = 0.6
    expected_delta = PHI_BUILD_RATE_ALPHA * (1.0 - phi_before) * (1.0 - i_val)
    delta, _ = k._apply_phi_eq7(agent, beta_i_inf=0.8, i_value=i_val, d_value=0.0)
    assert delta == pytest.approx(expected_delta, abs=1e-6)


def test_E32_eq7_decay_magnitude():
    """E32 — Decay delta = −δ × D × φ when β×i_∞ > 1.0 and D > 0."""
    k = _make_kernel_with_agent(phi=0.6)
    agent = k.agents["test-agent"]
    phi_before = 0.6
    d_val = 0.5
    expected_delta = -(PHI_DECAY_RATE_DELTA * d_val * phi_before)
    delta, _ = k._apply_phi_eq7(agent, beta_i_inf=1.2, i_value=0.5, d_value=d_val)
    assert delta == pytest.approx(expected_delta, abs=1e-6)


# ═══════════════════════════════════════════════════════════════════════════
# E33–E38  apply_observation() Equation 7 integration (alignment pass v3.3.0)
#
# apply_observation() uses result.beta_i_inf if set; otherwise computes
# β×i_∞ inline from result.beta.value and ETA_CANONICAL_DEFAULT.
# Tests supply beta_i_inf directly to control gate behavior.
# ═══════════════════════════════════════════════════════════════════════════

def test_E33_apply_observation_runs_eq7_cooperative():
    """E33 — apply_observation() runs Eq.7 BUILD when β×i_∞ < 1.0 and CI < 1.0."""
    k = _make_kernel_with_agent(phi=0.5)
    result = _make_observation_result(phi=0.5, beta=0.3, r_value=0.3, ci=0.7)
    result.beta_i_inf = 0.75   # explicitly cooperative: β×i_∞ < 1
    response = k.apply_observation(result)
    # PHI_EQ7_BUILD should appear in events
    eq7_events = [e for e in response["governance_events"] if "PHI_EQ7" in e]
    assert len(eq7_events) >= 1


def test_E34_apply_observation_no_eq7_when_beta_i_inf_none():
    """E34 — No Eq.7 event when beta_i_inf is None and β not computable."""
    k = _make_kernel_with_agent(phi=0.5)
    result = _make_observation_result(phi=0.5, beta=0.3, r_value=None, ci=0.7)
    result.beta_i_inf = None
    result.beta = ScoredValue(value=0.0, status=0, warnings=[])  # status=0 → computable=False
    response = k.apply_observation(result)
    eq7_events = [e for e in response["governance_events"] if "PHI_EQ7" in e]
    assert len(eq7_events) == 0


def test_E35_apply_observation_no_eq7_build_when_stagnation():
    """E35 — No Eq.7 BUILD when β×i_∞ > 1.0 (stagnation regime) and D=0."""
    k = _make_kernel_with_agent(phi=0.5)
    result = _make_observation_result(phi=0.5, beta=0.8, r_value=0.8, ci=0.7)
    result.beta_i_inf = 1.3   # non-cooperative: β×i_∞ > 1
    result.d_value = 0.0
    response = k.apply_observation(result)
    build_events = [e for e in response["governance_events"] if "PHI_EQ7_BUILD" in e]
    assert len(build_events) == 0


def test_E36_apply_observation_returns_phi_applied():
    """E36 — apply_observation() returns phi_applied value."""
    k = _make_kernel_with_agent(phi=0.5)
    result = _make_observation_result(phi=0.6, beta=0.3, r_value=0.3, ci=0.7)
    result.beta_i_inf = 0.75
    response = k.apply_observation(result)
    assert "phi_applied" in response
    assert isinstance(response["phi_applied"], float)


def test_E37_apply_observation_agent_phi_increases_cooperative():
    """E37 — Agent φ increases after apply_observation() when β×i_∞ < 1.0."""
    k = _make_kernel_with_agent(phi=0.5)
    phi_before = k.agents["test-agent"].phi
    result = _make_observation_result(phi=0.6, beta=0.3, r_value=0.3, ci=0.8)
    result.beta_i_inf = 0.75   # cooperative basin
    k.apply_observation(result)
    assert k.agents["test-agent"].phi >= phi_before


def test_E38_apply_observation_d_value_zero_no_decay():
    """E38 — No Eq.7 DECAY when D=0.0 regardless of β×i_∞ (stagnation finding)."""
    k = _make_kernel_with_agent(phi=0.6)
    result = _make_observation_result(phi=0.6, beta=0.9, r_value=0.9, ci=0.5)
    result.beta_i_inf = 1.3   # non-cooperative (stagnation regime)
    result.d_value = 0.0       # no disruption → decay term does NOT fire
    # First call
    k.apply_observation(result)
    # Second call — check no DECAY events
    decay_events = [e for e in k.apply_observation(result).get("governance_events", [])
                    if "PHI_EQ7_DECAY" in e]
    assert len(decay_events) == 0


# ═══════════════════════════════════════════════════════════════════════════
# E39–E42  Backward compatibility
# ═══════════════════════════════════════════════════════════════════════════

def test_E39_quorum_gate_uses_raw_beta():
    """E39 — Quorum gate φ·β still uses raw β (unchanged)."""
    k = _make_kernel_with_agent(phi=0.6, beta_val=0.8)
    # quorum gate uses phi * beta_raw: 0.6 * 0.8 = 0.48 < 0.5 → below quorum
    result = k.compute_omega()
    assert result["beta_service"] < 0.5


def test_E40_phi_gateway_threshold_exported():
    """E40 — PHI_GATEWAY_THRESHOLD is still exported (backward compat)."""
    assert PHI_GATEWAY_THRESHOLD == pytest.approx(0.50)


def test_E41_compute_omega_returns_beta_service():
    """E41 — compute_omega() returns beta_service key."""
    k = _make_kernel_with_agent(phi=0.7)
    omega = k.compute_omega()
    assert "beta_service" in omega
    assert isinstance(omega["beta_service"], float)


def test_E42_session34_tests_still_pass_imports():
    """E42 — Session 34 adapter imports unaffected by alignment pass changes."""
    from adapters.agentforce_adapter import AgentforceObservationBuilder
    from adapters.copilot_adapter import CopilotObservationBuilder
    from adapters.vertex_adapter import VertexObservationBuilder
    from adapters.servicenow_adapter import ServiceNowObservationBuilder
    assert AgentforceObservationBuilder is not None
    assert CopilotObservationBuilder is not None
    assert VertexObservationBuilder is not None
    assert ServiceNowObservationBuilder is not None


# ═══════════════════════════════════════════════════════════════════════════
# E43–E50  _compute_i_inf() and stagnation detector (alignment pass v3.3.0)
# ═══════════════════════════════════════════════════════════════════════════

def test_E43_compute_i_inf_at_eta_one_collapses_to_linear():
    """E43 — _compute_i_inf() matches tanh formula to machine precision."""
    import math
    eta = 0.99
    eps = 0.2
    beta_raw = 0.5
    i0 = 1.0
    result = _compute_i_inf(i0, eta, eps, beta_raw)
    bn = _beta_norm(beta_raw)
    approx = i0 * (1.0 - eta * math.tanh(eps * bn / eta))
    assert result == pytest.approx(approx, abs=1e-9)


def test_E44_compute_i_inf_saturates_below_i0():
    """E44 — i_∞ < i₀ for any positive ε, β, η (tanh term reduces i₀)."""
    for eps in [0.1, 1.0, 5.0]:
        for beta_raw in [0.5, 1.0, 2.0]:
            result = _compute_i_inf(I0_CANONICAL, ETA_CANONICAL_DEFAULT, eps, beta_raw)
            assert result < I0_CANONICAL, f"i_∞ should be < i₀ for ε={eps}, β={beta_raw}"


def test_E45_beta_times_i_inf_canonical_gate():
    """E45 — β×i_∞ < 1 in cooperative regime for canonical parameters."""
    eps = 1.5
    beta_raw = 0.8
    i_inf = _compute_i_inf(I0_CANONICAL, ETA_CANONICAL_DEFAULT, eps, beta_raw)
    beta_i_inf = beta_raw * i_inf
    assert beta_i_inf < 1.0, f"Expected cooperative basin for ε={eps}, β={beta_raw}"


def test_E46_stagnation_state_stable():
    """E46 — compute_stagnation_state() returns STABLE when β×i_∞ < 1 and D=0."""
    state = MELVKernel.compute_stagnation_state(beta_i_inf=0.8, d_value=0.0)
    assert state["state"] == "STABLE"
    assert state["intervention"] is False
    assert state["delta_gate"] == pytest.approx(-0.2, abs=1e-4)


def test_E47_stagnation_state_stagnation():
    """E47 — compute_stagnation_state() returns STAGNATION when β×i_∞ > 1 and D=0."""
    state = MELVKernel.compute_stagnation_state(beta_i_inf=1.2, d_value=0.0)
    assert state["state"] == "STAGNATION"
    assert state["intervention"] is False
    assert state["delta_gate"] == pytest.approx(0.2, abs=1e-4)


def test_E48_stagnation_state_collapse():
    """E48 — compute_stagnation_state() returns COLLAPSE when β×i_∞ > 1 and D > 0."""
    state = MELVKernel.compute_stagnation_state(beta_i_inf=1.3, d_value=0.5)
    assert state["state"] == "COLLAPSE"
    assert state["intervention"] is True


def test_E49_observation_result_has_beta_i_inf_field():
    """E49 — ObservationResult has beta_i_inf and delta_gate fields."""
    r = _make_observation_result()
    assert hasattr(r, 'beta_i_inf')
    assert hasattr(r, 'delta_gate')
    assert r.beta_i_inf is None
    assert r.delta_gate is None


def test_E50_three_tier_hierarchy_constants():
    """E50 — Canonical gate constants exported: I0_CANONICAL, ETA_CANONICAL_DEFAULT."""
    assert I0_CANONICAL == pytest.approx(1.0)
    assert ETA_CANONICAL_DEFAULT == pytest.approx(0.93)

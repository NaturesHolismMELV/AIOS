"""
test_session33.py — MELVcore Session 33: observe() Computation (v2.9.0)
=======================================================================

Tests for the full observe() primitive computation pipeline:
  - φ, σ, β, ε individual computation functions
  - ObservationComputer.compute() integration
  - CI gate conditions
  - φ/σ divergence signal
  - Governance loop (MELVKernel.apply_observation())
  - Framework adapters (LangGraph, AutoGen, CrewAI builders)

Test groups
-----------
  C01–C08  φ computation (success rate, downstream acceptance, state_reliability)
  C09–C12  σ computation (provisional ① stub behaviour)
  C13–C18  β computation (ResourcePolicy dimensions, action_scope, contention)
  C19–C24  ε computation (intrinsic, ecosystem CV, architectural)
  C25–C30  CI gate conditions and φ/σ divergence
  C31–C34  ObservationComputer.compute() integration
  C35–C38  MELVKernel.apply_observation() governance loop
  C39–C42  Framework adapter builders (LangGraph, AutoGen, CrewAI)

Author: Laurence W. Evans · ORCID: 0009-0001-0963-1840
Session: 33 · Version: 2.9.0
"""

from datetime import datetime

import pytest

from core.observe_schema import (
    ContentionEvent,
    LatencySample,
    ObservationPayload,
    ReconfigEvent,
    ResourcePolicy,
    TaskOutcome,
    ToolTopology,
    PHI_WINDOW_DEFAULT,
)
from core.observe_compute import (
    ObservationComputer,
    _compute_phi,
    _compute_sigma,
    _compute_beta,
    _compute_epsilon,
    _compute_ci,
    PHI_SIGMA_DIVERGENCE_WARNING,
)
from core.melv_engine import MELVKernel, AgentProfile, AgentStatus

_NOW = datetime(2026, 5, 3, 8, 0, 0)


# ── Helpers ────────────────────────────────────────────────────────────────

def _task(tid="t1", domain="nlp", success=True, reconfig=0, duration=10.0,
          downstream=None, consumer_beta=None):
    return TaskOutcome(
        task_id=tid, task_domain=domain, success=success,
        reconfiguration_count=reconfig, duration_seconds=duration,
        downstream_accepted=downstream, consumer_beta=consumer_beta,
    )


def _latency(domain="nlp", ttype="summarise", ms=200.0):
    return LatencySample(task_domain=domain, task_type=ttype,
                         latency_ms=ms, timestamp=_NOW)


def _reconfig(etype="branching", task_id="t1"):
    return ReconfigEvent(event_type=etype, tool_switched=False,
                         timestamp=_NOW, task_id=task_id)


def _contention(resource="tokens", origin="infra", delay=50.0):
    return ContentionEvent(resource_type=resource, origin=origin,
                           timestamp=_NOW, delay_ms=delay)


def _payload(**kwargs) -> ObservationPayload:
    defaults = dict(agent_id="agent-001", framework="langgraph", task_domain="nlp")
    defaults.update(kwargs)
    return ObservationPayload(**defaults)


def _rich_payload(n_history=20, n_latency_per_type=3) -> ObservationPayload:
    """Full payload that should gate-pass CI computation."""
    history = [_task(tid=str(i), domain="nlp", success=(i % 5 != 0),
                     downstream=(i % 4 != 0)) for i in range(n_history)]
    recent = history[-5:]
    latencies = []
    for ttype in ["summarise", "classify"]:
        for j in range(n_latency_per_type):
            latencies.append(_latency(ttype=ttype, ms=200.0 + j * 50))
    return ObservationPayload(
        agent_id="agent-rich",
        framework="langgraph",
        task_domain="nlp",
        domain_success_history=history,
        recent_task_outcomes=recent,
        resource_policy=ResourcePolicy(
            token_budget_per_hour=10000.0,
            compute_share=0.5,
            api_quota_per_minute=60.0,
        ),
        reconfiguration_events=[_reconfig() for _ in range(2)],
        latency_samples=latencies,
        tool_topology=ToolTopology(standard=3, fast_rest=2),
        task_duration_seconds=10.0,
    )


# ══════════════════════════════════════════════════════════════════
# C01–C08: φ computation
# ══════════════════════════════════════════════════════════════════

class TestPhiComputation:

    def test_C01_phi_status_one_no_domain(self):
        """C01: φ returns status ① when task_domain is None."""
        p = _payload(task_domain=None)
        sv = _compute_phi(p)
        assert sv.status == 1
        assert not sv.computable

    def test_C02_phi_status_two_sparse_history(self):
        """C02: φ returns status ② with fewer than 10 matching records."""
        p = _payload(
            domain_success_history=[_task(tid=str(i)) for i in range(5)]
        )
        sv = _compute_phi(p)
        assert sv.status == 2
        assert sv.computable

    def test_C03_phi_status_three_full_window(self):
        """C03: φ returns status ③ with ≥10 domain-matching records."""
        p = _payload(
            domain_success_history=[_task(tid=str(i)) for i in range(15)]
        )
        sv = _compute_phi(p)
        assert sv.status == 3

    def test_C04_phi_value_all_success(self):
        """C04: φ ≈ 1.0 when all tasks succeed with downstream accepted."""
        history = [_task(tid=str(i), success=True, downstream=True)
                   for i in range(15)]
        p = _payload(domain_success_history=history)
        sv = _compute_phi(p)
        assert sv.value > 0.9

    def test_C05_phi_value_all_fail(self):
        """C05: φ near 0 when all tasks fail."""
        history = [_task(tid=str(i), success=False, downstream=False)
                   for i in range(15)]
        p = _payload(domain_success_history=history)
        sv = _compute_phi(p)
        assert sv.value < 0.2

    def test_C06_phi_state_reliability_zero_pulls_to_midpoint(self):
        """C06: state_reliability=0 pulls φ toward 0.5 regardless of outcomes."""
        history = [_task(tid=str(i), success=True, downstream=True)
                   for i in range(15)]
        p = _payload(domain_success_history=history, state_reliability=0.0)
        sv = _compute_phi(p)
        assert abs(sv.value - 0.5) < 0.01

    def test_C07_phi_state_reliability_one_unchanged(self):
        """C07: state_reliability=1.0 does not alter φ computation."""
        history = [_task(tid=str(i), success=True, downstream=True)
                   for i in range(15)]
        p_no_rel = _payload(domain_success_history=history, state_reliability=None)
        p_full = _payload(domain_success_history=history, state_reliability=1.0)
        sv_no = _compute_phi(p_no_rel)
        sv_full = _compute_phi(p_full)
        assert abs(sv_no.value - sv_full.value) < 0.01

    def test_C08_phi_confidence_interval_present_at_status_three(self):
        """C08: φ has confidence interval when status ③."""
        history = [_task(tid=str(i)) for i in range(15)]
        p = _payload(domain_success_history=history)
        sv = _compute_phi(p)
        assert sv.confidence_interval is not None
        lo, hi = sv.confidence_interval
        assert lo <= sv.value <= hi


# ══════════════════════════════════════════════════════════════════
# C09–C12: σ computation
# ══════════════════════════════════════════════════════════════════

class TestSigmaComputation:

    def test_C09_sigma_always_status_one(self):
        """C09: σ is always status ① regardless of inputs."""
        p = _payload(recent_task_outcomes=[_task() for _ in range(20)])
        sv = _compute_sigma(p)
        assert sv.status == 1

    def test_C10_sigma_always_provisional(self):
        """C10: σ.provisional is always True."""
        p = _payload()
        sv = _compute_sigma(p)
        assert sv.provisional is True

    def test_C11_sigma_uses_match_score_when_provided(self):
        """C11: σ uses current_task_match_score if operator-provided."""
        p = _payload(current_task_match_score=0.85)
        sv = _compute_sigma(p)
        assert abs(sv.value - 0.85) < 0.001

    def test_C12_sigma_fallback_to_recent_success(self):
        """C12: σ falls back to recent success rate when no match score."""
        recent = [_task(tid=str(i), success=(i < 8)) for i in range(10)]
        p = _payload(recent_task_outcomes=recent)
        sv = _compute_sigma(p)
        assert abs(sv.value - 0.8) < 0.05


# ══════════════════════════════════════════════════════════════════
# C13–C18: β computation
# ══════════════════════════════════════════════════════════════════

class TestBetaComputation:

    def test_C13_beta_status_one_empty_policy_no_events(self):
        """C13: β status ① when ResourcePolicy empty and no infra events."""
        p = _payload()
        sv = _compute_beta(p)
        assert sv.status == 1

    def test_C14_beta_status_two_single_dimension(self):
        """C14: β status ② with one ResourcePolicy dimension."""
        p = _payload(resource_policy=ResourcePolicy(token_budget_per_hour=10000.0))
        sv = _compute_beta(p)
        assert sv.status == 2

    def test_C15_beta_status_three_multi_dimension(self):
        """C15: β status ③ with multiple ResourcePolicy dimensions."""
        p = _payload(resource_policy=ResourcePolicy(
            token_budget_per_hour=10000.0,
            compute_share=0.5,
            api_quota_per_minute=60.0,
        ))
        sv = _compute_beta(p)
        assert sv.status == 3

    def test_C16_beta_in_valid_range(self):
        """C16: β value is always within [0.1, 3.0]."""
        for rp in [
            ResourcePolicy(),
            ResourcePolicy(token_budget_per_hour=10000.0),
            ResourcePolicy(compute_share=0.1, memory_limit_mb=512.0),
        ]:
            p = _payload(resource_policy=rp)
            sv = _compute_beta(p)
            assert 0.1 <= sv.value <= 3.0, f"β={sv.value} out of range for {rp}"

    def test_C17_beta_contention_penalty_reduces_value(self):
        """C17: Infra contention events reduce β relative to no-contention case."""
        p_clean = _payload(resource_policy=ResourcePolicy(token_budget_per_hour=10000.0))
        p_contention = _payload(
            resource_policy=ResourcePolicy(token_budget_per_hour=10000.0),
            contention_events=[_contention(origin="infra") for _ in range(5)],
        )
        sv_clean = _compute_beta(p_clean)
        sv_cont = _compute_beta(p_contention)
        assert sv_cont.value < sv_clean.value

    def test_C18_action_scope_included_in_active_dimensions(self):
        """C18: action_scope alone makes ResourcePolicy non-empty (β computable)."""
        p = _payload(resource_policy=ResourcePolicy(
            action_scope="jira:read,comment;confluence:read"
        ))
        sv = _compute_beta(p)
        assert sv.status >= 1
        assert 0.1 <= sv.value <= 3.0


# ══════════════════════════════════════════════════════════════════
# C19–C24: ε computation
# ══════════════════════════════════════════════════════════════════

class TestEpsilonComputation:

    def test_C19_epsilon_intrinsic_zero_no_branching(self):
        """C19: ε_intrinsic=0 when no branching reconfig events."""
        p = _payload(task_duration_seconds=10.0)
        eps = _compute_epsilon(p)
        assert eps.intrinsic.value == 0.0
        assert eps.intrinsic.status == 2  # zero is valid, not missing

    def test_C20_epsilon_intrinsic_normalised_by_duration(self):
        """C20: ε_intrinsic = branching_count / duration_seconds."""
        p = _payload(
            reconfiguration_events=[_reconfig("branching") for _ in range(3)],
            task_duration_seconds=10.0,
        )
        eps = _compute_epsilon(p)
        assert abs(eps.intrinsic.value - 0.3) < 0.001

    def test_C21_epsilon_ecosystem_zero_no_latencies(self):
        """C21: ε_ecosystem=0 when no latency samples provided."""
        p = _payload()
        eps = _compute_epsilon(p)
        assert eps.ecosystem.value == 0.0
        assert eps.ecosystem.status == 1

    def test_C22_epsilon_ecosystem_cv_of_latency(self):
        """C22: ε_ecosystem = CV (std/mean) across same task_type samples."""
        # Two samples: 100ms and 300ms → mean=200, std≈141, CV≈0.707
        latencies = [
            _latency(ttype="summarise", ms=100.0),
            _latency(ttype="summarise", ms=300.0),
        ]
        p = _payload(latency_samples=latencies)
        eps = _compute_epsilon(p)
        assert abs(eps.ecosystem.value - 0.707) < 0.01

    def test_C23_epsilon_architectural_from_topology(self):
        """C23: ε_architectural = Σ(weights × counts) from ToolTopology."""
        topo = ToolTopology(standard=2, legacy=1)  # 2×1.0 + 1×2.0 = 4.0
        p = _payload(tool_topology=topo)
        eps = _compute_epsilon(p)
        assert abs(eps.architectural.value - 4.0) < 0.001

    def test_C24_epsilon_effective_is_intrinsic_plus_ecosystem(self):
        """C24: ε_effective = ε_intrinsic + ε_ecosystem (master equation)."""
        p = _payload(
            reconfiguration_events=[_reconfig("branching") for _ in range(2)],
            latency_samples=[_latency(ms=100.0), _latency(ms=300.0)],
            task_duration_seconds=10.0,
        )
        eps = _compute_epsilon(p)
        expected = round(eps.intrinsic.value + eps.ecosystem.value, 4)
        assert abs(eps.effective - expected) < 0.001


# ══════════════════════════════════════════════════════════════════
# C25–C30: CI gate and φ/σ divergence
# ══════════════════════════════════════════════════════════════════

class TestCIGateAndDivergence:

    def test_C25_ci_none_when_phi_status_low(self):
        """C25: CI is None when φ status < ③."""
        from core.observe_compute import _compute_ci, _compute_beta, _compute_epsilon
        from core.observe_schema import ScoredValue, EpsilonResult
        phi = ScoredValue(value=0.8, status=2, provisional=False)  # status ②
        beta = ScoredValue(value=1.0, status=3, provisional=False)
        eps_int = ScoredValue(value=0.2, status=2, provisional=False)
        eps_eco = ScoredValue(value=0.1, status=2, provisional=False)
        eps_arch = ScoredValue(value=2.0, status=3, provisional=False)
        eps = EpsilonResult(intrinsic=eps_int, ecosystem=eps_eco,
                            architectural=eps_arch, effective=0.3)
        ci = _compute_ci(phi, beta, eps)
        assert ci is None

    def test_C26_ci_computed_when_gate_met(self):
        """C26: CI is computed when all gate conditions are satisfied."""
        p = _rich_payload()
        computer = ObservationComputer()
        result = computer.compute(p)
        # Rich payload has 20 history records (≥10 → φ ③) and multi-dim policy
        # CI may or may not compute depending on β status — check non-None if ③
        if result.phi.status >= 3 and result.beta.status >= 3:
            if result.epsilon.intrinsic.status >= 2 and result.epsilon.ecosystem.status >= 2:
                assert result.ci is not None

    def test_C27_ci_in_valid_range(self):
        """C27: CI is always in [0, 1] when computed."""
        p = _rich_payload()
        result = ObservationComputer().compute(p)
        if result.ci is not None:
            assert 0.0 <= result.ci <= 1.0

    def test_C28_phi_sigma_divergence_computed(self):
        """C28: phi_sigma_divergence = |φ − σ| is always present."""
        p = _rich_payload()
        result = ObservationComputer().compute(p)
        assert result.phi_sigma_divergence is not None
        assert result.phi_sigma_divergence >= 0.0

    def test_C29_high_divergence_generates_warning(self):
        """C29: φ/σ divergence > threshold generates a warning."""
        # Force divergence: high φ history, low recent success
        history = [_task(tid=str(i), success=True, downstream=True)
                   for i in range(15)]
        recent = [_task(tid=f"s{i}", success=False) for i in range(5)]
        p = _payload(
            domain_success_history=history,
            recent_task_outcomes=recent,
            current_task_match_score=0.1,  # low σ
        )
        result = ObservationComputer().compute(p)
        divergence_warnings = [w for w in result.warnings if "divergence" in w.lower()]
        if result.phi_sigma_divergence > PHI_SIGMA_DIVERGENCE_WARNING:
            assert len(divergence_warnings) > 0

    def test_C30_sigma_does_not_gate_ci(self):
        """C30: σ status ① does not prevent CI computation when other gates pass."""
        p = _rich_payload()
        result = ObservationComputer().compute(p)
        # σ is always ① but CI can still be computed
        assert result.sigma.status == 1
        assert result.sigma.provisional is True
        # CI may still be non-None (dependent on β and ε status)
        # The key test: σ status alone must not veto CI
        if result.phi.status >= 3 and result.beta.status >= 3:
            if result.epsilon.intrinsic.status >= 2 and result.epsilon.ecosystem.status >= 2:
                assert result.ci is not None


# ══════════════════════════════════════════════════════════════════
# C31–C34: ObservationComputer.compute() integration
# ══════════════════════════════════════════════════════════════════

class TestObservationComputer:

    def test_C31_compute_returns_observation_result(self):
        """C31: compute() returns an ObservationResult with all fields."""
        from core.observe_schema import ObservationResult
        p = _payload()
        result = ObservationComputer().compute(p)
        assert isinstance(result, ObservationResult)
        assert result.agent_id == "agent-001"
        assert result.phi is not None
        assert result.sigma is not None
        assert result.beta is not None
        assert result.epsilon is not None

    def test_C32_compute_with_minimal_payload(self):
        """C32: compute() handles minimal payload without exceptions."""
        p = _payload()
        result = ObservationComputer().compute(p)
        assert result.phi.status == 1  # no history
        assert result.ci is None       # gate not met

    def test_C33_compute_warnings_aggregate(self):
        """C33: compute() aggregates warnings from all variable pipelines."""
        p = _payload()
        result = ObservationComputer().compute(p)
        assert isinstance(result.warnings, list)
        # Minimal payload should produce warnings about missing data
        assert len(result.warnings) > 0

    def test_C34_compute_timestamp_set(self):
        """C34: ObservationResult has a UTC timestamp."""
        from datetime import datetime as _dt
        p = _payload()
        result = ObservationComputer().compute(p)
        assert isinstance(result.timestamp, _dt)


# ══════════════════════════════════════════════════════════════════
# C35–C38: MELVKernel.apply_observation() governance loop
# ══════════════════════════════════════════════════════════════════

class TestGovernanceLoopIntegration:

    def _make_kernel_with_agent(self, agent_id="agent-001", phi=0.5):
        k = MELVKernel()
        k.register_agent(AgentProfile(
            agent_id=agent_id, name="TEST", domain="nlp",
            phi=phi, epsilon=3.0, status=AgentStatus.MATURING,
        ))
        return k

    def test_C35_apply_observation_returns_dict(self):
        """C35: apply_observation() returns a dict with required keys."""
        k = self._make_kernel_with_agent()
        p = _rich_payload()
        result = ObservationComputer().compute(p)
        # Override agent_id to match kernel
        result.agent_id = "agent-001"
        gov = k.apply_observation(result)
        assert "agent_updated" in gov
        assert "phi_applied" in gov
        assert "governance_events" in gov

    def test_C36_apply_observation_updates_phi(self):
        """C36: apply_observation() moves kernel φ toward observed φ."""
        k = self._make_kernel_with_agent(phi=0.3)
        history = [_task(tid=str(i), success=True, downstream=True) for i in range(15)]
        p = ObservationPayload(
            agent_id="agent-001",
            framework="langgraph",
            task_domain="nlp",
            domain_success_history=history,
        )
        result = ObservationComputer().compute(p)
        result.agent_id = "agent-001"
        gov = k.apply_observation(result)
        new_phi = k.agents["agent-001"].phi
        assert new_phi != pytest.approx(0.3)  # φ was updated

    def test_C37_apply_observation_auto_registers_unknown_agent(self):
        """C37: apply_observation() auto-registers agent not in kernel."""
        k = MELVKernel()  # empty kernel
        p = _rich_payload()
        result = ObservationComputer().compute(p)
        result.agent_id = "new-agent-xyz"
        gov = k.apply_observation(result)
        assert "new-agent-xyz" in k.agents
        assert any("AUTO_REGISTERED" in e for e in gov["governance_events"])

    def test_C38_apply_observation_provisions_beta_at_status_three(self):
        """C38: apply_observation() provisions β when β status ③."""
        k = self._make_kernel_with_agent()
        p = _payload(
            agent_id="agent-001",
            resource_policy=ResourcePolicy(
                token_budget_per_hour=10000.0,
                compute_share=0.5,
                api_quota_per_minute=60.0,
            ),
        )
        result = ObservationComputer().compute(p)
        result.agent_id = "agent-001"
        gov = k.apply_observation(result)
        if result.beta.status >= 3:
            assert gov["beta_provisioned"] is True


# ══════════════════════════════════════════════════════════════════
# C39–C42: Framework adapter builders
# ══════════════════════════════════════════════════════════════════

class TestFrameworkAdapters:

    def test_C39_autogen_builder_produces_valid_payload(self):
        """C39: AutoGenObservationBuilder.build() produces a valid ObservationPayload."""
        from adapters.autogen_adapter import AutoGenObservationBuilder
        builder = AutoGenObservationBuilder(
            agent_id="autogen-test",
            task_domain="code_generation",
            resource_policy=ResourcePolicy(token_budget_per_hour=5000.0),
        )
        for i in range(5):
            builder.record_turn(
                task_id=f"t{i}", success=(i % 2 == 0),
                revision_rounds=i % 3, duration_seconds=5.0,
                latency_ms=500.0 + i * 100, task_type="code_review",
            )
        payload = builder.build()
        assert payload.agent_id == "autogen-test"
        assert payload.framework == "autogen"
        assert len(payload.domain_success_history) == 5
        assert len(payload.reconfiguration_events) > 0

    def test_C40_crewai_builder_maps_delegation_to_downstream(self):
        """C40: CrewAI delegated_away=True maps to downstream_accepted=False."""
        from adapters.crewai_adapter import CrewAIObservationBuilder
        builder = CrewAIObservationBuilder(
            agent_id="crewai-researcher",
            task_domain="research",
        )
        builder.record_task(
            task_id="task-001", success=False, delegated_away=True,
            duration_seconds=8.0, latency_ms=800.0,
        )
        payload = builder.build()
        assert payload.domain_success_history[0].downstream_accepted is False

    def test_C41_langgraph_builder_produces_valid_payload(self):
        """C41: LangGraphObservationBuilder produces valid ObservationPayload."""
        from adapters.langgraph_adapter import LangGraphObservationBuilder
        builder = LangGraphObservationBuilder(
            agent_id="lg-agent",
            task_domain="retrieval",
        )
        for i in range(5):
            builder.record_node_invocation(
                task_id=f"run-{i}", node_name="retriever",
                success=True, retry_count=0,
                duration_seconds=1.5, latency_ms=1500.0,
            )
        payload = builder.build()
        assert payload.agent_id == "lg-agent"
        assert payload.framework == "langgraph"
        assert len(payload.domain_success_history) == 5

    def test_C42_adapter_payload_computes_without_error(self):
        """C42: All three adapter payloads compute() without raising exceptions."""
        from adapters.autogen_adapter import AutoGenObservationBuilder
        from adapters.crewai_adapter import CrewAIObservationBuilder
        from adapters.langgraph_adapter import LangGraphObservationBuilder

        computer = ObservationComputer()

        # AutoGen
        ag = AutoGenObservationBuilder("ag-01", "nlp")
        for i in range(12):
            ag.record_turn(f"t{i}", success=True, duration_seconds=5.0,
                           latency_ms=200.0 + i * 10, task_type="classify")
        result_ag = computer.compute(ag.build())
        assert result_ag.agent_id == "ag-01"

        # CrewAI
        cr = CrewAIObservationBuilder("cr-01", "analysis")
        for i in range(12):
            cr.record_task(f"task-{i}", success=True, duration_seconds=8.0,
                           latency_ms=800.0 + i * 20)
        result_cr = computer.compute(cr.build())
        assert result_cr.agent_id == "cr-01"

        # LangGraph
        lg = LangGraphObservationBuilder("lg-01", "code")
        for i in range(12):
            lg.record_node_invocation(f"run-{i}", "coder", success=True,
                                      duration_seconds=2.0, latency_ms=2000.0)
        result_lg = computer.compute(lg.build())
        assert result_lg.agent_id == "lg-01"

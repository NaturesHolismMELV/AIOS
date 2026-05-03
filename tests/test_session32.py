"""
test_session32.py — MELVcore Session 32: observe() Primitive Schema (v2.8.0)
=============================================================================

Validates the ObservationPayload input schema, ObservationResult output schema,
boundary enforcement guards, epistemic status propagation, and the
POST /api/observe/schema-validate endpoint.

Test groups
-----------
  T01–T08  ObservationPayload schema validation (fields, types, constraints)
  T09–T14  Boundary enforcement unit tests (all five guards)
  T15–T18  Epistemic status propagation (None / sparse / full signals)
  T19–T21  φ/σ temporal window separation
  T22–T25  ToolTopology ε_architectural computation (ARCH_CATEGORY_WEIGHTS)
  T26–T29  ResourcePolicy β signal derivation and beta_active_dimensions
  T30–T34  ContentionEvent origin classification and β pipeline guard
  T35–T40  ObservationValidator.validate() integration (full ValidationResult)

Author: Laurence W. Evans · ORCID: 0009-0001-0963-1840
Session: 32 · Version: 2.8.0
"""

import warnings
from datetime import datetime, timezone

import pytest

from core.observe_schema import (
    ARCH_CATEGORY_WEIGHTS,
    PHI_WINDOW_DEFAULT,
    SIGMA_WINDOW_DEFAULT,
    ArchitecturalMutationWarning,
    BetaBoundaryViolation,
    ContentionEvent,
    EpsilonEcosystemViolation,
    LatencySample,
    ObservationPayload,
    PhiBoundaryViolation,
    ReconfigEvent,
    ResourcePolicy,
    TaskOutcome,
    ToolTopology,
)
from core.observe_validator import (
    ObservationValidator,
    assert_architectural_not_mutated,
    assert_beta_pipeline_clean,
    assert_epsilon_ecosystem_uses_variance,
    assert_phi_domain_clean,
)

# ── Helpers ────────────────────────────────────────────────────────────────

_NOW = datetime(2026, 5, 2, 12, 0, 0)


def _task(
    task_id: str = "t1",
    domain: str = "nlp",
    success: bool = True,
    reconfig: int = 0,
    duration: float = 10.0,
    downstream: bool | None = True,
    consumer_beta: float | None = None,
) -> TaskOutcome:
    return TaskOutcome(
        task_id=task_id,
        task_domain=domain,
        success=success,
        reconfiguration_count=reconfig,
        duration_seconds=duration,
        downstream_accepted=downstream,
        consumer_beta=consumer_beta,
    )


def _contention(
    resource: str = "tokens",
    origin: str = "infra",
    delay: float = 50.0,
) -> ContentionEvent:
    return ContentionEvent(
        resource_type=resource,
        origin=origin,
        timestamp=_NOW,
        delay_ms=delay,
    )


def _reconfig(
    etype: str = "branching",
    tool_switched: bool = True,
    task_id: str = "t1",
) -> ReconfigEvent:
    return ReconfigEvent(
        event_type=etype,
        tool_switched=tool_switched,
        timestamp=_NOW,
        task_id=task_id,
    )


def _latency(
    domain: str = "nlp",
    task_type: str = "summarise",
    latency_ms: float = 200.0,
) -> LatencySample:
    return LatencySample(
        task_domain=domain,
        task_type=task_type,
        latency_ms=latency_ms,
        timestamp=_NOW,
    )


def _minimal_payload(**kwargs) -> ObservationPayload:
    defaults = dict(
        agent_id="agent-001",
        framework="langgraph",
        task_domain="nlp",
    )
    defaults.update(kwargs)
    return ObservationPayload(**defaults)


# ══════════════════════════════════════════════════════════════════
# T01–T08: ObservationPayload schema validation
# ══════════════════════════════════════════════════════════════════

class TestObservationPayloadSchema:

    def test_T01_minimal_payload_constructs(self):
        """T01: Minimal payload with only required fields constructs without error."""
        p = _minimal_payload()
        assert p.agent_id == "agent-001"
        assert p.framework == "langgraph"
        assert p.task_domain == "nlp"
        assert p.domain_success_history == []
        assert p.recent_task_outcomes == []

    def test_T02_empty_agent_id_raises(self):
        """T02: Empty agent_id raises ValueError."""
        with pytest.raises(ValueError, match="agent_id"):
            ObservationPayload(agent_id="", framework="langgraph", task_domain="nlp")

    def test_T03_whitespace_agent_id_raises(self):
        """T03: Whitespace-only agent_id raises ValueError."""
        with pytest.raises(ValueError, match="agent_id"):
            ObservationPayload(agent_id="   ", framework="langgraph", task_domain="nlp")

    def test_T04_task_domain_none_allowed(self):
        """T04: task_domain=None is valid (φ returns status ① in that case)."""
        p = _minimal_payload(task_domain=None)
        assert p.task_domain is None

    def test_T05_negative_task_duration_raises(self):
        """T05: Negative task_duration_seconds raises ValueError."""
        with pytest.raises(ValueError, match="task_duration_seconds"):
            _minimal_payload(task_duration_seconds=-1.0)

    def test_T06_current_task_match_score_out_of_range_raises(self):
        """T06: current_task_match_score outside [0,1] raises ValueError."""
        with pytest.raises(ValueError, match="current_task_match_score"):
            _minimal_payload(current_task_match_score=1.5)

    def test_T07_task_outcome_negative_duration_raises(self):
        """T07: TaskOutcome with negative duration_seconds raises ValueError."""
        with pytest.raises(ValueError, match="duration_seconds"):
            _task(duration=-1.0)

    def test_T08_task_outcome_negative_reconfig_raises(self):
        """T08: TaskOutcome with negative reconfiguration_count raises ValueError."""
        with pytest.raises(ValueError, match="reconfiguration_count"):
            _task(reconfig=-1)

    def test_T08b_consumer_beta_out_of_range_raises(self):
        """T08b: TaskOutcome consumer_beta outside [0.1, 3.0] raises ValueError."""
        with pytest.raises(ValueError, match="consumer_beta"):
            _task(consumer_beta=5.0)


# ══════════════════════════════════════════════════════════════════
# T09–T14: Boundary enforcement unit tests
# ══════════════════════════════════════════════════════════════════

class TestBoundaryGuards:

    def test_T09_beta_guard_passes_with_policy(self):
        """T09: β guard passes when ResourcePolicy has at least one dimension."""
        p = _minimal_payload(
            resource_policy=ResourcePolicy(token_budget_per_hour=10000.0),
        )
        assert_beta_pipeline_clean(p)  # must not raise

    def test_T10_beta_guard_passes_with_infra_events(self):
        """T10: β guard passes when infra ContentionEvents are present."""
        p = _minimal_payload(
            contention_events=[_contention(origin="infra")],
        )
        assert_beta_pipeline_clean(p)  # must not raise

    def test_T11_beta_guard_raises_all_agent_origin_no_policy(self):
        """T11: β guard raises BetaBoundaryViolation when only agent-origin events and no policy."""
        p = _minimal_payload(
            contention_events=[_contention(origin="agent")],
        )
        with pytest.raises(BetaBoundaryViolation):
            assert_beta_pipeline_clean(p)

    def test_T12_phi_guard_raises_zero_domain_match(self):
        """T12: φ guard raises PhiBoundaryViolation when zero history records match task_domain."""
        p = _minimal_payload(
            task_domain="nlp",
            domain_success_history=[_task(domain="finance"), _task(domain="code")],
        )
        with pytest.raises(PhiBoundaryViolation):
            assert_phi_domain_clean(p)

    def test_T13_phi_guard_passes_with_matching_records(self):
        """T13: φ guard passes when history contains domain-matching records."""
        p = _minimal_payload(
            task_domain="nlp",
            domain_success_history=[_task(domain="nlp"), _task(domain="finance")],
        )
        assert_phi_domain_clean(p)  # must not raise (partial match is OK)

    def test_T14_epsilon_ecosystem_guard_raises_on_mean_latency(self):
        """T14: ε_ecosystem guard raises EpsilonEcosystemViolation if mean latency passed."""
        p = _minimal_payload(latency_samples=[_latency(), _latency(latency_ms=300.0)])
        with pytest.raises(EpsilonEcosystemViolation):
            assert_epsilon_ecosystem_uses_variance(p, explicit_mean_latency=250.0)

    def test_T14b_epsilon_ecosystem_guard_raises_all_singletons(self):
        """T14b: ε_ecosystem guard raises when all task_types have exactly 1 sample."""
        p = _minimal_payload(
            latency_samples=[
                _latency(task_type="a", latency_ms=200.0),
                _latency(task_type="b", latency_ms=300.0),
            ]
        )
        with pytest.raises(EpsilonEcosystemViolation):
            assert_epsilon_ecosystem_uses_variance(p)

    def test_T14c_epsilon_ecosystem_guard_passes_with_multiple_samples(self):
        """T14c: ε_ecosystem guard passes with ≥2 samples per task_type."""
        p = _minimal_payload(
            latency_samples=[
                _latency(task_type="summarise", latency_ms=200.0),
                _latency(task_type="summarise", latency_ms=400.0),
            ]
        )
        assert_epsilon_ecosystem_uses_variance(p)  # must not raise

    def test_T14d_architectural_mutation_warning_fires(self):
        """T14d: ArchitecturalMutationWarning issued when topology differs from registered."""
        registered = ToolTopology(standard=3)
        p = _minimal_payload(
            tool_topology=ToolTopology(standard=5)  # different from registered
        )
        with pytest.warns(ArchitecturalMutationWarning):
            assert_architectural_not_mutated(p, registered_topology=registered)

    def test_T14e_architectural_guard_no_warning_when_topology_matches(self):
        """T14e: No ArchitecturalMutationWarning when topology matches registered."""
        topo = ToolTopology(standard=3, fast_rest=1)
        p = _minimal_payload(tool_topology=ToolTopology(standard=3, fast_rest=1))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            assert_architectural_not_mutated(p, registered_topology=topo)
        arch_warnings = [w for w in caught if issubclass(w.category, ArchitecturalMutationWarning)]
        assert len(arch_warnings) == 0


# ══════════════════════════════════════════════════════════════════
# T15–T18: Epistemic status propagation
# ══════════════════════════════════════════════════════════════════

class TestEpistemicStatusPropagation:

    def test_T15_phi_status_one_when_domain_none(self):
        """T15: φ returns status ① when task_domain is None."""
        p = _minimal_payload(task_domain=None)
        v = ObservationValidator()
        result = v.validate(p)
        phi_fs = next(fs for fs in result.field_statuses if fs.variable == "phi")
        assert phi_fs.status == 1
        assert not phi_fs.computable

    def test_T16_phi_status_two_with_sparse_history(self):
        """T16: φ returns status ② with fewer than 10 domain-matching records."""
        p = _minimal_payload(
            task_domain="nlp",
            domain_success_history=[_task(task_id=str(i), domain="nlp") for i in range(5)],
        )
        result = ObservationValidator().validate(p)
        phi_fs = next(fs for fs in result.field_statuses if fs.variable == "phi")
        assert phi_fs.status == 2
        assert phi_fs.computable

    def test_T17_phi_status_three_with_full_window(self):
        """T17: φ returns status ③ with PHI_WINDOW_DEFAULT domain-matching records."""
        p = _minimal_payload(
            task_domain="nlp",
            domain_success_history=[
                _task(task_id=str(i), domain="nlp") for i in range(PHI_WINDOW_DEFAULT)
            ],
        )
        result = ObservationValidator().validate(p)
        phi_fs = next(fs for fs in result.field_statuses if fs.variable == "phi")
        assert phi_fs.status == 3

    def test_T18_sigma_always_provisional_status_one(self):
        """T18: σ is always status ① provisional (MAIES-007 pending)."""
        p = _minimal_payload(
            recent_task_outcomes=[_task(task_id="s1", domain="nlp") for _ in range(20)],
        )
        result = ObservationValidator().validate(p)
        sigma_fs = next(fs for fs in result.field_statuses if fs.variable == "sigma")
        assert sigma_fs.status == 1
        assert sigma_fs.computable  # computable but provisional


# ══════════════════════════════════════════════════════════════════
# T19–T21: φ/σ temporal window separation
# ══════════════════════════════════════════════════════════════════

class TestPhiSigmaWindowSeparation:

    def test_T19_window_overlap_detected_in_boundary_check(self):
        """T19: Overlapping task_ids between φ and σ windows flagged in boundary checks."""
        shared_task = _task(task_id="shared-1", domain="nlp")
        p = _minimal_payload(
            task_domain="nlp",
            domain_success_history=[shared_task],
            recent_task_outcomes=[shared_task],
        )
        result = ObservationValidator().validate(p)
        window_check = next(
            bc for bc in result.boundary_checks
            if bc.guard == "phi_sigma_window_separation_guard"
        )
        assert not window_check.passed
        assert "WindowOverlapWarning" in window_check.violation_type

    def test_T20_no_overlap_when_windows_disjoint(self):
        """T20: No overlap flag when φ and σ windows use different task_ids."""
        p = _minimal_payload(
            task_domain="nlp",
            domain_success_history=[_task(task_id="phi-1", domain="nlp")],
            recent_task_outcomes=[_task(task_id="sigma-1", domain="nlp")],
        )
        result = ObservationValidator().validate(p)
        window_check = next(
            bc for bc in result.boundary_checks
            if bc.guard == "phi_sigma_window_separation_guard"
        )
        assert window_check.passed

    def test_T21_payload_domain_filter_excludes_cross_domain(self):
        """T21: domain_filtered_history excludes records not matching task_domain."""
        p = _minimal_payload(
            task_domain="nlp",
            domain_success_history=[
                _task(task_id="1", domain="nlp"),
                _task(task_id="2", domain="finance"),
                _task(task_id="3", domain="nlp"),
            ],
        )
        filtered = p.domain_filtered_history()
        assert len(filtered) == 2
        assert all(t.task_domain == "nlp" for t in filtered)


# ══════════════════════════════════════════════════════════════════
# T22–T25: ToolTopology ε_architectural computation
# ══════════════════════════════════════════════════════════════════

class TestToolTopologyEpsilonArchitectural:

    def test_T22_empty_topology_returns_zero(self):
        """T22: Empty ToolTopology yields ε_architectural = 0.0."""
        topo = ToolTopology()
        assert topo.epsilon_architectural() == 0.0

    def test_T23_single_category_computed_correctly(self):
        """T23: Single-category topology matches ARCH_CATEGORY_WEIGHTS exactly."""
        topo = ToolTopology(legacy=2)
        expected = 2 * ARCH_CATEGORY_WEIGHTS["legacy"]  # 2 × 2.0 = 4.0
        assert topo.epsilon_architectural() == pytest.approx(expected)

    def test_T24_mixed_topology_sums_correctly(self):
        """T24: Mixed topology sums weighted tool counts correctly."""
        topo = ToolTopology(
            agent_native=1,      # 1 × 0.2 = 0.2
            fast_rest=2,         # 2 × 0.5 = 1.0
            standard=1,          # 1 × 1.0 = 1.0
            human_bottlenecked=1, # 1 × 1.5 = 1.5
            legacy=1,            # 1 × 2.0 = 2.0
        )                        # total = 5.7
        assert topo.epsilon_architectural() == pytest.approx(5.7)

    def test_T25_negative_tool_count_raises(self):
        """T25: Negative tool count in ToolTopology raises ValueError."""
        with pytest.raises(ValueError):
            ToolTopology(standard=-1)

    def test_T25b_total_tools_counts_correctly(self):
        """T25b: total_tools() returns sum of all category counts."""
        topo = ToolTopology(agent_native=2, standard=3, legacy=1)
        assert topo.total_tools() == 6

    def test_T25c_topology_equality_check(self):
        """T25c: ToolTopology equality compares all category counts."""
        t1 = ToolTopology(standard=3, fast_rest=1)
        t2 = ToolTopology(standard=3, fast_rest=1)
        t3 = ToolTopology(standard=4, fast_rest=1)
        assert t1 == t2
        assert t1 != t3


# ══════════════════════════════════════════════════════════════════
# T26–T29: ResourcePolicy β signal derivation
# ══════════════════════════════════════════════════════════════════

class TestResourcePolicy:

    def test_T26_empty_resource_policy_is_empty(self):
        """T26: ResourcePolicy with all None fields reports is_empty=True."""
        rp = ResourcePolicy()
        assert rp.is_empty()

    def test_T27_single_dimension_not_empty(self):
        """T27: ResourcePolicy with one configured field reports is_empty=False."""
        rp = ResourcePolicy(token_budget_per_hour=50000.0)
        assert not rp.is_empty()
        assert rp.active_dimensions() == ["token_budget_per_hour"]

    def test_T28_all_dimensions_active(self):
        """T28: ResourcePolicy with all fields returns all four active dimensions."""
        rp = ResourcePolicy(
            token_budget_per_hour=50000.0,
            compute_share=0.25,
            memory_limit_mb=2048.0,
            api_quota_per_minute=60.0,
        )
        dims = rp.active_dimensions()
        assert len(dims) == 4
        assert "token_budget_per_hour" in dims
        assert "compute_share" in dims
        assert "memory_limit_mb" in dims
        assert "api_quota_per_minute" in dims

    def test_T29_compute_share_out_of_range_raises(self):
        """T29: compute_share outside [0,1] raises ValueError."""
        with pytest.raises(ValueError, match="compute_share"):
            ResourcePolicy(compute_share=1.5)


# ══════════════════════════════════════════════════════════════════
# T30–T34: ContentionEvent origin classification
# ══════════════════════════════════════════════════════════════════

class TestContentionEventClassification:

    def test_T30_infra_contention_events_filtered_correctly(self):
        """T30: infra_contention_events() returns only origin='infra' events."""
        p = _minimal_payload(
            contention_events=[
                _contention(origin="infra"),
                _contention(origin="agent"),
                _contention(origin="infra"),
            ]
        )
        infra = p.infra_contention_events()
        agent = p.agent_contention_events()
        assert len(infra) == 2
        assert len(agent) == 1

    def test_T31_branching_reconfig_events_filtered_correctly(self):
        """T31: branching_reconfig_events() returns only event_type='branching'."""
        p = _minimal_payload(
            reconfiguration_events=[
                _reconfig(etype="branching"),
                _reconfig(etype="repair"),
                _reconfig(etype="infra_induced"),
                _reconfig(etype="branching"),
            ]
        )
        branching = p.branching_reconfig_events()
        assert len(branching) == 2
        assert all(e.event_type == "branching" for e in branching)

    def test_T32_has_beta_signals_false_when_policy_empty_no_infra(self):
        """T32: has_beta_signals=False when ResourcePolicy empty and no infra events."""
        p = _minimal_payload()
        assert not p.has_beta_signals()

    def test_T33_has_beta_signals_true_with_policy(self):
        """T33: has_beta_signals=True when ResourcePolicy has configured dimensions."""
        p = _minimal_payload(
            resource_policy=ResourcePolicy(compute_share=0.5)
        )
        assert p.has_beta_signals()

    def test_T34_contention_event_negative_delay_raises(self):
        """T34: ContentionEvent with negative delay_ms raises ValueError."""
        with pytest.raises(ValueError, match="delay_ms"):
            _contention(delay=-1.0)


# ══════════════════════════════════════════════════════════════════
# T35–T40: ObservationValidator.validate() integration
# ══════════════════════════════════════════════════════════════════

class TestObservationValidatorIntegration:

    def test_T35_validate_returns_all_five_boundary_checks(self):
        """T35: validate() always returns exactly 5 boundary check results."""
        p = _minimal_payload()
        result = ObservationValidator().validate(p)
        assert len(result.boundary_checks) == 5

    def test_T36_validate_returns_six_field_statuses(self):
        """T36: validate() returns exactly 6 field status entries (φ, σ, β, ε×3)."""
        p = _minimal_payload()
        result = ObservationValidator().validate(p)
        variables = {fs.variable for fs in result.field_statuses}
        expected = {
            "phi", "sigma", "beta",
            "epsilon_intrinsic", "epsilon_ecosystem", "epsilon_architectural"
        }
        assert variables == expected

    def test_T37_validate_schema_valid_clean_payload(self):
        """T37: schema_valid=True for a clean payload with no violations."""
        p = _minimal_payload(
            task_domain="nlp",
            resource_policy=ResourcePolicy(token_budget_per_hour=10000.0),
            domain_success_history=[_task(task_id=str(i), domain="nlp") for i in range(15)],
            recent_task_outcomes=[_task(task_id=f"s{i}", domain="nlp") for i in range(5)],
        )
        result = ObservationValidator().validate(p)
        assert result.schema_valid

    def test_T38_validate_schema_invalid_on_beta_violation(self):
        """T38: schema_valid=False when BetaBoundaryViolation detected."""
        p = _minimal_payload(
            contention_events=[_contention(origin="agent")],
            # no ResourcePolicy and no infra events → violation
        )
        result = ObservationValidator().validate(p)
        beta_check = next(
            bc for bc in result.boundary_checks
            if bc.guard == "beta_pipeline_guard"
        )
        assert not beta_check.passed
        assert beta_check.violation_type == "BetaBoundaryViolation"
        assert not result.schema_valid

    def test_T39_signal_counts_accurate(self):
        """T39: ValidationResult signal_counts match payload contents."""
        p = _minimal_payload(
            task_domain="nlp",
            domain_success_history=[_task(task_id="p1", domain="nlp"),
                                     _task(task_id="p2", domain="finance")],
            recent_task_outcomes=[_task(task_id="s1", domain="nlp")],
            contention_events=[_contention(origin="infra"),
                               _contention(origin="agent")],
            reconfiguration_events=[_reconfig("branching"), _reconfig("repair")],
            latency_samples=[_latency(), _latency(latency_ms=300.0)],
        )
        result = ObservationValidator().validate(p)
        assert result.phi_history_count == 2
        assert result.phi_domain_filtered_count == 1   # only "nlp" matches
        assert result.sigma_recent_count == 1
        assert result.infra_contention_count == 1
        assert result.agent_contention_count == 1
        assert result.branching_reconfig_count == 1
        assert result.latency_sample_count == 2

    def test_T40_epsilon_architectural_in_result(self):
        """T40: ValidationResult.epsilon_architectural reflects ToolTopology computation."""
        topo = ToolTopology(standard=2, legacy=1)   # 2×1.0 + 1×2.0 = 4.0
        p = _minimal_payload(tool_topology=topo)
        result = ObservationValidator().validate(p)
        assert result.epsilon_architectural == pytest.approx(4.0)

    def test_T40b_beta_status_three_with_multi_dimension_policy(self):
        """T40b: β field status ③ when ResourcePolicy has 3+ active dimensions."""
        p = _minimal_payload(
            resource_policy=ResourcePolicy(
                token_budget_per_hour=10000.0,
                compute_share=0.5,
                memory_limit_mb=1024.0,
            )
        )
        result = ObservationValidator().validate(p)
        beta_fs = next(fs for fs in result.field_statuses if fs.variable == "beta")
        assert beta_fs.status == 3
        assert beta_fs.computable

    def test_T40c_phi_guard_passes_when_domain_none(self):
        """T40c: φ guard does not raise when task_domain is None (no domain to violate)."""
        p = _minimal_payload(
            task_domain=None,
            domain_success_history=[_task(domain="finance")],
        )
        assert_phi_domain_clean(p)  # must not raise — None domain skips the guard

    def test_T40d_validate_with_registered_topology_triggers_arch_mutation(self):
        """T40d: Arch mutation check appears in boundary results when topology differs."""
        registered = ToolTopology(standard=2)
        p = _minimal_payload(tool_topology=ToolTopology(standard=5))
        result = ObservationValidator().validate(p, registered_topology=registered)
        arch_check = next(
            bc for bc in result.boundary_checks
            if bc.guard == "architectural_mutation_guard"
        )
        assert not arch_check.passed
        assert arch_check.violation_type == "ArchitecturalMutationWarning"
        # Note: arch mutation is a Warning, not a fatal violation
        # schema_valid may still be True depending on other guards


# ══════════════════════════════════════════════════════════════════
# PATCH v2.8.1 TESTS — action_scope and state_reliability
# ══════════════════════════════════════════════════════════════════

class TestPatchV281:

    def test_P01_action_scope_accepted_in_resource_policy(self):
        """P01: ResourcePolicy accepts action_scope string."""
        rp = ResourcePolicy(action_scope="jira:read,comment;confluence:read")
        assert rp.action_scope == "jira:read,comment;confluence:read"
        assert not rp.is_empty()

    def test_P02_action_scope_in_active_dimensions(self):
        """P02: action_scope appears in active_dimensions when set."""
        rp = ResourcePolicy(action_scope="crm:read,update_fields")
        assert "action_scope" in rp.active_dimensions()

    def test_P03_action_scope_alone_sufficient_for_beta(self):
        """P03: ResourcePolicy with only action_scope is not empty — β computable."""
        rp = ResourcePolicy(action_scope="servicenow:read,create_incident")
        assert not rp.is_empty()
        assert rp.active_dimensions() == ["action_scope"]

    def test_P04_action_scope_with_quota_dims(self):
        """P04: action_scope combines with quota dimensions correctly."""
        rp = ResourcePolicy(
            token_budget_per_hour=5000.0,
            action_scope="jira:read,comment",
        )
        dims = rp.active_dimensions()
        assert "token_budget_per_hour" in dims
        assert "action_scope" in dims
        assert len(dims) == 2

    def test_P05_state_reliability_accepted_on_payload(self):
        """P05: ObservationPayload accepts state_reliability in [0,1]."""
        p = _minimal_payload(state_reliability=0.8)
        assert p.state_reliability == 0.8

    def test_P06_state_reliability_none_accepted(self):
        """P06: state_reliability=None is valid (no effect on φ status)."""
        p = _minimal_payload(state_reliability=None)
        assert p.state_reliability is None

    def test_P07_state_reliability_out_of_range_raises(self):
        """P07: state_reliability outside [0,1] raises ValueError."""
        with pytest.raises(ValueError, match="state_reliability"):
            _minimal_payload(state_reliability=1.5)

    def test_P08_state_reliability_negative_raises(self):
        """P08: Negative state_reliability raises ValueError."""
        with pytest.raises(ValueError, match="state_reliability"):
            _minimal_payload(state_reliability=-0.1)

    def test_P09_state_reliability_propagated_to_validation_result(self):
        """P09: ValidationResult carries state_reliability from payload."""
        p = _minimal_payload(state_reliability=0.6)
        result = ObservationValidator().validate(p)
        assert result.state_reliability == pytest.approx(0.6)

    def test_P10_state_reliability_none_propagated_correctly(self):
        """P10: None state_reliability propagates as None in ValidationResult."""
        p = _minimal_payload(state_reliability=None)
        result = ObservationValidator().validate(p)
        assert result.state_reliability is None

    def test_P11_beta_status_two_with_action_scope_only(self):
        """P11: β status ② when only action_scope provided (single dimension)."""
        p = _minimal_payload(
            resource_policy=ResourcePolicy(
                action_scope="jira:read,comment;confluence:read"
            )
        )
        result = ObservationValidator().validate(p)
        beta_fs = next(fs for fs in result.field_statuses if fs.variable == "beta")
        assert beta_fs.status == 2
        assert beta_fs.computable

    def test_P12_beta_status_three_with_action_scope_plus_quota(self):
        """P12: β status ③ with action_scope plus at least one quota dimension."""
        p = _minimal_payload(
            resource_policy=ResourcePolicy(
                action_scope="jira:read,comment",
                token_budget_per_hour=10000.0,
                compute_share=0.25,
            )
        )
        result = ObservationValidator().validate(p)
        beta_fs = next(fs for fs in result.field_statuses if fs.variable == "beta")
        assert beta_fs.status == 3

"""
observe_validator.py — MELVcore Session 32 patch · v2.8.1
====================================================
Schema validation and boundary enforcement for the observe() primitive.

Validates ObservationPayload and returns a structured ValidationResult
describing field-level epistemic status, boundary guard outcomes, and
data-quality warnings. Does NOT compute φ/σ/β/ε values (Session 33 scope).

Boundary guards are implemented as mandatory pre-checks, not advisories.
Each guard raises the appropriate exception if a violation is detected.
The validate() method runs all guards and collects results into a report
so the schema-validate endpoint can return the full picture rather than
failing on the first violation.

Author: Laurence W. Evans · ORCID: 0009-0001-0963-1840
        Cape Town, South Africa
Session: 32 patch · Version: 2.8.1
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Optional

from core.observe_schema import (
    ObservationPayload,
    ToolTopology,
    BetaBoundaryViolation,
    PhiBoundaryViolation,
    EpsilonEcosystemViolation,
    ArchitecturalMutationWarning,
    ARCH_CATEGORY_WEIGHTS,
    PHI_WINDOW_DEFAULT,
    SIGMA_WINDOW_DEFAULT,
)


# ══════════════════════════════════════════════════════════════════
# VALIDATION RESULT TYPES
# ══════════════════════════════════════════════════════════════════

@dataclass
class FieldStatus:
    """Epistemic status and data-quality flags for a single variable pipeline."""
    variable: str               # phi | sigma | beta | epsilon_intrinsic | ...
    status: int                 # 1 2 3 4
    computable: bool            # can this variable be computed?
    reason: str                 # human-readable explanation
    warnings: list[str] = field(default_factory=list)


@dataclass
class BoundaryCheckResult:
    """Outcome of a single boundary enforcement check."""
    guard: str                  # e.g. "beta_pipeline_guard"
    passed: bool
    violation_type: Optional[str] = None   # exception class name if failed
    detail: Optional[str] = None


@dataclass
class ValidationResult:
    """
    Full output of ObservationValidator.validate().

    Returned by POST /api/observe/schema-validate.
    Carries field-level epistemic status, boundary guard outcomes,
    and aggregate data-quality warnings — no φ/σ/β/ε values.
    """
    agent_id: str
    framework: str
    task_domain: Optional[str]

    # Field-level epistemic status
    field_statuses: list[FieldStatus] = field(default_factory=list)

    # Boundary guard outcomes
    boundary_checks: list[BoundaryCheckResult] = field(default_factory=list)

    # Aggregate warnings (non-fatal)
    warnings: list[str] = field(default_factory=list)

    # Signal counts (diagnostic)
    phi_history_count: int = 0
    phi_domain_filtered_count: int = 0
    state_reliability: Optional[float] = None   # v2.8.1 — propagated from payload
    sigma_recent_count: int = 0
    infra_contention_count: int = 0
    agent_contention_count: int = 0
    branching_reconfig_count: int = 0
    latency_sample_count: int = 0
    epsilon_architectural: float = 0.0
    beta_active_dimensions: list[str] = field(default_factory=list)

    # Overall verdict
    schema_valid: bool = True
    boundary_violations: int = 0


# ══════════════════════════════════════════════════════════════════
# BOUNDARY GUARDS (standalone — callable independently in Session 33)
# ══════════════════════════════════════════════════════════════════

def assert_beta_pipeline_clean(payload: ObservationPayload) -> None:
    """
    β pipeline guard.

    Asserts that the β computation inputs contain NO agent behavioural
    trace data. Only ResourcePolicy and infra-originated ContentionEvents
    are valid β inputs.

    This guard validates the data sources declared in the payload, not the
    computation itself (which is Session 33 scope). In Session 32 it checks
    that the payload does not mix agent-origin contention events into a
    context where they could corrupt β.

    Raises
    ------
    BetaBoundaryViolation
        If all contention events are agent-originated AND ResourcePolicy
        is empty — caller has no valid β signal and must not estimate β
        from agent traces as a fallback.
    """
    has_policy = not payload.resource_policy.is_empty()
    has_infra_events = len(payload.infra_contention_events()) > 0
    has_any_events = len(payload.contention_events) > 0
    all_agent_origin = (
        has_any_events and
        len(payload.agent_contention_events()) == len(payload.contention_events)
    )

    if not has_policy and not has_infra_events and all_agent_origin:
        raise BetaBoundaryViolation(
            "β pipeline guard: All contention events are agent-originated and "
            "ResourcePolicy is empty. β cannot be computed without ResourcePolicy "
            "or infra-originated contention events. Do not estimate β from agent "
            "behavioural traces."
        )


def assert_phi_domain_clean(payload: ObservationPayload) -> None:
    """
    φ pipeline domain guard.

    Asserts that domain_success_history records do not systematically
    represent a different domain than task_domain. Cross-domain records
    are filtered (valid) but if ALL history records are from a different
    domain, φ computation would be invalid.

    Raises
    ------
    PhiBoundaryViolation
        If task_domain is set, domain_success_history is non-empty, and
        ZERO records match task_domain (100% cross-domain contamination).
    """
    if payload.task_domain is None:
        return  # No domain set — φ will return status ① (not a violation)

    if not payload.domain_success_history:
        return  # Empty history — φ will return status ① (not a violation)

    matching = payload.domain_filtered_history()
    if len(matching) == 0:
        non_matching_domains = {t.task_domain for t in payload.domain_success_history}
        raise PhiBoundaryViolation(
            f"φ pipeline guard: task_domain='{payload.task_domain}' but all "
            f"{len(payload.domain_success_history)} history records are from "
            f"different domain(s): {non_matching_domains}. "
            f"Cross-domain outcomes must not be used as the sole φ input."
        )


def assert_epsilon_ecosystem_uses_variance(
    payload: ObservationPayload,
    explicit_mean_latency: Optional[float] = None,
) -> None:
    """
    ε_ecosystem guard.

    Asserts that mean latency is NOT used as an ε_ecosystem proxy.
    In the schema layer this checks for the degenerate case of only a
    single latency sample per task_type (CV cannot be computed — single
    samples should not be treated as ε signals).

    The explicit_mean_latency parameter allows callers to pass in a
    pre-computed mean for explicit rejection (used by the computation
    layer in Session 33).

    Raises
    ------
    EpsilonEcosystemViolation
        If explicit_mean_latency is provided (caller is attempting to use
        mean latency as ε_ecosystem directly).
    """
    if explicit_mean_latency is not None:
        raise EpsilonEcosystemViolation(
            f"ε_ecosystem guard: Mean latency ({explicit_mean_latency:.1f} ms) "
            "must not be used as ε_ecosystem proxy. "
            "Use coefficient of variation (std / mean) across same task_type samples. "
            "Mean latency reflects β (environmental richness) or stable infrastructure."
        )

    # Schema-layer check: warn if any task_type has only 1 sample (CV undefined)
    if payload.latency_samples:
        from collections import Counter
        type_counts: Counter = Counter(
            (s.task_domain, s.task_type) for s in payload.latency_samples
        )
        singleton_types = [k for k, v in type_counts.items() if v == 1]
        if singleton_types and len(type_counts) > 0:
            # All task_types are singletons — CV cannot be computed for any
            if len(singleton_types) == len(type_counts):
                raise EpsilonEcosystemViolation(
                    f"ε_ecosystem guard: All {len(singleton_types)} task_type(s) have "
                    "exactly 1 latency sample. CV (std/mean) requires ≥2 samples per "
                    "task_type. ε_ecosystem cannot be computed — do not substitute "
                    "mean latency."
                )


def assert_architectural_not_mutated(
    payload: ObservationPayload,
    registered_topology: Optional[ToolTopology] = None,
) -> None:
    """
    ε_architectural immutability guard.

    If a registered ToolTopology is provided (as it will be by the
    governance kernel in Session 33), asserts that the topology in this
    payload matches the registered value exactly.

    In schema-validate (Session 32 scope), registered_topology is None
    because there is no agent registry — the guard is advisory only.
    The warning is issued to alert integrators that the guard will be
    enforced at compute time.

    Issues
    ------
    ArchitecturalMutationWarning
        If registered_topology is provided and differs from payload topology.
    """
    if registered_topology is None:
        return  # No registry available at schema-validate time

    if payload.tool_topology != registered_topology:
        warnings.warn(
            f"ε_architectural guard: Observed ToolTopology differs from registered "
            f"topology for agent '{payload.agent_id}'. "
            f"Registered: {registered_topology}. "
            f"Observed: {payload.tool_topology}. "
            "ε_architectural is immutable until redeployment. "
            "Do not recompute from runtime tool-call logs.",
            ArchitecturalMutationWarning,
            stacklevel=2,
        )


# ══════════════════════════════════════════════════════════════════
# EPISTEMIC STATUS DERIVATION HELPERS
# ══════════════════════════════════════════════════════════════════

def _phi_status(payload: ObservationPayload) -> FieldStatus:
    """Derive epistemic status for φ from payload signals."""
    warnings_list: list[str] = []

    if payload.task_domain is None:
        return FieldStatus(
            variable="phi", status=1, computable=False,
            reason="task_domain is None — φ requires domain context",
            warnings=["Set task_domain to enable φ computation"],
        )

    filtered = payload.domain_filtered_history()
    total = len(payload.domain_success_history)
    cross_domain = total - len(filtered)

    if cross_domain > 0:
        warnings_list.append(
            f"{cross_domain}/{total} history records are cross-domain "
            f"(excluded from φ pipeline)"
        )

    if len(filtered) == 0:
        return FieldStatus(
            variable="phi", status=1, computable=False,
            reason="No domain-matching history records — φ cannot be computed",
            warnings=warnings_list,
        )

    if len(filtered) < 10:
        return FieldStatus(
            variable="phi", status=2, computable=True,
            reason=f"Sparse history ({len(filtered)} domain-matching records < 10)",
            warnings=warnings_list + [
                f"φ status ②: increase domain history to ≥10 for ③"
            ],
        )

    if len(filtered) >= PHI_WINDOW_DEFAULT:
        reliability_note = (
            f"; state_reliability={payload.state_reliability:.2f}"
            if payload.state_reliability is not None else ""
        )
        return FieldStatus(
            variable="phi", status=3, computable=True,
            reason=f"Full φ window ({len(filtered)} records ≥ {PHI_WINDOW_DEFAULT})"
                   + reliability_note,
            warnings=warnings_list,
        )

    return FieldStatus(
        variable="phi", status=2, computable=True,
        reason=f"Partial φ window ({len(filtered)}/{PHI_WINDOW_DEFAULT} records)",
        warnings=warnings_list + [
            f"φ status ②: {PHI_WINDOW_DEFAULT - len(filtered)} more records "
            f"needed for full window"
        ],
    )


def _sigma_status(payload: ObservationPayload) -> FieldStatus:
    """
    Derive epistemic status for σ.

    σ is always provisional ① until MAIES-007 validates σ proxies.
    """
    return FieldStatus(
        variable="sigma", status=1, computable=True,
        reason=(
            "σ (niche matching) is provisional ① pending MAIES-007 proxy validation. "
            "Does not gate CI computation."
        ),
        warnings=[
            "σ is a provisional ① stub — MAIES-007 required for promotion",
            f"recent_task_outcomes: {len(payload.recent_task_outcomes)} records "
            f"(σ window default: {SIGMA_WINDOW_DEFAULT})",
        ],
        # provisional flag always True for σ
    )


def _beta_status(payload: ObservationPayload) -> FieldStatus:
    """Derive epistemic status for β from ResourcePolicy and infra events."""
    warnings_list: list[str] = []
    active_dims = payload.resource_policy.active_dimensions()
    infra_events = payload.infra_contention_events()
    agent_events = payload.agent_contention_events()

    if agent_events:
        warnings_list.append(
            f"{len(agent_events)} agent-originated contention event(s) excluded "
            "from β pipeline"
        )

    if payload.resource_policy.is_empty() and not infra_events:
        return FieldStatus(
            variable="beta", status=1, computable=False,
            reason="ResourcePolicy is empty and no infra contention events",
            warnings=warnings_list + [
                "Provide ResourcePolicy quotas to enable β computation"
            ],
        )

    if payload.resource_policy.is_empty():
        return FieldStatus(
            variable="beta", status=1, computable=True,
            reason=f"ResourcePolicy empty; β estimated from {len(infra_events)} "
                   "infra contention events only (low confidence)",
            warnings=warnings_list + [
                "β status ①: ResourcePolicy not provided — add quota dimensions "
                "for ② or higher"
            ],
        )

    if len(active_dims) == 1:
        return FieldStatus(
            variable="beta", status=2, computable=True,
            reason=f"Single ResourcePolicy dimension: {active_dims[0]}",
            warnings=warnings_list + [
                "β status ②: add more ResourcePolicy dimensions for ③"
            ],
        )

    return FieldStatus(
        variable="beta", status=3, computable=True,
        reason=f"ResourcePolicy with {len(active_dims)} dimension(s): "
               + ", ".join(active_dims),
        warnings=warnings_list,
    )


def _epsilon_intrinsic_status(payload: ObservationPayload) -> FieldStatus:
    """Derive epistemic status for ε_intrinsic."""
    branching = payload.branching_reconfig_events()
    total_reconfig = len(payload.reconfiguration_events)
    non_branching = total_reconfig - len(branching)

    warnings_list: list[str] = []
    if non_branching > 0:
        warnings_list.append(
            f"{non_branching} repair/infra_induced reconfig event(s) excluded "
            "from ε_intrinsic pipeline"
        )

    if payload.task_duration_seconds == 0:
        warnings_list.append(
            "task_duration_seconds=0 — ε_intrinsic normalisation will produce "
            "undefined result if branching events exist"
        )

    if not branching:
        return FieldStatus(
            variable="epsilon_intrinsic", status=2, computable=True,
            reason="No branching reconfiguration events — ε_intrinsic defaults to 0",
            warnings=warnings_list + [
                "ε_intrinsic=0 (no branching) is a valid signal, not missing data"
            ],
        )

    return FieldStatus(
        variable="epsilon_intrinsic", status=3, computable=True,
        reason=f"{len(branching)} branching event(s) available for ε_intrinsic",
        warnings=warnings_list,
    )


def _epsilon_ecosystem_status(payload: ObservationPayload) -> FieldStatus:
    """Derive epistemic status for ε_ecosystem (CV of latency)."""
    from collections import Counter

    warnings_list: list[str] = []

    if not payload.latency_samples:
        return FieldStatus(
            variable="epsilon_ecosystem", status=1, computable=False,
            reason="No latency samples — ε_ecosystem cannot be computed",
            warnings=["Provide latency_samples to enable ε_ecosystem computation"],
        )

    type_counts: Counter = Counter(
        (s.task_domain, s.task_type) for s in payload.latency_samples
    )
    singleton_types = [k for k, v in type_counts.items() if v == 1]
    valid_types = [k for k, v in type_counts.items() if v >= 2]

    if singleton_types:
        warnings_list.append(
            f"{len(singleton_types)} task_type(s) have only 1 sample — "
            "CV requires ≥2 per task_type"
        )

    if not valid_types:
        return FieldStatus(
            variable="epsilon_ecosystem", status=1, computable=False,
            reason="All task_types have <2 samples — CV cannot be computed for any",
            warnings=warnings_list,
        )

    if len(valid_types) < len(type_counts):
        return FieldStatus(
            variable="epsilon_ecosystem", status=2, computable=True,
            reason=f"{len(valid_types)}/{len(type_counts)} task_types have ≥2 samples",
            warnings=warnings_list,
        )

    return FieldStatus(
        variable="epsilon_ecosystem", status=3, computable=True,
        reason=f"All {len(type_counts)} task_type(s) have ≥2 latency samples",
        warnings=warnings_list,
    )


def _epsilon_architectural_status(payload: ObservationPayload) -> FieldStatus:
    """Derive epistemic status for ε_architectural."""
    topo = payload.tool_topology
    eps_arch = topo.epsilon_architectural()

    if topo.total_tools() == 0:
        return FieldStatus(
            variable="epsilon_architectural", status=1, computable=True,
            reason="ToolTopology has no registered tools — ε_architectural=0.0",
            warnings=[
                "Register tool topology at agent init for accurate ε_architectural"
            ],
        )

    return FieldStatus(
        variable="epsilon_architectural", status=3, computable=True,
        reason=f"ε_architectural={eps_arch:.3f} from {topo.total_tools()} "
               "registered tool(s)",
        warnings=[],
    )


# ══════════════════════════════════════════════════════════════════
# MAIN VALIDATOR
# ══════════════════════════════════════════════════════════════════

class ObservationValidator:
    """
    Validates an ObservationPayload and returns a ValidationResult.

    Runs all five boundary guards and derives epistemic status for each
    MELVcore variable pipeline. The validate() method never raises on
    violations — it collects them into BoundaryCheckResult records so
    the API endpoint can return the complete picture.

    Use the standalone guard functions (assert_beta_pipeline_clean etc.)
    when strict enforcement with exceptions is required (Session 33 compute
    layer).
    """

    def validate(
        self,
        payload: ObservationPayload,
        registered_topology: Optional[ToolTopology] = None,
    ) -> ValidationResult:
        """
        Validate payload and return full ValidationResult.

        Parameters
        ----------
        payload:              The ObservationPayload to validate.
        registered_topology:  Optional ToolTopology from agent registry.
                              If provided, the architectural mutation guard
                              compares against it. None in Session 32 (no
                              registry at schema-validate time).
        """
        result = ValidationResult(
            agent_id=payload.agent_id,
            framework=payload.framework,
            task_domain=payload.task_domain,
        )

        # ── Signal counts ─────────────────────────────────────────────────
        result.phi_history_count = len(payload.domain_success_history)
        result.state_reliability = payload.state_reliability
        result.phi_domain_filtered_count = len(payload.domain_filtered_history())
        result.sigma_recent_count = len(payload.recent_task_outcomes)
        result.infra_contention_count = len(payload.infra_contention_events())
        result.agent_contention_count = len(payload.agent_contention_events())
        result.branching_reconfig_count = len(payload.branching_reconfig_events())
        result.latency_sample_count = len(payload.latency_samples)
        result.epsilon_architectural = payload.tool_topology.epsilon_architectural()
        result.beta_active_dimensions = payload.resource_policy.active_dimensions()

        # ── Boundary guards ───────────────────────────────────────────────
        result.boundary_checks = self._run_boundary_checks(
            payload, registered_topology
        )
        result.boundary_violations = sum(
            1 for c in result.boundary_checks if not c.passed
        )

        # ── Field-level epistemic status ──────────────────────────────────
        result.field_statuses = [
            _phi_status(payload),
            _sigma_status(payload),
            _beta_status(payload),
            _epsilon_intrinsic_status(payload),
            _epsilon_ecosystem_status(payload),
            _epsilon_architectural_status(payload),
        ]

        # ── Aggregate warnings ────────────────────────────────────────────
        for fs in result.field_statuses:
            result.warnings.extend(fs.warnings)

        # ── Overall verdict ───────────────────────────────────────────────
        fatal_violations = [
            c for c in result.boundary_checks
            if not c.passed and c.violation_type not in (
                "ArchitecturalMutationWarning",  # Warning, not Error
            )
        ]
        result.schema_valid = len(fatal_violations) == 0

        return result

    def _run_boundary_checks(
        self,
        payload: ObservationPayload,
        registered_topology: Optional[ToolTopology],
    ) -> list[BoundaryCheckResult]:
        checks: list[BoundaryCheckResult] = []

        # ── 1. β pipeline guard ───────────────────────────────────────────
        try:
            assert_beta_pipeline_clean(payload)
            checks.append(BoundaryCheckResult(
                guard="beta_pipeline_guard",
                passed=True,
                detail="β inputs are ResourcePolicy and/or infra contention events only",
            ))
        except BetaBoundaryViolation as e:
            checks.append(BoundaryCheckResult(
                guard="beta_pipeline_guard",
                passed=False,
                violation_type="BetaBoundaryViolation",
                detail=str(e),
            ))

        # ── 2. φ domain guard ─────────────────────────────────────────────
        try:
            assert_phi_domain_clean(payload)
            checks.append(BoundaryCheckResult(
                guard="phi_domain_guard",
                passed=True,
                detail="φ history contains domain-matching records",
            ))
        except PhiBoundaryViolation as e:
            checks.append(BoundaryCheckResult(
                guard="phi_domain_guard",
                passed=False,
                violation_type="PhiBoundaryViolation",
                detail=str(e),
            ))

        # ── 3. ε_ecosystem variance guard ────────────────────────────────
        try:
            assert_epsilon_ecosystem_uses_variance(payload)
            checks.append(BoundaryCheckResult(
                guard="epsilon_ecosystem_variance_guard",
                passed=True,
                detail="Latency samples present with potential for CV computation",
            ))
        except EpsilonEcosystemViolation as e:
            checks.append(BoundaryCheckResult(
                guard="epsilon_ecosystem_variance_guard",
                passed=False,
                violation_type="EpsilonEcosystemViolation",
                detail=str(e),
            ))

        # ── 4. φ/σ temporal window separation guard ───────────────────────
        # Schema-level check: warn if recent_task_outcomes overlap with
        # domain_success_history by task_id.
        phi_ids = {t.task_id for t in payload.domain_success_history}
        sigma_ids = {t.task_id for t in payload.recent_task_outcomes}
        overlap = phi_ids & sigma_ids
        if overlap:
            checks.append(BoundaryCheckResult(
                guard="phi_sigma_window_separation_guard",
                passed=False,
                violation_type="WindowOverlapWarning",
                detail=(
                    f"{len(overlap)} task_id(s) appear in both domain_success_history "
                    "(φ window) and recent_task_outcomes (σ window). "
                    "φ and σ temporal windows must not overlap."
                ),
            ))
        else:
            checks.append(BoundaryCheckResult(
                guard="phi_sigma_window_separation_guard",
                passed=True,
                detail="φ and σ windows have no task_id overlap",
            ))

        # ── 5. ε_architectural mutation guard ────────────────────────────
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                assert_architectural_not_mutated(payload, registered_topology)
                if caught:
                    checks.append(BoundaryCheckResult(
                        guard="architectural_mutation_guard",
                        passed=False,
                        violation_type="ArchitecturalMutationWarning",
                        detail=str(caught[0].message),
                    ))
                else:
                    checks.append(BoundaryCheckResult(
                        guard="architectural_mutation_guard",
                        passed=True,
                        detail=(
                            "ToolTopology matches registered topology"
                            if registered_topology else
                            "No registered topology provided (schema-validate mode)"
                        ),
                    ))
        except Exception as e:
            checks.append(BoundaryCheckResult(
                guard="architectural_mutation_guard",
                passed=False,
                violation_type=type(e).__name__,
                detail=str(e),
            ))

        return checks

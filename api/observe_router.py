"""
observe_router.py — MELVcore Session 32 patch · v2.8.1
=================================================
FastAPI router for the observe() primitive schema layer.

Session 32 endpoint:
  POST /api/observe/schema-validate
    Accepts an ObservationPayload and returns full ValidationResult:
    field-level epistemic status, boundary guard outcomes, signal counts,
    and aggregate warnings. Does NOT compute φ/σ/β/ε values.

Session 33 scope:
  POST /api/observe
    Full computation endpoint (not implemented here).

Author: Laurence W. Evans · ORCID: 0009-0001-0963-1840
        Cape Town, South Africa
Session: 32 patch · Version: 2.8.1
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal, Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field, field_validator, model_validator

from core.observe_schema import (
    ARCH_CATEGORY_WEIGHTS,
    PHI_WINDOW_DEFAULT,
    SIGMA_WINDOW_DEFAULT,
    BetaBoundaryViolation,
    ContentionEvent,
    ContentionOrigin,
    LatencySample,
    ObservationPayload,
    ReconfigEvent,
    ReconfigType,
    ResourcePolicy,
    ResourceType,
    SupportedFramework,
    TaskOutcome,
    ToolTopology,
)
from core.observe_validator import ObservationValidator
from core.observe_compute import ObservationComputer

router = APIRouter()


# ══════════════════════════════════════════════════════════════════
# PYDANTIC REQUEST/RESPONSE MODELS
# (Pydantic v2 models for FastAPI — separate from the dataclasses
#  which are the canonical Python schema. The Pydantic models handle
#  JSON (de)serialisation and input validation for the HTTP layer.)
# ══════════════════════════════════════════════════════════════════

class TaskOutcomeIn(BaseModel):
    task_id: str
    task_domain: str
    success: bool
    reconfiguration_count: int = Field(ge=0)
    duration_seconds: float = Field(ge=0)
    downstream_accepted: Optional[bool] = None
    consumer_beta: Optional[float] = Field(default=None, ge=0.1, le=3.0)


class ContentionEventIn(BaseModel):
    resource_type: ResourceType
    origin: ContentionOrigin
    timestamp: datetime
    delay_ms: float = Field(ge=0)


class ReconfigEventIn(BaseModel):
    event_type: ReconfigType
    tool_switched: bool
    timestamp: datetime
    task_id: str


class LatencySampleIn(BaseModel):
    task_domain: str
    task_type: str
    latency_ms: float = Field(ge=0)
    timestamp: datetime


class ToolTopologyIn(BaseModel):
    agent_native: int = Field(default=0, ge=0)
    fast_rest: int = Field(default=0, ge=0)
    standard: int = Field(default=0, ge=0)
    human_bottlenecked: int = Field(default=0, ge=0)
    legacy: int = Field(default=0, ge=0)

    def to_dataclass(self) -> ToolTopology:
        return ToolTopology(
            agent_native=self.agent_native,
            fast_rest=self.fast_rest,
            standard=self.standard,
            human_bottlenecked=self.human_bottlenecked,
            legacy=self.legacy,
        )


class ResourcePolicyIn(BaseModel):
    token_budget_per_hour: Optional[float] = Field(default=None, ge=0)
    compute_share: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    memory_limit_mb: Optional[float] = Field(default=None, ge=0)
    api_quota_per_minute: Optional[float] = Field(default=None, ge=0)
    action_scope: Optional[str] = None           # v2.8.1 — enterprise verb/resource permissions


class ObservationPayloadIn(BaseModel):
    """HTTP input schema for ObservationPayload."""
    agent_id: str = Field(min_length=1)
    framework: SupportedFramework
    task_domain: Optional[str] = None

    domain_success_history: list[TaskOutcomeIn] = Field(default_factory=list)
    recent_task_outcomes: list[TaskOutcomeIn] = Field(default_factory=list)
    current_task_match_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)

    resource_policy: ResourcePolicyIn = Field(default_factory=ResourcePolicyIn)
    contention_events: list[ContentionEventIn] = Field(default_factory=list)

    reconfiguration_events: list[ReconfigEventIn] = Field(default_factory=list)
    latency_samples: list[LatencySampleIn] = Field(default_factory=list)
    tool_topology: ToolTopologyIn = Field(default_factory=ToolTopologyIn)
    task_duration_seconds: float = Field(default=0.0, ge=0)
    state_reliability: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    # v2.8.1 — operator data-quality estimate for domain_success_history


# ── Response models ────────────────────────────────────────────────────────

class FieldStatusOut(BaseModel):
    variable: str
    status: int
    computable: bool
    reason: str
    warnings: list[str]


class BoundaryCheckOut(BaseModel):
    guard: str
    passed: bool
    violation_type: Optional[str]
    detail: Optional[str]


class SchemaValidateResponse(BaseModel):
    agent_id: str
    framework: str
    task_domain: Optional[str]
    schema_valid: bool
    boundary_violations: int
    field_statuses: list[FieldStatusOut]
    boundary_checks: list[BoundaryCheckOut]
    warnings: list[str]
    signal_counts: dict[str, Any]
    epsilon_architectural: float
    beta_active_dimensions: list[str]
    melv_version: str
    session: int
    timestamp: str


# ══════════════════════════════════════════════════════════════════
# CONVERSION HELPERS
# ══════════════════════════════════════════════════════════════════

def _pydantic_to_dataclass(p: ObservationPayloadIn) -> ObservationPayload:
    """Convert Pydantic HTTP model to canonical dataclass schema."""

    def _task_outcome(t: TaskOutcomeIn) -> TaskOutcome:
        return TaskOutcome(
            task_id=t.task_id,
            task_domain=t.task_domain,
            success=t.success,
            reconfiguration_count=t.reconfiguration_count,
            duration_seconds=t.duration_seconds,
            downstream_accepted=t.downstream_accepted,
            consumer_beta=t.consumer_beta,
        )

    def _contention(c: ContentionEventIn) -> ContentionEvent:
        return ContentionEvent(
            resource_type=c.resource_type,
            origin=c.origin,
            timestamp=c.timestamp,
            delay_ms=c.delay_ms,
        )

    def _reconfig(r: ReconfigEventIn) -> ReconfigEvent:
        return ReconfigEvent(
            event_type=r.event_type,
            tool_switched=r.tool_switched,
            timestamp=r.timestamp,
            task_id=r.task_id,
        )

    def _latency(l: LatencySampleIn) -> LatencySample:
        return LatencySample(
            task_domain=l.task_domain,
            task_type=l.task_type,
            latency_ms=l.latency_ms,
            timestamp=l.timestamp,
        )

    return ObservationPayload(
        agent_id=p.agent_id,
        framework=p.framework,
        task_domain=p.task_domain,
        domain_success_history=[_task_outcome(t) for t in p.domain_success_history],
        recent_task_outcomes=[_task_outcome(t) for t in p.recent_task_outcomes],
        current_task_match_score=p.current_task_match_score,
        resource_policy=ResourcePolicy(
            token_budget_per_hour=p.resource_policy.token_budget_per_hour,
            compute_share=p.resource_policy.compute_share,
            memory_limit_mb=p.resource_policy.memory_limit_mb,
            api_quota_per_minute=p.resource_policy.api_quota_per_minute,
            action_scope=p.resource_policy.action_scope,          # v2.8.1
        ),
        contention_events=[_contention(c) for c in p.contention_events],
        reconfiguration_events=[_reconfig(r) for r in p.reconfiguration_events],
        latency_samples=[_latency(l) for l in p.latency_samples],
        tool_topology=p.tool_topology.to_dataclass(),
        task_duration_seconds=p.task_duration_seconds,
        state_reliability=p.state_reliability,                    # v2.8.1
    )


# ══════════════════════════════════════════════════════════════════
# ENDPOINTS
# ══════════════════════════════════════════════════════════════════


# ── POST /api/observe response model ──────────────────────────────────────

class ScoredValueOut(BaseModel):
    value: float
    status: int
    confidence_interval: Optional[tuple[float, float]]
    warnings: list[str]
    provisional: bool

class EpsilonResultOut(BaseModel):
    intrinsic:     ScoredValueOut
    ecosystem:     ScoredValueOut
    architectural: ScoredValueOut
    effective:     float

class ObserveResponse(BaseModel):
    agent_id:              str
    framework:             str
    task_domain:           Optional[str]
    phi:                   ScoredValueOut
    sigma:                 ScoredValueOut
    beta:                  ScoredValueOut
    epsilon:               EpsilonResultOut
    ci:                    Optional[float]
    phi_sigma_divergence:  Optional[float]
    warnings:              list[str]
    governance:            Optional[dict]   # apply_observation result if kernel available
    melv_version:          str
    session:               int
    timestamp:             str


_validator = ObservationValidator()
_computer  = ObservationComputer()


@router.post(
    "/schema-validate",
    response_model=SchemaValidateResponse,
    summary="Validate ObservationPayload schema",
    description=(
        "Session 32 endpoint. Accepts an ObservationPayload and returns "
        "field-level epistemic status, boundary guard outcomes, signal counts, "
        "and aggregate warnings. Does NOT compute φ/σ/β/ε values — "
        "that is POST /api/observe (Session 33 scope)."
    ),
    tags=["Observe Primitive"],
)
async def schema_validate(payload_in: ObservationPayloadIn) -> SchemaValidateResponse:
    """
    POST /api/observe/schema-validate

    Validates an ObservationPayload and returns the full ValidationResult.

    Returns
    -------
    SchemaValidateResponse with:
      - schema_valid: True if no fatal boundary violations detected
      - boundary_violations: count of violations (including warnings)
      - field_statuses: per-variable epistemic status (φ, σ, β, ε×3)
      - boundary_checks: result of all five boundary guards
      - signal_counts: diagnostic counts of each signal class
      - warnings: aggregate list of data-quality warnings
    """
    try:
        payload = _pydantic_to_dataclass(payload_in)
    except (ValueError, TypeError) as e:
        raise HTTPException(status_code=422, detail=str(e))

    vr = _validator.validate(payload)

    return SchemaValidateResponse(
        agent_id=vr.agent_id,
        framework=vr.framework,
        task_domain=vr.task_domain,
        schema_valid=vr.schema_valid,
        boundary_violations=vr.boundary_violations,
        field_statuses=[
            FieldStatusOut(
                variable=fs.variable,
                status=fs.status,
                computable=fs.computable,
                reason=fs.reason,
                warnings=fs.warnings,
            )
            for fs in vr.field_statuses
        ],
        boundary_checks=[
            BoundaryCheckOut(
                guard=bc.guard,
                passed=bc.passed,
                violation_type=bc.violation_type,
                detail=bc.detail,
            )
            for bc in vr.boundary_checks
        ],
        warnings=vr.warnings,
        signal_counts={
            "phi_history_total": vr.phi_history_count,
            "phi_history_domain_filtered": vr.phi_domain_filtered_count,
            "phi_state_reliability": vr.state_reliability,        # v2.8.1
            "sigma_recent": vr.sigma_recent_count,
            "infra_contention_events": vr.infra_contention_count,
            "agent_contention_events": vr.agent_contention_count,
            "branching_reconfig_events": vr.branching_reconfig_count,
            "latency_samples": vr.latency_sample_count,
        },
        epsilon_architectural=vr.epsilon_architectural,
        beta_active_dimensions=vr.beta_active_dimensions,
        melv_version="2.8.1",
        session=32,
        timestamp=datetime.utcnow().isoformat() + "Z",
    )




@router.post(
    "/",
    response_model=ObserveResponse,
    summary="Observe agent signals and compute φ/σ/β/ε",
    description=(
        "Session 33 endpoint. Accepts an ObservationPayload, computes φ, σ, β, "
        "and ε values, and returns an ObservationResult. If the MELVKernel is "
        "available via app state, applies the result to the governance loop. "
        "CI is computed when φ ③+, β ③+, ε_intrinsic ②+, ε_ecosystem ②+."
    ),
    tags=["Observe Primitive"],
)
async def observe(
    payload_in: ObservationPayloadIn,
    request: Request,
) -> ObserveResponse:
    """
    POST /api/observe

    Full observe() primitive. Validates payload, computes all MELVcore
    variables, and optionally applies the result to the governance kernel.

    Returns ObserveResponse with:
      - phi, sigma, beta, epsilon: computed values with epistemic status
      - ci: Cooperation Index (None if gate not met)
      - phi_sigma_divergence: domain-shift signal
      - governance: kernel apply_observation() result (if kernel available)
      - warnings: aggregate data-quality and boundary warnings
    """
    try:
        payload = _pydantic_to_dataclass(payload_in)
    except (ValueError, TypeError) as e:
        raise HTTPException(status_code=422, detail=str(e))

    result = _computer.compute(payload)

    # ── Optional governance loop integration ──────────────────────────────
    governance: Optional[dict] = None
    try:
        kernel = getattr(request.app.state, "kernel", None)
        if kernel is not None:
            governance = kernel.apply_observation(result)
    except Exception as gov_exc:
        governance = {"error": str(gov_exc)}

    def _sv(sv) -> ScoredValueOut:
        return ScoredValueOut(
            value=sv.value,
            status=sv.status,
            confidence_interval=sv.confidence_interval,
            warnings=sv.warnings,
            provisional=sv.provisional,
        )

    eps_out = EpsilonResultOut(
        intrinsic=_sv(result.epsilon.intrinsic),
        ecosystem=_sv(result.epsilon.ecosystem),
        architectural=_sv(result.epsilon.architectural),
        effective=result.epsilon.effective,
    )

    return ObserveResponse(
        agent_id=result.agent_id,
        framework=payload_in.framework,
        task_domain=payload_in.task_domain,
        phi=_sv(result.phi),
        sigma=_sv(result.sigma),
        beta=_sv(result.beta),
        epsilon=eps_out,
        ci=result.ci,
        phi_sigma_divergence=result.phi_sigma_divergence,
        warnings=result.warnings,
        governance=governance,
        melv_version="2.9.0",
        session=33,
        timestamp=result.timestamp.isoformat() + "Z",
    )

@router.get(
    "/schema",
    summary="Describe ObservationPayload schema",
    tags=["Observe Primitive"],
)
async def describe_schema() -> dict:
    """
    GET /api/observe/schema

    Returns a human-readable description of the ObservationPayload schema
    with variable definitions, epistemic status scale, boundary rules,
    and framework signal extraction notes.
    """
    return {
        "melv_version": "2.8.1",
        "session": 32,
        "description": (
            "ObservationPayload schema for the MELVcore observe() primitive. "
            "Translates multi-agent AI framework signals into φ, σ, β, ε inputs."
        ),
        "variable_measurability_classes": {
            "epsilon": "Directly observable from behavioural traces",
            "phi": "Conditionally derivable — requires domain-conditioned history",
            "beta": "Latent environmental variable — requires ResourcePolicy reconstruction",
            "sigma": "Provisional ① — MAIES-007 required for proxy validation",
        },
        "epistemic_status_scale": {
            "1": "Defined in Blueprint for Harmony / theory; not yet MAIES-validated",
            "2": "One MAIES system confirmed; plausible but not converged",
            "3": "Three+ MAIES systems confirmed; strong convergence",
            "4": "Full cross-system convergence + empirical grounding",
        },
        "boundary_rules": {
            "beta": (
                "β estimators may ONLY read from ResourcePolicy and "
                "infra-originated ContentionEvents. Agent traces are excluded."
            ),
            "phi": (
                "φ computation is domain-conditioned. TaskOutcome records must "
                "match task_domain. Cross-domain mixing raises PhiBoundaryViolation."
            ),
            "epsilon_ecosystem": (
                "ε_ecosystem = CV (std/mean) across same task_type latency samples. "
                "Mean latency must NOT be used — it reflects β, not ε."
            ),
            "epsilon_architectural": (
                "ε_architectural is computed once at agent registration. "
                "Immutable until redeployment. Runtime tool logs must not recompute it."
            ),
            "sigma_phi_temporal": (
                "σ reads recent_task_outcomes (last 20). "
                "φ reads domain_success_history (last 200). Windows must not overlap."
            ),
        },
        "arch_category_weights": ARCH_CATEGORY_WEIGHTS,
        "temporal_windows": {
            "phi_default": PHI_WINDOW_DEFAULT,
            "sigma_default": SIGMA_WINDOW_DEFAULT,
        },
        "supported_frameworks": [
            "langgraph", "autogen", "crewai",
            "agentforce", "copilot_studio", "vertex_ai", "servicenow", "other",
        ],
        "ci_gate": (
            "CI is computed only when phi.status ③+, beta.status ③+, "
            "and epsilon intrinsic/ecosystem both ③+. "
            "sigma (provisional ①) does NOT gate CI computation."
        ),
        "patch_v2_8_1": {
            "state_reliability": (
                "Optional[float] in [0,1] on ObservationPayload. "
                "Operator-provided φ history quality weight. "
                "0=known-unreliable, 1=known-reliable, None=not assessed."
            ),
            "action_scope": (
                "Optional[str] on ResourcePolicy. Enterprise verb/resource "
                "permission string (e.g. 'jira:read,comment;confluence:read'). "
                "For platforms where β is bounded by role permissions "
                "rather than compute quota (Atlassian, Salesforce, ServiceNow)."
            ),
        },
        "session_33_scope": "POST /api/observe — full φ/σ/β/ε computation",
    }

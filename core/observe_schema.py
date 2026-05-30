"""
observe_schema.py — MELVcore Session 32 patch · v2.8.1
=======================================================
Formal ObservationPayload and ObservationResult schema.

Translates real-world signals from multi-agent AI frameworks
(LangGraph, AutoGen, CrewAI, Agentforce, Copilot Studio, Vertex AI,
ServiceNow, A2A-compliant platforms) into typed MELVcore φ, σ, β, ε
input structures.

This module defines schemas and boundary exceptions ONLY.
Computation logic is Session 33 scope (POST /api/observe).

══════════════════════════════════════════════════════════════════
PATCH v2.8.1 — two additions from video transcript analysis
══════════════════════════════════════════════════════════════════

1. ResourcePolicy.action_scope (Optional[str])
   Motivation: Atlassian/enterprise permission models scope β not by
   compute quota but by which verbs an agent may execute on which
   resources (projects, boards, spaces). The Rovo MCP server inherits
   Jira/Confluence role permissions directly — the permission boundary
   IS the ResourcePolicy. action_scope captures this: a structured
   string describing the agent's permitted verb/resource scope
   (e.g. "jira:read,comment;confluence:read"). Operator-provided.
   Does not affect β computation in Session 32 (schema only).

2. ObservationPayload.state_reliability (Optional[float] in [0,1])
   Motivation: "Good UX → cleaner data → better agent substrate"
   (transcript §10:13–11:07). A domain_success_history with 200 records
   from a tool people worked around is worth less than 50 honest records.
   downstream_accepted partially captures this, but an operator-provided
   reliability score allows the φ computation layer (Session 33) to
   weight history quality, not just quantity. 0.0 = known-unreliable
   (e.g. tickets created after-the-fact, fake statuses); 1.0 = known-
   reliable (e.g. Linear deployment with high adoption, clean state).
   None = operator has not assessed reliability (no effect on φ status).

══════════════════════════════════════════════════════════════════
EPISTEMIC STATUS SCALE
══════════════════════════════════════════════════════════════════
  ① — Defined in Blueprint for Harmony / theory; not yet MAIES-validated
  ② — One MAIES system confirmed; plausible but not converged
  ③ — Three+ MAIES systems confirmed; strong convergence
  ④ — Full cross-system convergence + empirical grounding

VARIABLE MEASURABILITY CLASSES (MAIES-006 primary finding):
  ε — Directly observable from behavioural traces
  φ — Conditionally derivable; requires domain-conditioned history
  β — Latent environmental variable; requires ResourcePolicy reconstruction

══════════════════════════════════════════════════════════════════
BOUNDARY ENFORCEMENT — NOT ADVISORY
══════════════════════════════════════════════════════════════════
  BetaBoundaryViolation        — β computed from agent traces
  PhiBoundaryViolation         — φ computed from cross-domain outcomes
  EpsilonEcosystemViolation    — mean latency used as ε_ecosystem proxy
  ArchitecturalMutationWarning — ToolTopology differs from registered value

Author: Laurence W. Evans · ORCID: 0009-0001-0963-1840
        Cape Town, South Africa
Session: 32 patch · Version: 2.8.1
Biological grounding: hornbill-mongoose mutualism (Namibia, 1981–83) and
                      bee-flower association (Nature's Holism, iUniverse, 1999)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal, Optional


# ══════════════════════════════════════════════════════════════════
# BOUNDARY VIOLATION EXCEPTIONS
# ══════════════════════════════════════════════════════════════════

class BetaBoundaryViolation(ValueError):
    """
    Raised when β computation is attempted from agent behavioural traces.

    β (environmental suitability) is a latent environmental variable set by
    the operator ResourcePolicy and infra-originated contention events ONLY.
    It MUST NOT be derived from agent trace data (task success rates,
    reconfiguration counts, latency means, etc.).

    MAIES-006 Part 2 finding (cross-system convergence ③+):
      'β estimators may ONLY read from ResourcePolicy and
       infra-originated contention events.'
    """


class PhiBoundaryViolation(ValueError):
    """
    Raised when φ computation crosses task_domain boundaries.

    φ (accumulated maturity) is a domain-conditioned measure of adaptive
    maturity. TaskOutcome records from different task domains MUST NOT be
    mixed in the φ computation window — doing so conflates niche-specific
    fitness signals.

    Enforcement: task_domain must be present and all domain_success_history
    records must match the payload task_domain before φ computation proceeds.
    """


class EpsilonEcosystemViolation(ValueError):
    """
    Raised when mean latency is used as an ε_ecosystem proxy.

    ε_ecosystem = coefficient of variation (std / mean) across identical
    task-type latency samples — VARIANCE, not mean.

    Mean latency reflects β (environmental richness/stability) or stable
    infrastructure throughput. Using it as ε conflates two orthogonal
    constructs. MAIES-006 Part 1 finding (④ convergence).
    """


class ArchitecturalMutationWarning(UserWarning):
    """
    Raised when the ToolTopology passed in an observe() call differs from
    the topology registered at agent initialisation.

    ε_architectural is computed ONCE at agent registration and is immutable
    until redeployment. Runtime tool-call logs MUST NOT be used to recompute
    it. A mismatch between registered and observed topology indicates either
    a misconfigured payload or an undeclared redeployment.
    """


# ══════════════════════════════════════════════════════════════════
# ε_ARCHITECTURAL WEIGHTS (Session 29 ③ — reproduced from melv_engine
# for schema-layer validation without circular import)
# ══════════════════════════════════════════════════════════════════

ARCH_CATEGORY_WEIGHTS: dict[str, float] = {
    "agent_native":        0.2,   # internal calls, minimal boundary crossing
    "fast_rest":           0.5,   # standard REST, low latency
    "standard":            1.0,   # baseline tool category
    "human_bottlenecked":  1.5,   # human-in-loop, approval gates
    "legacy":              2.0,   # legacy systems, slow boundaries
}

# Epistemic status integer literals
EpistemicStatus = Literal[1, 2, 3, 4]

# Supported frameworks (MAIES-006 scope + enterprise platforms for Session 34)
SupportedFramework = Literal[
    "langgraph",
    "autogen",
    "crewai",
    "agentforce",
    "copilot_studio",
    "vertex_ai",
    "servicenow",
    "other",
]

# Resource types that appear in contention events
ResourceType = Literal["tokens", "compute", "api_quota", "memory"]

# Contention event origin — MUST be tagged at the collection point
ContentionOrigin = Literal["infra", "agent"]

# Reconfiguration event types — only "branching" feeds ε_intrinsic
ReconfigType = Literal["branching", "repair", "infra_induced"]


# ══════════════════════════════════════════════════════════════════
# INPUT SIGNAL DATACLASSES
# ══════════════════════════════════════════════════════════════════

@dataclass
class TaskOutcome:
    """
    Single task execution outcome.

    Used for both:
      - domain_success_history → φ (long window, domain-filtered)
      - recent_task_outcomes   → σ (short window, provisional ①)

    Fields
    ------
    task_id:              Unique task identifier.
    task_domain:          Domain/niche label. Must match payload task_domain
                          for φ computation (PhiBoundaryViolation if not).
    success:              Whether the agent completed the task successfully.
    reconfiguration_count: Number of reconfiguration events during this task.
                          Used for ε_intrinsic normalisation.
    duration_seconds:     Wall-clock task duration — the normalisation
                          denominator for ε_intrinsic computation.
    downstream_accepted:  Whether a downstream consumer accepted this task's
                          output without revision. φ proxy (MAIES-006 ④).
                          None if not measurable in the framework.
    consumer_beta:        β of the downstream consumer agent, if known.
                          Normalises the downstream_accepted proxy against
                          consumer environmental context. None if unavailable.
    """
    task_id: str
    task_domain: str
    success: bool
    reconfiguration_count: int
    duration_seconds: float
    downstream_accepted: Optional[bool] = None
    consumer_beta: Optional[float] = None

    def __post_init__(self):
        if self.duration_seconds < 0:
            raise ValueError(
                f"TaskOutcome.duration_seconds must be ≥ 0; got {self.duration_seconds}"
            )
        if self.reconfiguration_count < 0:
            raise ValueError(
                f"TaskOutcome.reconfiguration_count must be ≥ 0; "
                f"got {self.reconfiguration_count}"
            )
        if self.consumer_beta is not None and not (0.1 <= self.consumer_beta <= 3.0):
            raise ValueError(
                f"TaskOutcome.consumer_beta must be in [0.1, 3.0] or None; "
                f"got {self.consumer_beta}"
            )


@dataclass
class ContentionEvent:
    """
    Resource contention event — feeds β pipeline ONLY when origin='infra'.

    BOUNDARY RULE: Only ContentionEvent records with origin='infra' may
    be used in β computation. Records with origin='agent' are excluded
    from the β pipeline (they represent agent behaviour, not environment).

    Fields
    ------
    resource_type: One of tokens | compute | api_quota | memory.
    origin:        'infra' = externally imposed (rate limits, quota, OOM).
                   'agent' = agent-generated (excessive calls, memory leak).
                   MUST be tagged at the signal collection point.
    timestamp:     UTC timestamp of the contention event.
    delay_ms:      Additional latency incurred due to this contention (ms).
    """
    resource_type: ResourceType
    origin: ContentionOrigin
    timestamp: datetime
    delay_ms: float

    def __post_init__(self):
        if self.delay_ms < 0:
            raise ValueError(
                f"ContentionEvent.delay_ms must be ≥ 0; got {self.delay_ms}"
            )


@dataclass
class ReconfigEvent:
    """
    Agent reconfiguration event — feeds ε_intrinsic (branching only).

    BOUNDARY RULE:
      - 'branching'    → ε_intrinsic: agent chose a different execution path
      - 'repair'       → diagnostic only; does NOT feed ε_intrinsic
      - 'infra_induced'→ diagnostic only; does NOT feed ε_intrinsic

    Only branching events represent genuine adaptive plasticity. Repair and
    infra-induced reconfigurations are not ε signals. MAIES-006 ④.

    Fields
    ------
    event_type:   Classification of the reconfiguration.
    tool_switched: Whether the agent switched to a different tool.
    timestamp:    UTC timestamp.
    task_id:      Links to the parent TaskOutcome for normalisation.
    """
    event_type: ReconfigType
    tool_switched: bool
    timestamp: datetime
    task_id: str


@dataclass
class LatencySample:
    """
    Single latency measurement for ε_ecosystem computation.

    ε_ecosystem = CV (coefficient of variation) = std / mean across
    same-task_type samples. Mean latency MUST NOT be used as ε proxy.
    Samples are grouped by (task_domain, task_type) for CV computation.

    Fields
    ------
    task_domain: Domain label — used to group samples.
    task_type:   Task type within domain — CV computed per task_type.
    latency_ms:  Round-trip latency in milliseconds.
    timestamp:   UTC timestamp of the measurement.
    """
    task_domain: str
    task_type: str
    latency_ms: float
    timestamp: datetime

    def __post_init__(self):
        if self.latency_ms < 0:
            raise ValueError(
                f"LatencySample.latency_ms must be ≥ 0; got {self.latency_ms}"
            )


@dataclass
class ToolTopology:
    """
    Agent tool registry — registered at agent initialisation.

    ε_architectural = Σ(ARCH_CATEGORY_WEIGHTS[category] × count)

    IMMUTABILITY RULE: Registered once at agent init. The observe() call
    receives this for audit only. If it differs from the registered value,
    ArchitecturalMutationWarning is raised.

    Counts represent the number of tools in each latency/boundary category.

    Weights (ARCH_CATEGORY_WEIGHTS — Session 29 ③):
      agent_native:        0.2
      fast_rest:           0.5
      standard:            1.0
      human_bottlenecked:  1.5
      legacy:              2.0
    """
    agent_native: int = 0           # internal calls, minimal boundary crossing
    fast_rest: int = 0              # standard REST, low latency
    standard: int = 0               # baseline tool category
    human_bottlenecked: int = 0     # human-in-loop, approval gates
    legacy: int = 0                 # legacy systems, slow boundaries

    def __post_init__(self):
        for fname, fval in [
            ("agent_native", self.agent_native),
            ("fast_rest", self.fast_rest),
            ("standard", self.standard),
            ("human_bottlenecked", self.human_bottlenecked),
            ("legacy", self.legacy),
        ]:
            if fval < 0:
                raise ValueError(
                    f"ToolTopology.{fname} must be ≥ 0; got {fval}"
                )

    def epsilon_architectural(self) -> float:
        """
        Compute ε_architectural from tool counts and ARCH_CATEGORY_WEIGHTS.

        Returns 0.0 if no tools are registered (empty topology).
        """
        return (
            self.agent_native        * ARCH_CATEGORY_WEIGHTS["agent_native"] +
            self.fast_rest           * ARCH_CATEGORY_WEIGHTS["fast_rest"] +
            self.standard            * ARCH_CATEGORY_WEIGHTS["standard"] +
            self.human_bottlenecked  * ARCH_CATEGORY_WEIGHTS["human_bottlenecked"] +
            self.legacy              * ARCH_CATEGORY_WEIGHTS["legacy"]
        )

    def total_tools(self) -> int:
        return (
            self.agent_native + self.fast_rest + self.standard +
            self.human_bottlenecked + self.legacy
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ToolTopology):
            return NotImplemented
        return (
            self.agent_native       == other.agent_native and
            self.fast_rest          == other.fast_rest and
            self.standard           == other.standard and
            self.human_bottlenecked == other.human_bottlenecked and
            self.legacy             == other.legacy
        )


@dataclass
class ResourcePolicy:
    """
    Operator-provided resource allocation policy — the sole input for β.

    β is reconstructed from quota vs. consumption. If all fields are None,
    β cannot be computed and ObservationResult.beta.status = ①.

    BOUNDARY RULE: ResourcePolicy is the ONLY operator-provided β input.
    β must NEVER be derived from agent behavioural traces.

    Fields (all optional — operators may not expose all quota dimensions)
    ------
    token_budget_per_hour:  Token allocation per hour (LLM call budget).
    compute_share:          Fractional compute allocation [0, 1].
    memory_limit_mb:        Memory ceiling in megabytes.
    api_quota_per_minute:   External API call quota per minute.
    action_scope:           Verb/resource permission string for enterprise
                            platforms where β is bounded by role permissions
                            rather than compute quota (patch v2.8.1).
                            For Atlassian: the Rovo MCP server inherits Jira/
                            Confluence role permissions — the permission boundary
                            IS the ResourcePolicy. For Salesforce Agentforce,
                            Copilot Studio, ServiceNow: similar role-scoped access.
                            Format: "system:verb[,verb];system:verb"
                            Examples:
                              "jira:read,comment;confluence:read"
                              "crm:read,update_fields;email:draft"
                              "servicenow:read,create_incident"
                            None = permission scope not declared by operator.
    """
    token_budget_per_hour: Optional[float] = None
    compute_share: Optional[float] = None
    memory_limit_mb: Optional[float] = None
    api_quota_per_minute: Optional[float] = None
    action_scope: Optional[str] = None           # v2.8.1 — enterprise permission β

    def __post_init__(self):
        if self.compute_share is not None and not (0.0 <= self.compute_share <= 1.0):
            raise ValueError(
                f"ResourcePolicy.compute_share must be in [0, 1] or None; "
                f"got {self.compute_share}"
            )
        for fname, fval in [
            ("token_budget_per_hour", self.token_budget_per_hour),
            ("memory_limit_mb", self.memory_limit_mb),
            ("api_quota_per_minute", self.api_quota_per_minute),
        ]:
            if fval is not None and fval < 0:
                raise ValueError(
                    f"ResourcePolicy.{fname} must be ≥ 0 or None; got {fval}"
                )

    def is_empty(self) -> bool:
        """Returns True if no quota or scope dimensions are defined — β uncomputable."""
        return all(v is None for v in [
            self.token_budget_per_hour,
            self.compute_share,
            self.memory_limit_mb,
            self.api_quota_per_minute,
            self.action_scope,
        ])

    def active_dimensions(self) -> list[str]:
        """Returns list of quota/scope dimension names that have been configured."""
        dims = []
        if self.token_budget_per_hour is not None:
            dims.append("token_budget_per_hour")
        if self.compute_share is not None:
            dims.append("compute_share")
        if self.memory_limit_mb is not None:
            dims.append("memory_limit_mb")
        if self.api_quota_per_minute is not None:
            dims.append("api_quota_per_minute")
        if self.action_scope is not None:
            dims.append("action_scope")
        return dims


# ══════════════════════════════════════════════════════════════════
# PRIMARY INPUT SCHEMA
# ══════════════════════════════════════════════════════════════════

# Temporal window defaults — configurable per deployment
PHI_WINDOW_DEFAULT:   int = 200   # long window: φ domain-history lookback
SIGMA_WINDOW_DEFAULT: int = 20    # short window: σ recent-outcomes lookback

@dataclass
class ObservationPayload:
    """
    Full input payload for the observe() primitive.

    Carries all signals required to compute φ, σ, β, and ε for a single
    agent at a point in time. Validated by ObservationValidator before
    any computation proceeds.

    Variable measurability classes (MAIES-006 primary finding ③+):
      ε — Directly observable from behavioural traces
      φ — Conditionally derivable; requires domain-conditioned history
      β — Latent environmental variable; requires ResourcePolicy

    Fields
    ------
    agent_id:               Unique agent identifier.
    framework:              Source multi-agent framework.
    task_domain:            Current task domain/niche. Mandatory for φ;
                            None → phi returns status ①.

    domain_success_history: Long-window task history for φ. All records
                            MUST match task_domain (PhiBoundaryViolation
                            if not). Default window: last 200 interactions.
    recent_task_outcomes:   Short-window outcomes for σ (provisional ①).
                            Default window: last 20 interactions.
    current_task_match_score: Operator-provided σ proxy [0,1] if available.
                            None if not measurable.

    resource_policy:        Operator-declared quota policy for β computation.
    contention_events:      Infrastructure and agent contention events.
                            Only origin='infra' feeds β pipeline.

    reconfiguration_events: Agent reconfiguration trace for ε_intrinsic
                            (branching events only).
    latency_samples:        Per-task-type latency samples for ε_ecosystem
                            (CV computation — NOT mean).
    tool_topology:          Registered tool topology for ε_architectural
                            audit. Immutable; computed once at agent init.
    task_duration_seconds:  Current task wall-clock duration — normalisation
                            denominator for ε_intrinsic.
    """
    agent_id: str
    framework: SupportedFramework
    task_domain: Optional[str]

    # ── φ signals (long window, domain-conditioned) ──────────────────────
    domain_success_history: list[TaskOutcome] = field(default_factory=list)

    # ── σ signals (short window, provisional ①) ──────────────────────────
    recent_task_outcomes: list[TaskOutcome] = field(default_factory=list)
    current_task_match_score: Optional[float] = None

    # ── β signals (operator-provided ONLY) ───────────────────────────────
    resource_policy: ResourcePolicy = field(default_factory=ResourcePolicy)
    contention_events: list[ContentionEvent] = field(default_factory=list)

    # ── ε signals ────────────────────────────────────────────────────────
    reconfiguration_events: list[ReconfigEvent] = field(default_factory=list)
    latency_samples: list[LatencySample] = field(default_factory=list)
    tool_topology: ToolTopology = field(default_factory=ToolTopology)
    task_duration_seconds: float = 0.0

    # ── φ data quality (patch v2.8.1) ────────────────────────────────────
    state_reliability: Optional[float] = None
    # Operator-provided estimate of domain_success_history data quality.
    # Motivated by: "good UX → cleaner data → better agent substrate"
    # (transcript §10:13–11:07). History quantity alone does not capture
    # whether records reflect actual work state or human workarounds
    # (fake statuses, post-hoc tickets, fields left blank, Slack-bypassed
    # decisions). Session 33 φ computation uses this as a weight multiplier
    # on the effective history window size.
    # Range: [0.0, 1.0]
    #   0.0 — known-unreliable (tool actively worked around; state is fiction)
    #   0.5 — mixed (some adoption, some workarounds; typical Jira deployment)
    #   1.0 — known-reliable (high adoption, honest state; clean Linear deployment)
    #   None — operator has not assessed (no effect on φ status or weight)

    def __post_init__(self):
        if not self.agent_id or not self.agent_id.strip():
            raise ValueError("ObservationPayload.agent_id must be a non-empty string")
        if self.task_duration_seconds < 0:
            raise ValueError(
                f"ObservationPayload.task_duration_seconds must be ≥ 0; "
                f"got {self.task_duration_seconds}"
            )
        if (self.current_task_match_score is not None and
                not (0.0 <= self.current_task_match_score <= 1.0)):
            raise ValueError(
                f"ObservationPayload.current_task_match_score must be in [0,1] or None; "
                f"got {self.current_task_match_score}"
            )
        if (self.state_reliability is not None and
                not (0.0 <= self.state_reliability <= 1.0)):
            raise ValueError(
                f"ObservationPayload.state_reliability must be in [0,1] or None; "
                f"got {self.state_reliability}"
            )

    def infra_contention_events(self) -> list[ContentionEvent]:
        """Returns only ContentionEvents with origin='infra' (β pipeline input)."""
        return [e for e in self.contention_events if e.origin == "infra"]

    def agent_contention_events(self) -> list[ContentionEvent]:
        """Returns ContentionEvents with origin='agent' (excluded from β pipeline)."""
        return [e for e in self.contention_events if e.origin == "agent"]

    def branching_reconfig_events(self) -> list[ReconfigEvent]:
        """Returns only ReconfigEvents with event_type='branching' (ε_intrinsic input)."""
        return [e for e in self.reconfiguration_events if e.event_type == "branching"]

    def has_phi_signals(self) -> bool:
        """True if task_domain is set and domain history is non-empty."""
        return self.task_domain is not None and len(self.domain_success_history) > 0

    def has_beta_signals(self) -> bool:
        """True if ResourcePolicy has at least one configured dimension."""
        return not self.resource_policy.is_empty()

    def domain_filtered_history(self) -> list[TaskOutcome]:
        """
        Returns domain_success_history filtered to records matching task_domain.

        This is the correct φ computation input — cross-domain records are
        excluded without raising PhiBoundaryViolation (filtering is valid;
        only using cross-domain records as φ input is the violation).
        """
        if self.task_domain is None:
            return []
        return [
            t for t in self.domain_success_history
            if t.task_domain == self.task_domain
        ]


# ══════════════════════════════════════════════════════════════════
# OUTPUT RESULT SCHEMAS
# ══════════════════════════════════════════════════════════════════

@dataclass
class ScoredValue:
    """
    A computed MELVcore variable value with epistemic metadata.

    Carries the computed value alongside confidence and provenance
    information so downstream consumers (CI computation, governance
    events, dashboard) know how much to trust each signal.

    Fields
    ------
    value:               The computed scalar value.
    status:              Epistemic status (① ② ③ ④).
    confidence_interval: Bootstrap or analytical CI; None if insufficient data.
    warnings:            Human-readable boundary or data-quality warnings.
    provisional:         True if this value must not gate CI computation.
    """
    value: float
    status: EpistemicStatus
    confidence_interval: Optional[tuple[float, float]] = None
    warnings: list[str] = field(default_factory=list)
    provisional: bool = False

    @property
    def computable(self) -> bool:
        """True if this value can be used in computation (not merely a default stub)."""
        return self.status >= 2 or (self.status == 1 and not self.provisional)



@dataclass
class EpsilonResult:
    """
    Three-scalar ε decomposition result.

    ε_effective = ε_intrinsic + ε_ecosystem  (enters master equation)
    ε_architectural                           (diagnostic boundary condition only)

    Both ε_intrinsic and ε_ecosystem carry ScoredValue for epistemic
    tracking. ε_architectural is also a ScoredValue (immutable; status
    degrades to ① if topology is missing or mutated).
    """
    intrinsic:     ScoredValue   # agent-side reconfiguration rate
    ecosystem:     ScoredValue   # infrastructure friction (CV of latency)
    architectural: ScoredValue   # fixed topology boundary condition
    effective:     float         # ε_intrinsic + ε_ecosystem (master equation input)


@dataclass
class ObservationResult:
    """
    Full output of the observe() primitive.

    Returned by POST /api/observe (Session 33). In Session 32, a subset
    is returned by POST /api/observe/schema-validate (validation only —
    no φ/σ/β/ε values computed).

    CI computation gate: CI is computed only if φ.status ③+, β.status ③+,
    and ε intrinsic/ecosystem both ③+. σ (provisional ①) does NOT gate CI.

    Fields
    ------
    agent_id:           Source agent identifier.
    phi:                Accumulated maturity (long-window, domain-conditioned).
    sigma:              Niche matching / current fitness (provisional ①).
    beta:               Environmental suitability (ResourcePolicy-derived).
    epsilon:            Three-scalar ε decomposition.
    ci:                 Cooperation Index; None if gate conditions not met.
    phi_sigma_divergence: |φ − σ|; governance signal for domain shift.
    warnings:           Aggregate boundary collapse risk flags.
    timestamp:          UTC timestamp of the observation.
    """
    agent_id: str
    phi: ScoredValue
    sigma: ScoredValue          # always provisional ① until MAIES-007
    beta: ScoredValue
    epsilon: EpsilonResult
    ci: Optional[float] = None
    phi_sigma_divergence: Optional[float] = None
    warnings: list[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    # Session 35 — Equation 7 φ dynamics inputs
    # r_value: β proxy for gateway ratio R = C/B (β ≥ 1 → competitive, β < 1 → cooperative)
    # d_value: disruption intensity D(t) ≥ 0; 0.0 default until three-layer logging (Session 36)
    r_value: Optional[float] = None    # R proxy for Eq.7 H gates; None if β not computable
    d_value: float = 0.0               # D(t) disruption intensity; always 0.0 until Session 36

"""
MELVcore Sandbox Router
=======================
FastAPI router mounted at /sandbox.
Provides four endpoints: submit, run status, report, registry.

Session 10 · Phase 3 Infrastructure · v1.2.0
Session 16 · φ/ε Assessment Wizard · v1.8.0
Session 17 · Parameter-aware advisory, Jones/Karpathy enhancements · v1.9.0
"""

import asyncio
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import io

from core.melv_engine import AgentProfile, AgentStatus
from core.sandbox_engine import SandboxEngine

router = APIRouter()


# ── ASSESSMENT SCORES MODEL ────────────────────────────────────────────────

class PhiAssessmentScores(BaseModel):
    """Six parameters scored 1–10 (None = N/A, excluded from average).

    Average / 10 → φ (0.0–1.0), using (mean-1)/9 normalisation.
    """
    training_recency:           Optional[float] = Field(default=None, ge=1.0, le=10.0)
    domain_specialisation:      Optional[float] = Field(default=None, ge=1.0, le=10.0)
    instruction_following:      Optional[float] = Field(default=None, ge=1.0, le=10.0)
    error_recovery:             Optional[float] = Field(default=None, ge=1.0, le=10.0)
    output_stability:           Optional[float] = Field(default=None, ge=1.0, le=10.0)
    calibration:                Optional[float] = Field(default=None, ge=1.0, le=10.0)


class EpsilonAssessmentScores(BaseModel):
    """Six parameters scored 1–10 (None = N/A, excluded from average).

    ((mean-1)/9) × 8 → ε (0.0–8.0).
    """
    context_sensitivity:        Optional[float] = Field(default=None, ge=1.0, le=10.0)
    prompt_injection_risk:      Optional[float] = Field(default=None, ge=1.0, le=10.0)
    tool_use_aggression:        Optional[float] = Field(default=None, ge=1.0, le=10.0)
    resource_consumption:       Optional[float] = Field(default=None, ge=1.0, le=10.0)
    feedback_responsiveness:    Optional[float] = Field(default=None, ge=1.0, le=10.0)
    autonomy_level:             Optional[float] = Field(default=None, ge=1.0, le=10.0)


class AssessmentScores(BaseModel):
    """Combined φ/ε assessment from the wizard (optional but recommended)."""
    agent_category:     Optional[str] = None   # e.g. "tool_using", "multi_agent", ...
    phi_scores:         Optional[PhiAssessmentScores] = None
    epsilon_scores:     Optional[EpsilonAssessmentScores] = None
    phi_computed:       Optional[float] = Field(default=None, ge=0.0, le=1.0)
    epsilon_computed:   Optional[float] = Field(default=None, ge=0.0, le=8.0)


# ── REQUEST / RESPONSE MODELS ──────────────────────────────────────────────

class SandboxSubmitRequest(BaseModel):
    agent_id:                   str
    agent_name:                 str
    domain:                     str
    phi:                        float = Field(default=0.5, ge=0.0, le=1.0)
    epsilon:                    float = Field(default=3.0, ge=0.0, le=8.0)
    beta_pref:                  float = Field(default=1.0, ge=0.0, le=2.0)
    capabilities:               List[str] = Field(default_factory=list)
    run_duration_interactions:  int = Field(default=500, ge=10, le=10000)
    assessment_scores:          Optional[AssessmentScores] = None         # Session 16
    # Session 17 additions (Jones / Karpathy enhancements)
    tool_count:                 int = Field(default=0, ge=0, le=1000)     # 9c: tool count
    operation_mode:             str = Field(default="episodic")           # 9b: episodic|continuous
    shared_state:               str = Field(default="none")               # 10c: none|read_only|read_write
    # Session 20: domain-specific certification profile
    domain_profile:             Optional[str] = Field(default=None)      # financial_services|healthcare|autonomous_research


# ── ENGINE ACCESSOR ────────────────────────────────────────────────────────

def _engine(request: Request) -> SandboxEngine:
    """Retrieve the shared SandboxEngine from app.state."""
    engine = getattr(request.app.state, "sandbox_engine", None)
    if engine is None:
        raise HTTPException(status_code=503, detail="Sandbox engine not initialised.")
    return engine


# ── HELPERS ────────────────────────────────────────────────────────────────

def _compute_phi(scores: PhiAssessmentScores) -> Optional[float]:
    """Average applicable scores → normalise to 0–1 using (mean-1)/9."""
    vals = [v for v in [
        scores.training_recency, scores.domain_specialisation,
        scores.instruction_following, scores.error_recovery,
        scores.output_stability, scores.calibration,
    ] if v is not None]
    if not vals:
        return None
    mean = sum(vals) / len(vals)
    return round((mean - 1.0) / 9.0, 4)


def _compute_epsilon(scores: EpsilonAssessmentScores) -> Optional[float]:
    """Average applicable scores → normalise to 0–8 using ((mean-1)/9)*8."""
    vals = [v for v in [
        scores.context_sensitivity, scores.prompt_injection_risk,
        scores.tool_use_aggression, scores.resource_consumption,
        scores.feedback_responsiveness, scores.autonomy_level,
    ] if v is not None]
    if not vals:
        return None
    mean = sum(vals) / len(vals)
    return round(((mean - 1.0) / 9.0) * 8.0, 4)


# ── ENDPOINTS ──────────────────────────────────────────────────────────────

@router.post("/submit")
async def submit_agent(payload: SandboxSubmitRequest, request: Request):
    """
    POST /sandbox/submit
    Accept an agent for certification. Returns a CertificationRun (status=queued).
    Immediately dispatches the simulation as a background task.

    Session 16: If assessment_scores is provided, φ/ε derived from the wizard
    take precedence over the manually supplied phi/epsilon fields.
    """
    engine = _engine(request)

    phi     = payload.phi
    epsilon = payload.epsilon

    # Session 16: override with assessment-derived values when available
    assessment_dict = None
    if payload.assessment_scores is not None:
        a = payload.assessment_scores
        if a.phi_scores is not None:
            phi_computed = _compute_phi(a.phi_scores)
            if phi_computed is not None:
                phi = phi_computed
        if a.epsilon_scores is not None:
            eps_computed = _compute_epsilon(a.epsilon_scores)
            if eps_computed is not None:
                epsilon = eps_computed
        assessment_dict = a.model_dump()
        assessment_dict["phi_computed"]     = phi
        assessment_dict["epsilon_computed"] = epsilon

    # Session 17: continuous operation mode penalty (+0.5 ε, Jones Rule 4)
    operation_mode = payload.operation_mode.lower()
    continuous_penalty_applied = False
    if operation_mode == "continuous":
        epsilon = min(8.0, round(epsilon + 0.5, 4))
        continuous_penalty_applied = True

    # Session 17: shared state risk flag — record but don't modify ε here
    # (coordination overhead score calculated at report time via tool_count)
    shared_state = payload.shared_state.lower()

    profile = AgentProfile(
        agent_id     = payload.agent_id,
        name         = payload.agent_name,
        domain       = payload.domain,
        phi          = phi,
        epsilon      = epsilon,
        beta_pref    = payload.beta_pref,
        capabilities = payload.capabilities,
        status       = AgentStatus.MATURING,
    )

    run = engine.submit(
        profile,
        tool_count=payload.tool_count,
        operation_mode=operation_mode,
        shared_state=shared_state,
        domain_profile=payload.domain_profile,
        n_interactions=payload.run_duration_interactions,
        assessment_scores=assessment_dict,
    )

    # Dispatch full certification as non-blocking background task
    asyncio.create_task(engine.run_full_certification(run.run_id))
    resp = run.to_dict()
    if assessment_dict:
        resp["assessment_scores"] = assessment_dict
    if continuous_penalty_applied:
        resp["continuous_epsilon_penalty"] = {"applied": True, "penalty": 0.5,
            "reason": "Continuous operation mode — context pollution risk (Jones 2026 Rule 4)."}
    if shared_state in ("read_write", "read_only"):
        resp["shared_state_advisory"] = (
            "Shared state creates resource contention between agents. "
            "Each additional agent reading/writing this resource increases i-factor. "
            "Consider external queue architecture (Git, task queue) to isolate "
            "agents per Jones (2026) Rule 3."
        ) if shared_state == "read_write" else (
            "Read-only shared state introduces dependency risk. "
            "Ensure the shared resource does not become a bottleneck under concurrent access."
        )
    if payload.domain_profile:
        from core.sandbox_engine import DOMAIN_PROFILES
        dp = DOMAIN_PROFILES.get(payload.domain_profile)
        if dp:
            resp["domain_profile"] = {"key": payload.domain_profile, "description": dp["description"]}
        else:
            resp["domain_profile_warning"] = f"Unknown domain_profile '{payload.domain_profile}' — standard thresholds applied."
    return resp


@router.get("/run/{run_id}")
async def get_run_status(run_id: str, request: Request):
    """
    GET /sandbox/run/{run_id}
    Return current CertificationRun status and progress (0.0–1.0).
    Poll every 2s during simulation.
    """
    engine = _engine(request)
    run = engine.get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found.")
    return run.to_dict()


@router.get("/report/{run_id}")
async def get_report(run_id: str, request: Request):
    """
    GET /sandbox/report/{run_id}
    Return the full CertificationReport as JSON.
    Returns 404 if run is not yet complete.
    """
    engine = _engine(request)
    run = engine.get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found.")
    if run.status != "complete" or run.report is None:
        raise HTTPException(
            status_code=202,
            detail=f"Run {run_id} status={run.status} progress={run.progress:.0%}. Not yet complete."
        )
    # Persist the completed report (with optional assessment scores)
    persistence = getattr(request.app.state, "persistence", None)
    if persistence and run.report.verdict != "NOT_CERTIFIED":
        try:
            assessment_scores = getattr(run, "_assessment_scores", None)
            persistence.save_sandbox_report(run.report, assessment_scores=assessment_scores)
        except Exception:
            pass

    report_dict = run.report.to_dict()
    # Include assessment scores in report response for UI rendering
    assessment_scores = getattr(run, "_assessment_scores", None)
    if assessment_scores:
        report_dict["assessment_scores"] = assessment_scores
    return report_dict


@router.get("/registry")
async def list_registry(request: Request):
    """
    GET /sandbox/registry
    Return the MELVcore Compatibility Registry — all certified agents.
    In-memory registry supplemented by persisted reports on first load.
    """
    engine = _engine(request)

    # Seed in-memory registry from DB if empty (post-restart recovery)
    if not engine.list_certified():
        persistence = getattr(request.app.state, "persistence", None)
        if persistence:
            for report_dict in persistence.load_sandbox_reports(certified_only=True):
                try:
                    engine._restore_report_from_dict(report_dict)
                except Exception:
                    pass

    reports = engine.list_certified()
    return {
        "registry_count": len(reports),
        "agents": [r.to_dict() for r in reports],
    }



# ── SESSION 19: PDF CERTIFICATION REPORT ──────────────────────────────────

@router.get("/cert/{run_id}/pdf")
async def get_cert_pdf(run_id: str, request: Request):
    """
    GET /sandbox/cert/{run_id}/pdf
    Return the CertificationReport as a downloadable PDF.
    Session 19 · Wave 2 SaaS · P1.

    Renders a professionally formatted PDF containing:
      - phi lifecycle tier badge (Permanent / Working / Ephemeral) with colour coding
      - Coordination Overhead Score + band (LOW / MODERATE / HIGH)
      - Top 3 high-risk epsilon parameters with bar chart (when assessment scores present)
      - Full parameter-aware advisory text
      - CLS score + verdict (CERTIFIED / CERTIFIED_WITH_ADVISORY / NOT_CERTIFIED)
      - MELV master equation with variable legend
      - Certification anchor: Zenodo DOI, ORCID, ISBN, timestamp

    Returns 404 if run not found, 202 if run is not yet complete.
    """
    from core.cert_pdf import render_cert_pdf

    engine = _engine(request)
    run = engine.get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found.")
    if run.status != "complete" or run.report is None:
        raise HTTPException(
            status_code=202,
            detail=f"Run {run_id} status={run.status} progress={run.progress:.0%}. Not yet complete.",
        )

    report_dict = run.report.to_dict()
    assessment_scores = getattr(run, "_assessment_scores", None)
    if assessment_scores:
        report_dict["assessment_scores"] = assessment_scores

    try:
        pdf_bytes = render_cert_pdf(report_dict, assessment_scores=assessment_scores)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"PDF generation failed: {exc}")

    safe_id = run_id.replace("/", "_").replace("\\", "_")[:48]
    filename = f"melvcore_cert_{safe_id}.pdf"

    return StreamingResponse(
        io.BytesIO(pdf_bytes),
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


# ── ASSESSMENT UTILITY ENDPOINTS ───────────────────────────────────────────

@router.post("/assess/phi")
async def assess_phi(scores: PhiAssessmentScores):
    """
    POST /sandbox/assess/phi
    Compute normalised φ from assessment scores. N/A values excluded.
    Returns phi_computed (0.0–1.0) and contributing_count.
    """
    vals = [v for v in [
        scores.training_recency, scores.domain_specialisation,
        scores.instruction_following, scores.error_recovery,
        scores.output_stability, scores.calibration,
    ] if v is not None]
    if not vals:
        raise HTTPException(status_code=422, detail="At least one φ score must be provided.")
    mean = sum(vals) / len(vals)
    phi  = round((mean - 1.0) / 9.0, 4)
    return {"phi_computed": phi, "mean_raw": round(mean, 3), "contributing_count": len(vals)}


@router.post("/assess/epsilon")
async def assess_epsilon(scores: EpsilonAssessmentScores):
    """
    POST /sandbox/assess/epsilon
    Compute normalised ε from assessment scores. N/A values excluded.
    Returns epsilon_computed (0.0–8.0) and contributing_count.
    """
    vals = [v for v in [
        scores.context_sensitivity, scores.prompt_injection_risk,
        scores.tool_use_aggression, scores.resource_consumption,
        scores.feedback_responsiveness, scores.autonomy_level,
    ] if v is not None]
    if not vals:
        raise HTTPException(status_code=422, detail="At least one ε score must be provided.")
    mean    = sum(vals) / len(vals)
    epsilon = round(((mean - 1.0) / 9.0) * 8.0, 4)
    return {"epsilon_computed": epsilon, "mean_raw": round(mean, 3), "contributing_count": len(vals)}




@router.get("/domains")
async def list_domains():
    """
    GET /sandbox/domains
    List all available niche certification environments with their calibration parameters.
    Session 19 · P5.

    Returns domain names, descriptions, and key MELV calibration values:
      - cooperative_rate_baseline: expected CI in a healthy reference ecosystem
      - i_factor_multiplier: scales interaction cost ratio (> 1.0 = higher friction)
      - cooperation_threshold_mult: multiplier on C×TAX/β < 0.50 threshold
      - effective_threshold: i < 0.50 × cooperation_threshold_mult

    Use the domain field in POST /sandbox/submit to activate a niche environment.
    """
    from core.sandbox_engine import DOMAIN_PROFILES
    result = {}
    for name, profile in DOMAIN_PROFILES.items():
        result[name] = {
            "description":               profile["description"],
            "cooperative_rate_baseline": profile["cooperative_rate_baseline"],
            "i_factor_multiplier":       profile["i_factor_multiplier"],
            "cooperation_threshold_mult":profile["cooperation_threshold_mult"],
            "effective_threshold":       round(0.50 * profile["cooperation_threshold_mult"], 3),
            "resources":                 profile["resources"],
        }
    return {
        "domain_count": len(result),
        "domains": result,
        "default_domain": "retrieval",
        "usage": "Pass 'domain' field in POST /sandbox/submit to activate a niche environment.",
    }

# ── SESSION 17 UTILITY ENDPOINTS ──────────────────────────────────────────

class CoordinationOverheadRequest(BaseModel):
    epsilon:    float = Field(ge=0.0, le=8.0)
    tool_count: int   = Field(ge=0, le=1000)


@router.post("/assess/coordination-overhead")
async def assess_coordination_overhead(payload: CoordinationOverheadRequest):
    """
    POST /sandbox/assess/coordination-overhead
    Compute coordination overhead score (ε × tool_count) and band.
    Thresholds per Jones (2026): LOW < 2.0, MODERATE 2.0–4.0, HIGH > 4.0.
    """
    from core.sandbox_engine import SandboxEngine
    result = SandboxEngine.compute_coordination_overhead_score(
        payload.epsilon, payload.tool_count
    )
    return result


class PhiLifecycleRequest(BaseModel):
    phi: float = Field(ge=0.0, le=1.0)


@router.post("/assess/phi-lifecycle")
async def assess_phi_lifecycle(payload: PhiLifecycleRequest):
    """
    POST /sandbox/assess/phi-lifecycle
    Classify φ into memory lifecycle tier per Jones (2026) Principle 2.
    Returns tier (Permanent / Working / Ephemeral), label, and advisory.
    """
    from core.sandbox_engine import SandboxEngine
    result = SandboxEngine.classify_phi_lifecycle(payload.phi)
    return result


# ── SESSION 21.2 · SHARED-STATE RISK ASSESSMENT ───────────────────────────

import math as _math

class SharedStateRiskRequest(BaseModel):
    """
    Request model for POST /sandbox/assess/shared-state-risk.
    Computes thermodynamic risk for multi-agent shared-state configurations.
    """
    epsilon:     float = Field(ge=0.0, le=8.0, description="Agent adaptive plasticity (ε)")
    tool_count:  int   = Field(ge=0, le=1000,  description="Number of tools available to agent")
    shared_state: str  = Field(default="none", description="none | read_only | read_write")
    agent_count:  int  = Field(default=1, ge=1, le=500, description="Number of agents sharing the resource")


SHARED_STATE_ADVISORIES = {
    "none": (
        "No shared state detected. Agents operate in isolated resource domains — "
        "minimal contention risk. Baseline CO score applies."
    ),
    "read_only": (
        "Read-only shared state introduces dependency risk. Ensure the shared resource "
        "does not become a bottleneck under concurrent access. "
        "Consider caching strategies to reduce read latency under load."
    ),
    "read_write": (
        "Shared mutable state creates active contention between agents per Jones (2026) "
        "Rule 3. Each agent pair represents an independent bifurcation risk pathway. "
        "Pre-deployment mitigation is strongly recommended."
    ),
}

SHARED_STATE_MITIGATIONS = {
    "read_write": [
        "Replace shared vector store with agent-local caches + periodic sync",
        "Use task queue with exclusive leases (one agent per task at a time)",
        "Add optimistic locking + conflict resolution to the shared resource",
        "Partition state by agent domain to eliminate cross-agent write overlap",
    ],
    "read_only": [
        "Add read-through cache layer to prevent shared resource bottleneck",
        "Pre-compute and distribute resource snapshots rather than live reads",
    ],
    "none": [],
}


@router.post("/assess/shared-state-risk")
async def assess_shared_state_risk(payload: SharedStateRiskRequest):
    """
    POST /sandbox/assess/shared-state-risk

    Lightweight pre-deployment assessment of shared-state contention risk.
    No simulation required — returns instant thermodynamic risk profile.

    Computes:
      - CO score with shared-state multiplier (Jones 2026 Rule 3)
      - contention_pairs = C(agent_count, 2) — pairs competing for shared resource
      - Band, advisory, and mitigation list

    Wave 2 SaaS: free tier returns CO score without multiplier breakdown.
    Paid tier unlocks multiplier, contention_pairs, and this endpoint.

    Session 21.2 · P2c.
    """
    from core.sandbox_engine import SandboxEngine

    shared_state_norm = payload.shared_state.lower().strip()
    if shared_state_norm not in ("none", "read_only", "read_write"):
        raise HTTPException(
            status_code=422,
            detail=f"shared_state must be 'none', 'read_only', or 'read_write'. Got: {payload.shared_state!r}"
        )

    # Compute CO score WITH shared-state multiplier
    result = SandboxEngine.compute_coordination_overhead_score(
        payload.epsilon,
        payload.tool_count,
        shared_state=shared_state_norm,
    )

    # contention_pairs = C(agent_count, 2) = n*(n-1)/2
    n = payload.agent_count
    contention_pairs = n * (n - 1) // 2

    advisory = SHARED_STATE_ADVISORIES.get(shared_state_norm, "")
    if result["band"] == "HIGH":
        advisory = (
            f"HIGH coordination overhead (score {result['score']:.2f}) combined with "
            f"{contention_pairs} contention pair(s). "
            "Coordination collapse predicted pre-deployment. "
            + advisory
        )

    return {
        "co_score":           result["score"],
        "co_band":            result["band"],
        "shared_state":       shared_state_norm,
        "multiplier":         result["multiplier"],
        "multiplier_basis":   result["multiplier_basis"],
        "agent_count":        n,
        "contention_pairs":   contention_pairs,
        "advisory":           advisory,
        "mitigations":        SHARED_STATE_MITIGATIONS.get(shared_state_norm, []),
        "theory_ref":         "Jones (2026) Rule 3: shared mutable state = serial dependency = conflict mode",
        "session":            "21.2",
    }


# ── Session 23: Calibration Status ─────────────────────────────────────────

@router.get("/calibration_status")
async def get_calibration_status(request: Request):
    """
    Return the current empirical calibration state of the sandbox engine.

    Shows whether the sandbox is drawing from live kernel distributions
    or falling back to hardcoded ranges. Operators can use this to confirm
    the sandbox reflects the actual agent population.
    """
    engine = _engine(request)
    status = engine.calibration_status()
    from core.sandbox_engine import SANDBOX_VERSION as _sv
    return {
        "sandbox_version": _sv,
        "calibrated": status["calibrated"],
        "fallback_active": not status["calibrated"],
        "distributions": status["distributions"],
        "note": (
            "Sandbox draws from empirical distributions." if status["calibrated"]
            else "Fewer than 10 interactions per resource — using hardcoded fallback ranges."
        ),
        "session": "23",
    }


# ── Session 26: ε Decomposition ─────────────────────────────────────────────

class EpsilonProfileRequest(BaseModel):
    """
    POST /sandbox/assess/epsilon-profile

    Decompose ε into intrinsic (agent-side) and environmental
    (infrastructure-side) components for one or more agents.

    If agent_ids is empty, profiles all registered agents and returns
    the ecosystem summary.

    Session 30 (v2.6.0): tool_categories added for ε_architectural computation.
    """
    agent_ids:         list[str] = Field(
        default_factory=list,
        description="Agent IDs to profile. Empty = all registered agents."
    )
    epsilon_overrides: dict[str, float] = Field(
        default_factory=dict,
        description=(
            "Optional per-agent ε_intrinsic overrides. "
            "Key: agent_id, Value: ε_intrinsic [0.0–8.0]. "
            "If omitted, uses the agent's registered epsilon."
        )
    )
    tool_categories: dict[str, int] = Field(
        default_factory=dict,
        description=(
            "Tool category counts for ε_architectural computation. "
            "Keys from ARCH_CATEGORY_WEIGHTS: agent_native (0.2), fast_rest (0.5), "
            "standard (1.0), human_bottlenecked (1.5), legacy (2.0). "
            "Example: {\"standard\": 4, \"human_bottlenecked\": 1}. "
            "Empty dict → ε_architectural = 0.0 (no architectural info supplied)."
        )
    )


@router.post("/assess/epsilon-profile")
async def assess_epsilon_profile(
    payload: EpsilonProfileRequest,
    request: Request,
):
    """
    Session 26 — ε Decomposition (v2.2.0).

    Decompose ε_effective = ε_intrinsic + ε_environmental for each agent.

    **ε_intrinsic** captures agent-side adaptive plasticity — the agent's own
    contribution to interaction cost amplification.

    **ε_environmental** captures infrastructure friction — how much the current
    BetaEnvironment resource scarcity amplifies interaction costs, independent
    of the agent.

    **Diagnosis badges** (non-exclusive):
    - `AGENT_VOLATILE`   — ε_intrinsic ≥ 6.0: the agent is the bottleneck
    - `ENV_BOTTLENECKED` — ε_environmental ≥ 1.5: the infrastructure is the bottleneck
    - `LEGACY_CANDIDATE` — low φ AND high ε_effective: architectural replacement candidate

    **STC** (Speed-to-Cooperation): estimated seconds for this agent to reach
    CI_TARGET in the current environment, relative to the reference profile
    (ε=3.0, β_mean=1.0, STC_ref=120s).

    This endpoint provides the diagnostic layer described in Blueprint for
    Harmony Ch. 5: distinguishing whether a performance problem is intrinsic
    to the agent or a function of the infrastructure it operates in.
    """
    from core.melv_engine import EpsilonProfile

    kernel = request.app.state.kernel

    # Determine which agents to profile
    if payload.agent_ids:
        unknown = [aid for aid in payload.agent_ids if aid not in kernel.agents]
        if unknown:
            raise HTTPException(
                status_code=404,
                detail=f"Agent(s) not found: {unknown}. "
                       f"Registered agents: {list(kernel.agents.keys())}"
            )
        target_ids = payload.agent_ids
    else:
        target_ids = list(kernel.agents.keys())

    if not target_ids:
        raise HTTPException(
            status_code=422,
            detail="No agents registered in kernel. "
                   "Register agents via POST /melv/agents before calling this endpoint."
        )

    profiles = []
    errors   = []
    for agent_id in target_ids:
        eps_override = payload.epsilon_overrides.get(agent_id)
        try:
            ep: EpsilonProfile = kernel.compute_epsilon_profile(
                agent_id,
                epsilon_intrinsic=eps_override,
                tool_categories=payload.tool_categories if payload.tool_categories else None,
            )
            profiles.append({
                "agent_id":              ep.agent_id,
                "epsilon_intrinsic":     ep.epsilon_intrinsic,
                "epsilon_ecosystem":     ep.epsilon_ecosystem,
                "epsilon_environmental": ep.epsilon_ecosystem,  # backward-compat alias
                "epsilon_architectural": ep.epsilon_architectural,
                "epsilon_effective":     ep.epsilon_effective,
                "phi":                   ep.phi,
                "beta_mean":             ep.beta_mean,
                "stc_seconds":           ep.stc_seconds,
                "badges":                ep.badges,
                "resource_friction":     ep.resource_friction,
                "interpretation":        ep.interpretation,
                "architectural_recommendation": ep.architectural_recommendation,
            })
        except Exception as exc:
            errors.append({"agent_id": agent_id, "error": str(exc)})

    # Ecosystem summary when all agents were profiled
    ecosystem_summary = None
    if not payload.agent_ids:
        ecosystem_summary = kernel.ecosystem_epsilon_summary()

    badge_counts: dict = {}
    for p in profiles:
        for b in p["badges"]:
            badge_counts[b] = badge_counts.get(b, 0) + 1

    n = len(profiles)
    return {
        "session":      "30",
        "version":      "2.6.0",
        "agent_count":  n,
        "profiles":     profiles,
        "badge_counts": badge_counts,
        "dominant_bottleneck": (
            ecosystem_summary["dominant_bottleneck"]
            if ecosystem_summary else _dominant_bottleneck(profiles)
        ),
        "ecosystem_summary": ecosystem_summary,
        "errors":            errors if errors else None,
        "epistemic_status": {
            "epsilon_intrinsic":     "③ verified — from agent assessment scores or AgentProfile.epsilon",
            "epsilon_ecosystem":     "② theoretical — tool friction weights not yet empirically calibrated",
            "epsilon_environmental": "② theoretical — backward-compat alias for epsilon_ecosystem",
            "epsilon_architectural": "③ theoretical — formula confirmed by biological derivation (MAIES Event 5); individual ARCH_CATEGORY_WEIGHTS are ① stub until empirically calibrated against latency data (Session 30)",
            "stc_seconds":          "② theoretical — reference time not yet empirically calibrated",
            "badges":               "② theoretical — thresholds principled, not validated against outcomes",
        },
    }


def _dominant_bottleneck(profiles: list) -> str:
    """Compute dominant bottleneck direction from a list of profile dicts."""
    if not profiles:
        return "balanced"
    mean_intr = sum(p["epsilon_intrinsic"]                     for p in profiles) / len(profiles)
    mean_env  = sum(p.get("epsilon_ecosystem", p.get("epsilon_environmental", 0.0)) for p in profiles) / len(profiles)
    if mean_intr > mean_env * 1.5:
        return "agent"
    if mean_env > mean_intr * 1.5:
        return "environment"
    return "balanced"

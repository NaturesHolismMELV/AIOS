"""
MELVcore Gateway API
====================
External agent registration and interaction reporting endpoint.
Any developer's existing agent becomes MELVcore-governed via this JSON API.

Built on the Modified Energetic Lotka-Volterra (MELV) framework.
Blueprint for Harmony — L.W. Evans (Ecotao Enterprises, Cape Town)

CANONICAL VARIABLE DEFINITIONS (do not deviate):
─────────────────────────────────────────────────
  φ (phi)   — INTERNAL to the agent. Evolutionary maturity / domain fitness.
              Increases as the agent matures, specialises, or adapts.
              The giraffe's long neck raises its φ. NOT set by environment.

  β (beta)  — EXTERNAL. Environmental suitability. Set by environment
              configuration. The acacia crown niche has high β because it
              is rich and uncontested. NOT something an agent can set.

  i         — Interaction Cost Ratio. i = C_AB / B_AB. Computed from the
              cost and benefit of an interaction between two agents.
              NOT a property of a single agent.

  CI        — Cooperation Index. System-level measure of cooperative
              equilibrium. Derived from all pairwise i-factors.

VALIDATION RULE (enforced at every POST /melv/interact):
  beta must NEVER appear in the agent interaction payload.
  Agents report phi only. The kernel reads beta from environment config.
  Violation → HTTP 422 with clear error message.
"""

import time
import random
import uuid
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, model_validator, Field

from core.melv_engine import MELVKernel, AgentProfile, AgentStatus, KernelAction
from core.cost_calculator import get_calculator, CostCalculator
from core.nudge_engine import NudgeEngine

router = APIRouter()

# ── PYDANTIC MODELS ────────────────────────────────────────────────────────

class GatewayRegistration(BaseModel):
    """
    Register an external agent with the MELVcore kernel.

    Decorator equivalent:
        @melv.register_agent(domain="research", phi=0.82)
        class ResearchAgent: ...
    """
    name:         str           = Field(..., description="Human-readable agent name")
    domain:       str           = Field(..., description="Agent's specialisation domain")
    phi:          float         = Field(0.5,  ge=0.0, le=1.0,
                                        description="Initial maturity/fitness φ ∈ [0,1]")
    epsilon:      float         = Field(3.0,  ge=0.0, le=8.0,
                                        description="Adaptive plasticity ε ∈ [0,8]")
    capabilities: list[str]     = Field(default_factory=list)

    # ENFORCEMENT: beta must NOT appear in registration payload
    @model_validator(mode='before')
    @classmethod
    def reject_beta_in_payload(cls, values):
        if 'beta' in values:
            raise ValueError(
                "MELV VIOLATION: 'beta' (environmental suitability) must NOT be "
                "set by the agent. β is owned by the kernel/environment config. "
                "Remove 'beta' from your payload. Agents report φ (phi) only."
            )
        if 'beta_pref' in values:
            raise ValueError(
                "MELV VIOLATION: 'beta_pref' must NOT be set via the Gateway API. "
                "Environmental suitability β is managed by the kernel. "
                "Agents report φ (phi) only."
            )
        return values


class InteractionReport(BaseModel):
    """
    Report an interaction between two agents to the MELVcore kernel.
    The kernel computes i = C/B, modulates by β (from environment config),
    and returns a governance decision.

    CRITICAL: 'beta' is NOT accepted here. The kernel reads β from its
    own environment configuration. Agents report cost, benefit, and phi only.
    """
    agent_a:       str   = Field(..., description="Initiating agent name or ID")
    agent_b:       str   = Field(..., description="Receiving agent name or ID")
    cost:          float = Field(..., gt=0,  description="Interaction cost C_AB > 0")
    benefit:       float = Field(..., gt=0,  description="Interaction benefit B_AB > 0")
    phi_a:         Optional[float] = Field(None, ge=0.0, le=1.0,
                                           description="Agent A current maturity φ_a")
    phi_b:         Optional[float] = Field(None, ge=0.0, le=1.0,
                                           description="Agent B current maturity φ_b")
    resource_type: str   = Field("compute",
                                  description="Resource contested: compute | api_quota | "
                                              "vector_db | storage | token_budget")

    @model_validator(mode='before')
    @classmethod
    def reject_beta_in_interaction(cls, values):
        forbidden = {'beta', 'beta_a', 'beta_b', 'beta_pref', 'environmental_suitability'}
        found = forbidden.intersection(values.keys())
        if found:
            raise ValueError(
                f"MELV VIOLATION: Fields {found} must NOT appear in interaction "
                f"reports. Environmental suitability β is managed internally by the "
                f"MELVcore kernel and is NEVER supplied by agents. "
                f"Agents report: agent_a, agent_b, cost, benefit, phi_a, phi_b only."
            )
        return values


class BetaProvision(BaseModel):
    """Human portal: adjust environmental β for a resource type."""
    resource: str
    value:    float = Field(..., ge=0.1, le=3.0)


# ── ACTIONABLE NUDGE BUILDER ──────────────────────────────────────────────

VALID_RESOURCES = {"compute", "api_quota", "vector_db", "storage", "token_budget", "context_window"}

# Module-level NudgeEngine singleton
_nudge_engine = NudgeEngine()

def _build_nudge_v2(
    action: KernelAction,
    beta_i: float,
    resource: str,
    contention_depth: int,
    agent_phi: float = 0.5,
) -> Optional[dict]:
    """
    Build a structured Nudge v2 payload using NudgeEngine.
    Returns a dict (serialisable) or None if no nudge needed.
    Also triggers the oxpecker effect when niche_diverge fires.
    """
    if action == KernelAction.NONE:
        return None

    # Map KernelAction → NudgeEngine action string
    # (NudgeEngine uses contention_depth to select the exact nudge type)
    nudge_resp = _nudge_engine.build_nudge_v2(
        action=action.value,
        beta_i=beta_i,
        resource=resource,
        contention_depth=contention_depth,
        agent_phi=agent_phi,
    )
    return nudge_resp.to_dict()


# ── GATEWAY SINGLETON (shares kernel with server.py) ─────────────────────
# The kernel instance is injected at startup via set_kernel()

_kernel: Optional[MELVKernel] = None

def set_kernel(kernel: MELVKernel):
    """Called by api/server.py at startup to inject the shared kernel."""
    global _kernel
    _kernel = kernel

def get_kernel() -> MELVKernel:
    if _kernel is None:
        raise RuntimeError("MELVcore kernel not initialised. Call set_kernel() first.")
    return _kernel


# ── ENDPOINTS ─────────────────────────────────────────────────────────────

@router.get("/")
async def gateway_info():
    """MELVcore Gateway API — registration and interaction reporting."""
    return {
        "gateway":     "MELVcore",
        "version":     "0.1.0",
        "description": "External agent registration and thermodynamic governance",
        "endpoints": {
            "POST /melv/register":  "Register an external agent",
            "POST /melv/interact":  "Report an agent interaction, receive governance decision",
            "GET  /melv/agents":    "List all Gateway-registered agents",
            "GET  /melv/status":    "Gateway health + cooperation index",
            "POST /melv/beta":      "Human portal — adjust environmental β",
        },
        "note": (
            "beta (β) is NEVER accepted in agent payloads. "
            "Agents report phi (φ) only. The kernel owns β."
        )
    }


@router.post("/register", status_code=201)
async def register_agent(registration: GatewayRegistration):
    """
    Register an external agent with MELVcore.

    Equivalent to the decorator pattern:
        @melv.register_agent(domain="research", phi=0.82)
        class ResearchAgent: ...

    Returns the assigned agent_id for use in /melv/interact calls.
    """
    kernel = get_kernel()

    agent_id = f"gateway_{registration.domain}_{uuid.uuid4().hex[:8]}"
    profile = AgentProfile(
        agent_id=agent_id,
        name=registration.name,
        domain=registration.domain,
        phi=registration.phi,
        epsilon=registration.epsilon,
        capabilities=registration.capabilities,
        status=AgentStatus.ACTIVE if registration.phi >= 0.5 else AgentStatus.MATURING,
    )
    kernel.register_agent(profile)

    return {
        "registered":  True,
        "agent_id":    agent_id,
        "name":        registration.name,
        "domain":      registration.domain,
        "phi":         registration.phi,
        "status":      profile.status.value,
        "message": (
            f"Agent '{registration.name}' registered with MELVcore. "
            f"Use agent_id='{agent_id}' in /melv/interact calls."
        )
    }


@router.post("/interact")
async def report_interaction(report: InteractionReport):
    """
    Report an agent interaction and receive a MELVcore governance decision.

    The kernel computes:
      i = C_AB / B_AB
      βi = β(resource) · i

    Returns:
      status    — "cooperative" | "threshold" | "bifurcation"
      i_factor  — computed interaction cost ratio
      action    — null | route_service | nudge | provision_beta | niche_diverge
      nudge     — structured actionable instruction (when action is not null)

    ENFORCEMENT: 'beta' must NOT appear in the payload.
    β is read from the kernel's environment config, never from the agent.
    """
    kernel = get_kernel()

    # Validate resource type
    if report.resource_type not in VALID_RESOURCES:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid resource_type '{report.resource_type}'. "
                   f"Valid: {sorted(VALID_RESOURCES)}"
        )

    # Update phi if agent self-reports (phi is agent-internal — this is correct MELV)
    if report.phi_a is not None:
        agent_a = kernel.get_agent(report.agent_a)
        if agent_a:
            agent_a.phi = report.phi_a

    if report.phi_b is not None:
        agent_b = kernel.get_agent(report.agent_b)
        if agent_b:
            agent_b.phi = report.phi_b

    # Record the interaction — kernel reads β from its own environment
    record = kernel.record_interaction(
        agent_a=report.agent_a,
        agent_b=report.agent_b,
        cost=report.cost,
        benefit=report.benefit,
        resource_type=report.resource_type,
    )

    i_factor = record.i_factor
    beta_i   = record.beta_i
    itype    = record.interaction_type.value

    # Map interaction type to governance status and action
    if itype == "cooperative":
        status = "cooperative"
        action = KernelAction.NONE
    elif itype == "threshold":
        status = "threshold"
        action = KernelAction.NUDGE
    else:
        status = "bifurcation"
        action = KernelAction.NICHE_DIVERGENCE if beta_i < 1.6 else KernelAction.PROVISION_BETA

    # Session 7: get contention depth for this pair
    contention_depth = kernel.get_contention_depth(report.agent_a, report.agent_b)

    # Get agent φ for nudge calibration (use phi_a as primary)
    agent_phi = report.phi_a if report.phi_a is not None else 0.5

    # Build Nudge v2 payload via NudgeEngine
    nudge = _build_nudge_v2(action, beta_i, report.resource_type, contention_depth, agent_phi)

    # Oxpecker / Channel 2: when niche_diverge fires, apply β lift to adjacent domain
    oxpecker_report = None
    if nudge and nudge.get("nudge_type") == "niche_diverge":
        oxpecker_report = _nudge_engine.apply_oxpecker_effect(
            vacating_agent=report.agent_a,
            resource_type=report.resource_type,
            environment=kernel.beta,
        )

    phi_delta       = nudge.get("phi_delta", 0.0)      if nudge else 0.0
    niche_suggestion = nudge.get("niche_suggestion", "") if nudge else ""

    return {
        "status":            status,
        "i_factor":          round(i_factor, 4),
        "beta_i":            round(beta_i, 4),
        "beta_env":          round(record.beta, 4),
        "action":            action.value if action != KernelAction.NONE else None,
        "nudge":             nudge,
        "contention_depth":  contention_depth,
        "phi_delta":         phi_delta,
        "niche_suggestion":  niche_suggestion,
        "oxpecker_effect":   oxpecker_report,
        "timestamp":         record.timestamp,
        "melv_note": (
            "β was read from kernel environment config. "
            "φ updated if supplied by agents (φ is agent-internal). "
            "β is NEVER set by agents. Nudge v2: contention depth escalation active."
        )
    }


@router.get("/agents")
async def list_gateway_agents():
    """List all agents registered via the Gateway API."""
    kernel = get_kernel()
    gateway_agents = [
        a for a in kernel.get_all_agents()
        if a['agent_id'].startswith('gateway_')
    ]
    return {
        "gateway_agents": gateway_agents,
        "count": len(gateway_agents),
    }


@router.get("/status")
async def gateway_status():
    """Gateway health snapshot with cooperation index."""
    kernel = get_kernel()
    ci = kernel.cooperation_index()
    return {
        "gateway":           "MELVcore",
        "cooperation_index": round(ci, 4),
        "healthy":           ci >= 0.75,
        "total_agents":      len(kernel.agents),
        "total_interactions": len(kernel.interactions),
        "total_events":      len(kernel.events),
        "beta_environment":  kernel.beta.to_dict(),
        "uptime_note":       "β is managed by the kernel. Agents report φ only.",
    }


@router.get("/costs")
async def cost_breakdown():
    """
    CostCalculator state — current weight profiles and recent cost breakdown.

    Returns the per-task-type weight profiles and a breakdown of the most
    recent LLM cost computations recorded by the shared CostCalculator.
    """
    calc = get_calculator()
    return {
        "profiles":       calc.all_profiles(),
        "summary":        calc.summary_by_type(),
        "recent_records": calc.recent_breakdown(n=20),
        "constants": {
            "input_price_per_token":  CostCalculator.INPUT_PRICE_PER_TOKEN,
            "output_price_per_token": CostCalculator.OUTPUT_PRICE_PER_TOKEN,
            "cost_cap":               CostCalculator.COST_CAP,
            "base_formula":           "min(2.0, token_cost*1000*token_w + latency*0.1*latency_w)",
            "note":                   "Base formula locked Session 4. Weights are Session 6 addition.",
        },
    }


@router.post("/beta")
async def provision_beta(provision: BetaProvision):
    """
    Human portal: adjust environmental suitability β for a resource type.
    This is the ONLY correct way to change β — the kernel/operator sets it,
    never the agents.
    """
    kernel = get_kernel()
    if provision.resource not in VALID_RESOURCES:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid resource '{provision.resource}'. Valid: {sorted(VALID_RESOURCES)}"
        )
    kernel.provision_beta(provision.resource, provision.value)
    return {
        "updated":  provision.resource,
        "value":    provision.value,
        "beta":     kernel.beta.to_dict(),
        "note":     "β updated by kernel operator (human portal). Agents cannot set β.",
    }

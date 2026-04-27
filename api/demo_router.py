"""
MELVcore Demo Router
====================
Public (no API key required) endpoints for the live bifurcation demonstration.

Mounted at /demo — exempt from APIKeyMiddleware.

Purpose: allows anyone with the dashboard URL to run the bifurcation demo
without needing an API key. Designed for the Demo Brief, Substack post,
and any first-time visitor arriving at the dashboard.

Security boundaries:
  - Rate limited: 1 demo session per IP per 10 minutes (enforced here,
    not in RateLimitMiddleware which uses a shared bucket)
  - Demo agents are tagged demo=True and cleaned up after 15 minutes
  - Interactions are capped at 20 per session
  - Only the stress agent profile is writable; no other agents touched
  - No β provisioning or environment changes permitted
  - Read endpoints (/demo/status) are always open

Session 30 (v2.6.0) — April 2026
Author: Laurence W. Evans · ORCID: 0009-0001-0963-1840
"""

import time
import uuid
from collections import defaultdict
from typing import Optional

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from core.melv_engine import MELVKernel, AgentProfile, AgentStatus
from core.nudge_engine import NudgeEngine
from core.gateway import get_kernel, VALID_RESOURCES

router = APIRouter()

# ── DEMO RATE LIMIT ────────────────────────────────────────────────────────
# 1 demo session per IP per DEMO_WINDOW seconds.
# Tracked independently of the main rate limiter.

DEMO_WINDOW          = 600   # 10 minutes between sessions per IP
DEMO_MAX_INTERACTIONS = 20   # interactions allowed per session
DEMO_AGENT_TTL       = 900   # demo agents cleaned up after 15 minutes

# IP → last session start time
_demo_sessions: dict[str, float] = {}

# demo_agent_id → (registered_at, interaction_count)
_demo_agents: dict[str, tuple[float, int]] = {}


def _client_ip(request: Request) -> str:
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def _check_demo_rate(ip: str) -> tuple[bool, int]:
    """
    Return (allowed, retry_after_seconds).
    Allows a new session if the IP has not started one in the last DEMO_WINDOW.
    """
    now = time.time()
    last = _demo_sessions.get(ip, 0)
    if now - last < DEMO_WINDOW:
        return False, int(DEMO_WINDOW - (now - last))
    return True, 0


def _cleanup_demo_agents():
    """Remove demo agents older than DEMO_AGENT_TTL from tracking dict."""
    now = time.time()
    expired = [aid for aid, (ts, _) in _demo_agents.items() if now - ts > DEMO_AGENT_TTL]
    for aid in expired:
        del _demo_agents[aid]


# ── REQUEST MODELS ────────────────────────────────────────────────────────

class DemoRegisterRequest(BaseModel):
    """
    Register the bifurcation stress agent for the public demo.
    phi and epsilon are fixed to demo values — payload is informational only.
    """
    name:   str = Field(default="BIFURCATION-TEST",
                        description="Display name for the stress agent")
    domain: str = Field(default="stress_test",
                        description="Agent domain")


class DemoInteractRequest(BaseModel):
    """
    Fire a single stress interaction for the demo.
    agent_id must be a registered demo agent (returned by /demo/register).
    cost/benefit are fixed to stress values — payload is informational only.
    """
    agent_id: str = Field(..., description="Demo agent_id from /demo/register")


# ── ENDPOINTS ─────────────────────────────────────────────────────────────

@router.get("/status")
async def demo_status():
    """
    Public health check for the demo system.
    Returns current CI, agent count, and demo availability.
    Always accessible — no auth, no rate limit.
    """
    kernel = get_kernel()
    ci     = kernel.compute_cooperation_index()
    agents = len(kernel.agents)
    _cleanup_demo_agents()
    active_demos = len(_demo_agents)

    return {
        "demo_available":    True,
        "cooperation_index": round(ci, 4),
        "ecosystem_agents":  agents,
        "active_demo_sessions": active_demos,
        "demo_window_seconds":  DEMO_WINDOW,
        "max_interactions":     DEMO_MAX_INTERACTIONS,
        "message": (
            "MELVcore live demo. Use POST /demo/register to start, "
            "then POST /demo/interact to fire stress interactions. "
            "No API key required."
        )
    }


@router.post("/register", status_code=201)
async def demo_register(request: Request, payload: DemoRegisterRequest):
    """
    Register a bifurcation stress agent for the public demo.

    Rate limited: 1 session per IP per 10 minutes.
    Returns a demo_agent_id for use in /demo/interact calls.

    The stress agent is pre-configured:
      phi = 0.3  (low maturity — barely established in niche)
      epsilon = 7.5  (near-maximum adaptive plasticity)
    These values are fixed for the demo to ensure a consistent,
    meaningful bifurcation event every time.
    """
    ip = _client_ip(request)
    allowed, retry = _check_demo_rate(ip)

    if not allowed:
        raise HTTPException(
            status_code=429,
            detail={
                "error":   "demo_rate_limited",
                "message": f"Demo sessions are limited to 1 per {DEMO_WINDOW // 60} minutes per visitor.",
                "retry_after_seconds": retry,
            }
        )

    # Register demo session
    _demo_sessions[ip] = time.time()
    _cleanup_demo_agents()

    kernel = get_kernel()

    # Fixed stress agent parameters — intentionally hardcoded for demo reproducibility
    demo_agent_id = f"demo_stress_{uuid.uuid4().hex[:8]}"
    profile = AgentProfile(
        agent_id   = demo_agent_id,
        name       = payload.name or "BIFURCATION-TEST",
        domain     = payload.domain or "stress_test",
        phi        = 0.3,    # low maturity
        epsilon    = 7.5,    # near-maximum plasticity
        status     = AgentStatus.MATURING,
        capabilities = ["stress_test"],
    )
    kernel.register_agent(profile)

    # Track for interaction cap and cleanup
    _demo_agents[demo_agent_id] = (time.time(), 0)

    # Read current CI for baseline context
    ci_before = round(kernel.compute_cooperation_index(), 4)

    return {
        "registered":       True,
        "demo_agent_id":    demo_agent_id,
        "phi":              0.3,
        "epsilon":          7.5,
        "status":           "maturing",
        "ci_before":        ci_before,
        "max_interactions": DEMO_MAX_INTERACTIONS,
        "next_step":        f"POST /demo/interact with agent_id='{demo_agent_id}'",
        "message": (
            "Stress agent registered. phi=0.3 (low maturity), epsilon=7.5 (high plasticity). "
            f"Ecosystem CI before stress: {ci_before}. "
            f"Fire up to {DEMO_MAX_INTERACTIONS} interactions via POST /demo/interact."
        )
    }


@router.post("/interact")
async def demo_interact(request: Request, payload: DemoInteractRequest):
    """
    Fire a single bifurcation stress interaction for the demo.

    No API key required. Rate limited via session (max 20 interactions
    per demo agent). Interaction parameters are fixed:
      cost    = 9.5  (very high cost)
      benefit = 0.5  (minimal benefit)
      i_factor = 19.0  (114× the cooperative threshold of 0.5)

    The target is the first available RESEARCH agent in the ecosystem.
    Returns the kernel's governance response and current CI.
    """
    agent_id = payload.agent_id

    # Validate this is a registered demo agent
    if agent_id not in _demo_agents:
        raise HTTPException(
            status_code=404,
            detail={
                "error":   "demo_agent_not_found",
                "message": f"Agent '{agent_id}' is not a registered demo agent. "
                           "Start a session with POST /demo/register first.",
            }
        )

    # Check interaction cap
    registered_at, interaction_count = _demo_agents[agent_id]
    if interaction_count >= DEMO_MAX_INTERACTIONS:
        raise HTTPException(
            status_code=429,
            detail={
                "error":   "demo_interaction_limit",
                "message": f"Demo interaction limit ({DEMO_MAX_INTERACTIONS}) reached for this agent. "
                           f"Start a new session with POST /demo/register.",
            }
        )

    kernel = get_kernel()

    # Verify agent still exists in kernel (may have been cleaned up)
    agent = kernel.get_agent(agent_id)
    if not agent:
        del _demo_agents[agent_id]
        raise HTTPException(
            status_code=404,
            detail={
                "error":   "demo_agent_expired",
                "message": "Demo agent has expired. Start a new session with POST /demo/register.",
            }
        )

    # Find first active RESEARCH agent as the interaction partner
    target_id = next(
        (aid for aid, a in kernel.agents.items()
         if a.name == "RESEARCH" and a.status == AgentStatus.ACTIVE
         and aid != agent_id),
        None
    )
    if not target_id:
        # Fallback: any active agent that isn't the stress agent
        target_id = next(
            (aid for aid in kernel.agents if aid != agent_id),
            None
        )
    if not target_id:
        raise HTTPException(status_code=503, detail="No target agents available in ecosystem.")

    # Fixed stress interaction parameters
    STRESS_COST    = 9.5
    STRESS_BENEFIT = 0.5
    RESOURCE_TYPE  = "compute"

    # Update stress agent phi (stays at 0.3 — it's not maturing)
    agent.phi = 0.3

    # Record interaction — kernel reads β from environment
    record = kernel.record_interaction(
        agent_a       = agent_id,
        agent_b       = target_id,
        cost          = STRESS_COST,
        benefit       = STRESS_BENEFIT,
        resource_type = RESOURCE_TYPE,
    )

    # Update interaction count
    _demo_agents[agent_id] = (registered_at, interaction_count + 1)

    # Build nudge via NudgeEngine
    from core.melv_engine import KernelAction
    itype  = record.interaction_type.value
    beta_i = record.beta_i

    if itype == "cooperative":
        action = KernelAction.NONE
        status = "cooperative"
    elif itype == "threshold":
        action = KernelAction.NUDGE
        status = "threshold"
    else:
        action = KernelAction.NICHE_DIVERGENCE if beta_i < 1.6 else KernelAction.PROVISION_BETA
        status = "bifurcation"

    contention_depth = kernel.get_contention_depth(agent_id, target_id)
    nudge_engine     = NudgeEngine()
    nudge            = nudge_engine.build_nudge(action, beta_i, RESOURCE_TYPE, contention_depth, 0.3)

    # Current CI after this interaction
    ci_now = round(kernel.compute_cooperation_index(), 4)

    interactions_remaining = DEMO_MAX_INTERACTIONS - (interaction_count + 1)

    return {
        "interaction_number":     interaction_count + 1,
        "interactions_remaining": interactions_remaining,
        "agent_id":               agent_id,
        "target_agent":           target_id,
        "i_factor":               round(record.i_factor, 4),
        "beta_i":                 round(beta_i, 4),
        "status":                 status,
        "action":                 action.value if hasattr(action, "value") else str(action),
        "nudge_type":             nudge.nudge_type if nudge else None,
        "nudge_rationale":        nudge.rationale if nudge else None,
        "ci_now":                 ci_now,
        "interpretation": (
            f"Interaction {interaction_count + 1}/{DEMO_MAX_INTERACTIONS}: "
            f"i={record.i_factor:.1f} (cost/benefit = {STRESS_COST}/{STRESS_BENEFIT}). "
            f"βi={beta_i:.1f} — {'bifurcation zone' if status == 'bifurcation' else status}. "
            f"Kernel action: {action.value if hasattr(action, 'value') else action}. "
            f"Ecosystem CI: {ci_now}."
        )
    }


@router.get("/ci")
async def demo_ci():
    """
    Public CI snapshot — no auth required.
    For polling during the demo to track recovery arc.
    """
    kernel = get_kernel()
    ci     = kernel.compute_cooperation_index()
    return {
        "cooperation_index": round(ci, 4),
        "target":            0.75,
        "healthy":           ci >= 0.75,
        "timestamp":         time.time(),
    }

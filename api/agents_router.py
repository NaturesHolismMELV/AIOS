"""
agents_router.py — Session 5 API routes for WriterAgent and PlannerAgent.

Mounted at /agents/ in api/server.py.

Endpoints
─────────
    POST /agents/writer/execute    — generate written content (real Haiku call)
    POST /agents/planner/execute   — decompose a goal into a plan (real Haiku call)
    GET  /agents/writer/status     — WriterAgent MELVcore state
    GET  /agents/planner/status    — PlannerAgent MELVcore state
    GET  /agents/llm/summary       — combined token budget summary (3 LLM agents)

All agent calls go through BaseAgent.run_task() so cost/benefit is correctly
reported to the MELVcore kernel and φ is updated on each call.

Author: L.W. Evans | ORCID: 0009-0001-0963-1840 | Cape Town, South Africa
"""

from fastapi import APIRouter, Request, HTTPException
from pydantic import BaseModel
from core.cost_calculator import get_calculator, COST_PROFILES

router = APIRouter()


# ── Request schemas ───────────────────────────────────────────────────────────

class WriterRequest(BaseModel):
    task_text: str
    context: str = ""
    max_tokens: int = 512


class PlannerRequest(BaseModel):
    goal: str
    constraints: str = ""
    max_tokens: int = 400


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get(request: Request, attr: str, label: str):
    agent = getattr(request.app.state, attr, None)
    if agent is None:
        raise HTTPException(status_code=503, detail=f"{label} not initialised")
    return agent


# ── Routes ────────────────────────────────────────────────────────────────────

@router.post("/writer/execute")
async def writer_execute(body: WriterRequest, request: Request):
    """
    Generate written content via a real Haiku LLM call.

    The call goes through BaseAgent.run_task() which:
    - Records cost/benefit to MELVcore kernel
    - Updates WriterAgent φ based on outcome quality
    - Returns kernel-standard result dict

    Body fields: task_text (required), context (optional), max_tokens (optional)
    """
    agent = _get(request, "writer_agent", "WriterAgent")
    task = {
        "task_text":  body.task_text,
        "context":    body.context,
        "max_tokens": body.max_tokens,
    }
    # run_task() handles kernel reporting and φ update
    result = await agent.run_task(task, peer_agent_id="TOKEN_BUDGET")
    return result


@router.post("/planner/execute")
async def planner_execute(body: PlannerRequest, request: Request):
    """
    Decompose a goal into an ordered agent execution plan via Haiku.

    Returns a structured JSON plan plus MELVcore cost metrics.
    """
    agent = _get(request, "planner_agent", "PlannerAgent")
    task = {
        "goal":        body.goal,
        "constraints": body.constraints,
        "max_tokens":  body.max_tokens,
    }
    result = await agent.run_task(task, peer_agent_id="TOKEN_BUDGET")
    return result


@router.get("/writer/status")
async def writer_status(request: Request):
    """Current WriterAgent MELVcore state."""
    agent = _get(request, "writer_agent", "WriterAgent")
    p = agent.profile
    return {
        "agent":         agent.name,
        "agent_id":      agent.agent_id,
        "resource_type": "token_budget",
        "phi":           round(p.phi, 4),
        "epsilon":       round(p.epsilon, 4),
        "status":        p.status.value,
        "task_count":    p.task_count,
        "success_rate":  round(p.success_rate, 4),
        "real_llm":      True,
        "model":         "claude-haiku-4-5-20251001",
    }


@router.get("/planner/status")
async def planner_status(request: Request):
    """Current PlannerAgent MELVcore state."""
    agent = _get(request, "planner_agent", "PlannerAgent")
    p = agent.profile
    return {
        "agent":         agent.name,
        "agent_id":      agent.agent_id,
        "resource_type": "token_budget",
        "phi":           round(p.phi, 4),
        "epsilon":       round(p.epsilon, 4),
        "status":        p.status.value,
        "task_count":    p.task_count,
        "success_rate":  round(p.success_rate, 4),
        "real_llm":      True,
        "model":         "claude-haiku-4-5-20251001",
    }


@router.get("/llm/summary")
async def llm_budget_summary(request: Request):
    """
    Combined token-budget summary across all three LLM agents
    (ANALYSIS, WRITER, PLANNER).

    Computes mean_i from the kernel's recent interactions involving
    token_budget resources — the authoritative MELVcore view.
    """
    kernel = getattr(request.app.state, "kernel", None)
    if kernel is None:
        raise HTTPException(status_code=503, detail="Kernel not initialised")

    # Pull token_budget interactions from recent history
    recent_token = [
        r for r in kernel.interactions[-100:]
        if hasattr(r, "beta") and r.agent_b == "TOKEN_BUDGET"
           or getattr(r, "_resource_type", None) == "token_budget"
    ]

    # Fallback: all recent interactions of token_budget resource type
    # (resource_type is stored in beta lookup but not on the record itself)
    # Use all recent interactions and filter by agents we know are LLM-based
    llm_agent_names = {"WRITER", "ANALYSIS", "PLANNER"}
    recent_llm = [
        r for r in kernel.interactions[-200:]
        if r.agent_a in llm_agent_names or r.agent_b in llm_agent_names
    ]

    # Get agent status for each LLM agent
    agents_data = []
    for agent_name in ["ANALYSIS", "WRITER", "PLANNER"]:
        # Find the agent profile by name
        profile = next(
            (a for a in kernel.agents.values() if a.name == agent_name), None
        )
        if profile:
            cp = COST_PROFILES.get(agent_name)
            agents_data.append({
                "agent":        profile.name,
                "phi":          round(profile.phi, 4),
                "task_count":   profile.task_count,
                "status":       profile.status.value,
                "real_llm":     True,
                "model":        "claude-haiku-4-5-20251001",
                "cost_type":    cp.label() if cp else "balanced",
                "token_weight": cp.token_weight if cp else 1.0,
                "latency_weight": cp.latency_weight if cp else 1.0,
            })

    mean_i = (
        sum(r.i_factor for r in recent_llm) / len(recent_llm)
        if recent_llm else 0.0
    )
    ci_proxy = round(1.0 - mean_i, 4)

    return {
        "resource":             "token_budget",
        "llm_agents":           agents_data,
        "recent_llm_interactions": len(recent_llm),
        "mean_i_factor":        round(mean_i, 6),
        "ci_proxy":             ci_proxy,
        "contention_risk":      "HIGH" if mean_i > 0.9 else ("MEDIUM" if mean_i > 0.5 else "LOW"),
        "bifurcation_approaching": mean_i >= 0.95,
        "authoritative_ci":     round(kernel.cooperation_index(), 4),
        "note": (
            "mean_i_factor and ci_proxy are filtered to LLM agents only. "
            "authoritative_ci covers all agents — computed by MELVcore kernel."
        ),
    }

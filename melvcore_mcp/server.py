"""
mcp/server.py — MELVcore MCP Server
=====================================
Session 14 · v1.7.0

Exposes MELVcore governance as an MCP (Model Context Protocol) server,
allowing any MCP-compatible AI client — Claude Desktop, Cursor, Zed,
or any LangChain/LangGraph MCP tool node — to discover and invoke
thermodynamic agent certification natively.

Four tools
----------
1. get_cooperation_index
   Returns the live CI of the running AIOS ecosystem.
   Use before deploying an agent to check if the environment is healthy.

2. certify_agent
   Full sandbox certification: submit profile → poll → return report.
   Synchronous from the caller's perspective (blocks until complete,
   typically 5–15 seconds for 500 interactions).

3. record_interaction
   Report a cost/benefit interaction directly to the MELVKernel.
   Enables an MCP client to act as a governed agent in real time.

4. provision_beta
   Human operator portal: adjust environmental suitability (β) for
   a resource type. Useful for testing how β changes affect CI.

Two MCP resources
-----------------
melvcore://ecosystem/health
   Live ecosystem health snapshot (same as GET /api/health).

melvcore://registry
   MELVcore Compatibility Registry — all certified agents.

Transport
---------
Mounted as a Starlette sub-application inside the main FastAPI server:
   /mcp   → Streamable HTTP transport (recommended, MCP 2025-03-26 spec)
   /mcp/sse → SSE transport (legacy, for older clients)

Standalone stdio mode (for Claude Desktop config):
   python -m mcp.server  (runs stdio transport against a live kernel)

Usage — Claude Desktop (claude_desktop_config.json)
------------------------------------------------------
{
  "mcpServers": {
    "melvcore": {
      "command": "uvicorn",
      "args": ["api.server:app", "--port", "8000"],
      "env": {}
    }
  }
}

Or with the hosted demo URL:
{
  "mcpServers": {
    "melvcore": {
      "type": "sse",
      "url": "https://YOUR-APP.railway.app/mcp/sse"
    }
  }
}
"""

import asyncio
import json
import logging
import os
import sys
import time
from typing import Any, Optional

# AIOS root is added to sys.path by the caller (api/server.py) before this
# module is loaded via importlib. Do NOT modify sys.path here — on Windows /
# Python 3.14 inserting at position 0 causes 'mcp' to resolve to the local
# melvcore_mcp/ directory instead of the installed SDK, causing infinite recursion.

from mcp.server.fastmcp import FastMCP

logger = logging.getLogger("aios.mcp")

# ── MCP SERVER INSTANCE ───────────────────────────────────────────────────

mcp = FastMCP(
    name="MELVcore",
    instructions=(
        "MELVcore is a thermodynamic agent certification platform. "
        "Use certify_agent to test whether a new AI agent will cooperate "
        "or degrade a multi-agent ecosystem. Use get_cooperation_index to "
        "check current ecosystem health before deploying. "
        "Use record_interaction to participate as a governed agent. "
        "Cooperation Index (CI) ≥ 0.75 indicates a healthy cooperative basin. "
        "CLS (Composite Longevity Score) ≥ 80 = CERTIFIED, ≥ 50 = advisory, < 50 = not certified."
    ),
    # Session 15 fix: set explicit paths so both transports mount cleanly under /mcp
    # streamable_http_path='/' → app.mount("/mcp", sh_app) → serves POST /mcp
    # sse_path='/sse'          → app.mount("/mcp", sse_app) → serves GET /mcp/sse
    streamable_http_path="/",
    sse_path="/sse",
    message_path="/messages",
    # Session 17 fix: stateless mode — new transport per request, no session ID required.
    # This makes /mcp work with MCP Inspector, curl, and any client that doesn't
    # manage session lifecycle. Stateful clients (Claude Desktop) also work fine.
    stateless_http=True,
)

# ── KERNEL ACCESS ─────────────────────────────────────────────────────────

# The kernel and sandbox_engine are injected at startup from server.py
# via set_kernel() and set_sandbox_engine(). This avoids circular imports
# and allows the MCP server to work with the same live state as the REST API.

_kernel = None
_sandbox_engine = None


def set_kernel(kernel) -> None:
    global _kernel
    _kernel = kernel


def set_sandbox_engine(engine) -> None:
    global _sandbox_engine
    _sandbox_engine = engine


def _require_kernel():
    if _kernel is None:
        raise RuntimeError(
            "MELVcore kernel not initialised. "
            "Call mcp.server.set_kernel(kernel) at startup."
        )
    return _kernel


def _require_sandbox():
    if _sandbox_engine is None:
        raise RuntimeError(
            "SandboxEngine not initialised. "
            "Call mcp.server.set_sandbox_engine(engine) at startup."
        )
    return _sandbox_engine


# ══════════════════════════════════════════════════════════════════════════
# TOOL 1 — get_cooperation_index
# ══════════════════════════════════════════════════════════════════════════

@mcp.tool()
async def get_cooperation_index() -> str:
    """
    Get the live Cooperation Index (CI) of the MELVcore ecosystem.

    CI = 1 − mean(β·i) across all recent agent interactions.
    Range: 0.0 (pure conflict) → 1.0 (full cooperation).
    Target: CI ≥ 0.75 (cooperative basin).

    Use this before deploying a new agent to check whether the
    environment is currently healthy or under stress.

    Returns a JSON object with:
      - cooperation_index: float [0, 1]
      - target: 0.75
      - healthy: bool (true if CI ≥ 0.75)
      - regime: str ("cooperative" | "threshold" | "conflict")
      - n_agents: int
      - n_interactions: int
      - recommendation: str (plain-language guidance)
    """
    kernel = _require_kernel()
    ci     = kernel.cooperation_index()
    health = kernel.ecosystem_health()

    if ci >= 0.75:
        regime = "cooperative"
        recommendation = (
            f"Ecosystem is in the cooperative basin (CI={ci:.4f}). "
            "Safe to introduce new agents."
        )
    elif ci >= 0.50:
        regime = "threshold"
        recommendation = (
            f"Ecosystem is in the threshold zone (CI={ci:.4f}). "
            "New agents should be certified before deployment. "
            "Monitor closely for bifurcation events."
        )
    else:
        regime = "conflict"
        recommendation = (
            f"Ecosystem is in the conflict zone (CI={ci:.4f}). "
            "Do not introduce new agents without certification. "
            "Consider reducing agent load or adjusting β provisioning."
        )

    result = {
        "cooperation_index":  round(ci, 4),
        "target":             0.75,
        "healthy":            ci >= 0.75,
        "regime":             regime,
        "n_agents":           health.get("n_agents", 0),
        "n_interactions":     health.get("n_interactions_total", 0),
        "mean_phi":           round(health.get("mean_phi", 0.0), 4),
        "recommendation":     recommendation,
    }
    return json.dumps(result, indent=2)


# ══════════════════════════════════════════════════════════════════════════
# TOOL 2 — certify_agent
# ══════════════════════════════════════════════════════════════════════════

@mcp.tool()
async def certify_agent(
    agent_id:   str,
    domain:     str,
    phi:        float = 0.5,
    epsilon:    float = 3.0,
    agent_name: str   = "",
    n_interactions: int = 500,
) -> str:
    """
    Run a full MELVcore sandbox certification for an agent profile.

    Submits the agent to a reference ecosystem simulation, runs
    n_interactions interactions, and returns a CertificationReport
    with a CERTIFIED / CERTIFIED_WITH_ADVISORY / NOT_CERTIFIED verdict.

    Parameters
    ----------
    agent_id : str
        Unique identifier for the agent (alphanumeric, hyphens, underscores).
    domain : str
        Agent's functional domain. One of: retrieval, generation, planning,
        analysis, coding, search, system, custom.
    phi : float
        Evolutionary maturity [0.0–1.0]. Higher = more domain-optimised.
        0.0=novice, 0.65=proficient, 0.85+=expert.
    epsilon : float
        Adaptive plasticity [0.0–8.0]. Higher = more adaptive but potentially
        destabilising. Values > 4.5 often generate advisories.
    agent_name : str
        Human-readable display name (optional, defaults to agent_id.upper()).
    n_interactions : int
        Number of simulation interactions (default 500, range 10–2000).
        More interactions = more accurate but slower (500 ≈ 10–15 seconds).

    Returns
    -------
    JSON with: verdict, cls_score, baseline_ci, with_agent_ci, ci_delta,
    narrative, advisory (if any), run_id, and certification_anchor.

    CLS thresholds:
      ≥ 80  →  CERTIFIED         (agent cooperates or is neutral)
      ≥ 50  →  CERTIFIED_WITH_ADVISORY  (marginal — monitor)
      < 50  →  NOT_CERTIFIED     (agent degrades ecosystem)
    """
    engine = _require_sandbox()

    # Validate inputs
    phi     = max(0.0, min(1.0, float(phi)))
    epsilon = max(0.0, min(8.0, float(epsilon)))
    n_interactions = max(10, min(2000, int(n_interactions)))
    name    = agent_name.strip() or agent_id.upper()

    # Build profile
    from core.melv_engine import AgentProfile, AgentStatus
    profile = AgentProfile(
        agent_id    = agent_id,
        name        = name,
        domain      = domain,
        phi         = phi,
        epsilon     = epsilon,
        beta_pref   = 1.0,
        status      = AgentStatus.MATURING,
    )

    # Submit and run synchronously (poll in tight async loop)
    run = engine.submit(profile)
    run.n_interactions = n_interactions

    # Run certification in executor to avoid blocking event loop
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, _run_certification_sync, engine, run.run_id)

    # Fetch report
    final_run = engine.get_run(run.run_id)
    if final_run is None or final_run.status != "complete" or final_run.report is None:
        return json.dumps({
            "error":   "certification_failed",
            "run_id":  run.run_id,
            "status":  final_run.status if final_run else "unknown",
            "detail":  "Simulation did not complete. Try again or reduce n_interactions.",
        }, indent=2)

    report = final_run.report
    baseline_ci   = report.baseline.final_ci
    with_agent_ci = report.with_agent.final_ci
    ci_delta      = with_agent_ci - baseline_ci

    result = {
        "verdict":          report.verdict,
        "cls_score":        round(report.cls_score, 2),
        "certified":        report.verdict != "NOT_CERTIFIED",
        "baseline_ci":      round(baseline_ci, 4),
        "with_agent_ci":    round(with_agent_ci, 4),
        "ci_delta":         round(ci_delta, 4),
        "ci_delta_pct":     f"{ci_delta:+.2%}",
        "narrative":        report.narrative,
        "advisory":         report.advisory,
        "implicated_resources": report.implicated_resources,
        "certification_anchor": report.certification_anchor,
        "run_id":           report.run_id,
        "agent_profile": {
            "agent_id": agent_id,
            "domain":   domain,
            "phi":      phi,
            "epsilon":  epsilon,
        },
    }
    return json.dumps(result, indent=2)


def _run_certification_sync(engine, run_id: str) -> None:
    """
    Run a full certification synchronously (for executor offload).
    Mirrors run_full_certification but without asyncio.sleep yield points.
    """
    run = engine.get_run(run_id)
    if run is None:
        return

    run.status   = "running"
    run.progress = 0.0

    try:
        baseline = engine._run_baseline_tracked(run)
        run.progress = 0.5

        agent_snapshot = engine._run_agent_tracked(run)
        run.progress = 0.9

        run.baseline_metrics = baseline
        run.agent_metrics    = agent_snapshot

        report = engine.compute_report(run_id)
        run.report   = report
        run.progress = 1.0
        run.status   = "complete"

        # Add to certified registry if applicable
        if report.verdict != "NOT_CERTIFIED":
            if report not in engine._reports:
                engine._reports.append(report)

    except Exception as e:
        run.status = "failed"
        logger.error("certify_agent sync run failed: %s", e, exc_info=True)


# ══════════════════════════════════════════════════════════════════════════
# TOOL 3 — record_interaction
# ══════════════════════════════════════════════════════════════════════════

@mcp.tool()
async def record_interaction(
    agent_a:       str,
    agent_b:       str,
    cost:          float,
    benefit:       float,
    resource_type: str = "compute",
) -> str:
    """
    Record a cost/benefit interaction between two agents in the live kernel.

    This is the governance hook that makes MELVcore a real-time governor
    rather than a post-hoc auditor. Call this from your agent logic after
    every significant operation to keep CI tracking current.

    Parameters
    ----------
    agent_a : str
        ID of the initiating agent.
    agent_b : str
        ID of the responding agent.
    cost : float
        Measurable cost of the interaction [0.0–2.0].
        Examples: token usage normalised to [0,2], latency in seconds * 0.3,
        or a domain-specific cost metric.
    benefit : float
        Measurable benefit of the interaction [0.01–2.0].
        Examples: task completion quality [0,1], information gain, user rating.
    resource_type : str
        Resource contended for. One of: compute, api_quota, vector_db,
        storage, token_budget, context_window. Default: compute.

    Returns
    -------
    JSON with: i_factor (cost/benefit), beta_i (modulated threshold),
    interaction_type (cooperative/threshold/conflict), and updated CI.

    The kernel will automatically nudge agents if beta_i ≥ 0.70
    (threshold zone) or ≥ 1.0 (conflict zone).
    """
    kernel = _require_kernel()

    # Clamp to valid ranges
    cost    = max(0.0, min(2.0, float(cost)))
    benefit = max(0.01, float(benefit))

    record = kernel.record_interaction(
        agent_a=agent_a,
        agent_b=agent_b,
        cost=cost,
        benefit=benefit,
        resource_type=resource_type,
    )

    ci_after = kernel.cooperation_index()

    result = {
        "recorded":         True,
        "agent_a":          agent_a,
        "agent_b":          agent_b,
        "i_factor":         round(record.i_factor, 4),
        "beta_i":           round(record.beta_i, 4),
        "interaction_type": record.interaction_type.value,
        "cost":             round(cost, 4),
        "benefit":          round(benefit, 4),
        "resource_type":    resource_type,
        "ci_after":         round(ci_after, 4),
        "ci_healthy":       ci_after >= 0.75,
        "interpretation": (
            "Cooperative — below threshold (βi < 0.70)"
            if record.beta_i < 0.70 else
            "Threshold zone — monitor (0.70 ≤ βi < 1.0)"
            if record.beta_i < 1.00 else
            "Conflict zone — kernel nudge triggered (βi ≥ 1.0)"
        ),
    }
    return json.dumps(result, indent=2)


# ══════════════════════════════════════════════════════════════════════════
# TOOL 4 — provision_beta
# ══════════════════════════════════════════════════════════════════════════

@mcp.tool()
async def provision_beta(
    resource: str,
    value:    float,
) -> str:
    """
    Adjust the environmental suitability (β) for a resource type.

    β modulates how effectively agents can interact over a given resource.
    High β = abundant resource = lower effective i-factor = more cooperative.
    Low β = scarce resource = higher effective i-factor = more conflict.

    This is the human operator portal. Use it to simulate resource
    scarcity (β < 1.0) or abundance (β > 1.0) and observe CI response.

    Parameters
    ----------
    resource : str
        Resource to adjust. One of:
        compute, api_quota, vector_db, storage, token_budget, context_window.
    value : float
        New β value for this resource [0.1–3.0].
        1.0 = neutral, < 1.0 = scarce, > 1.0 = abundant.

    Returns
    -------
    JSON with: updated resource, new value, full beta environment state,
    and predicted CI direction (higher β generally raises CI).
    """
    kernel = _require_kernel()

    valid = ["compute", "api_quota", "vector_db", "storage",
             "token_budget", "context_window"]
    if resource not in valid:
        return json.dumps({
            "error":   "invalid_resource",
            "detail":  f"Resource must be one of: {valid}",
            "provided": resource,
        }, indent=2)

    value = max(0.1, min(3.0, float(value)))
    kernel.provision_beta(resource, value)

    ci_now = kernel.cooperation_index()
    beta   = kernel.beta.to_dict()

    # Simple directional prediction
    if value > 1.0:
        prediction = f"β={value:.2f} (abundant) should raise CI over upcoming interactions."
    elif value < 0.8:
        prediction = f"β={value:.2f} (scarce) may lower CI — watch for bifurcation events."
    else:
        prediction = f"β={value:.2f} (neutral) — minimal CI impact expected."

    result = {
        "updated":          True,
        "resource":         resource,
        "new_value":        round(value, 3),
        "beta_environment": {k: round(v, 3) for k, v in beta.items()},
        "ci_current":       round(ci_now, 4),
        "prediction":       prediction,
    }
    return json.dumps(result, indent=2)


# ══════════════════════════════════════════════════════════════════════════
# RESOURCES
# ══════════════════════════════════════════════════════════════════════════

@mcp.resource("melvcore://ecosystem/health")
async def ecosystem_health_resource() -> str:
    """Live ecosystem health snapshot — CI, agents, interactions, events."""
    kernel = _require_kernel()
    health = kernel.ecosystem_health()
    return json.dumps(health, indent=2)


@mcp.resource("melvcore://registry")
async def registry_resource() -> str:
    """MELVcore Compatibility Registry — all certified agents."""
    engine = _require_sandbox()
    reports = engine.list_certified()
    return json.dumps({
        "registry_count": len(reports),
        "agents": [r.to_dict() for r in reports],
    }, indent=2)


# ══════════════════════════════════════════════════════════════════════════
# STANDALONE STDIO MODE
# ══════════════════════════════════════════════════════════════════════════

def run_stdio():
    """
    Run the MCP server in stdio mode (for Claude Desktop integration).

    Claude Desktop config (claude_desktop_config.json):
    {
      "mcpServers": {
        "melvcore": {
          "command": "python",
          "args": ["-m", "mcp.server"],
          "cwd": "/path/to/AIOS"
        }
      }
    }

    In stdio mode the server creates its own MELVKernel and SandboxEngine
    (no persistence, ephemeral — suitable for desktop tool use).
    """
    from core.melv_engine import MELVKernel
    from core.sandbox_engine import SandboxEngine
    from agents.implementations import create_default_ecosystem

    kernel = MELVKernel()
    create_default_ecosystem(kernel)
    engine = SandboxEngine()

    set_kernel(kernel)
    set_sandbox_engine(engine)

    logger.info("MELVcore MCP Server starting in stdio mode")
    mcp.run(transport="stdio")


if __name__ == "__main__":
    run_stdio()

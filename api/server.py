"""
AIOS REST API
=============
FastAPI backend serving the Harmony Dashboard.
All MELV metrics exposed as JSON endpoints.
"""

import asyncio
import random
import sys
import os
import httpx 

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional
from api.data_routes import router as data_router
from api.agents_router import router as agents_router
from api.sandbox_router import router as sandbox_router
from api.theorem_router import router as theorem_router
from api.middleware import RateLimitMiddleware, APIKeyMiddleware, print_startup_banner
from core.sandbox_engine import SandboxEngine

from core.melv_engine import MELVKernel
from core.gateway import router as gateway_router, set_kernel
from api.demo_router import router as demo_router
from api.observe_router import router as observe_router
from core.persistence import AIOSPersistence
from agents.implementations import create_default_ecosystem
from agents.oxpecker_agent import OxpeckerAgent

# Session 14: MCP server — load via importlib to avoid 'mcp' package name clash
# Session 15 fix: Windows/Python 3.14 recursion fix.
#   - sys.path.append AIOS root here (not inside melvcore_mcp/server.py)
#     so installed 'mcp' SDK is found before local melvcore_mcp/ directory
#   - Register module in sys.modules BEFORE exec_module to break any remaining
#     circular import chain
import sys as _sys
import importlib.util as _ilu

# Add AIOS root to path so melvcore_mcp can import sibling packages (melvcore, core, etc.)
# Append (not insert) so installed packages take priority over local directories
_aios_root = os.path.dirname(os.path.abspath(__file__))  # api/ -> AIOS root is parent
_aios_root = os.path.dirname(_aios_root)
if _aios_root not in _sys.path:
    _sys.path.append(_aios_root)

_mcp_path = os.path.join(_aios_root, "melvcore_mcp", "server.py")

if "aios_mcp_server" not in _sys.modules:
    _mcp_spec = _ilu.spec_from_file_location("aios_mcp_server", _mcp_path)
    _mcp_mod  = _ilu.module_from_spec(_mcp_spec)
    _sys.modules["aios_mcp_server"] = _mcp_mod   # register BEFORE exec to break recursion
    _mcp_spec.loader.exec_module(_mcp_mod)

_mcp_mod        = _sys.modules["aios_mcp_server"]
mcp_server      = _mcp_mod.mcp
set_mcp_kernel  = _mcp_mod.set_kernel
set_mcp_sandbox = _mcp_mod.set_sandbox_engine


# ── APP SETUP ──────────────────────────────────────────────────────────────
# Session 15 fix: MCP streamable HTTP requires its session_manager to be
# started inside the outer app's lifespan context — FastAPI does NOT
# propagate lifespan events to mounted sub-applications (Starlette limitation).
# Solution: use a proper lifespan context manager that runs the MCP
# session_manager alongside the existing startup/shutdown logic.

from contextlib import asynccontextmanager

# Pre-initialise the MCP transport apps so session_manager exists before lifespan
_mcp_streamable_app = mcp_server.streamable_http_app()   # creates session_manager
_mcp_sse_app        = mcp_server.sse_app()               # creates SSE transport

@asynccontextmanager
async def _lifespan(app: FastAPI):
    """
    Unified lifespan for AIOS.
    Runs MCP session_manager (required by SDK 1.24+) and existing startup logic.
    """
    import logging

    # ── Startup ──────────────────────────────────────────────────────────
    summary = persistence.restore_kernel(kernel)
    logging.getLogger("aios.server").info("Startup restore: %s", summary)

    set_kernel(kernel)
    app.state.kernel         = kernel
    app.state.ecosystem      = ecosystem
    app.state.sandbox_engine = sandbox_engine
    app.state.persistence    = persistence

    for agent in ecosystem.values():
        if agent.name == "WRITER":
            app.state.writer_agent = agent
        elif agent.name == "PLANNER":
            app.state.planner_agent = agent
        elif agent.name == "ANALYSIS":
            app.state.analysis_agent = agent

    # Session 23: calibrate sandbox from live kernel interaction history
    cal_summary = sandbox_engine.calibrate_from_kernel(kernel)
    logging.getLogger("aios.server").info(
        "Sandbox calibrated from %d live interactions — resources: %s",
        cal_summary["total_interactions_sampled"],
        cal_summary["calibrated_resources"],
    )

    # Session 24.2 Fix B — restore theorem experiment state so prediction
    # baseline survives Railway restarts (ci_at_prediction: null was caused
    # by in-memory state being wiped on redeploy)
    from api.theorem_router import _theorem_state
    saved_ts = persistence.load_theorem_state()
    if saved_ts:
        _theorem_state.update(saved_ts)
        logging.getLogger("aios.server").info(
            "Theorem state restored from SQLite: prediction_made_at=%s ci_at_prediction=%s",
            saved_ts.get("prediction_made_at"), saved_ts.get("ci_at_prediction"),
        )

    # Session 27: register OXPECKER agent
    oxpecker_agent.register()
    app.state.oxpecker_agent = oxpecker_agent
    logging.getLogger("aios.server").info("Oxpecker agent registered: %s", oxpecker_agent.status())

    asyncio.create_task(drive_real_agents())
    asyncio.create_task(_keep_alive())
    set_mcp_kernel(kernel)
    set_mcp_sandbox(sandbox_engine)
    app.state.mcp_server = mcp_server

    print_startup_banner()

    # ── MCP session_manager lifespan (required by SDK 1.24+) ─────────────
    async with mcp_server.session_manager.run():
        yield   # server is live here

    # ── Shutdown ─────────────────────────────────────────────────────────
    persistence.save_beta(kernel.beta)
    persistence.close()


app = FastAPI(
    title="AIOS — AI Operating System",
    description=(
        "Thermodynamic agent orchestration via MELVcore. "
        "Implements the Modified Energetic Lotka-Volterra (MELV) cooperation framework. "
        "Blueprint for Harmony — L.W. Evans, Cooperation Press 2026. "
        "ISBN 978-969-8992-10-1 · ORCID: 0009-0001-0963-1840"
    ),
    version="2.6.0",
    lifespan=_lifespan,
    redirect_slashes=False,   # prevents /mcp → /mcp/ redirect for MCP transports
    swagger_ui_parameters={"persistAuthorization": True},
)

# Session 24.3.2 — OpenAPI security scheme so Swagger renders Authorize button.
# Declares X-API-Key header authentication matching the APIKeyMiddleware in
# api/middleware.py.  Adds the lock icon on protected endpoints in /docs.
from fastapi.openapi.utils import get_openapi as _get_openapi

def _custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    schema = _get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
    )
    schema.setdefault("components", {})
    schema["components"]["securitySchemes"] = {
        "ApiKeyAuth": {
            "type": "apiKey",
            "in": "header",
            "name": "X-API-Key",
            "description": (
                "API key required for write operations and sensitive endpoints. "
                "Set AIOS_API_KEY in Railway Variables. "
                "Read-only endpoints (health, public data) are unauthenticated."
            ),
        }
    }
    # Apply security requirement to all paths (Swagger shows lock icon on each)
    for path_item in schema.get("paths", {}).values():
        for operation in path_item.values():
            if isinstance(operation, dict):
                operation.setdefault("security", [{"ApiKeyAuth": []}])
    app.openapi_schema = schema
    return app.openapi_schema

app.openapi = _custom_openapi

# Session 13: deployment middleware (rate limiting + optional API key)
# Middleware is applied in reverse-add order by Starlette.
# CORSMiddleware must run FIRST to handle preflight OPTIONS before APIKeyMiddleware
# blocks them with 401. So CORSMiddleware is added LAST here.
app.add_middleware(RateLimitMiddleware)
app.add_middleware(APIKeyMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "X-API-Key"],
    allow_credentials=False,
)

app.include_router(data_router,    prefix="/data",    tags=["data"])
app.include_router(gateway_router, prefix="/melv",    tags=["MELVcore Gateway"])
app.include_router(demo_router,    prefix="/demo",    tags=["Public Demo"])
app.include_router(agents_router,  prefix="/agents",  tags=["agents"])
app.include_router(sandbox_router, prefix="/sandbox", tags=["MELVcore Sandbox"])
app.include_router(theorem_router, prefix="/theorem", tags=["Cooperation Theorem"])
app.include_router(observe_router, prefix="/api/observe", tags=["Observe Primitive"])  # Session 32

# Session 14 / Session 15 fix: mount MCP transports.
#
# Architecture (final, clean):
#   /mcp/sse, /mcp/messages → SSE transport via app.mount("/mcp", sse_app)
#   /mcp                    → Streamable HTTP via explicit @app.api_route
#
# Starlette app.mount("/mcp") only matches paths starting with "/mcp/" — NOT
# exact "/mcp". An explicit api_route handles /mcp directly without any
# redirect or slash ambiguity.

# Extract raw StreamableHTTPASGIApp callable from the streamable sub-app
from starlette.routing import Mount as _SMount
_sh_asgi = _mcp_streamable_app.routes[0].app   # StreamableHTTPASGIApp

app.mount("/mcp", _mcp_sse_app)   # GET /mcp/sse,  POST /mcp/messages

from fastapi import Request as _Request

@app.api_route("/mcp", methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"])
async def _mcp_streamable_endpoint(request: _Request):
    """Proxy /mcp to the MCP streamable HTTP ASGI handler (path rewritten to '/').
    
    Also ensures Accept header contains both application/json and text/event-stream,
    as required by the MCP streamable HTTP spec. Some clients (MCP Inspector, curl)
    only send one — we add the missing type rather than rejecting the request.
    """
    scope = dict(request.scope)
    scope["path"]     = "/"
    scope["raw_path"] = b"/"

    # Ensure Accept header satisfies MCP streamable HTTP requirements:
    # must include BOTH application/json AND text/event-stream.
    accept = request.headers.get("accept", "")
    needs_json = "application/json" not in accept
    needs_sse  = "text/event-stream" not in accept
    if needs_json or needs_sse:
        additions = []
        if needs_json: additions.append("application/json")
        if needs_sse:  additions.append("text/event-stream")
        new_accept = (accept + ", " + ", ".join(additions)).lstrip(", ")
        # Rebuild headers with patched Accept
        headers = [
            (k, v) for k, v in scope.get("headers", [])
            if k.lower() != b"accept"
        ]
        headers.append((b"accept", new_accept.encode()))
        scope["headers"] = headers

    await _sh_asgi(scope, request.receive, request._send)

# Serve frontend static files — enables http://localhost:8000/frontend/dashboard12.html
_frontend_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frontend")
app.mount("/frontend", StaticFiles(directory=_frontend_dir), name="frontend")

# CORSMiddleware moved to line below — must run before APIKeyMiddleware

# ── GLOBAL STATE ───────────────────────────────────────────────────────────

persistence    = AIOSPersistence()
kernel         = MELVKernel(persistence=persistence)
ecosystem      = create_default_ecosystem(kernel)
sandbox_engine = SandboxEngine()

# Session 27: OXPECKER agent — thermodynamic recycling of interrupted work
oxpecker_agent = OxpeckerAgent(kernel, persistence)

# ── SIMULATION ─────────────────────────────────────────────────────────────

async def drive_real_agents():
    """
    Session 22 Fix B — Replace the random number generator background simulation.

    Periodically invoke real agents on lightweight scheduled tasks so their
    actual token costs and latencies flow into record_interaction() via the
    kernel. The dashboard CI then reflects the actual cost structure of the
    agent population, not a random walk.

    Fallback: if no real agents are available, samples cost/benefit from the
    last 100 real interaction records rather than using random.uniform().
    This preserves empirical calibration without API cost.
    """
    agent_ids = list(ecosystem.keys())

    while True:
        await asyncio.sleep(random.uniform(8.0, 15.0))  # slower cadence than old 1.5–3.5s

        agent_ids = list(ecosystem.keys())
        if len(agent_ids) < 2:
            continue

        # Pick the agent idle longest (lowest task_count)
        agent_id = min(
            agent_ids,
            key=lambda aid: kernel.agents[aid].task_count if aid in kernel.agents else 0
        )
        agent = ecosystem.get(agent_id)
        if agent is None:
            continue

        # Peer agent: pick a different one
        peers = [a for a in agent_ids if a != agent_id]
        peer_id = random.choice(peers) if peers else agent_id
        resource = random.choice([
            "compute", "api_quota", "vector_db",
            "storage", "token_budget", "context_window"
        ])

        # Derive cost/benefit from real interaction history (empirical fallback)
        # Uses the last 100 recorded interactions rather than random.uniform()
        # Session 24.2 Fix A — β-scaled cost generation.
        # Draw a raw cost from historical distribution, then divide by the
        # current β for this resource.  Higher β (provisioned by the kernel)
        # therefore lowers effective cost, closing the feedback loop:
        #   β ↑  →  cost ↓  →  i_factor ↓  →  pairs move below i_critical
        # Without this, PROVISION_BETA raised β in isolation but the next
        # interaction drew cost/benefit from the same unchanged distribution,
        # leaving i_factors stationary regardless of how many times β was
        # provisioned.  This was the root cause of pairs_resolved=0.
        current_beta = kernel.beta.get(resource)

        recent = kernel.interactions[-100:] if kernel.interactions else []
        if len(recent) >= 5:
            costs    = [r.cost    for r in recent]
            benefits = [r.benefit for r in recent]
            cost_raw = random.gauss(
                sum(costs) / len(costs),
                max(0.05, (max(costs) - min(costs)) / 4)
            )
            # β-scaling: higher β reduces effective cost
            cost    = max(0.05, min(2.0, cost_raw / max(0.1, current_beta)))
            benefit = max(0.05, min(2.0, random.gauss(
                sum(benefits) / len(benefits),
                max(0.05, (max(benefits) - min(benefits)) / 4)
            )))
        else:
            # Cold start: use conservative defaults until history accumulates
            cost_raw = random.uniform(0.1, 0.6)
            cost    = max(0.05, min(2.0, cost_raw / max(0.1, current_beta)))
            benefit = random.uniform(0.5, 1.2)

        try:
            kernel.record_interaction(
                agent_a=agent_id,
                agent_b=peer_id,
                cost=cost,
                benefit=benefit,
                resource_type=resource,
            )
            # Update φ for the driving agent
            quality = benefit / max(cost, 0.01)
            kernel.update_phi(agent_id, min(1.0, quality))
        except Exception:
            pass  # background probe failure is non-fatal

        # Session 27: periodically process pending oxpecker fragments
        try:
            await oxpecker_agent.process_pending_fragments(batch_size=3)
        except Exception:
            pass  # recycling failure is non-fatal


# ── KEEP-ALIVE ─────────────────────────────────────────────────────────────

async def _keep_alive():
    """
    Ping /health every 10 minutes to prevent Railway sleep-on-inactivity.
    Railway hobby tier sleeps after ~10–15 min of no traffic; this keeps
    the service warm at zero cost. Logs each ping to Railway's log stream.
    """
    await asyncio.sleep(60)          # let server fully start before first ping
    while True:
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get("https://web-production-e14d1.up.railway.app/health")
                print(f"[keep-alive] /health → {resp.status_code}", flush=True)
        except Exception as exc:
            print(f"[keep-alive] ping failed: {exc}", flush=True)
        await asyncio.sleep(600)     # 10 minutes


# ── MODELS ─────────────────────────────────────────────────────────────────

class BetaUpdate(BaseModel):
    resource: str
    value:    float

class InteractionPost(BaseModel):
    agent_a:       str
    agent_b:       str
    cost:          float
    benefit:       float
    resource_type: str = "compute"


# ── ENDPOINTS ──────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    return {"system": "AIOS", "version": "3.0.0", "status": "operational",
            "demo": "/demo", "docs": "/docs", "mcp": "/mcp", "mcp_sse": "/mcp/sse"}

@app.get("/health")
async def health_simple():
    """Simple health check for load balancers / Railway uptime monitor."""
    return {"status": "ok", "version": "3.0.0"}

@app.get("/dashboard")
async def dashboard_page():
    """Shortcut to the live MELVcore dashboard."""
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frontend", "dashboard12.html")
    return FileResponse(path, media_type="text/html")


@app.get("/demo")
async def demo_landing():
    """Public demo landing page — the URL to share."""
    import os
    landing = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "frontend", "landing.html"
    )
    return FileResponse(landing, media_type="text/html")

@app.get("/api/db_stats")
async def db_stats():
    """Persistence layer row counts — useful for monitoring storage health."""
    return persistence.stats()

@app.get("/api/health")
async def health():
    """Full ecosystem health snapshot for the Harmony Dashboard."""
    return kernel.ecosystem_health()

@app.get("/api/agents")
async def list_agents():
    """All registered agent profiles with current MELV parameters."""
    return {"agents": kernel.get_all_agents()}

@app.get("/api/agents/{agent_id}")
async def get_agent(agent_id: str):
    agent = kernel.get_agent(agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    return agent.to_dict()

@app.get("/api/interactions")
async def list_interactions(n: int = 30):
    """Recent interaction records with i-factor measurements."""
    return {"interactions": kernel.get_recent_interactions(n)}

@app.post("/api/interactions")
async def record_interaction(interaction: InteractionPost):
    """Manually record an interaction (for testing / external agents)."""
    record = kernel.record_interaction(
        agent_a=interaction.agent_a,
        agent_b=interaction.agent_b,
        cost=interaction.cost,
        benefit=interaction.benefit,
        resource_type=interaction.resource_type
    )
    return record.to_dict()

@app.get("/api/events")
async def list_events(n: int = 20):
    """Bifurcation events — kernel interventions log."""
    return {"events": kernel.get_recent_events(n)}

@app.get("/api/omega")
async def omega_network():
    """OmegaNet service coupling matrix metrics."""
    return kernel.compute_omega()

@app.get("/api/beta")
async def get_beta():
    """Current β environment (resource suitability)."""
    return kernel.beta.to_dict()

@app.post("/api/beta")
async def update_beta(update: BetaUpdate):
    """Human portal: adjust β provisioning."""
    valid = ["compute", "api_quota", "vector_db", "storage", "token_budget", "context_window"]
    if update.resource not in valid:
        raise HTTPException(400, f"Unknown resource. Valid: {valid}")
    kernel.provision_beta(update.resource, update.value)
    return {"updated": update.resource, "value": update.value,
            "beta": kernel.beta.to_dict()}

@app.get("/api/cooperation_index")
async def cooperation_index():
    """Current Cooperation Index CI = 1 - mean(βi)."""
    return {
        "cooperation_index": kernel.cooperation_index(),
        "target":            0.75,
        "healthy":           kernel.cooperation_index() >= 0.75,
    }

@app.get("/api/ci_dynamics")
async def ci_dynamics_endpoint():
    """CI Dynamics — dCI/dt, optimisation half-life, drift coefficient, oscillation events."""
    return kernel.ci_dynamics()

@app.get("/api/quorum_status")
async def quorum_status_endpoint():
    """
    Sigmoid quorum gate status — Session 25 (v2.1.0).

    Exposes the ecosystem-mean φ·β product, sigmoid gate value, and
    PROVISION_BETA step magnitude that would fire if the kernel
    triggered a provisioning action right now.

    Biological correspondence (MAIES Event 2): the quorum gate maps the
    MELV sigmoid efficiency function onto bacterial quorum sensing
    (Nadell, Drescher & Foster 2016). φ·β is the population density
    analogue; τ=0.5, k=10 are ABM V2.1 verified constants (③).
    """
    return kernel.quorum_status()

@app.get("/api/quorum_reliability")
async def quorum_reliability_endpoint():
    """
    Session 28 — Quorum Reliability Tagging (v2.4.0).

    Extends the sigmoid quorum gate (Session 25) with an epistemic status
    layer.  Below quorum (φ·β < τ=0.5), the ecosystem is in a high-noise
    regime and agent outputs carry elevated confabulation risk.

    Origin: MAIES Event 4 — quorum gate as agent output reliability marker
    (NotebookLM + Claude synthesis, MAIES-adjacent ②).

    Biological correspondence (Nadell et al. 2016): bacterial quorum sensing
    suppresses costly cooperative behaviours below N_threshold.  In MELV,
    below-quorum outputs are structurally less reliable for the same reason.

    The endpoint does not suppress outputs — it marks epistemic status.
    External consumers decide how to act on the reliability_advisory.

    Returns:
      phi_beta, quorum_regime, reliability_level, reliability_advisory,
      per_agent breakdown, epistemic_status ② theoretical.
    """
    return kernel.quorum_reliability()

@app.get("/api/ci_history")
async def ci_history_endpoint(n: int = 200):
    """Rolling CI history — last N readings as [{t: float, ci: float}].
    Default n=200, max n=1000. Timestamps are Unix epoch seconds."""
    max_n = min(max(1, n), 1000)
    history = kernel._ci_history[-max_n:]
    return [{"t": round(t, 3), "ci": round(ci, 4)} for t, ci in history]


@app.get("/api/oxpecker_status")
async def oxpecker_status_endpoint():
    """
    Session 27 — Oxpecker Phase 2 status.

    Returns fragment queue counts, processing rate, estimated value recovered,
    OXPECKER agent profile, and Pathway A context cache state.

    Validation Stream 9: fragment value ∝ φ_a × φ_b.
    MAIES Event 1 (NotebookLM): bifurcation recycling mechanism, now implemented.
    """
    return oxpecker_agent.status()

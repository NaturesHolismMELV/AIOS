"""
AIOS REST API
=============
FastAPI backend serving the Harmony Dashboard.
All MELV metrics exposed as JSON endpoints.
"""

import asyncio
import random
import sys
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional
from api.data_routes import router as data_router

from core.melv_engine import MELVKernel
from agents.implementations import create_default_ecosystem

# ── APP SETUP ──────────────────────────────────────────────────────────────

app = FastAPI(
    title="AIOS — AI Operating System",
    description="Thermodynamic agent orchestration via MELVcore",
    version="0.1.0"
)
app.include_router(data_router, prefix="/data", tags=["data"])

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── GLOBAL STATE ───────────────────────────────────────────────────────────

kernel    = MELVKernel()
ecosystem = create_default_ecosystem(kernel)

# ── SIMULATION ─────────────────────────────────────────────────────────────

async def simulate_interactions():
    """
    Background task: simulates ongoing agent interactions.
    Drives the ecosystem through realistic βi dynamics.
    """
    agent_ids = list(ecosystem.keys())

    while True:
        await asyncio.sleep(random.uniform(1.5, 3.5))

        if len(agent_ids) < 2:
            continue

        # Pick two random agents to interact
        a_id, b_id = random.sample(agent_ids, 2)
        agent_a    = ecosystem[a_id]
        agent_b    = ecosystem[b_id]

        # Occasionally inject a high-cost interaction to exercise the kernel
        if random.random() < 0.12:
            # threshold/conflict scenario
            cost    = random.uniform(0.8, 1.4)
            benefit = random.uniform(0.6, 1.0)
        else:
            # normal cooperative interaction
            cost    = random.uniform(0.1, 0.6)
            benefit = random.uniform(0.5, 1.2)

        resource = random.choice(["compute", "api_quota", "vector_db",
                                   "storage", "token_budget"])

        kernel.record_interaction(
            agent_a=a_id,
            agent_b=b_id,
            cost=cost,
            benefit=benefit,
            resource_type=resource
        )

        # Occasionally update φ
        if random.random() < 0.3:
            quality = benefit / max(cost, 0.01)
            kernel.update_phi(a_id, min(1.0, quality))


@app.on_event("startup")
async def startup():
    asyncio.create_task(simulate_interactions())


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
    return {"system": "AIOS", "version": "0.1.0", "status": "operational"}

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
    valid = ["compute", "api_quota", "vector_db", "storage", "token_budget"]
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

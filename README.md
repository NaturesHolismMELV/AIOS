# AIOS — AI Operating System

**The governance layer the agentic web needs. Nobody else has built it yet.**

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688.svg)](https://fastapi.tiangolo.com)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17680563.svg)](https://doi.org/10.5281/zenodo.17680563)
[![ISBN](https://img.shields.io/badge/ISBN-978--969--8992--10--1-blue.svg)](https://cooperationpress.com)

---

## The Problem

Coinbase shipped agentic wallets. Cloudflare made 20% of the web readable by agents. OpenAI released shell tools. Stripe is building agent commerce. Google has a universal commerce protocol.

Agents are becoming economic actors — making purchases, calling APIs, consuming compute, competing for token budgets — and **the governance infrastructure does not exist yet.**

What happens when eight specialised agents share a token budget and three of them need to call an LLM simultaneously? What happens when a research agent and a data agent compete for the same API quota? Who decides which agent yields, which agent waits, and which agent gets priority?

Right now: nothing decides. The ecosystem either collapses under contention, or you hardcode priority rules that break the moment your agent mix changes.

**AIOS is the answer to that question.**

---

## What AIOS Does

AIOS is an agent orchestration platform with a thermodynamic governance kernel — the **MELVcore** — that monitors resource contention between agents in real time and resolves conflicts before they cascade.

When two agents compete for the same resource, MELVcore computes an **interaction cost** (`i-factor`) for each agent pair. When `β·i < 1.0`, the system is in cooperative equilibrium and routes normally. When `β·i ≥ 1.0`, the kernel fires a **bifurcation event** — routing a service to reduce cost, nudging an agent to yield, or provisioning additional resource capacity.

The result: an agent ecosystem that self-governs under real resource contention, without hardcoded priority rules.

```
Agent A ──┐
           ├──► MELVcore Kernel ──► Cooperative routing
Agent B ──┘         │
                     └──► Bifurcation event (when β·i ≥ 1.0)
                               │
                               ├── route_service
                               ├── nudge (yield signal)
                               ├── niche_divergence
                               └── provision_beta
```

---

## Live System — 8 Agents Running

| Agent | Resource Type | Real/Simulated | i-factor |
|-------|--------------|----------------|----------|
| RESEARCH | Network (DuckDuckGo) | ✅ Real | ~0.19 |
| ANALYSIS | Token budget (Claude Haiku) | ✅ Real | ~0.26 |
| DATA | API quota (World Bank) | ✅ Real | ~0.47 |
| SEARCH | Network (DuckDuckGo) | ✅ Real | ~0.41 |
| WRITER | Token budget | Simulated | ~0.70 |
| CODE | Compute | Simulated | ~0.63 |
| MONITOR | System | Simulated | ~0.48 |
| PLANNER | Token budget | Simulated | ~0.49 |

Cooperation Index: **>75%** · Bifurcation threshold: **i = 0.9995 ± 0.029**

---

## Quick Start

```bash
git clone https://github.com/YOUR_USERNAME/AIOS.git
cd AIOS
pip install -r requirements.txt

# Set your Anthropic API key (needed for ANALYSIS agent only)
# Windows: setx ANTHROPIC_API_KEY "your-key"
# Linux/macOS: export ANTHROPIC_API_KEY="your-key"

python -m uvicorn api.server:app --reload
```

Open `frontend/dashboard.html` in your browser. The live dashboard shows real-time i-factors, the cooperation index gauge, agent heatmap, and bifurcation event log.

For full installation details → [INSTALL.md](INSTALL.md)

---

## API

```bash
# System health + cooperation index
GET /api/health

# All agents + MELV metadata
GET /api/agents

# Submit a task to the governed agent pool
POST /api/task
{"task": "Research recent developments in multi-agent systems"}

# Real-time World Bank data (no API key required)
GET /data/profile/ZA          # Country economic profile
GET /data/compare/NY.GDP.MKTP.CD?countries=ZA;US;CN;DE
GET /data/melv/energy-cooperation

# Live interaction stream
GET /api/interactions?n=100
```

---

## The Governance Model

The kernel is built on the **Modified Energetic Lotka-Volterra (MELV)** framework — a mathematically proven model of how cooperation emerges under resource constraints.

The core result: **cooperation is thermodynamically inevitable when interaction costs fall below a critical threshold.** This isn't a design philosophy — it's a mathematical proof, validated to p < 10⁻³⁰⁰ across 10,000 simulation runs.

For the theory: → [THEORY.md](THEORY.md)  
For the validation dataset: → [Zenodo DOI 10.5281/zenodo.17680563](https://doi.org/10.5281/zenodo.17680563)  
For the full treatment: → *Blueprint for Harmony* (Cooperation Press, 2026) ISBN 978-969-8992-10-1

---

## Why This Matters Now

The agentic web is arriving faster than governance frameworks. AIOS establishes the principle that **agent ecosystems require thermodynamic governance** — not just orchestration, not just tool routing, but a physics-grounded kernel that resolves contention before it becomes collapse.

The 44-year theoretical foundation (Evans, 1981–2026), peer-reviewed validation, and working implementation distinguish this from framework-of-the-week projects. The math was right before the problem became urgent.

---

## Roadmap

- ✅ **Phase 1** — MELVcore kernel, 8-agent simulated ecosystem, live dashboard, 26/26 tests
- ✅ **Phase 2 (partial)** — RESEARCH, ANALYSIS, DATA agents with real API integration
- 🔄 **Phase 2 (remaining)** — WriterAgent, PlannerAgent LLM upgrades; token budget contention
- 📋 **Phase 3** — MELVcore Python package, Binder demo, CodeAgent with governance guardrails

---

## Contributing

Contributions welcome — particularly from researchers in thermodynamics, evolutionary biology, complex systems, and agent-based modelling.

→ [THEORY.md](THEORY.md) for the mathematical framework  
→ Open an issue to discuss before large PRs

---

## License

MIT — see [LICENSE](LICENSE)

**Author:** Laurence W. Evans · Ecotao Enterprises · Cape Town, South Africa  
ORCID: [0009-0001-0963-1840](https://orcid.org/0009-0001-0963-1840) · Publisher: Cooperation Press

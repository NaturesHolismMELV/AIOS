# MELVcore — Thermodynamic Governance Kernel for Multi-Agent AI Systems

> **Research prototype — v2.7.0 — Active development**  
> MELVcore is an open-source governance kernel for multi-agent AI ecosystems.  
> AIOS is the reference platform built on MELVcore.

[![Version](https://img.shields.io/badge/version-3.0.0-brightgreen.svg)](https://github.com/NaturesHolismMELV/AIOS/blob/main/melvcore/pyproject.toml)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://github.com/NaturesHolismMELV/AIOS/blob/main/LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://python.org)
[![PyPI](https://img.shields.io/badge/PyPI-melvcore-orange.svg)](https://pypi.org/project/melvcore/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20011156.svg)](https://doi.org/10.5281/zenodo.20011156)
[![ORCID](https://img.shields.io/badge/ORCID-0009--0001--0963--1840-green.svg)](https://orcid.org/0009-0001-0963-1840)

---

## The Problem

Agents are becoming economic actors — making purchases, calling APIs, consuming
compute, competing for token budgets. Coinbase ships agentic wallets. Cloudflare
makes web content agent-readable. Stripe is building agent commerce.

The governance infrastructure does not yet exist.

What happens when eight specialised agents share a token budget and three of them
need to call an LLM simultaneously? Who decides which agent yields, which waits,
which gets priority? Without a governance layer, ecosystems either collapse under
contention or rely on hardcoded priority rules that break as the agent mix changes.

MELVcore is a thermodynamic approach to that problem.

---

## What MELVcore Does

MELVcore is a **governance kernel** that monitors resource contention between
agents in real time and resolves conflicts before they cascade. It is derived from
the Modified Energetic Lotka-Volterra (MELV) framework — a theoretical programme
grounded in ecological field observations (hornbill-mongoose mutualism, Namibia
1981–83; bee-flower association, published *Nature's Holism*, iUniverse, 1999).

The core mechanism:

- Every agent interaction has a **cost** (C) and a **benefit** (B)
- The **i-factor** = C/B measures interaction efficiency
- **β** scales against resource availability in the current environment
- When **β·i < 1.0**: cooperative equilibrium — the system routes normally
- When **β·i ≥ 1.0**: bifurcation event — the kernel intervenes

Empirically derived bifurcation threshold: **i_critical = 0.9995 ± 0.029**
(R² = 0.9248, 405 ABM runs — see [Validation](#validation)).

---

## Quick Start

```bash
pip install melvcore
```

```python
from melvcore import MELVKernel, NudgeEngine, CostCalculator, integrate_agent

# 1. Create the kernel
kernel = MELVKernel()
nudge  = NudgeEngine()
calc   = CostCalculator()

# 2. Register agents (phi = evolutionary maturity; beta is kernel-managed)
profile_r = integrate_agent(kernel, "r01", "ResearchAgent",
                             domain="research", phi=0.82)
profile_w = integrate_agent(kernel, "w01", "WriterAgent",
                             domain="writing",  phi=0.71)

# 3. Record an interaction
cost = calc.compute_cost(in_tok=250, out_tok=120, latency_s=0.85,
                         task_type="RESEARCH")
rec  = kernel.record_interaction("r01", "w01",
                                  cost=cost, benefit=2.0,
                                  resource_type="token_budget")

print(rec.interaction_type)        # "cooperative"
print(kernel.cooperation_index())  # 0.82 > 0.75 target — healthy ecosystem
```

Or via the Gateway API (AIOS server running):

```python
import requests
r = requests.post("http://localhost:8000/melv/interact", json={
    "agent_a": "r01", "agent_b": "w01",
    "cost": 0.9, "benefit": 2.0,
    "phi_a": 0.82, "phi_b": 0.71,
    # beta is never in this payload — the kernel reads it from BetaEnvironment
})
print(r.json()["status"])  # "cooperative"
```

See [INSTALL.md](INSTALL.md) and [DEPLOY.md](DEPLOY.md) for full setup.

---

## Features

- **Real-time i-factor monitoring** and bifurcation detection
- **Cooperation Index (CI)** tracking against the 0.75 cooperative basin threshold
- **ε-decomposition** (intrinsic · ecosystem · architectural) — three-scalar
  adaptive plasticity with correct biological semantics
- **Quorum reliability tagging** — sigmoid gate as output reliability marker
- **Oxpecker Phase 2** — thermodynamic recycling of interrupted work fragments
- **Sandbox certification engine** — Composite Longevity Score (CLS) for agents
- **NudgeEngine** — graduated intervention ladder before hard governance actions
- **Live dashboard** and REST API (Railway deployment)
- **Seven framework adapters** — LangGraph, AutoGen, CrewAI (open-source);
  Agentforce, Copilot Studio, Vertex AI, ServiceNow (enterprise)
- **653 passing tests** across the kernel and governance layer

---

## Limitations and Current Status

MELVcore is an **active research prototype** (v2.7.0, Sessions 1–30c, May 2026).
The core kernel and ABM validation are mature. Users should be aware of the
following:

**Parameter calibration.** Several weights — tool friction category values
(agent_native=0.2, fast_rest=0.5, standard=1.0, human_bottlenecked=1.5,
legacy=2.0) and the per-agent ε Gaussian perturbation (σ=0.3) — are
theoretically grounded but not yet calibrated against large-scale production
latency or performance datasets. These are marked as epistemic status ① (stub)
in the theory documentation.

**Validation scope.** The cooperation theorem is supported by ABM simulations
(405 runs, DOI: [10.5281/zenodo.19422174](https://doi.org/10.5281/zenodo.19422174)),
a blind DeepSeek axiom reconstruction (March 2026, see [VALIDATION.md](VALIDATION.md)),
and live platform events on a ~313-agent deployment. Independent reproduction by
external teams is encouraged and has not yet occurred.

**AI-assisted development.** MELVcore has been developed using a multi-AI
consultation methodology (MAIES). Insights from Gemini, Grok, NotebookLM, and
Claude informed specific design decisions and are documented as AI Synthesis
Points A–D in [VALIDATION.md](VALIDATION.md). These are treated as hypothesis
generators rather than formal proofs, except where blind reconstruction was
performed (DeepSeek, MAIES Event 3).

**Deployment maturity.** The live Railway instance demonstrates autonomous
recovery under adversarial load but is not yet hardened for arbitrary adversarial
conditions or high-stakes production use. Security review is ongoing;
see [SECURITY.md](SECURITY.md).

**Scope.** Currently optimised for token/compute/API contention in LLM-based
agent systems. Applicability to other domains requires signal mapping — the
subject of the planned MAIES-006 investigation.

---

## Theory Summary

The MELV framework extends classical Lotka-Volterra population dynamics:

1. **Replacing fixed interaction coefficients** with dynamic i-factors computed
   from real resource flows
2. **Adding thermodynamic β scaling** connecting agent interactions to
   environmental resource availability
3. **Introducing maturity φ** as a slow state variable capturing agent learning
   and specialisation (τ_φ >> τ_interaction)
4. **Proving the cooperation theorem** — below i_critical, cooperative equilibria
   are thermodynamically inevitable, not merely probable

**The master equation:**
```
i₁₂(t) = i₁₂⁰ × (1 − ε_effective × φ(t) × β(t))
```

where `ε_effective = ε_intrinsic + ε_ecosystem` (ε_architectural is a diagnostic
boundary condition, never enters the master equation).

Full derivation: *Blueprint for Harmony* (Cooperation Press, 2026).
ISBN 978-969-8992-10-1. See also [THEORY.md](THEORY.md).

---

## Validation

| Evidence Stream | Result |
|----------------|--------|
| Cooperation Theorem | CI=1.0 confirmed live 20 April 2026. Preprint DOI: [10.5281/zenodo.19665563](https://doi.org/10.5281/zenodo.19665563) |
| ABM V2.1 (405 runs) | 100% ESS invasion recovery (34/34). Hartigan dip p≈0. r = −0.866. φ×β sensitivity = 1.0, specificity = 0.997. DOI: [10.5281/zenodo.19422174](https://doi.org/10.5281/zenodo.19422174) |
| DeepSeek blind reconstruction (Mar 2026) | 8 MELV axioms provided; master equation not given. DeepSeek independently derived β·i < 1 as the cooperation viability condition and recovered φ timescale separation. Full transcript in [VALIDATION.md](VALIDATION.md). |
| Live bifurcation demo (27 Apr 2026) | Single stress agent drove CI to trough of 0.713. Recovery to 0.830 while stress agent still active. Theorem held. |
| Live unplanned stress event (27 Apr 2026) | CI 0.503 (mean i-factor 8.149, 26 alerts) → CI 0.957 (mean i-factor 0.131, 4 alerts) in 12 minutes. No manual intervention. |
| theorem_confirmed flag | `true` in production SQLite. Set 20 April 2026, 09:00 SAST. |

See [VALIDATION.md](VALIDATION.md) for full logs, the DeepSeek transcript, and
AI Synthesis Points A–D documentation.

---

## MELVcore Sandbox — Ecosystem Certification

The Sandbox answers the question every multi-agent developer faces:

> *"Will my agent remain stable and cooperative as the ecosystem around it grows?"*

Submit any agent for a thermodynamic certification run against the MELV reference
ecosystem. The Sandbox produces a **Composite Longevity Score (CLS)** in [0, 100]:

| CLS | Verdict | Interpretation |
|-----|---------|----------------|
| ≥ 80 | **CERTIFIED** | Agent is ecosystem-neutral or beneficial |
| 60–79 | **CERTIFIED_WITH_ADVISORY** | Minor degradation — monitor specific resources |
| < 60 | **NOT_CERTIFIED** | Agent degrades ecosystem CI dynamics |

```bash
# Sandbox API endpoints
POST /sandbox/submit              # submit agent for certification
GET  /sandbox/run/{run_id}        # poll progress (0.0–1.0)
GET  /sandbox/report/{run_id}     # full CertificationReport JSON
GET  /sandbox/registry            # MELVcore Compatibility Registry
```

---

## Documentation

| File | Contents |
|------|----------|
| [THEORY.md](THEORY.md) | Mathematical foundations, axioms, master equation derivation |
| [VALIDATION.md](VALIDATION.md) | DeepSeek transcript, ABM results, live event logs, AI Synthesis Points |
| [SANDBOX.md](SANDBOX.md) | Agent certification guide |
| [INSTALL.md](INSTALL.md) | Installation and local setup |
| [DEPLOY.md](DEPLOY.md) | Railway deployment |
| [SECURITY.md](SECURITY.md) | Security policy |

---

## Architecture


**Key components:** MELVKernel (governance loop), Agent Registry (agent lifecycle management), NudgeEngine (escalation), CostCalculator (C/B proxy), three-layer telemetry (L1/L2/L3).

**API Reference:** Full OpenAPI docs at `/docs` on the live platform. Core endpoints: `POST /api/observe`, `GET /api/status`, `POST /api/telemetry/l1`, `GET /api/telemetry/l3/{agent_id}`, `POST /api/telemetry/eta_cycle`.

**Roadmap:** Session 37 — Dungbeetle formalisation + irreversibility diagnostic (v3.2.0). Session 38 — ABM Equation 7 integration. Session 40 — Canonical document v1.3 + Zenodo preprint update.

> Legacy version 1.0.0 tagged at first PyPI release; current platform is v3.1.1.
MELVcore implements the Modified Energetic Lotka-Volterra (MELV) cooperation-evolution framework as a governance kernel for multi-agent AI systems. The master equation i(t) = i₀ × (1 − ε × φ(t) × β_norm(t)) governs cooperation dynamics, with three-layer telemetry (L1/L2/L3) for φ perpetuity tracking and η saturation estimation (BI-NLS).

## Repository Structure

```
AIOS/
├── core/           # MELVKernel, BetaEnvironment, CostCalculator
├── governance/     # NudgeEngine, bifurcation detection, kernel decisions
├── melvcore/       # PyPI-publishable package
├── melvcore_mcp/   # MCP server adapter
├── adapters/       # LangGraph, AutoGen, CrewAI, Agentforce, Copilot, Vertex, ServiceNow
├── agents/         # Reference agent implementations (OXPECKER-01, etc.)
├── api/            # FastAPI Gateway (AIOS server)
├── frontend/       # Live dashboard
└── tests/          # 517+ tests
```

Development history is recorded in commit messages and the session roadmap in
[THEORY.md](THEORY.md). [SESSION15_PLAN.md](SESSION15_PLAN.md) is a legacy
planning document from the early development phase.

---

## Live Platform

The AIOS reference deployment is live at:
**[web-production-e14d1.up.railway.app](https://web-production-e14d1.up.railway.app)**

Public demo endpoints (no API key required, rate-limited):
```
POST /demo/register    # register a demo agent
POST /demo/interact    # record a demo interaction
GET  /demo/ci          # current Cooperation Index
POST /demo/purge       # clean up demo agents
```

A reproducible bifurcation demonstration can be triggered via the `/demo/`
endpoints. See [VALIDATION.md](VALIDATION.md) for the protocol.

---

## Citation

```bibtex
@software{evans2026melvcore,
  author       = {Evans, Laurence W.},
  title        = {{MELVcore}: A Thermodynamic Governance Kernel for
                  Multi-Agent {AI} Systems},
  year         = {2026},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.20011156},
  url          = {https://doi.org/10.5281/zenodo.20011156},
  orcid        = {0009-0001-0963-1840}
}
```

See also [CITATION.cff](CITATION.cff) for machine-readable citation metadata.

---

## Contributing

Issues and pull requests are welcome. For discussion of the theoretical
framework, open an issue referencing the relevant section of [THEORY.md](THEORY.md).

---

## References

- Evans, L.W. (1999). *Nature's Holism*. iUniverse.
- Evans, L.W. (2026). *Blueprint for Harmony*. Cooperation Press.
  ISBN 978-969-8992-10-1.
- Evans, L.W. (2026). MELVcore AIOS Platform. Zenodo.
  DOI: [10.5281/zenodo.20011156](https://doi.org/10.5281/zenodo.20011156)
- Evans, L.W. (2026). MELV Agent-Based Model V2.1. Zenodo.
  DOI: [10.5281/zenodo.19422174](https://doi.org/10.5281/zenodo.19422174)
- Evans, L.W. (2026). The MELV Framework — What Has Changed and Why:
  A Guide for Non-Mathematical Readers. ResearchGate. CC BY 4.0.
  DOI: [10.13140/RG.2.2.17524.10880](https://www.researchgate.net/publication/405214441)
- Nadell, C.D., Drescher, K. & Foster, K.R. (2016). Spatial structure,
  cooperation and competition in biofilms. *Nature Reviews Microbiology*,
  14, 589–600.

---

*Laurence W. Evans · ORCID: [0009-0001-0963-1840](https://orcid.org/0009-0001-0963-1840) · Cape Town, South Africa*

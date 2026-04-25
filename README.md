# MELVcore — Thermodynamic Governance Kernel for the Agentic Web

**Built on 44 years of validated theory. Nobody else has the physics.**

> AIOS is the reference platform built on MELVcore.  
> MELVcore is to AIOS as Linux is to Ubuntu.

[![Version](https://img.shields.io/badge/version-1.9.1-brightgreen.svg)](melvcore/pyproject.toml)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://python.org)
[![PyPI](https://img.shields.io/badge/PyPI-melvcore-orange.svg)](https://pypi.org/project/melvcore/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19665563.svg)](https://doi.org/10.5281/zenodo.19665563)
[![ORCID](https://img.shields.io/badge/ORCID-0009--0001--0963--1840-green.svg)](https://orcid.org/0009-0001-0963-1840)
[![ISBN](https://img.shields.io/badge/ISBN-978--969--8992--10--1-blue.svg)](https://cooperationpress.com)

---

## The Problem

Coinbase shipped agentic wallets. Cloudflare made 20% of the web readable by agents. OpenAI released shell tools. Stripe is building agent commerce. Google has a universal commerce protocol.

Agents are becoming economic actors — making purchases, calling APIs, consuming compute, competing for token budgets — and **the governance infrastructure does not exist yet.**

What happens when eight specialised agents share a token budget and three of them need to call an LLM simultaneously? What happens when a research agent and a data agent compete for the same API quota? Who decides which agent yields, which agent waits, and which agent gets priority?

Right now: nothing decides. The ecosystem either collapses under contention, or you hardcode priority rules that break the moment your agent mix changes.

**MELVcore is the answer to that question.**

---

## What is MELVcore?

MELVcore is a **thermodynamic governance kernel** built on the Modified Energetic Lotka-Volterra (MELV) framework. It monitors resource contention between agents in real time and resolves conflicts before they cascade.

The physics is simple:

- Every agent interaction has a **cost** (C) and a **benefit** (B)
- The **i-factor** = C/B measures interaction efficiency
- The **environmental suitability** β scales against resource availability
- When **β·i < 1.0**: cooperative equilibrium — the system routes normally
- When **β·i ≥ 1.0**: bifurcation event — the kernel intervenes

Validated across 10,000 simulations: **78.0% cooperative equilibria**, bifurcation threshold at **i = 0.9995 ± 0.029** (p < 10⁻³⁰⁰).

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

# 2. Register agents (phi = evolutionary maturity; never pass beta)
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

# 4. Or use the Gateway API (AIOS server running)
import requests
r = requests.post("http://localhost:8000/melv/interact", json={
    "agent_a": "r01", "agent_b": "w01",
    "cost": 0.9, "benefit": 2.0,
    "phi_a": 0.82, "phi_b": 0.71,
    # beta is NEVER in this payload — the kernel reads it from BetaEnvironment
})
print(r.json()["status"])          # "cooperative"
```

---

## Theory Summary

The MELV framework extends classical Lotka-Volterra population dynamics by:

1. **Replacing fixed interaction coefficients** with dynamic i-factors computed from real resource flows
2. **Adding thermodynamic β scaling** that connects agent interactions to environmental resource availability  
3. **Introducing maturity φ** as a state variable that captures agent learning and specialisation
4. **Proving the cooperation theorem** — below i_critical, competitive equilibria are thermodynamically unstable

Full derivation: *Blueprint for Harmony: Thermodynamic Foundations of Cooperation and Conscious Evolution* (Cooperation Press, 2026). ISBN 978-969-8992-10-1.

---


## MELVcore Sandbox — Ecosystem Certification *(Session 10 — v1.2.0)*

The **MELVcore Sandbox** answers the question every multi-agent developer now faces:

> *"Will my agent remain stable and cooperative as the ecosystem around it grows?"*

Submit any agent for a thermodynamic certification run against the 8-agent MELV reference ecosystem. The Sandbox computes three CI Dynamics metrics — CI half-life delta (Δt½), Oscillation Impact Score (OIS), and Drift Degradation Coefficient (DDC) — and produces a **Composite Longevity Score (CLS)** in [0, 100]:

| CLS | Verdict | Interpretation |
|-----|---------|----------------|
| ≥ 80 | **CERTIFIED** | Agent is ecosystem-neutral or beneficial |
| 60–79 | **CERTIFIED_WITH_ADVISORY** | Minor degradation — monitor specific resources |
| < 60 | **NOT_CERTIFIED** | Agent degrades ecosystem CI dynamics |

Every report is anchored to Zenodo DOI `10.5281/zenodo.19665563` and ORCID `0009-0001-0963-1840` — the first agent certification platform grounded in published, peer-reviewed thermodynamic theory.

```bash
# Sandbox API endpoints
POST /sandbox/submit              # submit agent for certification
GET  /sandbox/run/{run_id}        # poll progress (0.0–1.0)
GET  /sandbox/report/{run_id}     # full CertificationReport JSON
GET  /sandbox/registry            # MELVcore Compatibility Registry
```

## tanh φ Enhancement *(Session 10 — DeepSeek independent derivation)*

The φ update equation has been upgraded from a linear rule to a tanh relaxation, derived independently by DeepSeek from the eight MELV axioms alone (March 2026):

```
dφ/dt = (1/τ_φ) · [tanh(γ · mean_surplus) − φ] + η_φ(t)
```

Where `τ_φ >> τ_interaction` (Axiom 3: slow timescale separation), `γ = 2.0` (PHI_GAIN), `mean_surplus = mean of last 10 outcomes − 0.5`, and `η_φ(t) ~ N(0, 0.002)` (Axiom 8: heterogeneity). Three advantages over the previous linear rule: natural boundedness in [0,1], diminishing returns at high maturity (the giraffe constraint), and surplus memory window. DeepSeek's independent reconstruction from axioms alone is the fourth AI validation of the MELV framework.

## Dashboard11 & CI History API *(Session 11 — v1.3.0)*

Session 11 delivers the production-quality `frontend/dashboard11.html` with two key enhancements:

**CI History Chart Panel** (CI Dynamics tab)
- Rolling time-series chart of the last 200 CI readings using Chart.js
- Cooperative threshold (0.75) shown as a dashed reference line
- Live-updating every 3 seconds alongside existing CI Dynamics metrics

**Sandbox UI Polish** (Sandbox tab)
- Client-side input validation: φ must be in [0, 1], ε in [0, 8]
- Animated progress bar with status labels during certification runs
- Colour-coded verdict badge (green/amber/red) with CLS gauge bar
- Registry auto-refreshes on tab navigation; shows CLS score, domain, certified_at

**CI History API endpoint** (`api/server.py`):
```
GET /api/ci_history?n=200
```
Returns the last N CI readings as `[{t: float, ci: float}]`. Default n=200, max n=1000. Timestamps are Unix epoch seconds derived from `kernel._ci_history`.

---

## CI Dynamics Framework *(Session 9 — v1.1.0)*

MELVcore v1.1.0 introduces the **CI Dynamics Framework** — the first theoretical extensions that go beyond the original MELV equations.
The live dashboard revealed that CI can rise toward the 0.75 target and then fall back. The kernel detected individual bifurcation events but had no instrument to characterise the *trajectory* of the ecosystem.
Three new derived metrics close that gap:

| Metric | Equation | Interpretation |
|--------|----------|----------------|
| `dCI/dt` — rate of change | linear regression slope over last 10 CI readings | Instantaneous momentum toward or away from the cooperative basin |
| CI Optimisation Half-Life `t½` | `ln(2) / k`, `k = dCI_dt / gap_to_target` | Seconds to close half the remaining gap to CI = 0.75 — pharmacokinetic analogy |
| CI Drift Coefficient `δ` | linear regression slope over last 500 CI readings | Long-run trend; positive = converging, negative = degrading |

The kernel classifies ecosystem **regime** from these three metrics: `cooperative`, `converging`, `underdamped`, `diverging`, or `stasis`.

**Oscillation detection:** when CI crosses the 0.75 threshold upward and falls back below it within 60 s by ≥ 0.05, an `OscillationEvent` is recorded — peak, trough, amplitude, period, and implicated agent pairs.
This characterises whether the system is overdamped, underdamped, or divergent around the cooperative attractor, directly extending the thermodynamic vocabulary of *Blueprint for Harmony*.

```python
snapshot = kernel.ci_dynamics()
print(snapshot["regime"])               # "converging"
print(snapshot["ci_half_life_sec"])     # 48.7  — single number enterprise buyers understand
print(snapshot["dci_dt"])               # 0.000142
print(snapshot["ci_drift_coefficient"]) # 0.000031
```

REST: `GET /api/ci_dynamics`

The half-life metric is the core commercial instrument for the MELVcore Sandbox certification platform. See [SANDBOX.md](SANDBOX.md).


---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│  AIOS — Reference Platform                                          │
│                                                                     │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐           │
│  │ Research │  │ Analysis │  │  Writer  │  │ Planner  │  ...       │
│  │  Agent   │  │  Agent   │  │  Agent   │  │  Agent   │           │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘           │
│       │              │              │              │                 │
│       └──────────────┴──────────────┴──────────────┘               │
│                              │                                      │
│                    ┌─────────▼──────────┐                          │
│                    │   Gateway API      │  POST /melv/interact      │
│                    │ POST /melv/interact│  GET  /melv/costs         │
│                    └─────────┬──────────┘                          │
│                              │                                      │
│  ════════════════════════════╪═══════════════════════════════════   │
│                              │   MELVcore Kernel (this library)    │
│                    ┌─────────▼──────────┐                          │
│                    │    MELVKernel      │  i-factor monitoring      │
│                    │  record_interact() │  bifurcation detection    │
│                    │  cooperation_index │  ecosystem health         │
│                    └──────┬──────┬──────┘                          │
│                           │      │                                  │
│              ┌────────────▼─┐  ┌─▼──────────────┐                 │
│              │  NudgeEngine │  │ CostCalculator  │                 │
│              │  Nudge v2    │  │ Weighted LLM    │                 │
│              │  4-stage esc │  │ cost profiles   │                 │
│              └──────────────┘  └─────────────────┘                 │
│                                                                     │
│              ┌──────────────────────────────────────┐              │
│              │         BetaEnvironment              │              │
│              │  β per resource (kernel-owned only)  │              │
│              └──────────────────────────────────────┘              │
└─────────────────────────────────────────────────────────────────────┘
```

**MELVcore is to AIOS as Linux is to Ubuntu.**

---

## Agent Registry (AIOS v0.8.0)

| Agent | Resource | i-factor | φ | Real API | Notes |
|-------|----------|----------|---|----------|-------|
| RESEARCH | Network (DDG) | ~0.19 | 0.82 | ✓ | latency_heavy profile |
| ANALYSIS | Token budget | ~0.26 | 0.78 | ✓ | token_heavy (1.4×) |
| DATA | API quota (World Bank) | ~0.47 | 0.58 | ✓ | maturing |
| SEARCH | Network (DDG) | ~0.41 | 0.75 | ✓ | parallel search |
| WRITER | Token budget | ~0.71 | 0.71 | ✓ | balanced profile |
| PLANNER | Token budget | ~0.85 | 0.85 | ✓ | token_heavy; cap events |
| CODE | Compute | ~0.63 | 0.91 | Phase 3 | no real execution yet |
| MONITOR | System | ~0.48 | 0.95 | simulated | stays simulated |

---

## API Reference Summary

### Gateway

```
POST /melv/interact       # Record agent interaction, get bifurcation response
GET  /melv/              # Ecosystem health snapshot (CI, βi, φ, events)
GET  /melv/costs         # CostCalculator history and profile weights
GET  /melv/nudge         # NudgeEngine documentation and escalation map
```

### Python (library)

```python
MELVKernel.record_interaction(agent_a, agent_b, cost, benefit, resource_type)
MELVKernel.update_phi(agent_id, outcome_quality)
MELVKernel.cooperation_index() → float      # CI ≥ 0.75 = healthy
MELVKernel.ecosystem_health() → dict        # full snapshot

NudgeEngine.build_nudge_v2(action, beta_i, resource, contention_depth, agent_phi)
NudgeEngine.apply_oxpecker_effect(vacating_agent, resource_type, environment)

CostCalculator.compute_cost(in_tok, out_tok, latency_s, task_type) → float

MELVKernel.dci_dt() → float                 # Rate of change (CI-units/sec)
MELVKernel.ci_half_life() → float | None    # Seconds to close half gap to 0.75
MELVKernel.ci_drift_coefficient() → float   # Long-run trend slope
MELVKernel.ci_dynamics() → dict             # Full CI Dynamics snapshot
```

### Running the server

```bash
cd C:\Users\web\AIOS
python -m uvicorn api.server:app --reload
# API docs: http://localhost:8000/docs
# Dashboard: http://localhost:8000/frontend/dashboard12.html
```

---

## Roadmap

| Phase | Session | Name | Status |
|-------|---------|------|--------|
| 2 | 4 | Gateway API + MELVcore rebranding | ✅ Complete |
| 2 | 5 | Writer + Planner real Haiku LLM | ✅ Complete |
| 2 | 6 | CostCalculator — weighted cost profiles | ✅ Complete |
| 2 | 7 | Nudge v2 — four-stage bifurcation response + oxpecker Channel 2 | ✅ Complete |
| 2 | 8 | v1.0.0 Release — governance library, PyPI, Theory-to-Code | ✅ Complete |
| 2 | 9 | v1.1.0 — CI Dynamics: dCI/dt, half-life, drift, oscillation, Sandbox design | ✅ Complete |
| 3 | 10 | v1.2.0 — MELVcore Sandbox backend + tanh φ enhancement (DeepSeek validation) | ✅ Complete |
| 3 | **11** | **v1.3.0 — Dashboard11 (CI History chart, Sandbox UI polish) + CI History API** | ✅ Complete |
| 3 | **12** | **v1.4.0 — SQLite Persistence (kernel restore, sandbox registry, db_stats API) + CI gauge fix** | ✅ Complete |
| 3 | **13** | **v1.5.0 — Hosted Demo (Railway/Render), Rate Limit + API Key middleware, LangGraph adapter scaffold, public landing page** | ✅ Code complete — deployment pending |
| 3 | **14** | **v1.6.0 — MCP Server (4 tools, 2 resources, SSE + Streamable HTTP, mcp.json manifest)** | ✅ Complete |
| 3 | **15** | **v1.7.0 — Adversarial sandbox hardening, startup grace period, MCP Inspector compliance, Windows/Python 3.14 fix** | ✅ Complete |
| 3 | **16** | **v1.8.0 — φ/ε Assessment Wizard (multi-step certification wizard, structured parameter scoring)** | ✅ Complete |
| 3 | **17–21.2** | **v1.9.2 — Jones/Karpathy governance suite, PDF certification (WeasyPrint), domain profiles, LangGraph workflow certification, shared-state risk scoring, security hardening** | ✅ Complete |
| 4 | — | Railway deployment + live public URL | 🔜 Next |
| 4 | — | SaaS hosted MELVcore (Wave 2), preprint publication (TechRxiv + Zenodo v3) | Planned |

---

## Citation

If you use MELVcore in research, please cite:

```bibtex
@software{evans2026melvcore,
  author       = {Evans, L.W.},
  title        = {MELVcore: Thermodynamic Governance Kernel for the Agentic Web},
  year         = {2026},
  version      = {2.0.1},
  publisher    = {Ecotao Enterprises},
  url          = {https://github.com/NaturesHolismMELV/AIOS},
  doi          = {10.5281/zenodo.19665563},
  orcid        = {0009-0001-0963-1840},
}

@book{evans2026blueprint,
  author    = {Evans, L.W.},
  title     = {Blueprint for Harmony: Thermodynamic Foundations of Cooperation and Conscious Evolution},
  year      = {2026},
  publisher = {Cooperation Press},
  isbn      = {978-969-8992-10-1},
}
```

---

## Identity

| | |
|---|---|
| **Author** | Laurence W. Evans (pub. name) / Zaid — Ecotao Enterprises, Cape Town |
| **ORCID** | [0009-0001-0963-1840](https://orcid.org/0009-0001-0963-1840) |
| **Email** | [laurence@ecotao.co.za](mailto:laurence@ecotao.co.za) |
| **Twitter/X** | @NaturesHolism |
| **Book** | [Blueprint for Harmony](https://www.amazon.com/Blueprint-Harmony-Thermodynamic-Foundations-Cooperation-ebook/dp/B0GNLBVHWS) — Cooperation Press, 2026. ISBN 978-969-8992-10-1 |
| **Substack** | [The Thermodynamics of Life](https://naturesholism.substack.com) |
| **Zenodo DOI** | [10.5281/zenodo.19665563](https://doi.org/10.5281/zenodo.19665563) (p < 10⁻³⁰⁰) |
| **GitHub** | [github.com/NaturesHolismMELV/AIOS](https://github.com/NaturesHolismMELV/AIOS) |
| **License** | Apache 2.0 — patent protection, contributor patent grant, enterprise-friendly |

---

*Ecotao Enterprises · L.W. Evans · Cape Town, South Africa · MELVcore v1.9.2 · Session 21.2 complete · 419 tests · April 2026*

<!-- trigger rebuild -->

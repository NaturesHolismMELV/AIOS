# MELVcore Sandbox — Architecture Design

**Phase 3 Platform · Session 9 Design Document**

L.W. Evans · Ecotao Enterprises, Cape Town  
ORCID: [0009-0001-0963-1840](https://orcid.org/0009-0001-0963-1840)  
Blueprint for Harmony · Cooperation Press, 2026 · ISBN 978-969-8992-10-1

---

## 1  Purpose and Problem Statement

The Agentic AI Summit Silicon Valley (2026) identified 500+ builders from Adobe, Netflix, PayPal, and OpenAI shipping production-ready multi-agent systems.
Their stated priorities — MCP modular design, observability, and security — reflect an unmet need that no current framework addresses:

> **"Will my agent remain stable and cooperative as the ecosystem around it grows?"**

This is a thermodynamic question, not an engineering one. An agent optimised for performance in isolation may become a disruptive actor in a heterogeneous ecosystem, driving up i-factors for neighbouring agents and forcing repeated bifurcation events.

The **MELVcore Sandbox** answers this question with a principled, mathematically grounded certification protocol.

---

## 2  Market Differentiators

MELVcore Sandbox has three differentiators no competitor can replicate:

| Differentiator | What it means |
|----------------|---------------|
| **Peer-reviewed methodology** | Zenodo DOI 10.5281/zenodo.17680563, ORCID, ISBN-anchored theory (p < 10⁻³⁰⁰) |
| **Quantitative longevity score** | CI optimisation half-life as a single publishable number |
| **Formal certification output** | Suitable for regulatory and due diligence use in finance, healthcare, legal |

---

## 3  Core Concept: Ecosystem Certification

A developer submits their agent to the Sandbox for a certification run. The Sandbox:

1. Registers the submitted agent alongside a reference ecosystem of 8 standardised agents (the AIOS default ecosystem)
2. Runs a controlled simulation over a configurable duration (default: 10,000 interactions or 30 minutes)
3. Computes the three CI Dynamics metrics with the submitted agent present vs absent
4. Detects any oscillation events triggered or suppressed by the submitted agent
5. Produces a **Certification Report** with a pass/fail verdict and a set of quantitative scores

The agent *certifies* if it does not degrade the ecosystem's CI Dynamics relative to baseline.

---

## 4  Certification Metrics

### 4.1  Primary: CI Half-Life Delta (Δt½)

```
Δt½ = t½_with_agent − t½_baseline
```

- **Δt½ < 0** (negative): agent *improves* ecosystem convergence — certifies with commendation
- **Δt½ ≈ 0** (within ±10%): agent is neutral — certifies
- **Δt½ > 0** (positive, small): agent slightly degrades convergence — certifies with advisory
- **Δt½ >> 0** (positive, large): agent significantly degrades convergence — does not certify

### 4.2  Secondary: Oscillation Impact Score (OIS)

```
OIS = (osc_count_with_agent − osc_count_baseline) / max(1, osc_count_baseline)
```

Measures whether the submitted agent increases oscillation frequency in the ecosystem.

### 4.3  Tertiary: Drift Degradation Coefficient (DDC)

```
DDC = drift_with_agent − drift_baseline
```

Long-run drift slope comparison. A negative DDC means the agent causes long-run degradation.

### 4.4  Composite Longevity Score (CLS)

A single dimensionless score in [0, 100]:

```
CLS = 100 × sigmoid(−α·Δt½_norm) × (1 − β·|OIS|) × sigmoid(γ·DDC_norm)
```

Where α, β, γ are calibration constants (to be set in Phase 3 implementation).
CLS ≥ 80 → Certified. CLS 60–79 → Conditional certification. CLS < 60 → Not certified.

---

## 5  API Specification

### 5.1  Submit Agent for Certification

```
POST /sandbox/submit
```

**Request:**
```json
{
  "agent_id":    "my_agent_01",
  "agent_name":  "ResearchAssistant",
  "domain":      "research",
  "phi":         0.72,
  "epsilon":     3.5,
  "beta_pref":   1.0,
  "capabilities": ["web_search", "summarise", "cite"],
  "run_duration_interactions": 10000,
  "run_duration_seconds": null
}
```

**Response:**
```json
{
  "run_id":    "RUN-20260303-001",
  "status":    "queued",
  "eta_sec":   45
}
```

---

### 5.2  Get Certification Run Status

```
GET /sandbox/run/{run_id}
```

**Response (running):**
```json
{
  "run_id":     "RUN-20260303-001",
  "status":     "running",
  "progress":   0.42,
  "ci_current": 0.681
}
```

**Response (complete):**
```json
{
  "run_id":    "RUN-20260303-001",
  "status":    "complete",
  "report_url": "/sandbox/report/RUN-20260303-001"
}
```

---

### 5.3  Get Certification Report

```
GET /sandbox/report/{run_id}
```

**Response:**
```json
{
  "run_id":        "RUN-20260303-001",
  "agent_id":      "my_agent_01",
  "timestamp":     1741042800,
  "verdict":       "CERTIFIED",
  "cls_score":     84.2,

  "baseline": {
    "ci_half_life_sec":     22.1,
    "ci_drift_coefficient": 0.000041,
    "oscillation_count":    3,
    "final_ci":             0.793,
    "regime":               "cooperative"
  },

  "with_agent": {
    "ci_half_life_sec":     24.8,
    "ci_drift_coefficient": 0.000038,
    "oscillation_count":    4,
    "final_ci":             0.781,
    "regime":               "cooperative"
  },

  "delta_metrics": {
    "delta_half_life_sec":     2.7,
    "delta_half_life_pct":     12.2,
    "oscillation_impact_score": 0.33,
    "drift_degradation_coeff":  -0.000003
  },

  "narrative": "Agent my_agent_01 (ResearchAssistant) introduces moderate half-life degradation (+12.2%) and one additional oscillation event relative to baseline. The ecosystem remains in cooperative regime throughout. Certified with advisory: monitor token_budget contention under sustained load.",

  "implicated_resources": ["token_budget", "api_quota"],
  "advisory": "Consider reducing epsilon (adaptive plasticity) from 3.5 to 2.8 to reduce oscillation induction.",

  "certification_anchor": {
    "framework":   "MELVcore v1.1.0",
    "theory":      "Blueprint for Harmony, Cooperation Press 2026, ISBN 978-969-8992-10-1",
    "validation":  "Zenodo DOI 10.5281/zenodo.17680563",
    "author_orcid": "0009-0001-0963-1840"
  }
}
```

---

### 5.4  List Certified Agents (Public Registry)

```
GET /sandbox/registry
```

Returns a public list of all certified agents — the MELVcore Compatibility Registry.
This is the commercial moat: a publicly visible trust signal that enterprise buyers can query before adopting an agent.

---

## 6  Reference Ecosystem

The Sandbox baseline uses the 8-agent AIOS default ecosystem (Sessions 1–9):

| Agent | Domain | φ | Resource Profile |
|-------|--------|----|-----------------|
| RESEARCH | research | 0.82 | network / latency-heavy |
| ANALYSIS | analysis | 0.78 | token_budget / token-heavy |
| DATA | data | 0.58 | api_quota / maturing |
| SEARCH | search | 0.75 | network / parallel |
| WRITER | writing | 0.71 | token_budget / balanced |
| PLANNER | planning | 0.85 | token_budget / token-heavy |
| CODE | coding | 0.91 | compute / phase 3 |
| MONITOR | system | 0.95 | system / simulated |

Submitted agents interact with this ecosystem over the run duration.
The reference ecosystem parameters are frozen at certification time and versioned with the Sandbox release.

---

## 7  Regime Classification in Certification

The report classifies the submitted agent's *contribution to regime* rather than just reporting the final regime:

| Regime Transition | Description | Advisory |
|-------------------|-------------|----------|
| cooperative → cooperative | Agent is ecosystem-neutral | None |
| converging → cooperative | Agent *accelerates* convergence | Commendation |
| converging → converging | Agent is neutral during approach | None |
| cooperative → underdamped | Agent induces oscillation | Monitor |
| converging → diverging | Agent *reverses* trajectory | Fail |
| any → stasis | Agent arrests ecosystem momentum | Advisory |

---

## 8  Stability Classification (Thermodynamic)

The Sandbox report includes a stability classification derived from the oscillation count and half-life:

| Class | Criterion | Interpretation |
|-------|-----------|----------------|
| **Overdamped** | zero oscillations, Δt½ ≈ 0 | Monotonic convergence — highest stability |
| **Critically damped** | ≤ 1 oscillation, Δt½ < 10% | Near-optimal convergence |
| **Underdamped** | 2–5 oscillations, converges | Oscillating but self-correcting |
| **Undamped** | > 5 oscillations, CI persists | Persistent oscillation — requires advisory |
| **Divergent** | negative drift, CI < 0.60 | Fails certification |

This language maps directly to the thermodynamic vocabulary of Chapter 12 of *Blueprint for Harmony* ("Damped Cooperation Systems") and provides the technical grounding for the certification claim.

---

## 9  Revenue Model

### Wave 1 — Developer Tier (Freemium)

- **Free**: 3 certification runs per month, public registry listing, basic report
- **$49/month**: unlimited runs, private reports, email delivery, CI history export
- **$199/month**: CI history API access, custom reference ecosystems, priority queue

### Wave 2 — Enterprise Tier

- **$2,000/year per agent**: formal certification letter on Ecotao letterhead, Zenodo-anchored report hash, named for regulatory / due diligence use
- **$10,000/year**: enterprise license — unlimited agents, white-label report, SLA

### Wave 3 — Ecosystem Certification

- Certification of entire multi-agent pipelines (LangGraph, CrewAI, AutoGen)
- CI Dynamics API integration into existing observability stacks (Prometheus, Datadog)
- Academic partnership: cite MELVcore Sandbox in published AI governance papers

---

## 10  Implementation Phases

| Phase | Session | Deliverable |
|-------|---------|-------------|
| Design | **9 (current)** | This architecture document + API specification |
| Infrastructure | 10 | `api/sandbox_router.py` — submit/status/report endpoints; `core/sandbox_engine.py` — CertificationRun, run isolation, baseline comparison |
| UI | 11 | Sandbox tab in dashboard9 → dashboard10; run progress live polling; report download |
| Public registry | 12 | Public-facing `/sandbox/registry` endpoint; MELVcore Compatibility Registry page; GitHub Pages |
| Monetisation | 13 | Stripe integration for Developer tier; PDF certification letter generation |

---

## 11  Connection to MELV Theory

The Sandbox is not an engineering artefact — it is a direct operationalisation of MELV Axiom 7:

> *"Below the critical interaction cost threshold i_critical, cooperation is not merely possible — it is thermodynamically inevitable."*

The Sandbox answers the inverse question: *given an agent with specific φ, ε, and domain profile, does its presence keep the ecosystem below i_critical?*

The CI half-life quantifies this as a measurable timescale.
The oscillation count quantifies how often the ecosystem is pushed above and falls back below the cooperative threshold.
The drift coefficient quantifies whether the agent is contributing to the long-run convergence that Axiom 7 predicts.

This makes MELVcore Sandbox the first agent certification platform grounded in a published, peer-reviewed thermodynamic theory — not a benchmarking heuristic.

---

## 12  Identity Block

```
MELVcore Sandbox — Architecture Design Document
Session 9 · Ecotao Enterprises · L.W. Evans · Cape Town · March 2026
ORCID: 0009-0001-0963-1840
Blueprint for Harmony · Cooperation Press, 2026 · ISBN 978-969-8992-10-1
Zenodo DOI: 10.5281/zenodo.17680563
```

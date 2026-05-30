---
name: melvcore
description: >
  Certify AI agents for cooperative stability before deploying them to production.
  MELVcore applies the Modified Energetic Lotka-Volterra (MELV) thermodynamic
  framework to predict whether an agent will remain cooperative as ecosystem
  competition grows. Use this skill when you need to: assess φ (cooperation
  level), ε (ecosystem sensitivity), and β (resource preference) for any agent;
  understand coordination overhead risk; download a PDF certification report;
  or interpret MELV constants (bifurcation threshold, cooperation basin,
  cooperation threshold). Do NOT use for general AI benchmarking, performance
  testing, or latency measurement — MELVcore certifies cooperative stability,
  not task performance.
version: "1.9.1"
author: "L.W. Evans | ORCID: 0009-0001-0963-1840 | Cape Town, South Africa"
orcid: "0009-0001-0963-1840"
zenodo_doi: "10.5281/zenodo.19029077"
concept_doi: "10.5281/zenodo.17535157"
isbn: "978-969-8992-10-1"
github: "https://github.com/NaturesHolismMELV/AIOS"
license: "Apache-2.0"
---

# MELVcore Certification Skill

**MELVcore** is the reference implementation of the Modified Energetic Lotka-Volterra
(MELV) framework — a thermodynamic model that predicts cooperative stability in
multi-agent AI systems.

> *"The same physics that governs ecological cooperation governs AI agent ecosystems."*
> — Blueprint for Harmony, L.W. Evans (Cooperation Press, 2026)

---

## MELV Master Equation

```
i(t) = i⁰ × (1 − ε × φ(t) × β(t))
```

| Parameter | Symbol | Range    | Meaning                                                        |
|-----------|--------|----------|----------------------------------------------------------------|
| i⁰        | i⁰     | 0–1      | Initial interaction intensity (baseline cooperation rate)       |
| ε         | ε      | 0–8      | Ecosystem sensitivity — how strongly the agent responds to competition |
| φ         | φ      | 0–1      | Cooperation level — fraction of interactions that are cooperative |
| β         | β      | 0–2      | Resource preference — appetite for contested resources          |

### Cooperation Threshold

An agent is cooperative when:

```
(C × TAX) / β < 0.50
```

where **C** = competition intensity, **TAX** = resource taxation coefficient.

---

## MELV Constants (v1.9.0)

| Constant                  | Value              | Source                        |
|---------------------------|--------------------|-------------------------------|
| Bifurcation threshold     | i = 0.9995 ± 0.029 | Zenodo 10.5281/zenodo.19029077 |
| Cooperation basin target  | CI ≥ 0.75          | Blueprint for Harmony §4.2    |
| Cooperative basin split   | 78.0% / 16.2% / 5.8% | Validated DeepSeek replication |
| Service coupling R²       | 0.9248             | Validated preprint             |
| Cooperation correlation   | r = −0.944         | Validated preprint             |

---

## φ Lifecycle Tiers

φ determines how an agent's cooperative context ages across interactions.

| Tier       | φ Range   | Meaning                                           | Advisory                                        |
|------------|-----------|---------------------------------------------------|-------------------------------------------------|
| Permanent  | φ ≥ 0.85  | Deep evergreen context — stable long-term memory  | None — optimal tier                             |
| Working    | 0.50–0.85 | Active project context — updates with interaction | Use `record_interaction()` regularly            |
| Ephemeral  | φ < 0.50  | Session-scoped — memory decays between sessions   | Implement external state persistence; high risk |

**Reference:** Jones/Murag (2026) Principle 2 — φ lifecycle governs memory retention
in deployed agent systems. Ephemeral agents must use external state management or
their cooperation index will degrade over time.

---

## Coordination Overhead (CO) Score

CO Score = ε × tool_count (Jones 2026)

| Band     | CO Score  | Action                                                        |
|----------|-----------|---------------------------------------------------------------|
| LOW      | < 2.0     | No action required                                            |
| MODERATE | 2.0 – 4.0 | Monitor tool interaction patterns under concurrent load       |
| HIGH     | > 4.0     | Reduce tool set to ≤ 5 core tools OR lower ε to < 3.0        |

HIGH CO triggers a deployment advisory. In `financial_services` and `healthcare`
domain profiles, HIGH CO is treated as NOT_CERTIFIED.

---

## Category-Default Parameter Profiles

When you don't have measured φ/ε values, use these as starting estimates
based on agent category:

| Category          | φ target | ε target | Tool count | Notes                              |
|-------------------|----------|----------|------------|------------------------------------|
| Iterative Loop    | 0.45     | 5.5      | 20         | AutoResearch-style; HIGH CO expected |
| Autonomous        | 0.55     | 4.5      | 10         | Extended autonomy; monitor closely  |
| Multi-Agent       | 0.65     | 3.5      | 8          | Coordination risk at scale          |
| Tool-Using (ReAct)| 0.70     | 3.0      | 5          | ReAct pattern; moderate overhead    |
| Task Executor     | 0.80     | 2.0      | 2          | Constrained; low overhead           |
| Simple/Reactive   | 0.85     | 1.5      | 0          | Permanent tier; lowest risk         |

---

## API Walkthrough

MELVcore exposes a REST API at `http://localhost:8000` (dev) or your deployment URL.

### Step 1 — Submit for certification

```http
POST /sandbox/submit
Content-Type: application/json

{
  "agent_id":   "my-agent-001",
  "agent_name": "MyResearchAgent",
  "domain":     "research",
  "phi":         0.65,
  "epsilon":     3.5,
  "beta_pref":   1.0,
  "capabilities": ["search", "summarise"],
  "run_duration_interactions": 500,
  "tool_count":  8,
  "operation_mode": "episodic",
  "shared_state": "none"
}
```

Response:
```json
{
  "run_id": "abc123",
  "status": "queued"
}
```

#### Optional: domain_profile field

Pass `"domain_profile": "financial_services"` (or `"healthcare"` or
`"autonomous_research"`) to apply stricter or relaxed thresholds:

```json
{
  "agent_id":      "fin-agent-01",
  "domain_profile": "financial_services",
  ...
}
```

See domain profile table in the next section.

---

### Step 2 — Poll until complete

```http
GET /sandbox/run/{run_id}
```

Response while running:
```json
{ "run_id": "abc123", "status": "running", "progress": 0.42 }
```

Response when complete:
```json
{ "run_id": "abc123", "status": "complete" }
```

Poll every 0.5–1 second. Typical run time: < 2 seconds.

---

### Step 3 — Retrieve certification report (JSON)

```http
GET /sandbox/certify/{run_id}
```

Key fields in the response:

| Field                    | Description                                                    |
|--------------------------|----------------------------------------------------------------|
| `verdict`                | `CERTIFIED` / `CERTIFIED_WITH_ADVISORY` / `NOT_CERTIFIED`      |
| `cls_score`              | Cooperative Lifecycle Score (0–100); ≥80 = CERTIFIED          |
| `delta_half_life_sec`    | Predicted cooperation half-life (seconds of interaction time)  |
| `oscillation_impact_score` | OIS; < 0 = dampening (good), > 0 = amplifying (risk)        |
| `drift_degradation_coeff`  | DDC; < 0 = convergent (good), > 0 = divergent (risk)        |
| `coordination_overhead`  | CO score, band, advisory (if tool_count > 0)                  |
| `phi_lifecycle`          | Tier, label, advisory                                          |
| `advisory`               | Human-readable deployment guidance                             |
| `certification_anchor`   | DOI, ORCID, ISBN, sandbox_version, certified_at               |

---

### Step 4 — Download PDF certification report

```http
GET /sandbox/cert/{run_id}/pdf
```

Returns `application/pdf`. The report contains:
- Verdict banner with CLS bar
- φ lifecycle badge (colour-coded)
- CO score badge
- Top-3 ε risk parameter bars
- MELV master equation with parameter values
- Advisory text
- Certification anchor with both DOIs

Save to file:
```python
import httpx

r = httpx.get(f"http://localhost:8000/sandbox/cert/{run_id}/pdf")
with open("certification_report.pdf", "wb") as f:
    f.write(r.content)
```

---

## Verdict Interpretation

| Verdict                   | CLS Score | Meaning                                            | Action                              |
|---------------------------|-----------|----------------------------------------------------|-------------------------------------|
| CERTIFIED                 | ≥ 80      | Agent is cooperatively stable                      | Deploy with standard monitoring     |
| CERTIFIED_WITH_ADVISORY   | 50–79     | Stable with caveats                                | Review advisory before deploying    |
| NOT_CERTIFIED             | < 50      | Agent poses ecosystem destabilisation risk         | Do not deploy; reduce φ/ε conflicts |

---

## Domain Profiles

MELVcore supports domain-specific certification thresholds via the optional
`domain_profile` field in submit requests.

| Profile               | Key Changes                                                              |
|-----------------------|--------------------------------------------------------------------------|
| `financial_services`  | φ ≥ 0.70 required; CO HIGH → NOT_CERTIFIED; β bounds tightened           |
| `healthcare`          | φ ≥ 0.75 required; autonomous operation_mode blocked; CO HIGH = NOT_CERTIFIED |
| `autonomous_research` | iterative_loop assumed; tool_count defaults to 20; CO HIGH threshold relaxed to > 5.0 |

Omitting `domain_profile` uses the standard MELV thresholds (CLS ≥ 80 = CERTIFIED).

---

## Quick Reference — curl Examples

```bash
# Submit
curl -s -X POST http://localhost:8000/sandbox/submit \
  -H "Content-Type: application/json" \
  -d '{"agent_id":"test-01","agent_name":"TestAgent","domain":"coding",
       "phi":0.70,"epsilon":3.0,"beta_pref":1.0,"tool_count":5,
       "run_duration_interactions":50}' | python -m json.tool

# Poll
RUN_ID=<from above>
curl -s http://localhost:8000/sandbox/run/$RUN_ID | python -m json.tool

# Certify (JSON)
curl -s http://localhost:8000/sandbox/certify/$RUN_ID | python -m json.tool

# Download PDF
curl -s http://localhost:8000/sandbox/cert/$RUN_ID/pdf -o report.pdf
echo "Saved report.pdf"
```

---

## Python Helper

```python
import time
import httpx

def certify_agent(base_url: str, agent_id: str, agent_name: str,
                  phi: float, epsilon: float, domain: str = "general",
                  tool_count: int = 0, domain_profile: str | None = None) -> dict:
    """Submit, poll, and return the certification report."""
    payload = {
        "agent_id":   agent_id,
        "agent_name": agent_name,
        "domain":     domain,
        "phi":         phi,
        "epsilon":     epsilon,
        "beta_pref":   1.0,
        "tool_count":  tool_count,
        "run_duration_interactions": 500,
    }
    if domain_profile:
        payload["domain_profile"] = domain_profile

    with httpx.Client() as client:
        r = client.post(f"{base_url}/sandbox/submit", json=payload, timeout=10)
        r.raise_for_status()
        run_id = r.json()["run_id"]

        for _ in range(60):
            time.sleep(0.5)
            status_r = client.get(f"{base_url}/sandbox/run/{run_id}", timeout=5)
            if status_r.json().get("status") == "complete":
                break

        cert_r = client.get(f"{base_url}/sandbox/certify/{run_id}", timeout=10)
        cert_r.raise_for_status()
        return cert_r.json()


# Example
report = certify_agent(
    base_url="http://localhost:8000",
    agent_id="research-agent-01",
    agent_name="ResearchAgent",
    phi=0.65,
    epsilon=3.5,
    tool_count=8,
)
print(f"Verdict: {report['verdict']}")
print(f"CLS:     {report['cls_score']:.1f}")
print(f"Advisory: {report.get('advisory','')}")
```

---

## Starting the MELVcore Server

From the AIOS root directory (`C:\Users\web\AIOS` on Windows):

```bash
python -m uvicorn api.server:app --reload
```

Default port: 8000. Dashboard at http://localhost:8000/dashboard.
Interactive certification wizard at http://localhost:8000/demo.

---

## Certification Anchor

All certifications issued by MELVcore reference:

| Item             | Value                                  |
|------------------|----------------------------------------|
| Framework        | MELV — Modified Energetic Lotka-Volterra |
| Preprint DOI     | 10.5281/zenodo.19029077                |
| Concept DOI      | 10.5281/zenodo.17535157                |
| Author ORCID     | 0009-0001-0963-1840                    |
| Book ISBN        | 978-969-8992-10-1                      |
| GitHub           | https://github.com/NaturesHolismMELV/AIOS |
| License          | Apache-2.0                             |

---

## Validation Streams

MELVcore v1.9.0 is validated by two independent streams:

1. **DeepSeek independent replication** — axiomatic reconstruction of the MELV
   master equation from first principles, bifurcation threshold confirmed at
   i = 0.9995 ± 0.029.

2. **Karpathy alignment** — MELV's Iterative Loop category maps to Karpathy's
   AutoResearch framework; CO score predicts the coordination collapse he
   observes empirically. MELV provides the theoretical substrate; practitioners
   are confirming it empirically.

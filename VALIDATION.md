# VALIDATION.md — MELVcore Validation Archive

**v2.7.0 · Sessions 1–30c · May 2026**  
**L.W. Evans · ORCID: 0009-0001-0963-1840 · Cape Town, South Africa**

This document archives the validation evidence for the MELVcore platform and
the MELV theoretical framework. It distinguishes between structured independent
validation tests and AI Synthesis Points (convergent design insights from
multi-AI consultation).

---

## 1. ABM V2.1 — Agent-Based Model Validation

**DOI:** [10.5281/zenodo.19422174](https://doi.org/10.5281/zenodo.19422174)  
**Published:** April 2026  
**Epistemic status:** ④ verified

405 simulation runs. Key results:

| Metric | Result |
|--------|--------|
| ESS invasion recovery | 100% (34/34 runs) |
| Hartigan dip test | p ≈ 0 (bimodal CI distribution confirmed) |
| φ×β vs CI correlation | r = −0.866 |
| φ×β sensitivity | 1.0 |
| φ×β specificity | 0.997 |
| Bifurcation boundary | 9% THRESH zone confirmed |
| i_critical (empirical) | 0.9995 ± 0.029 (R² = 0.9248) |

The ABM implements sigmoid Allee effect, ESS mutant invasion tests, and
bidirectional linking to the MELVcore preprint (DOI: 10.5281/zenodo.19029077).

---

## 2. DeepSeek Blind Axiom Reconstruction — March 2026

**Epistemic status:** ④ verified (structured independent test)  
**Classification:** MAIES Event 3

### Test Protocol

DeepSeek R1 was provided the eight MELV axioms. It was not given:
- The master equation: i₁₂(t) = i₁₂⁰ × (1 − ε × φ(t) × β(t))
- The cooperation threshold: β·i < 1
- Any other framework outputs or equations

**Prompt (exact):**
> "Create a mathematical formula for interacting, adaptive entities based on
> these axioms."

The eight axioms were provided as the complete input.

### DeepSeek Response — Key Results

DeepSeek independently derived:

| MELV Construct | DeepSeek Derivation | Convergence |
|---------------|---------------------|-------------|
| Cooperation threshold β·i < 1 | Viability condition: β_{ij,E} · C_{ij}/B_{ij} < 1 | ✅ Structural equivalence |
| φ timescale separation | τ_φ >> τ_interaction (Axiom 3: slow optimisation capacity evolution) | ✅ Exact |
| Bounded adaptation rate (ε) | \|dx_i/dt\| ≤ ε_max < ∞ from Axiom 2 | ✅ Exact |
| Emergent cooperation | Nash equilibrium without global ∇Σ = 0 — no global objective function | ✅ Structural equivalence |
| Perturbation term | η(t) with Var(x) > 0, Axiom 8 | ✅ Exact |

**Full dynamical system derived by DeepSeek:**

```
dx_i/dt = min(ε_max, α_i(φ_i, β_{ij,E}) · ((B_ij - C_ij)/(B_ij + C_ij)) · x_i) + η_i(t)

dφ_i/dt = (1/τ_φ) · [tanh(γ · ∫_{t-δ}^{t} (B_ij - C_ij)dτ) - φ_i] + η_φ(t)

where τ_φ >> τ_interaction

Viability condition: β_{ij,E} · C_{ij}/B_{ij} < 1  →  β·i < 1
```

### Honest Assessment

**What converges:** The cooperation threshold β·i < 1 emerges from the axioms
alone without being given it. The φ timescale separation and bounded ε are
recovered exactly. The emergence condition (Nash equilibrium without global
objective) is structurally equivalent.

**What diverges:** DeepSeek's formulation uses a differential equation form;
MELV uses a ratio form. The convergence is on the **threshold condition**, not
the full equation form. This is expected — different mathematical formalisms
can encode the same physical constraint.

**Significance:** The 8 axioms are sufficiently constrained that an independent
system recovers the cooperation threshold without being given it. β·i < 1 is
not an assumption of the framework — it is derivable from the axioms.

---

## 3. Live Platform — Cooperation Theorem Confirmation

**theorem_confirmed:** `true` in production SQLite  
**Date confirmed:** 20 April 2026, 09:00 SAST  
**Epistemic status:** ③ verified

CI reached 1.0 in the live deployed system. The `theorem_confirmed` flag was
set by the kernel. The cooperative basin is not merely a simulation result —
it has been reached in the live 300+ agent ecosystem.

---

## 4. Live Platform — Designed Bifurcation Demonstration

**Date:** 27 April 2026  
**Epistemic status:** ③ verified  
**Reproducible:** Yes — via `/demo/` endpoints (see below)

### Protocol

A single adversarial stress agent was registered with the following profile:

```
φ = 0.3  (low maturity)
ε = 7.5  (high plasticity — exploitative)
cost = 9.5, benefit = 0.5
i-factor = 19.0
β·i = 57.0  (114× the cooperative threshold of 0.5)
```

15 interactions were recorded against the live ecosystem (313 cooperative agents).

### Results

| Interaction | CI | Kernel Action | Significance |
|------------|-----|--------------|--------------|
| Baseline | 0.853 | — | Healthy cooperative ecosystem |
| 1 | 0.777 | provision_beta + niche_diverge | Immediate bifurcation detection |
| 2–3 | 0.768→0.755 | niche_diverge (escalating) | Approaching threshold |
| **4 (trough)** | **0.713** | provision_beta + niche_diverge | **Below 0.75 — theorem tested** |
| 5 | 0.767 | niche_diverge | **Recovery while stress agent still firing** |
| 6 | 0.830 | niche_diverge | Cooperative basin reasserting |
| 7–12 | 0.805–0.819 | niche_diverge | Stable above threshold under continuous stress |
| Post-demo | 0.800 | Agent auto-deregistered | Stress agent leaves no trace |

**Verbatim kernel rationale at CI trough (Interaction 4):**
> "Long-term structural adaptation required. Route toward 'data_retrieval' —
> lower contention, higher β. Thermodynamic equivalent: giraffe evolving a
> longer neck to access the uncontested acacia crown."

### Reproducing This Result

The `/demo/` endpoints support public reproduction (no API key, rate-limited):

```bash
# 1. Register a stress agent
POST /demo/register
{
  "agent_id": "stress_01",
  "agent_type": "ANALYSIS",
  "phi": 0.3,
  "epsilon": 7.5
}

# 2. Record adversarial interactions (repeat ~15 times)
POST /demo/interact
{
  "agent_a": "stress_01",
  "agent_b": "<any_registered_agent>",
  "cost": 9.5,
  "benefit": 0.5,
  "resource_type": "token_budget"
}

# 3. Monitor CI
GET /demo/ci

# 4. Clean up
POST /demo/purge
```

Rate limit: 1 session per IP per 10 minutes, 20 interactions per session.

---

## 5. Live Platform — Unplanned Stress Event

**Date:** 27 April 2026  
**Epistemic status:** ③ verified (real adversarial conditions, not designed test)

Accumulated demo stress agents from multiple test sessions caused genuine
ecosystem degradation with no manual monitoring. Autonomous kernel recovery:

| Time (approx) | CI | Mean i-factor | Ecosystem State |
|--------------|-----|--------------|-----------------|
| 12:20 | 0.602 | 4.737 | 19 alerts. OXPECKER-01 generating βi = 17.870. |
| 12:34 | 0.503 | 8.149 | 26 alerts. Entire heatmap red. |
| **12:46** | **0.957** | **0.131** | **4 alerts. Full autonomous recovery.** |

CI 0.503 → CI 0.957 in 12 minutes. No manual intervention. The kernel's
compound response — β provisioned (+0.050 per resource) plus permanent niche
tags on the worst-performing pairs — restructured the interaction network.
The cooperative density of 308 genuine agents then diluted the remaining
perturbation signal.

**Significance:** The designed demo confirms the theorem under controlled
conditions. The unplanned event confirms it under real adversarial conditions —
multiple bad actors, no human monitoring the situation. Both outcomes are
consistent with the theorem's claim that recovery is thermodynamically
inevitable below i_critical.

---

## 6. AI Synthesis Points (ASPs) — Convergent Design Insights

The following AI Synthesis Points arose during multi-AI development consultation
and informed specific implementation decisions. They are **not** structured
independent validations — inputs were not controlled and independence cannot
be claimed. Their value is convergent reasoning from multiple analytical
perspectives that raised confidence in specific design directions before
implementation.

| ASP | System | Insight | Design Decision | Session |
|-----|--------|---------|-----------------|---------|
| A | NotebookLM | Oxpecker-giraffe mutualism as analogue of bifurcation-state recovery: one organism's waste is another's resource. | OXPECKER-01 agent implemented as dedicated recycling agent for interrupted work fragments. | S27 |
| B | Gemini | MELV sigmoid function is structurally equivalent to bacterial quorum sensing activation (Nadell et al. 2016). | Sigmoid quorum gate formalised in running code. Form confirmed as biologically grounded. | S25 |
| C | Gemini + Claude | Quorum gate value is an output reliability marker, not merely a governance threshold. Below-quorum regime is high-noise. | Quorum reliability tagging — regime field and reliability_advisory on kernel responses. | S28 |
| D | Grok + Gemini + Evans (biological) | ε_architectural as fixed boundary condition (thermal resistance model) vs ε_ecosystem as dynamic governance term. Triple convergence from impedance-model, energetic-thermodynamic, and biological derivation. | ε_architectural adopted as diagnostic-only scalar, never entering master equation. Policy cap on β provisioning when ε_architectural > 3.0. | S29 |

**Note on ASP attribution:** ASP C was originally attributed to NotebookLM +
Claude in internal documentation. The correct attribution is Gemini + Claude.
This correction is recorded here for accuracy.

---

## 7. Epistemic Status Framework

MELVcore uses a four-level epistemic status system throughout its documentation:

| Level | Symbol | Meaning |
|-------|--------|---------|
| Empirically verified, reproducible | ④ | ABM results, assessment pipeline, theorem_confirmed flag |
| Live platform verified | ③ | Implemented, tested, live. Results observed but not yet independently reproduced. |
| Theoretically grounded | ② | Sound argument, not yet empirically validated |
| Stub / not yet calibrated | ① | Theoretically ordered, pending empirical calibration |

Claims marked ① or ② should not be treated as empirically established. Claims
marked ③ have been observed in the live system and are reproducible via the
`/demo/` endpoints or test suite. Claims marked ④ have been independently
validated (ABM, DeepSeek).

---

## 8. Test Suite Summary

**517+ tests passing** as of v2.7.0 (Sessions 1–30c).

```bash
# Run the full test suite
pytest tests/ -v

# Run kernel tests only
pytest tests/test_kernel/ -v

# Run governance tests only
pytest tests/test_governance/ -v
```

Key test categories: kernel mechanics, bifurcation detection, ε decomposition,
quorum gate, Oxpecker recycling, sandbox certification, API endpoints.

---

*VALIDATION.md · MELVcore v2.7.0 · May 2026*  
*L.W. Evans · ORCID: 0009-0001-0963-1840 · Cape Town, South Africa*

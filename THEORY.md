# THEORY.md — The MELV Framework

**Modified Energetic Lotka-Volterra (MELV)**  
*Mathematical foundations of cooperation in multi-agent systems*

L.W. Evans | ORCID: 0009-0001-0963-1840 | Cape Town, South Africa  
ORCID: [0009-0001-0963-1840](https://orcid.org/0009-0001-0963-1840)

---

## Origin

The MELV framework began with ecological observations in Namibia (1981–1983), where the author observed cooperative patterns between species that gain mutual benefit through coordinated behaviour despite occupying different ecological niches.

The core question: *under what conditions does cooperation become thermodynamically inevitable, rather than merely possible?*

First published conceptually in *Nature's Holism* (Evans, 1999). Formalised mathematically 2024–2026 through computational validation. Full treatment: *Blueprint for Harmony: Thermodynamic Foundations of Cooperation and Conscious Evolution* (Cooperation Press, 2026). ISBN 978-969-8992-10-1.

---

## The Seven Axioms

**Axiom 1 — Energy Conservation**  
All agent interactions involve energy exchange. No interaction is cost-free.

**Axiom 2 — Interaction Cost**  
The interaction cost `i` measures the energetic burden of an agent pair's exchange, normalised to [0, ∞). Low `i` → cheap interaction. High `i` → expensive interaction.

**Axiom 3 — Environmental Suitability**  
The environmental parameter `β` scales the interaction cost against available resources. β > 1 indicates resource scarcity; β < 1 indicates abundance.

**Axiom 4 — Bifurcation Condition**  
When `β·i ≥ 1.0`, the system crosses a bifurcation threshold. The interaction becomes energetically unsustainable and the system must reorganise.

**Axiom 5 — Cooperative Equilibrium**  
When `β·i < 1.0` for all agent pairs, the system settles into cooperative equilibrium — the thermodynamically preferred state.

**Axiom 6 — Maturity Dependence**  
Agent maturity `φ ∈ [0,1]` modulates interaction efficiency. Higher-maturity agents extract more benefit from the same interaction cost, lowering their effective `i`.

**Axiom 7 — Inevitability of Cooperation**  
Below the critical interaction cost threshold `i_critical`, cooperation is not merely possible — it is thermodynamically inevitable. The system cannot sustain competitive equilibria below this threshold.

---

## Core Equations

### Interaction Cost
```
i_AB = C_AB / B_AB
```

Where cost and benefit are measured in domain-appropriate units:
- Network agents: latency (cost), result quality (benefit)
- LLM agents: token cost (cost), structural quality of output (benefit)  
- Data agents: HTTP latency + response size (cost), data completeness (benefit)

### Effective Interaction Cost
```
β·i < 1.0  →  cooperative equilibrium
β·i ≥ 1.0  →  bifurcation required
```

### Cooperation Index
```
CI = 1 − mean(β·i across all agent pairs)
Target: CI ≥ 0.75
```

### Adaptive Dynamics (φ)
```
dφ/dt = ε(φ − φ_target) + η
```
Where η ~ N(0, σ) is stochastic perturbation (Axiom 8: necessary heterogeneity).

### Service Coupling (Omega)
```
β_svc = λ_max(Ω) / n
```
Where Ω is the weighted interaction adjacency matrix and λ_max its leading eigenvalue.

### LLM Cost Formula (locked Session 4)
```
token_cost = in_tok × 0.0000008 + out_tok × 0.000004
raw_cost   = token_cost × 1000 × w_token + latency × 0.1 × w_latency
cost       = min(2.0, raw_cost)
```

### Nudge Escalation (Nudge v2 — Session 7)
```
depth → {1: retry_with_jitter, 2: rephrase, 3: yield, 4+: niche_diverge}
eff_depth = depth + 1   if φ ≥ 0.75    (high-φ: early niche routing)
eff_depth = depth − 1   if φ < 0.50    (low-φ: extended retry)
```

### Oxpecker / Channel 2 (Session 7)
```
β_adj ∈ [+0.05, +0.10]  applied to vacated resource domain
β_adj_adjacent = β_adj × 0.5   for declared adjacent resources
```

---

## Validated Results

Computational validation published on Zenodo: [DOI 10.5281/zenodo.17680563](https://doi.org/10.5281/zenodo.17680563)

| Metric | Result |
|--------|--------|
| Cooperative equilibria | **78.0%** of 10,000 runs |
| Bifurcation threshold | **i = 0.9995 ± 0.029** |
| Statistical significance | **p < 10⁻³⁰⁰** |
| Precision of threshold | Extraordinary (±0.029) |

---

## Theory-to-Code Table

This table is the formal bridge between *Blueprint for Harmony* (L.W. Evans, Cooperation Press 2026) and the MELVcore open-source implementation. Every MELV axiom or equation maps directly to an implementing function.

| MELV Axiom / Variable | Equation | File | Function | Session |
|-----------------------|----------|------|----------|---------|
| **Interaction Cost Ratio (i)** | `i = C / B` | `core/melv_engine.py` | `InteractionRecord.i_factor` (property, lines ~130–133) | 1 |
| **Modulated Threshold (βi)** | `βi = β × i` | `core/melv_engine.py` | `InteractionRecord.beta_i` (property, lines ~135–137) | 1 |
| **Bifurcation Threshold** | `βi < 1.0 → cooperative` | `core/melv_engine.py` | `MELVKernel.record_interaction` (lines ~220–235) | 1 |
| **Adaptive Dynamics (φ)** | `dφ/dt = ε(φ − φ_target) + η` | `core/melv_engine.py` | `MELVKernel.update_phi` (lines ~197–215) | 1 |
| **Cooperation Index (CI)** | `CI = 1 − mean(βi)` | `core/melv_engine.py` | `MELVKernel.cooperation_index` (lines ~290–300) | 1 |
| **Service Coupling (Ω)** | `β_svc = λ_max(Ω) / n` | `core/melv_engine.py` | `MELVKernel.compute_omega` (lines ~255–285) | 1 |
| **LLM Cost Formula** | `min(2.0, token_cost×1000×w_t + latency×0.1×w_l)` | `core/cost_calculator.py` | `CostCalculator.compute_cost` (lines ~115–140) | 6 |
| **Nudge Escalation** | `depth → {1:jitter, 2:rephrase, 3:yield, 4+:niche}` | `core/nudge_engine.py` | `NudgeEngine.build_nudge_v2` (lines ~120–210) | 7 |
| **φ Depth Adjustment** | `eff_depth ± 1 based on φ threshold` | `core/nudge_engine.py` | `NudgeEngine._effective_depth` (lines ~85–100) | 7 |
| **Oxpecker β Lift** | `β_adj ∈ [+0.05, +0.10]` | `core/nudge_engine.py` | `NudgeEngine.apply_oxpecker_effect` (lines ~215–275) | 7 |
| **BetaEnvironment.set()** | `β owned by kernel only` | `core/melv_engine.py` | `BetaEnvironment.set` (lines ~168–170) | 7 |
| **CI Rate of Change (dCI/dt)** | `slope of CI(t) over rolling window` | `core/melv_engine.py` | `MELVKernel.dci_dt` | 9 |
| **CI Optimisation Half-Life** | `t½ = ln(2)/k, k = dCI\_dt/gap` | `core/melv_engine.py` | `MELVKernel.ci_half_life` | 9 |
| **CI Drift Coefficient** | `long-run linear regression slope of CI(t)` | `core/melv_engine.py` | `MELVKernel.ci_drift_coefficient` | 9 |
| **Oscillation Detection** | `CI crosses 0.75 then falls back within window` | `core/melv_engine.py` | `MELVKernel._detect_oscillation` | 9 |
| **tanh φ Update (Axioms 3 & 8)** | `dφ/dt = (1/τ_φ)·[tanh(γ·mean_surplus) − φ] + η_φ(t)` | `core/melv_engine.py` | `MELVKernel.update_phi` (tanh relaxation, Session 10) | 10 |
| **Composite Longevity Score (CLS)** | `CLS = 100·σ(−α·Δt½_norm)·(1−β·\|OIS\|)·σ(γ·DDC_norm)` | `core/sandbox_engine.py` | `SandboxEngine.compute_report` | 10 |
| **CI History API** | Rolling time-series `{t, ci}` of last N CI readings from `_ci_history`; enables live dashboard visualisation of CI trajectory | `api/server.py` | `GET /api/ci_history?n=<int>` — max n=200 default, max n=1000 | 11 |
| **SQLite Persistence** | Durable state store (WAL mode) — agents, interactions, CI history, bifurcation/oscillation events, beta, sandbox reports survive server restart. Kernel accepts optional `persistence=` arg; `AIOSPersistence.restore_kernel()` hydrates state at startup | `core/persistence.py` | `GET /api/db_stats` — row counts per table; `AIOSPersistence(db_path)` | 12 |

**Column definitions:**
- **MELV Axiom / Variable** — the theoretical concept from *Blueprint for Harmony*
- **Equation** — the canonical mathematical expression
- **File** — path relative to AIOS/ project root
- **Function** — implementing Python function/method with approximate line range
- **Session** — the AIOS development session that implemented this function

---

## Bifurcation Interventions

When `β·i ≥ 1.0`, MELVcore selects an intervention:

| Intervention | Mechanism | Effect on i |
|--------------|-----------|-------------|
| `route_service` | Route task to lower-cost agent | Reduces cost term |
| `nudge` | Signal agent to yield resource | Increases benefit term |
| `niche_divergence` | Separate agents into non-competing niches | Removes interaction |
| `provision_beta` | Increase resource allocation | Reduces β |
| `agent_substitute` | Replace agent with lower-cost equivalent | Replaces i entirely |

---

## Two-Channel Cooperation

**Channel 1 — Direct:** Agent–kernel interaction mediated by C, B, i-factor, βi, and nudge. Sessions 4–7 implement Channel 1 fully.

**Channel 2 — Indirect:** Environmental mediation. One agent's niche specialisation changes the β landscape for adjacent agents. No direct interaction required. Implemented via `apply_oxpecker_effect()` (Session 7).

Named for the oxpecker–mammal and hornbill–bee-eater mutualistic relationships observed in Namibia (1981–1983): cooperation as thermodynamic side effect, not deliberate intent.

---

## Connection to Classical Theory

The MELV framework extends classical Lotka-Volterra competition equations by:

1. **Replacing fixed interaction coefficients** with dynamic i-factors computed from real resource flows
2. **Adding thermodynamic β scaling** that connects agent interactions to environmental resource availability
3. **Introducing maturity φ** as a state variable that captures agent learning and specialisation
4. **Proving the cooperation theorem** — below i_critical, competitive equilibria are thermodynamically unstable

---

## Further Reading

- *Blueprint for Harmony: Thermodynamic Foundations of Cooperation and Conscious Evolution* — L.W. Evans (Cooperation Press, 2026). ISBN 978-969-8992-10-1
- Validation dataset: [Zenodo DOI 10.5281/zenodo.17680563](https://doi.org/10.5281/zenodo.17680563)
- Author ORCID: [0009-0001-0963-1840](https://orcid.org/0009-0001-0963-1840)
- *Nature's Holism* — L.W. Evans (1999) — original conceptual framework
- GitHub: [github.com/NaturesHolismMELV/AIOS](https://github.com/NaturesHolismMELV/AIOS)

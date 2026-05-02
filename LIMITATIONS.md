# Limitations and Current Status

**MELVcore is an active research prototype** (v2.7.0, Sessions 1–30c,
May 2026). While the core kernel and ABM validation are mature, users and
readers should be aware of the following.

---

## Parameter Calibration

Several weights are theoretically grounded but not yet calibrated against
empirical data:

- **Tool friction category weights** (agent_native=0.2, fast_rest=0.5,
  standard=1.0, human_bottlenecked=1.5, legacy=2.0): The relative ordering
  is theoretically justified, but individual values have not been validated
  against production latency measurements. Epistemic status: ① stub.

- **Per-agent ε_intrinsic Gaussian perturbation** (σ=0.3): The perturbation
  is deterministic (seeded from agent_id hash) and ensures unique ε values
  across agents of the same type. The σ value is principled but has not been
  validated against empirical agent performance data. Epistemic status: ②
  theoretical.

- **ε_ecosystem friction computation**: Recomputed each governance tick from
  the current β vector via tool category weights. The computation structure
  is sound; the individual weights are stubs pending calibration.
  Epistemic status: ② theoretical.

---

## Validation Scope

The cooperation theorem (below i_critical, cooperative equilibria are
thermodynamically inevitable) is supported by:

1. ABM simulations (405 runs, DOI: 10.5281/zenodo.19422174) — ④ verified
2. DeepSeek blind axiom reconstruction (March 2026) — ④ verified
3. Live platform events on a ~313-agent deployment — ③ verified (observed,
   not yet independently reproduced)

**Independent external reproduction has not yet occurred.** The `/demo/`
endpoints support reproduction of the bifurcation demonstration by external
parties. See [VALIDATION.md](VALIDATION.md) for the protocol.

---

## AI-Assisted Development

MELVcore was developed using a multi-AI consultation methodology (MAIES —
Multi-AI Emergent Synthesis). AI systems (Gemini, Grok, NotebookLM, Claude,
DeepSeek) were engaged at key decision points.

**Structured independent validation (DeepSeek, March 2026):** DeepSeek was
given only the 8 MELV axioms — not the master equation or threshold condition —
and independently recovered the cooperation threshold β·i < 1. This is the
strongest form of external corroboration available and is archived in full in
[VALIDATION.md](VALIDATION.md).

**AI Synthesis Points A–D:** Insights from Gemini, Grok, NotebookLM, and
Claude informed specific design decisions (Oxpecker mechanism, quorum gate
formalisation, reliability tagging, ε_architectural boundary condition). These
are documented as convergent reasoning that raised confidence in specific
directions — they are not treated as formal validation. See
[VALIDATION.md](VALIDATION.md) for full documentation.

---

## Deployment Maturity

The live Railway deployment demonstrates autonomous recovery under adversarial
load (CI 0.503 → 0.957 in 12 minutes, no manual intervention, 27 April 2026).
However:

- The system has not been hardened for arbitrary adversarial loads or
  high-stakes production use.
- Rate limiting is applied to public `/demo/` endpoints (1 session/IP/10 min,
  20 interactions/session).
- Security review is ongoing. See [SECURITY.md](SECURITY.md).
- The SQLite persistence layer is appropriate for the current scale; production
  deployments at higher agent counts would require a more robust store.

---

## Conceptual Scope

MELVcore is currently optimised for **token/compute/API contention** in
LLM-based agent systems. The MELV framework's master equation is domain-general
(the interaction coefficient is dimensionless, functioning analogously to a
refractive index prior to domain-specific calibration), but explicit signal
mapping to other multi-agent frameworks (LangGraph, AutoGen, CrewAI) has not
yet been performed. This is the subject of the planned MAIES-006 Signal Mapping
investigation.

The biological analogies (hornbill-mongoose mutualism, oxpecker-giraffe
mutualism, bacterial quorum sensing) are used as interpretive grounding and
design guidance — not as proof of mechanism. The mathematical convergences
(DeepSeek reconstruction, ABM validation) provide the evidentiary basis for the
framework's claims.

---

## What Has Not Changed

Despite the diagnostic corrections in Session 30c, the following are stable
and unchanged from their originally verified states:

- Master equation: i₁₂(t) = i₁₂⁰ × (1 − ε_effective × φ(t) × β(t))
- i_critical = 0.9995 ± 0.029 (R² = 0.9248)
- ABM V2.1 results (all ④ verified)
- Sessions 1–25 theory and implementation
- Cooperation theorem and theorem_confirmed flag

---

## Contributions and Feedback

Critical feedback and reproduction attempts are welcome. Open an issue on
GitHub referencing the specific claim, the relevant section of
[THEORY.md](THEORY.md) or [VALIDATION.md](VALIDATION.md), and the evidence
basis for your concern.

---

*LIMITATIONS.md · MELVcore v2.7.0 · May 2026*  
*L.W. Evans · ORCID: 0009-0001-0963-1840 · Cape Town, South Africa*

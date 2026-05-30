"""
observe_compute.py — MELVcore Session 33–36 · v3.1.1
======================================================
Full observe() primitive computation logic.

Accepts a validated ObservationPayload and computes φ, σ, β, and ε
values, returning an ObservationResult with epistemic status metadata,
boundary-enforced inputs, and a Cooperation Index if gate conditions
are satisfied.

══════════════════════════════════════════════════════════════════
COMPUTATION RULES (MAIES-006 ③+ convergence)
══════════════════════════════════════════════════════════════════

  φ  = domain_success_rate × downstream_acceptance_rate
       weighted by state_reliability (v2.8.1 patch)
       long window (default 200), domain-filtered

  σ  = current_task_match_score if provided, else
       recent success rate (short window, default 20)
       ALWAYS provisional ① — does not gate CI

  β  = min(quota_utilisation_ratios) across ResourcePolicy dimensions
       + infra_contention_penalty
       Range enforced: [0.1, 3.0]
       action_scope present → bonus +0.1 (bounded access = richer niche)

  ε_intrinsic     = branching_reconfigs / task_duration_seconds
                    (normalised reconfiguration rate)
  ε_ecosystem     = CV (std/mean) of latency samples per task_type,
                    averaged across task_types with ≥2 samples
  ε_architectural = Σ(ARCH_CATEGORY_WEIGHTS × tool_counts)  [immutable]
  ε_effective     = ε_intrinsic + ε_ecosystem  (master equation)

  CI = 1 − ε_effective × φ × β  [master equation i₁₂(t) → β_i → CI]
     Computed only when φ.status ③+, β.status ③+,
     ε_intrinsic.status ②+, ε_ecosystem.status ②+

  φ/σ divergence = |φ − σ|  (domain-shift governance signal)

══════════════════════════════════════════════════════════════════
WHAT IS NOT HERE (Session 33 scope boundary)
══════════════════════════════════════════════════════════════════
  - σ full computation — MAIES-007 pending (stub returns status ①)
  - Framework adapters — in adapters/ module
  - Dashboard — in frontend/dashboard12.html
  - Governance loop kernel integration — in MELVKernel.apply_observation()

Author: Laurence W. Evans · ORCID: 0009-0001-0963-1840
        Cape Town, South Africa
Session: 33 · Version: 2.9.0
"""

from __future__ import annotations

import math
import statistics
from collections import defaultdict
from datetime import datetime
from typing import Optional

from core.observe_schema import (
    ARCH_CATEGORY_WEIGHTS,
    ObservationPayload,
    ObservationResult,
    EpsilonResult,
    ScoredValue,
)
from core.observe_validator import ObservationValidator
from core.melv_engine import _beta_norm, I_FLOOR  # Session 35 — β_norm correction

# ── Constants ──────────────────────────────────────────────────────────────
BETA_FLOOR   = 0.1
BETA_CEILING = 3.0
PHI_FLOOR    = 0.0
PHI_CEILING  = 1.0
EPSILON_FLOOR   = 0.0
EPSILON_CEILING = 8.0

# β: low-utilisation (unconstrainted niche) = high β
# Full quota utilisation → β approaching floor
BETA_FULL_UTILISATION = BETA_FLOOR   # agent consuming 100% of quota
BETA_ZERO_UTILISATION = BETA_CEILING # agent consuming 0% of quota

# β bonus for having action_scope declared (bounded access = richer niche)
BETA_ACTION_SCOPE_BONUS = 0.1

# CI gate: minimum epistemic status required for CI computation
CI_GATE_PHI_MIN_STATUS     = 3
CI_GATE_BETA_MIN_STATUS    = 3
CI_GATE_EPS_INT_MIN_STATUS = 2
CI_GATE_EPS_ECO_MIN_STATUS = 2

# φ/σ divergence: above this threshold → domain shift warning
PHI_SIGMA_DIVERGENCE_WARNING = 0.2


# ══════════════════════════════════════════════════════════════════
# INDIVIDUAL VARIABLE COMPUTATION
# ══════════════════════════════════════════════════════════════════

def _compute_phi(payload: ObservationPayload) -> ScoredValue:
    """
    Compute φ (accumulated maturity) from domain_success_history.

    φ = domain_success_rate × downstream_acceptance_rate
        weighted by state_reliability if provided

    The two-factor product reflects the MELV master equation's
    requirement that φ captures both the agent's internal success
    AND whether downstream consumers accepted the output (mutualism
    requires both parties to benefit).

    Returns status ① if task_domain is None or history is empty.
    Returns status ② if history < 10 domain-matching records.
    Returns status ③ if history ≥ 10 domain-matching records.
    """
    warnings: list[str] = []

    if payload.task_domain is None:
        return ScoredValue(
            value=0.5, status=1,
            warnings=["φ requires task_domain — defaulting to 0.5"],
            provisional=True,
        )

    filtered = payload.domain_filtered_history()

    if not filtered:
        return ScoredValue(
            value=0.5, status=1,
            warnings=["No domain-matching history — φ defaulting to 0.5"],
            provisional=True,
        )

    # ── Success rate ──────────────────────────────────────────────────────
    success_rate = sum(1 for t in filtered if t.success) / len(filtered)

    # ── Downstream acceptance rate ────────────────────────────────────────
    with_acceptance = [t for t in filtered if t.downstream_accepted is not None]
    if with_acceptance:
        acceptance_rate = sum(
            1 for t in with_acceptance if t.downstream_accepted
        ) / len(with_acceptance)
        phi_raw = (success_rate + acceptance_rate) / 2.0
    else:
        phi_raw = success_rate
        warnings.append(
            "downstream_accepted not available — φ based on success_rate only"
        )

    # ── state_reliability weighting (v2.8.1) ─────────────────────────────
    # reliability < 1.0 pulls φ toward the neutral 0.5 midpoint.
    # A completely unreliable history (0.0) contributes nothing — φ = 0.5.
    # An unassessed history (None) is treated as fully reliable.
    reliability = payload.state_reliability if payload.state_reliability is not None else 1.0
    phi = reliability * phi_raw + (1.0 - reliability) * 0.5

    phi = max(PHI_FLOOR, min(PHI_CEILING, phi))

    # ── Epistemic status ──────────────────────────────────────────────────
    n = len(filtered)
    if n < 10:
        status = 2
        warnings.append(f"φ status ②: {n} domain records < 10")
    else:
        status = 3

    if payload.state_reliability is not None and payload.state_reliability < 0.5:
        warnings.append(
            f"state_reliability={payload.state_reliability:.2f} — "
            "φ weighted toward 0.5; improve data quality for accurate φ"
        )

    # ── Bootstrap confidence interval (simple Bernoulli) ─────────────────
    ci: Optional[tuple[float, float]] = None
    if n >= 10:
        se = math.sqrt(phi * (1 - phi) / n)
        ci = (max(0.0, round(phi - 1.96 * se, 4)),
              min(1.0, round(phi + 1.96 * se, 4)))

    return ScoredValue(
        value=round(phi, 4),
        status=status,
        confidence_interval=ci,
        warnings=warnings,
        provisional=False,
    )


def _compute_sigma(payload: ObservationPayload) -> ScoredValue:
    """
    Compute σ (niche matching / current fitness) — provisional ① stub.

    σ is always status ① until MAIES-007 validates proxy signals.
    Does not gate CI computation.

    If current_task_match_score is operator-provided, use it directly.
    Otherwise, use recent_task_outcomes success rate as a weak proxy.
    """
    warnings = [
        "σ is provisional ① — MAIES-007 required for promoted status",
        "σ does not gate CI computation",
    ]

    if payload.current_task_match_score is not None:
        sigma = payload.current_task_match_score
        warnings.append("σ from operator-provided current_task_match_score")
    elif payload.recent_task_outcomes:
        recent = payload.recent_task_outcomes
        sigma = sum(1 for t in recent if t.success) / len(recent)
        warnings.append(
            f"σ from recent success rate ({len(recent)} outcomes) — "
            "weak proxy pending MAIES-007"
        )
    else:
        sigma = 0.5
        warnings.append("No recent outcomes — σ defaulting to 0.5")

    sigma = max(0.0, min(1.0, sigma))

    return ScoredValue(
        value=round(sigma, 4),
        status=1,
        confidence_interval=None,
        warnings=warnings,
        provisional=True,
    )


def _compute_beta(payload: ObservationPayload) -> ScoredValue:
    """
    Compute β (environmental suitability) from ResourcePolicy.

    β is a latent environmental variable reconstructed from
    operator-provided quota vs. inferred consumption.

    Strategy:
      For each ResourcePolicy dimension with a quota, estimate
      utilisation from infra contention events for that resource.
      β_dimension = 1 / (1 + utilisation_rate)  → range (0.5, 1.0]
      Then scale to [BETA_FLOOR, BETA_CEILING] across all dimensions.

      β_proxy = mean(β_dimension) across active dimensions
              × (1 + action_scope_bonus)
              - infra_contention_penalty

    Range enforced: [0.1, 3.0].

    Returns status ① if no policy and no infra events.
    Returns status ② if single dimension.
    Returns status ③ if multiple dimensions.
    """
    warnings: list[str] = []
    rp = payload.resource_policy
    infra_events = payload.infra_contention_events()

    if rp.is_empty() and not infra_events:
        return ScoredValue(
            value=1.0, status=1,
            warnings=["β uncomputable — no ResourcePolicy and no infra events; defaulting to 1.0"],
            provisional=True,
        )

    # ── Contention penalty ────────────────────────────────────────────────
    # Each infra contention event reduces β by a small amount.
    # Reflects: infrastructure is constraining the agent's niche.
    contention_penalty = min(0.5, len(infra_events) * 0.05)
    if infra_events:
        warnings.append(
            f"{len(infra_events)} infra contention event(s) → "
            f"β penalty = {contention_penalty:.2f}"
        )

    # ── Dimension-based β estimates ───────────────────────────────────────
    dim_betas: list[float] = []

    # Group infra events by resource_type for utilisation estimation
    contention_by_resource: dict[str, int] = defaultdict(int)
    for ev in infra_events:
        contention_by_resource[ev.resource_type] += 1

    # Quota dimensions
    if rp.token_budget_per_hour is not None:
        token_events = contention_by_resource.get("tokens", 0)
        utilisation = min(1.0, token_events * 0.1)  # each token event = 10% utilisation
        dim_betas.append(1.0 / (1.0 + utilisation))

    if rp.compute_share is not None:
        compute_events = contention_by_resource.get("compute", 0)
        utilisation = min(1.0, compute_events * 0.1)
        # compute_share already encodes richness — high share = high β
        dim_betas.append(rp.compute_share * (1.0 / (1.0 + utilisation)))

    if rp.memory_limit_mb is not None:
        memory_events = contention_by_resource.get("memory", 0)
        utilisation = min(1.0, memory_events * 0.1)
        dim_betas.append(1.0 / (1.0 + utilisation))

    if rp.api_quota_per_minute is not None:
        api_events = contention_by_resource.get("api_quota", 0)
        utilisation = min(1.0, api_events * 0.1)
        dim_betas.append(1.0 / (1.0 + utilisation))

    # action_scope: bounded verb permissions = defined niche = bonus
    if rp.action_scope is not None:
        # More verbs/systems declared = richer niche access
        n_systems = len(rp.action_scope.split(";"))
        n_verbs = len(rp.action_scope.replace(";", ",").split(","))
        scope_richness = min(0.3, (n_systems * 0.05) + (n_verbs * 0.02))
        warnings.append(
            f"action_scope: {n_systems} system(s), {n_verbs} verb(s) → "
            f"niche richness bonus +{scope_richness:.2f}"
        )
    else:
        scope_richness = 0.0

    if not dim_betas and infra_events:
        # Only infra events, no policy — low β (constrained environment)
        beta_raw = 0.5 - contention_penalty + scope_richness
    elif dim_betas:
        beta_raw = (sum(dim_betas) / len(dim_betas)) - contention_penalty + scope_richness
    else:
        beta_raw = 1.0

    # Scale from (0, 1] to [BETA_FLOOR, BETA_CEILING]
    # Neutral point 0.5 → β=1.0; max → β approaches BETA_CEILING
    beta = BETA_FLOOR + (beta_raw / 1.0) * (BETA_CEILING - BETA_FLOOR)
    beta = max(BETA_FLOOR, min(BETA_CEILING, round(beta, 4)))

    # ── Epistemic status ──────────────────────────────────────────────────
    active_dims = rp.active_dimensions()
    n_dims = len(active_dims)

    if n_dims == 0:
        status = 1
    elif n_dims == 1:
        status = 2
    else:
        status = 3

    return ScoredValue(
        value=beta,
        status=status,
        confidence_interval=None,
        warnings=warnings,
        provisional=(status < 3),
    )


def _compute_epsilon(payload: ObservationPayload) -> EpsilonResult:
    """
    Compute three-scalar ε decomposition.

    ε_intrinsic  = branching_reconfigs / max(task_duration_seconds, 1.0)
    ε_ecosystem  = mean CV(latency) across task_types with ≥2 samples
    ε_architectural = Σ(ARCH_CATEGORY_WEIGHTS × tool_counts)
    ε_effective  = ε_intrinsic + ε_ecosystem
    """
    warnings_int: list[str] = []
    warnings_eco: list[str] = []
    warnings_arch: list[str] = []

    # ── ε_intrinsic ───────────────────────────────────────────────────────
    branching = payload.branching_reconfig_events()
    n_branch = len(branching)
    duration = max(payload.task_duration_seconds, 1.0)  # floor at 1s

    if payload.task_duration_seconds == 0:
        warnings_int.append(
            "task_duration_seconds=0 — floored to 1.0s for normalisation"
        )

    eps_intrinsic_raw = n_branch / duration
    eps_intrinsic = max(EPSILON_FLOOR, min(EPSILON_CEILING, round(eps_intrinsic_raw, 4)))

    non_branching = len(payload.reconfiguration_events) - n_branch
    if non_branching > 0:
        warnings_int.append(
            f"{non_branching} repair/infra_induced reconfig event(s) excluded "
            "from ε_intrinsic"
        )

    if n_branch == 0:
        status_int = 2
        warnings_int.append("ε_intrinsic=0.0 (no branching events) — valid signal")
    else:
        status_int = 3

    sv_intrinsic = ScoredValue(
        value=eps_intrinsic,
        status=status_int,
        warnings=warnings_int,
        provisional=False,
    )

    # ── ε_ecosystem (CV of latency per task_type) ─────────────────────────
    # Group samples by (task_domain, task_type)
    groups: dict[tuple, list[float]] = defaultdict(list)
    for s in payload.latency_samples:
        groups[(s.task_domain, s.task_type)].append(s.latency_ms)

    valid_cvs: list[float] = []
    singleton_count = 0
    for (domain, ttype), latencies in groups.items():
        if len(latencies) < 2:
            singleton_count += 1
            continue
        mean_lat = statistics.mean(latencies)
        if mean_lat == 0:
            continue
        cv = statistics.stdev(latencies) / mean_lat
        valid_cvs.append(cv)

    if singleton_count:
        warnings_eco.append(
            f"{singleton_count} task_type(s) have <2 samples — excluded from CV"
        )

    if not valid_cvs:
        eps_ecosystem = 0.0
        status_eco = 1
        warnings_eco.append(
            "ε_ecosystem=0.0 (insufficient latency samples) — add ≥2 per task_type"
        )
    else:
        eps_ecosystem = round(
            max(EPSILON_FLOOR, min(EPSILON_CEILING,
                statistics.mean(valid_cvs))), 4
        )
        status_eco = 3 if len(valid_cvs) == len(groups) else 2

    sv_ecosystem = ScoredValue(
        value=eps_ecosystem,
        status=status_eco,
        warnings=warnings_eco,
        provisional=(status_eco < 2),
    )

    # ── ε_architectural ───────────────────────────────────────────────────
    topo = payload.tool_topology
    eps_arch = round(topo.epsilon_architectural(), 4)

    if topo.total_tools() == 0:
        status_arch = 1
        warnings_arch.append(
            "ε_architectural=0.0 — no tools registered; "
            "declare tool topology at agent init"
        )
    else:
        status_arch = 3

    sv_architectural = ScoredValue(
        value=eps_arch,
        status=status_arch,
        warnings=warnings_arch,
        provisional=False,
    )

    # ── ε_effective (master equation term) ────────────────────────────────
    eps_effective = round(
        max(EPSILON_FLOOR, min(EPSILON_CEILING,
            eps_intrinsic + eps_ecosystem)), 4
    )

    return EpsilonResult(
        intrinsic=sv_intrinsic,
        ecosystem=sv_ecosystem,
        architectural=sv_architectural,
        effective=eps_effective,
    )


# ══════════════════════════════════════════════════════════════════
# COOPERATION INDEX GATE
# ══════════════════════════════════════════════════════════════════

def _compute_ci(
    phi: ScoredValue,
    beta: ScoredValue,
    epsilon: EpsilonResult,
) -> Optional[float]:
    """
    Compute Cooperation Index if gate conditions are satisfied.

    Master equation (Session 35 — β_norm correction C1):
      i₁₂(t) = i₁₂⁰ × (1 − ε × φ(t) × β_norm(t))
      β_norm(t) = β(t)/(1+β(t)) ∈ (0,1)
    Simplified for single-agent observe():
      CI = 1 − ε_effective × φ × β_norm

    Gate: φ ③+, β ③+, ε_intrinsic ②+, ε_ecosystem ②+
    σ (always ①) does NOT gate CI.

    Returns None if gate not met.
    Range enforced: [0, 1].
    """
    if phi.status < CI_GATE_PHI_MIN_STATUS:
        return None
    if beta.status < CI_GATE_BETA_MIN_STATUS:
        return None
    if epsilon.intrinsic.status < CI_GATE_EPS_INT_MIN_STATUS:
        return None
    if epsilon.ecosystem.status < CI_GATE_EPS_ECO_MIN_STATUS:
        return None

    # Session 35 (v3.1.0): β_norm correction C1 — use β/(1+β) ∈ (0,1)
    # I_FLOOR guards against unbounded negative i(t) for high-ε agents
    ci_raw     = 1.0 - (epsilon.effective * phi.value * _beta_norm(beta.value))
    ci_floored = max(I_FLOOR, ci_raw)
    return max(0.0, min(1.0, round(ci_floored, 4)))


# ══════════════════════════════════════════════════════════════════
# MAIN COMPUTATION ENTRY POINT
# ══════════════════════════════════════════════════════════════════

class ObservationComputer:
    """
    Computes φ, σ, β, and ε from a validated ObservationPayload.

    Usage
    -----
        computer = ObservationComputer()
        result = computer.compute(payload)
        # result: ObservationResult
    """

    def __init__(self):
        self._validator = ObservationValidator()

    def compute(
        self,
        payload: ObservationPayload,
        skip_validation: bool = False,
    ) -> ObservationResult:
        """
        Validate payload and compute ObservationResult.

        Parameters
        ----------
        payload:         The ObservationPayload to compute from.
        skip_validation: If True, skips the schema validation step
                         (use only when validator was already called upstream).

        Returns
        -------
        ObservationResult with all variable values, epistemic status,
        CI (if gate met), φ/σ divergence, and aggregate warnings.
        """
        all_warnings: list[str] = []

        # ── Schema validation ─────────────────────────────────────────────
        if not skip_validation:
            vr = self._validator.validate(payload)
            if not vr.schema_valid:
                for bc in vr.boundary_checks:
                    if not bc.passed:
                        all_warnings.append(
                            f"[{bc.guard}] {bc.violation_type}: {bc.detail}"
                        )

        # ── Compute each variable ─────────────────────────────────────────
        phi     = _compute_phi(payload)
        sigma   = _compute_sigma(payload)
        beta    = _compute_beta(payload)
        epsilon = _compute_epsilon(payload)

        # ── CI ────────────────────────────────────────────────────────────
        ci = _compute_ci(phi, beta, epsilon)
        if ci is None:
            all_warnings.append(
                "CI not computed — gate requires φ ③+, β ③+, "
                "ε_intrinsic ②+, ε_ecosystem ②+"
            )

        # ── φ/σ divergence ────────────────────────────────────────────────
        phi_sigma_divergence = round(abs(phi.value - sigma.value), 4)
        if phi_sigma_divergence > PHI_SIGMA_DIVERGENCE_WARNING:
            all_warnings.append(
                f"φ/σ divergence = {phi_sigma_divergence:.3f} > "
                f"{PHI_SIGMA_DIVERGENCE_WARNING} — possible domain shift; "
                "review task_domain assignment"
            )

        # ── Aggregate warnings from variable computations ─────────────────
        for sv in [phi, sigma, beta,
                   epsilon.intrinsic, epsilon.ecosystem, epsilon.architectural]:
            all_warnings.extend(sv.warnings)

        # Session 35 — Equation 7 inputs
        # r_value: use raw β as R proxy. β ≥ 1 → competitive (R ≥ 0.50),
        # β < 1 → cooperative (R < 0.50). None if β not computable.
        r_value = beta.value if beta.computable else None

        # Session 36 — d_value populated from telemetry if available.
        # Falls back to 0.0 when no L1 records exist (bootstrap).
        d_value = 0.0
        try:
            from core.telemetry import AIOSTelemetry as _AIOSTelemetry  # local import
            import os as _os
            _db = _os.environ.get(
                "AIOS_DB_PATH",
                _os.path.join(
                    _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))),
                    "aios_state.db",
                ),
            )
            _tel = _AIOSTelemetry(_db)
            d_value = _tel.compute_d_value_for_agent(payload.agent_id)
            _tel.close()
        except Exception:
            pass  # telemetry layer unavailable; keep d_value=0.0

        return ObservationResult(
            agent_id=payload.agent_id,
            phi=phi,
            sigma=sigma,
            beta=beta,
            epsilon=epsilon,
            ci=ci,
            phi_sigma_divergence=phi_sigma_divergence,
            warnings=all_warnings,
            timestamp=datetime.utcnow(),
            r_value=r_value,
            d_value=d_value,  # D(t) from L1 rolling mean (Session 36)
        )

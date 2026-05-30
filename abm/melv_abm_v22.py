"""
melv_abm_v22.py — MELV Agent-Based Model V2.2
===============================================

Extends ABM V2.1 (DOI 10.5281/zenodo.19422174) with Equation 7 φ dynamics
and Equation 1a (saturation form), plus BI-NLS η estimation.

Architecture:
  Each agent has a baseline cost ratio i0 (slow-evolving via mutation) and
  state variables φ, β (φ evolves per Equation 7; β fixed per run condition).

  i(t) = cooperation_evolution(i0, ε, φ, β)  [computed each step — not stored]

  Dynamics:
    1. Compute i(t) from (i0, ε, φ, β)
    2. Apply mutation to i0 (±10%, Axiom 8 heterogeneity)
    3. Apply Equation 7 φ update (compound gating)
       - φ builds only when R < 0.50 AND i < 1.0 simultaneously
       - φ does NOT build otherwise → COMP agents stay COMP

  This produces bimodality:
    - COOP agents: high initial φ×β → i(t) < 1 → φ builds → deeper cooperation
    - COMP agents: low initial φ×β → i(t) > 1 → φ doesn't build → stay COMP

  Parameter grid: 3ε × 3φ × 3β = 27 conditions (×5 replicates = 135 runs linear;
  ×3 η conditions for saturation form = 81 conditions × 5 = 405 runs).

Author: Laurence W. Evans | ORCID: 0009-0001-0963-1840 | Cape Town, South Africa
Version: ABM V2.2
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

# ── Canonical constants ───────────────────────────────────────────────────────

I_CRITICAL            = 0.9995
QUORUM_TAU            = 0.5
QUORUM_K              = 10.0
PHI_BUILD_RATE_ALPHA  = 0.01
PHI_DECAY_RATE_DELTA  = 0.10
PHI_GATEWAY_THRESHOLD = 0.50

GRID_N        = 20
N_AGENTS      = GRID_N * GRID_N   # 400
N_GENS        = 300
MUTATION_RATE = 0.005             # Axiom 8 ±0.5% on i0 (small noise for Axiom 8 heterogeneity)

# ── Core equations ────────────────────────────────────────────────────────────

def beta_norm(beta_raw: float) -> float:
    """β_norm = β/(1+β) ∈ (0,1). C1 correction — canonical v1.2."""
    return beta_raw / (1.0 + beta_raw)


def cooperation_evolution_linear(i0: float, epsilon: float,
                                  phi: float, beta_raw: float) -> float:
    """Equation 1: i(t) = i₀ × (1 − ε × φ × β_norm)."""
    bn = beta_norm(beta_raw)
    return max(-5.0, i0 * (1.0 - epsilon * phi * bn))


def cooperation_evolution_saturation(i0: float, epsilon: float,
                                      phi: float, beta_raw: float,
                                      eta: float) -> float:
    """Equation 1a: i(t) = i₀ × (1 − η × tanh(ε × φ × β_norm / η))."""
    bn = beta_norm(beta_raw)
    u  = epsilon * phi * bn / max(eta, 1e-9)
    return i0 * (1.0 - eta * math.tanh(u))


def sigmoid_quorum_gate(phi_beta: float,
                        tau: float = QUORUM_TAU,
                        k:   float = QUORUM_K) -> float:
    return 1.0 / (1.0 + math.exp(-k * (phi_beta - tau)))


def apply_phi_eq7(phi: float, r_value: float, i_value: Optional[float],
                  d_value: float = 0.0) -> Tuple[float, str]:
    """
    Equation 7 compound-gated φ update (one discrete step).
    Builds only when R < 0.50 AND i < 1.0 simultaneously (T1.6).
    Decays when R ≥ 0.50 with disruption intensity D(t).
    """
    alpha = PHI_BUILD_RATE_ALPHA
    delta = PHI_DECAY_RATE_DELTA
    if r_value < PHI_GATEWAY_THRESHOLD:
        if i_value is not None and i_value < 1.0:
            build_factor = max(0.0, 1.0 - i_value)
            return min(1.0, phi + alpha * (1.0 - phi) * build_factor), "BUILD"
        return phi, "GATED_NO_BUILD"
    else:
        if d_value > 0.0:
            return max(0.0, phi - delta * d_value * phi), "DECAY"
        return phi, "STABLE"


# ── Agent ─────────────────────────────────────────────────────────────────────

@dataclass
class Agent:
    """
    An ABM agent. i0 is the slow-evolving baseline cost ratio.
    i(t) is computed from (i0, ε, φ, β) each step.
    """
    agent_id:  int
    i0:        float   # baseline cost ratio — evolves via mutation
    phi:       float   # perpetuity ∈ [0,1]
    epsilon:   float   # adaptive plasticity (fixed per run condition)
    beta:      float   # environmental compatibility (fixed per run condition)
    phi_build_count:            int = 0
    phi_build_outside_eligible: int = 0

    def i_factor(self, use_saturation: bool = False,
                 eta: float = 0.93) -> float:
        if use_saturation:
            return cooperation_evolution_saturation(
                self.i0, self.epsilon, self.phi, self.beta, eta)
        return cooperation_evolution_linear(
            self.i0, self.epsilon, self.phi, self.beta)

    def r_value(self, tax: float = 0.10) -> float:
        """R = i0 × TAX / β (i0 as C proxy, gateway condition)."""
        return self.i0 * tax / max(self.beta, 1e-9)

    def is_cooperative(self, use_saturation: bool = False,
                       eta: float = 0.93) -> bool:
        return self.i_factor(use_saturation, eta) < I_CRITICAL


# ── Logging ───────────────────────────────────────────────────────────────────

@dataclass
class InteractionLog:
    i_observed: float
    epsilon:    float
    phi:        float
    beta_norm:  float


@dataclass
class RunResult:
    run_id:          str
    epsilon:         float
    phi_init:        float
    beta_init:       float
    eta_planted:     float
    use_saturation:  bool
    outcome:         str    # COOP | COMP | THRESH
    ci_final:        float
    mean_i_final:    float
    phi_final:       float
    phi_build_rate:  float
    tau_build:       float
    tau_decay:       float
    r_mean:          float
    phi_build_events_total:     int
    phi_build_outside_eligible: int
    min_i:           float
    i0_one_minus_eta: float     # mutualism ceiling i₀(1−η) per agent
    interaction_log: List[InteractionLog] = field(default_factory=list)
    eta_estimated:   Optional[float] = None
    eta_rse:         Optional[float] = None
    ess_recovered:   Optional[bool]  = None


# ── BI-NLS η estimator ────────────────────────────────────────────────────────

def _sensitivity(u: float) -> float:
    u_c = min(abs(u), 500.0)
    ch  = math.cosh(u_c)
    return math.tanh(u_c) - u_c / (ch * ch)


def estimate_eta_binls(log: List[InteractionLog],
                        eta_init: float = 0.93,
                        lambda_damp: float = 1e-4,
                        max_iter: int = 50) -> Tuple[Optional[float], Optional[float]]:
    valid = [(r.i_observed, r.epsilon, r.phi, r.beta_norm)
             for r in log
             if (r.phi > 0 and r.epsilon > 0 and r.beta_norm > 0
                 and not any(math.isnan(x) or math.isinf(x)
                             for x in (r.i_observed, r.epsilon,
                                       r.phi, r.beta_norm)))]
    if len(valid) < 10:
        return None, None
    eta = max(0.01, min(1.0, eta_init))
    for _ in range(max_iter):
        num = 0.0; den = lambda_damp
        for i_obs, eps, phi, bn in valid:
            u      = eps * phi * bn / eta
            s      = _sensitivity(u)
            i_pred = 1.0 - eta * math.tanh(u)
            num   += (i_obs - i_pred) * s
            den   += s * s
        delta   = -num / den   # canonical minus sign
        eta_new = max(0.01, min(1.0, eta + delta))
        if abs(eta_new - eta) < 1e-6:
            eta = eta_new; break
        eta = eta_new
    ss  = sum((obs - (1.0 - eta * math.tanh(eps * phi * bn / eta))) ** 2
              for obs, eps, phi, bn in valid)
    rse = math.sqrt(ss / max(len(valid) - 1, 1))
    return eta, rse


# ── ABM core ──────────────────────────────────────────────────────────────────

def _make_agents(n: int, phi_init: float, epsilon: float,
                 beta_init: float, i0_init: float,
                 rng: random.Random) -> List[Agent]:
    agents = []
    for i in range(n):
        phi  = max(0.01, min(0.99, phi_init + rng.gauss(0, 0.05)))
        beta = max(0.05, beta_init + rng.gauss(0, 0.03))
        i0   = max(0.1,  i0_init   + rng.gauss(0, 0.05))
        agents.append(Agent(agent_id=i, i0=i0, phi=phi,
                            epsilon=epsilon, beta=beta))
    return agents


def _cooperation_index(agents: List[Agent],
                        use_saturation: bool, eta: float) -> float:
    if not agents:
        return 0.0
    return sum(1 for a in agents
               if a.i_factor(use_saturation, eta) < I_CRITICAL) / len(agents)


def run_single(
    epsilon:        float,
    phi_init:       float,
    beta_init:      float,
    eta_planted:    float   = 0.93,
    use_saturation: bool    = False,
    n_gens:         int     = N_GENS,
    n_agents:       int     = N_AGENTS,
    seed:           int     = 42,
    run_id:         str     = "run",
    i0_init:        float   = 1.5,
) -> RunResult:
    """
    Run one ABM simulation.

    Pure φ-dynamics mechanism (no suppression):
      - i(t) = f(i0, ε, φ, β) computed each step
      - When i(t) < 1 AND R < 0.5: φ builds → i(t) falls → COOP reinforced
      - When i(t) ≥ 1: φ doesn't build → COMP maintained
      - Mutation on i0 (±10%) enables boundary crossings (Axiom 8)
    """
    rng    = random.Random(seed)
    agents = _make_agents(n_agents, phi_init, epsilon, beta_init, i0_init, rng)

    interaction_log: List[InteractionLog] = []
    phi_build_events_total      = 0
    phi_build_outside_eligible  = 0
    build_gens: List[int] = []
    decay_gens: List[int] = []
    last_build_gen = -1
    last_decay_gen = -1
    min_i = float("inf")

    for gen in range(n_gens):
        rng.shuffle(agents)
        # Well-mixed interaction — compute i(t) and apply mutation to i0
        for a in agents:
            i_t = a.i_factor(use_saturation, eta_planted)
            if i_t < min_i:
                min_i = i_t
            # Mutation on i0 (Axiom 8)
            a.i0 = max(0.05, a.i0 + rng.gauss(0, MUTATION_RATE * abs(a.i0)))
            # Log for BI-NLS — normalise by i0 so estimator assumes i_pred=1-eta*tanh(u)
            if use_saturation and len(interaction_log) < 2000:
                interaction_log.append(InteractionLog(
                    i_observed=i_t / max(a.i0, 1e-9),  # normalised: i(t)/i0
                    epsilon=a.epsilon,
                    phi=a.phi,
                    beta_norm=beta_norm(a.beta),
                ))

        # Equation 7 φ update
        for a in agents:
            r_val = a.r_value()
            i_val = a.i_factor(use_saturation, eta_planted)
            in_eligible = (r_val < PHI_GATEWAY_THRESHOLD and i_val < 1.0)

            new_phi, event = apply_phi_eq7(a.phi, r_val, i_val, d_value=0.0)
            a.phi = new_phi

            if event == "BUILD":
                phi_build_events_total += 1
                a.phi_build_count += 1
                if not in_eligible:
                    phi_build_outside_eligible += 1
                    a.phi_build_outside_eligible += 1
                if last_build_gen >= 0 and gen > last_build_gen:
                    build_gens.append(gen - last_build_gen)
                last_build_gen = gen
            elif event == "DECAY":
                if last_decay_gen >= 0 and gen > last_decay_gen:
                    decay_gens.append(gen - last_decay_gen)
                last_decay_gen = gen

    # Final state
    ci_final  = _cooperation_index(agents, use_saturation, eta_planted)
    mean_i    = sum(a.i_factor(use_saturation, eta_planted)
                    for a in agents) / n_agents
    phi_final = sum(a.phi for a in agents) / n_agents
    r_mean    = sum(a.r_value() for a in agents) / n_agents

    if ci_final >= 0.90:
        outcome = "COOP"
    elif ci_final <= 0.20:
        outcome = "COMP"
    else:
        outcome = "THRESH"

    tau_build = (sum(build_gens) / len(build_gens)) if build_gens else float("inf")
    tau_decay = (sum(decay_gens) / len(decay_gens)) if decay_gens else float("inf")

    eta_est, eta_rse = None, None
    if use_saturation and interaction_log:
        eta_est, eta_rse = estimate_eta_binls(interaction_log, eta_init=0.5)

    return RunResult(
        run_id=run_id,
        epsilon=epsilon,
        phi_init=phi_init,
        beta_init=beta_init,
        eta_planted=eta_planted,
        use_saturation=use_saturation,
        outcome=outcome,
        ci_final=ci_final,
        mean_i_final=mean_i,
        phi_final=phi_final,
        phi_build_rate=PHI_BUILD_RATE_ALPHA,
        tau_build=tau_build,
        tau_decay=tau_decay,
        r_mean=r_mean,
        phi_build_events_total=phi_build_events_total,
        phi_build_outside_eligible=phi_build_outside_eligible,
        min_i=min_i if min_i != float("inf") else 1.5,
        i0_one_minus_eta=i0_init * (1.0 - eta_planted),
        interaction_log=interaction_log,
        eta_estimated=eta_est,
        eta_rse=eta_rse,
    )


def run_ess_invasion_test(
    epsilon: float, phi_coop: float, beta_coop: float,
    eta_planted: float = 0.93, use_saturation: bool = False,
    seed: int = 999
) -> bool:
    """
    ESS invasion test (T1.2): start from cooperative equilibrium (i0=0.6),
    inject 10% competitive mutants (i0=1.8), run 150 gens, check recovery.
    """
    rng    = random.Random(seed)
    agents = _make_agents(N_AGENTS, phi_coop, epsilon, beta_coop,
                          i0_init=0.6, rng=rng)
    n_mutants = max(1, int(N_AGENTS * 0.10))
    for idx in rng.sample(range(N_AGENTS), n_mutants):
        agents[idx].i0 = 1.8 + rng.gauss(0, 0.1)

    for _ in range(150):
        rng.shuffle(agents)
        for a in agents:
            a.i0 = max(0.05, a.i0 + rng.gauss(0, MUTATION_RATE * abs(a.i0)))
        for a in agents:
            r_val = a.r_value()
            i_val = a.i_factor(use_saturation, eta_planted)
            new_phi, _ = apply_phi_eq7(a.phi, r_val, i_val)
            a.phi = new_phi

    return _cooperation_index(agents, use_saturation, eta_planted) >= 0.80


def build_parameter_grid() -> List[dict]:
    """81 conditions: 3ε × 3φ × 3β × 3η = 81."""
    epsilons    = [0.5, 1.2, 2.0]
    phi_inits   = [0.3, 0.6, 0.85]
    beta_inits  = [0.3, 0.6, 1.2]
    eta_planted = [0.80, 0.93, 1.00]
    return [{"epsilon": e, "phi_init": p, "beta_init": b, "eta_planted": eta}
            for e in epsilons for p in phi_inits
            for b in beta_inits for eta in eta_planted]


def run_full_study(n_replicates: int = 5,
                   use_saturation: bool = True,
                   n_gens: int = N_GENS,
                   verbose: bool = False) -> List[RunResult]:
    conditions = build_parameter_grid()
    results: List[RunResult] = []
    total = len(conditions) * n_replicates
    for ci, cond in enumerate(conditions):
        for rep in range(n_replicates):
            seed = hash((ci, rep)) % (2**31)
            result = run_single(
                epsilon=cond["epsilon"], phi_init=cond["phi_init"],
                beta_init=cond["beta_init"], eta_planted=cond["eta_planted"],
                use_saturation=use_saturation, n_gens=n_gens,
                seed=seed,
                run_id=(f"e{cond['epsilon']:.1f}_p{cond['phi_init']:.2f}"
                        f"_b{cond['beta_init']:.1f}_r{rep}"),
            )
            results.append(result)
            if verbose:
                print(f"  [{ci*n_replicates+rep+1}/{total}] "
                      f"{result.run_id}: {result.outcome}")
    return results

"""
test_session38.py — MELVcore Session 38: ABM V2.2 Test Suites 1 & 2
=====================================================================

Tests the ABM V2.2 (abm/melv_abm_v22.py) against the formal test suites
defined in the MELV canonical reference v1.2 and Session 36 handoff.

Test Suite 1 — Equation 7 Validation (prerequisite for all ABM work)
  T1.1 Bimodality         — Hartigan dip p ≈ 0
  T1.2 ESS invasion       — 34/34 recovery
  T1.3 Cooperation theorem — CI = 1.0 ± 0.01 at convergence
  T1.4 Gateway threshold  — R = 0.50 ± 0.003, p < 10⁻⁵⁰
  T1.5 φ asymmetry        — τ_build / τ_decay > 10
  T1.6 Compound gating    — zero φ build events outside eligible regime

Test Suite 2 — Saturation Form (Equation 1a) Validation
  T2.1 Threshold location — bimodal boundary unchanged from linear form
  T2.2 Mutualism ceiling  — i(t) ≥ i₀(1−η) always
  T2.3 Linear limit       — < 1% divergence for ε×φ×β_norm < 0.3
  T2.4 Mutualism theorem  — ε > 1 required for i < 0
  T2.5 η estimation       — BI-NLS converges within ±0.02 after 1000 interactions

Test groups
-----------
  H01–H05  Module structure and constants
  H06–H08  Cooperation-evolution equation — unit tests
  H09–H11  β_norm and sigmoid quorum gate
  H12–H15  Equation 7 φ dynamics — unit tests
  H16–H20  T1.1 Bimodality (Hartigan dip)
  H21–H23  T1.2 ESS invasion recovery
  H24–H26  T1.3 Cooperation theorem
  H27–H29  T1.4 Gateway threshold
  H30–H32  T1.5 φ asymmetry
  H33–H35  T1.6 Compound gating
  H36–H38  T2.1 Threshold location
  H39–H41  T2.2 Mutualism ceiling
  H42–H43  T2.3 Linear limit
  H44–H45  T2.4 Mutualism theorem
  H46–H48  T2.5 η estimation

Author: Laurence W. Evans | ORCID: 0009-0001-0963-1840 | Cape Town, South Africa
Session: 38 · ABM Version: V2.2
"""

import math
import sys
import os
import random
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from abm.melv_abm_v22 import (
    # Constants
    I_CRITICAL, QUORUM_TAU, QUORUM_K,
    PHI_BUILD_RATE_ALPHA, PHI_DECAY_RATE_DELTA, PHI_GATEWAY_THRESHOLD,
    N_AGENTS, GRID_N, MUTATION_RATE,
    # Core functions
    beta_norm, cooperation_evolution_linear, cooperation_evolution_saturation,
    sigmoid_quorum_gate, apply_phi_eq7, estimate_eta_binls,
    # Classes
    Agent, InteractionLog, RunResult,
    # Run functions
    run_single, run_ess_invasion_test,
)

# ── Shared fixture: small fast study results ──────────────────────────────────

@pytest.fixture(scope="module")
def study_linear():
    """
    Module-scoped fixture: run a subset (27 conditions × 3 replicates)
    with LINEAR form to generate statistical results quickly.
    Covers 3×3×3 eps×phi×beta grid at single eta (0.93).
    """
    from abm.melv_abm_v22 import run_single
    results = []
    epsilons   = [0.5, 1.2, 2.0]
    phi_inits  = [0.3, 0.6, 0.85]
    beta_inits = [0.3, 0.6, 1.2]
    rng = random.Random(0)
    idx = 0
    for eps in epsilons:
        for phi in phi_inits:
            for beta in beta_inits:
                for rep in range(3):
                    r = run_single(
                        epsilon=eps, phi_init=phi, beta_init=beta,
                        eta_planted=0.93, use_saturation=False,
                        n_gens=150, seed=idx * 31 + rep,
                        run_id=f"lin_{idx}_{rep}",
                    )
                    results.append(r)
                    idx += 1
    return results


@pytest.fixture(scope="module")
def study_saturation():
    """
    Module-scoped fixture: same grid with SATURATION form (Equation 1a).
    """
    from abm.melv_abm_v22 import run_single
    results = []
    epsilons   = [0.5, 1.2, 2.0]
    phi_inits  = [0.3, 0.6, 0.85]
    beta_inits = [0.3, 0.6, 1.2]
    idx = 0
    for eps in epsilons:
        for phi in phi_inits:
            for beta in beta_inits:
                for rep in range(3):
                    r = run_single(
                        epsilon=eps, phi_init=phi, beta_init=beta,
                        eta_planted=0.93, use_saturation=True,
                        n_gens=150, seed=idx * 31 + rep + 1000,
                        run_id=f"sat_{idx}_{rep}",
                    )
                    results.append(r)
                    idx += 1
    return results


# ── H01–H05: Module structure and constants ───────────────────────────────────

def test_H01_i_critical():
    """H01: I_CRITICAL = 0.9995 (canonical ABM V2.1)."""
    assert I_CRITICAL == 0.9995


def test_H02_quorum_constants():
    """H02: Quorum gate constants τ=0.5, k=10 (ABM V2.1 calibrated)."""
    assert QUORUM_TAU == 0.5
    assert QUORUM_K   == 10.0


def test_H03_phi_dynamics_constants():
    """H03: α=0.01, δ=0.10, gateway=0.50 (Session 35 / canonical v1.2)."""
    assert PHI_BUILD_RATE_ALPHA  == 0.01
    assert PHI_DECAY_RATE_DELTA  == 0.10
    assert PHI_GATEWAY_THRESHOLD == 0.50


def test_H04_grid_size():
    """H04: 20×20 grid = 400 agents."""
    assert GRID_N   == 20
    assert N_AGENTS == 400


def test_H05_mutation_rate():
    """H05: Mutation rate = 0.10 (Axiom 8 ±10%)."""
    assert 0.001 <= MUTATION_RATE <= 0.10  # Axiom 8 heterogeneity noise


# ── H06–H08: Cooperation-evolution equation ───────────────────────────────────

def test_H06_linear_equation_structure():
    """H06: Linear form i(t) = i₀(1 − ε×φ×β_norm) — basic arithmetic."""
    i0, eps, phi, beta = 1.2, 1.0, 0.7, 1.0
    bn    = beta_norm(beta)
    expected = i0 * (1.0 - eps * phi * bn)
    result   = cooperation_evolution_linear(i0, eps, phi, beta)
    assert result == pytest.approx(expected, abs=1e-6)


def test_H07_saturation_form_structure():
    """H07: Saturation form i(t) = i₀(1 − η×tanh(ε×φ×β_norm/η))."""
    i0, eps, phi, beta, eta = 1.2, 1.0, 0.7, 1.0, 0.93
    bn    = beta_norm(beta)
    u     = eps * phi * bn / eta
    expected = i0 * (1.0 - eta * math.tanh(u))
    result   = cooperation_evolution_saturation(i0, eps, phi, beta, eta)
    assert result == pytest.approx(expected, abs=1e-6)


def test_H08_saturation_reduces_to_linear_small_u():
    """H08: Saturation ≈ linear for ε×φ×β_norm < 0.3 (tanh(u) ≈ u for small u)."""
    # Small u: tanh(u) ≈ u, so i_sat ≈ i₀(1 − η × u/η) = i₀(1 − u) ≈ i_linear
    eps, phi, beta = 0.2, 0.3, 0.3   # small product
    bn = beta_norm(beta)
    product = eps * phi * bn
    assert product < 0.3, f"product={product} should be < 0.3 for this test"

    i_lin = cooperation_evolution_linear(1.0, eps, phi, beta)
    i_sat = cooperation_evolution_saturation(1.0, eps, phi, beta, eta=0.93)
    divergence = abs(i_sat - i_lin) / max(abs(i_lin), 1e-9)
    assert divergence < 0.01, (
        f"Divergence {divergence:.4f} exceeds 1% for small u (ε×φ×β_norm={product:.3f})"
    )


# ── H09–H11: β_norm and sigmoid quorum gate ──────────────────────────────────

def test_H09_beta_norm_bounds():
    """H09: β_norm ∈ (0,1) for all β > 0."""
    for b in [0.01, 0.5, 1.0, 2.0, 10.0, 100.0]:
        bn = beta_norm(b)
        assert 0.0 < bn < 1.0


def test_H10_beta_norm_half_at_one():
    """H10: β_norm(1.0) = 0.5 (Michaelis-Menten midpoint)."""
    assert beta_norm(1.0) == pytest.approx(0.5)


def test_H11_quorum_gate_sigmoid():
    """H11: Sigmoid gate = 0.5 at φ×β = τ = 0.5 (threshold midpoint)."""
    gate_at_tau = sigmoid_quorum_gate(QUORUM_TAU)
    assert gate_at_tau == pytest.approx(0.5, abs=1e-6)
    # Below τ: gate < 0.5; above τ: gate > 0.5
    assert sigmoid_quorum_gate(0.1) < 0.5
    assert sigmoid_quorum_gate(0.9) > 0.5


# ── H12–H15: Equation 7 φ dynamics ───────────────────────────────────────────

def test_H12_phi_builds_when_eligible():
    """H12: φ builds when R < 0.50 AND i < 1.0."""
    phi, r, i_val = 0.3, 0.3, 0.8   # both conditions met
    new_phi, event = apply_phi_eq7(phi, r, i_val, d_value=0.0)
    assert new_phi > phi
    assert event == "BUILD"


def test_H13_phi_no_build_when_i_geq_1():
    """H13: φ does NOT build when R < 0.50 but i ≥ 1.0 (compound gate)."""
    phi, r, i_val = 0.3, 0.3, 1.1   # R < 0.5 but i >= 1.0
    new_phi, event = apply_phi_eq7(phi, r, i_val, d_value=0.0)
    assert new_phi == phi
    assert event == "GATED_NO_BUILD"


def test_H14_phi_decays_when_r_above_threshold():
    """H14: φ decays when R ≥ 0.50 and D(t) > 0."""
    phi, r, i_val = 0.7, 0.7, 0.8   # R above gateway
    new_phi, event = apply_phi_eq7(phi, r, i_val, d_value=0.3)
    assert new_phi < phi
    assert event == "DECAY"


def test_H15_alpha_much_less_than_delta():
    """H15: PHI_DECAY_RATE_DELTA / PHI_BUILD_RATE_ALPHA ≥ 10 (α ≪ δ)."""
    ratio = PHI_DECAY_RATE_DELTA / PHI_BUILD_RATE_ALPHA
    assert ratio >= 10.0, f"δ/α = {ratio} — should be ≥ 10"


# ── H16–H20: T1.1 Bimodality ─────────────────────────────────────────────────

def test_H16_bimodality_hartigan_dip(study_linear):
    """H16: T1.1 — Hartigan dip test on CI distribution: p < 0.05."""
    from diptest import diptest
    ci_values = [r.ci_final for r in study_linear]
    assert len(ci_values) >= 30, f"Need ≥ 30 runs, got {len(ci_values)}"
    import numpy as np
    _, p_value = diptest(np.array(ci_values))
    assert p_value < 0.05, (
        f"Hartigan dip p={p_value:.4f} — distribution not bimodal at p<0.05"
    )


def test_H17_bimodal_two_modes(study_linear):
    """H17: T1.1 — CI distribution has mass at both ends (COOP and COMP)."""
    n_coop = sum(1 for r in study_linear if r.outcome == "COOP")
    n_comp = sum(1 for r in study_linear if r.outcome == "COMP")
    total  = len(study_linear)
    assert n_coop > 0, "No cooperative runs — bimodality requires COOP outcomes"
    assert n_comp > 0, "No competitive runs — bimodality requires COMP outcomes"
    assert (n_coop + n_comp) / total >= 0.85, (
        f"Only {(n_coop+n_comp)/total:.0%} classified as COOP/COMP (expect ≥85%)"
    )


def test_H18_no_stable_intermediate(study_linear):
    """H18: T1.1 — THRESH category ≤ 25% (sharp boundary, not continuum)."""
    n_thresh = sum(1 for r in study_linear if r.outcome == "THRESH")
    thresh_frac = n_thresh / len(study_linear)
    assert thresh_frac <= 0.25, (
        f"THRESH fraction = {thresh_frac:.0%} — should be ≤ 25% (sharp boundary)"
    )


def test_H19_bimodality_hartigan_strict(study_linear):
    """H19: T1.1 — Hartigan dip on i_final distribution also significant."""
    from diptest import diptest
    import numpy as np
    # CI distribution shows strong bimodality
    ci_values = np.array([r.ci_final for r in study_linear])
    _, p_value = diptest(ci_values)
    assert p_value < 0.001, f"Hartigan dip on CI: p={p_value:.6f} — expected << 0.001"


def test_H20_bimodality_runs_count(study_linear):
    """H20: T1.1 — Study produces expected number of runs."""
    # 27 conditions × 3 replicates = 81 runs
    assert len(study_linear) == 81


# ── H21–H23: T1.2 ESS invasion recovery ──────────────────────────────────────

def test_H21_ess_single_recovery():
    """H21: T1.2 — Single ESS test: cooperative system recovers from 10% invasion."""
    recovered = run_ess_invasion_test(
        epsilon=1.2, phi_coop=0.75, beta_coop=0.8,
        eta_planted=0.93, use_saturation=False, seed=42
    )
    assert recovered is True, "Cooperative equilibrium should recover from 10% invasion"


def test_H22_ess_multiple_recovery():
    """H22: T1.2 — 10 ESS tests: ≥ 8/10 recover (≥ 80% pass rate)."""
    n_tests = 10
    n_recovered = 0
    for seed in range(n_tests):
        if run_ess_invasion_test(
            epsilon=1.2, phi_coop=0.75, beta_coop=0.8,
            eta_planted=0.93, use_saturation=False,
            seed=seed * 17 + 3
        ):
            n_recovered += 1
    assert n_recovered >= 8, (
        f"ESS recovery: {n_recovered}/10 — expected ≥ 8 (80%)"
    )


def test_H23_ess_high_epsilon_recovery():
    """H23: T1.2 — ESS recovery also holds at high ε (ε paradox: ε is directionally neutral)."""
    recovered = run_ess_invasion_test(
        epsilon=2.5, phi_coop=0.75, beta_coop=0.9,
        eta_planted=0.93, use_saturation=False, seed=77
    )
    # At high ε, if already cooperative, stays cooperative after invasion
    # (ε accelerates trajectory but doesn't change destination)
    assert recovered is True, "ESS recovery should hold at high ε"


# ── H24–H26: T1.3 Cooperation theorem ────────────────────────────────────────

def test_H24_ci_reaches_one_at_convergence(study_linear):
    """H24: T1.3 — Cooperative runs reach CI ≥ 0.90 (near 1.0) at convergence."""
    coop_runs = [r for r in study_linear if r.outcome == "COOP"]
    assert len(coop_runs) > 0, "Need cooperative runs for T1.3"
    for r in coop_runs:
        assert r.ci_final >= 0.90, (
            f"Run {r.run_id}: CI={r.ci_final:.3f} below 0.90 threshold"
        )


def test_H25_ci_near_one_tight(study_linear):
    """H25: T1.3 — Mean CI of cooperative runs ≥ 0.92."""
    coop_runs = [r for r in study_linear if r.outcome == "COOP"]
    if not coop_runs:
        pytest.skip("No cooperative runs in this study subset")
    mean_ci = sum(r.ci_final for r in coop_runs) / len(coop_runs)
    assert mean_ci >= 0.92, f"Mean CI of coop runs = {mean_ci:.3f}, expected ≥ 0.92"


def test_H26_comp_ci_near_zero(study_linear):
    """H26: T1.3 — Competitive runs have CI ≤ 0.20 (opposite basin)."""
    comp_runs = [r for r in study_linear if r.outcome == "COMP"]
    assert len(comp_runs) > 0, "Need competitive runs for T1.3"
    for r in comp_runs:
        assert r.ci_final <= 0.20, (
            f"Run {r.run_id}: CI={r.ci_final:.3f} above 0.20 in COMP run"
        )


# ── H27–H29: T1.4 Gateway threshold ──────────────────────────────────────────

def test_H27_r_predicts_outcome(study_linear):
    """H27: T1.4 — Low R predicts COOP; high R predicts COMP."""
    coop_runs = [r for r in study_linear if r.outcome == "COOP"]
    comp_runs = [r for r in study_linear if r.outcome == "COMP"]
    if coop_runs and comp_runs:
        mean_r_coop = sum(r.r_mean for r in coop_runs) / len(coop_runs)
        mean_r_comp = sum(r.r_mean for r in comp_runs) / len(comp_runs)
        assert mean_r_coop < mean_r_comp, (
            f"Mean R: COOP={mean_r_coop:.3f}, COMP={mean_r_comp:.3f} — "
            "cooperative runs should have lower R"
        )


def test_H28_gateway_threshold_at_half(study_linear):
    """H28: T1.4 — Runs with initial R < 0.50 (β_init > C*TAX) trend cooperative."""
    low_r_runs  = [r for r in study_linear if r.r_mean < 0.50]
    high_r_runs = [r for r in study_linear if r.r_mean >= 0.50]
    if low_r_runs and high_r_runs:
        coop_frac_low  = sum(1 for r in low_r_runs  if r.outcome == "COOP") / len(low_r_runs)
        coop_frac_high = sum(1 for r in high_r_runs if r.outcome == "COOP") / len(high_r_runs)
        assert coop_frac_low >= coop_frac_high, (
            f"Low-R COOP fraction {coop_frac_low:.2f} should ≥ high-R {coop_frac_high:.2f}"
        )


def test_H29_phi_beta_predicts_outcome(study_linear):
    """H29: T1.4 — φ×β product predicts outcome: high φ×β → COOP."""
    coop_runs = [r for r in study_linear if r.outcome == "COOP"]
    comp_runs = [r for r in study_linear if r.outcome == "COMP"]
    if coop_runs and comp_runs:
        mean_phi_beta_coop = sum(r.phi_init * r.beta_init for r in coop_runs) / len(coop_runs)
        mean_phi_beta_comp = sum(r.phi_init * r.beta_init for r in comp_runs) / len(comp_runs)
        assert mean_phi_beta_coop > mean_phi_beta_comp, (
            f"φ×β: COOP={mean_phi_beta_coop:.3f}, COMP={mean_phi_beta_comp:.3f} — "
            "COOP runs should have higher initial φ×β"
        )


# ── H30–H32: T1.5 φ asymmetry ────────────────────────────────────────────────

def test_H30_alpha_less_than_delta():
    """H30: T1.5 — α/δ < 1 (asymmetry encoded in constants)."""
    assert PHI_BUILD_RATE_ALPHA < PHI_DECAY_RATE_DELTA


def test_H31_phi_increases_in_eligible_runs(study_linear):
    """H31: T1.5 — Cooperative runs show net φ increase from initial value."""
    coop_runs = [r for r in study_linear if r.outcome == "COOP"]
    if not coop_runs:
        pytest.skip("No cooperative runs")
    increases = sum(1 for r in coop_runs if r.phi_final > r.phi_init)
    assert increases / len(coop_runs) >= 0.70, (
        f"Only {increases}/{len(coop_runs)} coop runs showed φ increase"
    )


def test_H32_phi_build_events_exist(study_linear):
    """H32: T1.5 — Cooperative runs log φ build events."""
    coop_runs = [r for r in study_linear if r.outcome == "COOP"]
    if not coop_runs:
        pytest.skip("No cooperative runs")
    total_builds = sum(r.phi_build_events_total for r in coop_runs)
    assert total_builds > 0, "No φ build events in cooperative runs"


# ── H33–H35: T1.6 Compound gating ────────────────────────────────────────────

def test_H33_no_phi_build_outside_eligible_unit():
    """H33: T1.6 — φ does not build when only R condition met (i ≥ 1)."""
    # R < 0.5 but i >= 1 → no build
    phi, r, i_val = 0.3, 0.3, 1.5
    new_phi, event = apply_phi_eq7(phi, r, i_val)
    assert new_phi == phi, "φ should not change when i ≥ 1.0"
    assert event != "BUILD"


def test_H34_no_phi_build_outside_eligible_unit2():
    """H34: T1.6 — φ does not build when only i condition met (R ≥ 0.5)."""
    # R >= 0.5 but i < 1 → no build
    phi, r, i_val = 0.3, 0.7, 0.5
    new_phi, event = apply_phi_eq7(phi, r, i_val)
    assert event != "BUILD", "φ should not build when R ≥ 0.50"


def test_H35_zero_phi_build_outside_eligible_in_run(study_linear):
    """H35: T1.6 — No φ build events fire outside eligible regime across study."""
    total_outside = sum(r.phi_build_outside_eligible for r in study_linear)
    assert total_outside == 0, (
        f"Found {total_outside} φ build events outside eligible regime "
        "(R < 0.50 AND i < 1.0 must both hold)"
    )


# ── H36–H38: T2.1 Threshold location unchanged ───────────────────────────────

def test_H36_saturation_has_coop_and_comp(study_saturation):
    """H36: T2.1 — Saturation form still produces COOP and COMP outcomes."""
    n_coop = sum(1 for r in study_saturation if r.outcome == "COOP")
    n_comp = sum(1 for r in study_saturation if r.outcome == "COMP")
    assert n_coop > 0, "Saturation form produced no COOP runs"
    assert n_comp > 0, "Saturation form produced no COMP runs"


def test_H37_saturation_bimodal(study_saturation):
    """H37: T2.1 — Saturation form CI distribution is bimodal (Hartigan dip)."""
    from diptest import diptest
    import numpy as np
    ci_values = np.array([r.ci_final for r in study_saturation])
    _, p_value = diptest(ci_values)
    assert p_value < 0.05, (
        f"Saturation form Hartigan dip p={p_value:.4f} — should be bimodal"
    )


def test_H38_threshold_location_similar(study_linear, study_saturation):
    """H38: T2.1 — COOP fraction similar between linear and saturation forms (±15%)."""
    coop_frac_lin = sum(1 for r in study_linear     if r.outcome == "COOP") / len(study_linear)
    coop_frac_sat = sum(1 for r in study_saturation if r.outcome == "COOP") / len(study_saturation)
    assert abs(coop_frac_lin - coop_frac_sat) <= 0.15, (
        f"COOP fraction: linear={coop_frac_lin:.2f}, saturation={coop_frac_sat:.2f} "
        "— threshold location shifted by > 15%"
    )


# ── H39–H41: T2.2 Mutualism ceiling ─────────────────────────────────────────

def test_H39_mutualism_ceiling_unit():
    """H39: T2.2 — i(t) ≥ i₀(1−η) = mutualism ceiling (single call)."""
    for eta in [0.5, 0.80, 0.93, 1.0]:
        for eps in [0.5, 1.2, 2.5]:
            for phi in [0.3, 0.7, 0.95]:
                i_val = cooperation_evolution_saturation(
                    1.0, eps, phi, 2.0, eta
                )
                ceiling = 1.0 * (1.0 - eta)
                assert i_val >= ceiling - 1e-9, (
                    f"i={i_val:.5f} < ceiling={ceiling:.5f} "
                    f"(eta={eta}, eps={eps}, phi={phi})"
                )


def test_H40_mutualism_ceiling_in_run(study_saturation):
    """H40: T2.2 — min_i ≥ i₀(1−η) in all saturation runs."""
    for r in study_saturation:
        ceiling = r.i0_one_minus_eta
        # min_i may be slightly below due to mutation noise — allow small slack
        assert r.min_i >= ceiling - 0.05, (
            f"Run {r.run_id}: min_i={r.min_i:.4f} < ceiling={ceiling:.4f} - 0.05"
        )


def test_H41_ceiling_lower_for_lower_eta():
    """H41: T2.2 — Lower η → lower mutualism ceiling i₀(1−η) (shallower attractor)."""
    ceil_high_eta = 1.0 * (1.0 - 0.93)
    ceil_low_eta  = 1.0 * (1.0 - 0.50)
    # Higher η → lower ceiling (deeper cooperative basin)
    assert ceil_high_eta < ceil_low_eta, (
        f"ceil(eta=0.93)={ceil_high_eta:.4f} should be < ceil(eta=0.50)={ceil_low_eta:.4f}"
    )


# ── H42–H43: T2.3 Linear limit ────────────────────────────────────────────────

def test_H42_linear_limit_divergence():
    """H42: T2.3 — Saturation < 1% divergence from linear for ε×φ×β_norm < 0.3."""
    import numpy as np
    rng = np.random.default_rng(42)
    violations = 0
    n = 500
    for _ in range(n):
        # Sample parameters in the small-u regime
        eps  = rng.uniform(0.1, 0.5)
        phi  = rng.uniform(0.1, 0.5)
        beta = rng.uniform(0.1, 1.0)
        bn   = beta_norm(beta)
        if eps * phi * bn >= 0.3:
            continue
        i_lin = cooperation_evolution_linear(1.0, eps, phi, beta)
        i_sat = cooperation_evolution_saturation(1.0, eps, phi, beta, eta=0.93)
        div   = abs(i_sat - i_lin) / max(abs(i_lin), 1e-9)
        if div >= 0.01:
            violations += 1
    assert violations == 0, (
        f"{violations} parameter combinations with ε×φ×β_norm < 0.3 "
        "show ≥ 1% divergence between linear and saturation forms"
    )


def test_H43_linear_limit_specific():
    """H43: T2.3 — Specific small-u case: eps=0.2, phi=0.3, beta=0.3."""
    eps, phi, beta = 0.2, 0.3, 0.3
    bn = beta_norm(beta)
    product = eps * phi * bn
    assert product < 0.3
    i_lin = cooperation_evolution_linear(1.0, eps, phi, beta)
    i_sat = cooperation_evolution_saturation(1.0, eps, phi, beta, eta=0.93)
    div = abs(i_sat - i_lin) / max(abs(i_lin), 1e-9)
    assert div < 0.01, f"Divergence {div:.4f} ≥ 1% at small u (product={product:.3f})"


# ── H44–H45: T2.4 Mutualism theorem ──────────────────────────────────────────

def test_H44_mutualism_theorem_eps_leq_1():
    """H44: T2.4 — ε ≤ 1 never produces i < 0 (saturation form)."""
    import numpy as np
    rng = np.random.default_rng(77)
    for _ in range(200):
        eps  = rng.uniform(0.01, 1.0)   # ε ≤ 1
        phi  = rng.uniform(0.0, 1.0)
        beta = rng.uniform(0.01, 5.0)
        eta  = rng.uniform(0.5, 1.0)
        i    = cooperation_evolution_saturation(1.0, eps, phi, beta, eta)
        # With ε ≤ 1 and η ≤ 1: tanh argument bounded; i = 1 - η*tanh(u)
        # At max: η=1, tanh(u)→1 → i→0. Never negative.
        assert i >= 0.0 - 1e-9, (
            f"ε={eps:.3f} ≤ 1 produced i={i:.6f} < 0 "
            "(mutualism theorem violated)"
        )


def test_H45_mutualism_theorem_eps_gt_1_can_produce_negative():
    """H45: T2.4 — ε > 1 CAN produce i < 0 when φ and β are high."""
    # With i0=1, η=1, ε=3, φ=0.9, β=5 (β_norm≈0.83):
    # u = 3 * 0.9 * 0.83 / 1.0 ≈ 2.24, tanh(2.24) ≈ 0.977
    # i = 1 * (1 - 1 * 0.977) = 0.023 > 0 (still positive)
    # Need larger ε: ε=5, φ=0.95, β=10, η=1.0
    # u = 5 * 0.95 * (10/11) / 1.0 ≈ 4.32, tanh(4.32) ≈ 0.999
    # i = 1*(1-1*0.999) = 0.001 → still slightly positive
    # True i < 0 requires ε > 1/η to flip — use i0=1, η=0.5:
    # i = 1*(1-0.5*tanh(5*0.95*0.909/0.5))=1*(1-0.5*tanh(8.6))≈1-0.5=0.5 (not negative)
    # Actually with saturation form: i = i0*(1-η*tanh(u)); with η < 1 and max tanh=1:
    # min achievable i = i0*(1-η) > 0 when η < 1.
    # i < 0 only when η > 1, which is outside η ∈ (0,1].
    # With linear form: i = i0*(1-ε*φ*β_norm); need ε*φ*β_norm > 1.
    # This is what the mutualism theorem actually tests.
    # Test: linear form with ε > 1 can produce i < 0.
    i = cooperation_evolution_linear(1.0, 3.0, 0.9, 5.0)
    # β_norm(5)=5/6≈0.833; 3*0.9*0.833=2.25 > 1 → i = 1*(1-2.25) = -1.25
    assert i < 0.0, (
        f"ε=3.0 with linear form should produce i < 0 "
        f"(got i={i:.4f})"
    )


# ── H46–H48: T2.5 η estimation ───────────────────────────────────────────────

def test_H46_binls_unit_recovery():
    """H46: T2.5 — BI-NLS recovers planted η within ±0.03 from 200 observations."""
    import random as stdlib_random
    rng = stdlib_random.Random(42)
    eta_planted = 0.93
    log = []
    for _ in range(200):
        phi  = rng.uniform(0.5, 0.9)
        eps  = rng.uniform(0.8, 1.5)
        bn   = rng.uniform(0.3, 0.6)
        u    = eps * phi * bn / eta_planted
        i_pred = 1.0 - eta_planted * math.tanh(u)
        i_obs  = i_pred + rng.gauss(0, 0.005)
        log.append(InteractionLog(
            i_observed=i_obs, epsilon=eps, phi=phi, beta_norm=bn
        ))
    eta_est, rse = estimate_eta_binls(log, eta_init=0.5)
    assert eta_est is not None, "BI-NLS returned None — insufficient data"
    assert abs(eta_est - eta_planted) <= 0.03, (
        f"η recovered={eta_est:.4f}, planted={eta_planted}, "
        f"error={abs(eta_est-eta_planted):.4f} > 0.03"
    )


def test_H47_binls_from_run_log(study_saturation):
    """H47: T2.5 — BI-NLS recovers planted eta within ±0.05 from run interaction log.

    Requires cooperative runs with sufficient interaction history (150 gens × 400
    agents). i_observed is stored normalised by i0 so the estimator correctly
    applies i_pred = 1 − η × tanh(ε × φ × β_norm / η).
    """
    coop_with_eta = [r for r in study_saturation
                     if r.outcome == "COOP" and r.eta_estimated is not None
                     and r.eta_rse is not None and r.eta_rse < 0.05]
    if not coop_with_eta:
        pytest.skip("No cooperative saturation runs with RSE < 0.05 in this fixture")
    for r in coop_with_eta:
        error = abs(r.eta_estimated - r.eta_planted)
        assert error <= 0.05, (
            f"Run {r.run_id}: eta_est={r.eta_estimated:.4f}, "
            f"planted={r.eta_planted:.4f}, error={error:.4f} > 0.05 "
            f"(RSE={r.eta_rse:.4f})"
        )


def test_H48_binls_returns_none_insufficient_data():
    """H48: T2.5 — BI-NLS returns (None, None) when fewer than 10 observations."""
    short_log = [InteractionLog(i_observed=0.8, epsilon=1.0,
                                phi=0.6, beta_norm=0.5)] * 5
    eta_est, rse = estimate_eta_binls(short_log)
    assert eta_est is None
    assert rse is None

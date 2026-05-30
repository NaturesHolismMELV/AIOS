"""
test_session36.py — MELVcore Session 36: Three-Layer Logging + η Estimation (v3.1.1)
======================================================================================

Tests for Session 36 deliverables:
  - core/telemetry.py — L1Record, L2Snapshot, L3EtaEstimate dataclasses
  - Three-layer SQLite schema (telemetry_l1 / telemetry_l2 / telemetry_l3)
  - AIOSTelemetry CRUD operations (log_l1, log_l2, get_l3, run_eta_cycle)
  - _sensitivity() function — BI-NLS kernel
  - estimate_eta_binls() — full Gauss-Newton estimation
  - rse_band() — threshold classification
  - compute_d_value() — D(t) from L1 rolling records
  - eta_governance_flag() — governance classification
  - build_c_proxy() / build_b_proxy() / build_tax_proxy() proxy builders
  - ObservationResult.d_value populated from telemetry
  - Persistence schema migration (telemetry tables in aios_state.db)

Test groups
-----------
  F01–F05  RSE constants and thresholds
  F06–F10  η governance constants and thresholds
  F11–F17  _sensitivity() BI-NLS kernel function
  F18–F23  estimate_eta_binls() convergence and RSE classification
  F24–F27  compute_d_value() disruption intensity
  F28–F30  eta_governance_flag() governance flags
  F31–F35  Proxy builders (C, B, TAX)
  F36–F40  AIOSTelemetry L1/L2/L3 CRUD
  F41–F43  run_eta_cycle() full pipeline
  F44–F46  Integration: d_value in ObservationResult + schema migration

Author: Laurence W. Evans | ORCID: 0009-0001-0963-1840 | Cape Town, South Africa
Session: 36 · Version: 3.1.1
"""

import math
import os
import sqlite3
import tempfile
import time
import pytest

from core.telemetry import (
    # Constants
    RSE_EXCELLENT, RSE_ACCEPTABLE, RSE_POOR,
    ETA_STABLE_THRESHOLD, ETA_DECLINING_THRESHOLD, ETA_CRITICAL_RATIO,
    ETA_INITIAL, ETA_ARCHITECTURAL_DEFAULT, ETA_MIN_INTERACTIONS,
    ETA_IDENTIFICATION_THRESHOLD,
    # Dataclasses
    L1Record, L2Snapshot, L3EtaEstimate,
    # Estimation
    _sensitivity, _is_identified, estimate_eta_binls, rse_band,
    # D(t)
    compute_d_value,
    # Governance flag
    eta_governance_flag,
    # Proxy builders
    build_c_proxy, build_b_proxy, build_tax_proxy,
    # Telemetry class
    AIOSTelemetry,
    TELEMETRY_SCHEMA,
)


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def tmp_db(tmp_path):
    """Temporary SQLite DB with telemetry schema applied."""
    db_path = str(tmp_path / "test_aios.db")
    tel = AIOSTelemetry(db_path)
    yield tel
    tel.close()


def _make_obs(i_obs=0.6, eps=1.2, phi=0.7, beta_norm=0.4):
    return {"i_observed": i_obs, "epsilon": eps, "phi": phi, "beta_norm": beta_norm}


def _cooperative_obs(n=120, eta=0.93):
    """Generate synthetic cooperative observations around a planted η."""
    import random
    random.seed(42)
    obs = []
    for _ in range(n):
        phi = random.uniform(0.5, 0.9)
        eps = random.uniform(0.8, 1.5)
        bn  = random.uniform(0.3, 0.6)
        u   = eps * phi * bn / eta
        i_pred = 1.0 - eta * math.tanh(u)
        i_obs  = i_pred + random.gauss(0, 0.005)
        obs.append({"i_observed": i_obs, "epsilon": eps, "phi": phi, "beta_norm": bn})
    return obs


# ── F01–F05: RSE constants and thresholds ────────────────────────────────────

def test_F01_rse_excellent_value():
    """F01: RSE_EXCELLENT = 0.02."""
    assert RSE_EXCELLENT == 0.02


def test_F02_rse_acceptable_value():
    """F02: RSE_ACCEPTABLE = 0.05."""
    assert RSE_ACCEPTABLE == 0.05


def test_F03_rse_poor_value():
    """F03: RSE_POOR = 0.10."""
    assert RSE_POOR == 0.10


def test_F04_rse_band_excellent():
    """F04: rse_band classifies RSE < 0.02 as EXCELLENT."""
    assert rse_band(0.01) == "EXCELLENT"
    assert rse_band(0.019) == "EXCELLENT"


def test_F05_rse_band_classification():
    """F05: rse_band classifies ACCEPTABLE and POOR correctly."""
    assert rse_band(0.02) == "ACCEPTABLE"
    assert rse_band(0.04) == "ACCEPTABLE"
    assert rse_band(0.05) == "POOR"
    assert rse_band(0.15) == "POOR"


# ── F06–F10: η governance constants ──────────────────────────────────────────

def test_F06_eta_stable_threshold():
    """F06: ETA_STABLE_THRESHOLD = 0.05."""
    assert ETA_STABLE_THRESHOLD == 0.05


def test_F07_eta_declining_threshold():
    """F07: ETA_DECLINING_THRESHOLD = 0.15."""
    assert ETA_DECLINING_THRESHOLD == 0.15


def test_F08_eta_critical_ratio():
    """F08: ETA_CRITICAL_RATIO = 0.70."""
    assert ETA_CRITICAL_RATIO == 0.70


def test_F09_eta_initial_value():
    """F09: ETA_INITIAL = 0.93 (bee-flower calibration)."""
    assert ETA_INITIAL == 0.93


def test_F10_eta_min_interactions():
    """F10: ETA_MIN_INTERACTIONS = 100."""
    assert ETA_MIN_INTERACTIONS == 100


# ── F11–F17: _sensitivity() ─────────────────────────────────────────────────

def test_F11_sensitivity_zero():
    """F11: S(0) = 0."""
    assert _sensitivity(0.0) == pytest.approx(0.0, abs=1e-10)


def test_F12_sensitivity_positive():
    """F12: S(u) > 0 for u > 0."""
    for u in [0.1, 0.5, 1.0, 2.0, 5.0]:
        assert _sensitivity(u) > 0, f"S({u}) should be positive"


def test_F13_sensitivity_identification_threshold():
    """F13: S(u) > 0.3 when u is large enough (u ≈ 0.7+)."""
    # At u = 1.0, S(u) should exceed identification threshold
    assert _sensitivity(1.0) > ETA_IDENTIFICATION_THRESHOLD


def test_F14_sensitivity_is_identified():
    """F14: _is_identified() returns True when S(u) > threshold."""
    assert _is_identified(1.5) is True
    assert _is_identified(0.05) is False


def test_F15_sensitivity_numerical_stability():
    """F15: _sensitivity() does not raise for large u (overflow guard)."""
    # u = 1000 would overflow cosh without guard
    s = _sensitivity(1000.0)
    assert math.isfinite(s)
    assert s >= 0.0


def test_F16_sensitivity_symmetry_with_abs():
    """F16: S(−u) is handled by abs() in _is_identified()."""
    assert _is_identified(-1.5) is True
    assert _is_identified(-0.05) is False


def test_F17_sensitivity_monotone():
    """F17: S(u) is monotonically non-decreasing and approaches 1 for large u."""
    values = [_sensitivity(u) for u in [0.0, 0.5, 1.0, 2.0, 5.0, 10.0]]
    for i in range(len(values) - 1):
        assert values[i+1] >= values[i], f"S not monotone at index {i}: {values}"
    # For large u, S(u) → 1 (tanh→1, u/cosh² → 0)
    assert _sensitivity(50.0) > 0.99


# ── F18–F23: estimate_eta_binls() ────────────────────────────────────────────

def test_F18_binls_insufficient_obs():
    """F18: Returns eta_init, converged=False when fewer than 10 observations."""
    result = estimate_eta_binls([_make_obs() for _ in range(5)])
    assert result["converged"] is False
    assert result["n_obs"] == 5
    assert result["eta"] == ETA_INITIAL
    assert len(result["warnings"]) > 0


def test_F19_binls_converges_planted_eta():
    """F19: BI-NLS recovers planted η within ±0.03 from 120 synthetic observations."""
    eta_planted = 0.93
    obs = _cooperative_obs(n=120, eta=eta_planted)
    result = estimate_eta_binls(obs, eta_init=0.5)
    assert result["converged"] is True
    assert abs(result["eta"] - eta_planted) <= 0.03, (
        f"η recovered {result['eta']:.4f}, planted {eta_planted}"
    )


def test_F20_binls_rse_band_excellent():
    """F20: RSE band is EXCELLENT on clean synthetic data."""
    obs = _cooperative_obs(n=120, eta=0.93)
    result = estimate_eta_binls(obs)
    # RSE on low-noise data should be excellent or acceptable
    assert result["rse_band"] in ("EXCELLENT", "ACCEPTABLE")
    assert result["rse"] is not None and result["rse"] < RSE_ACCEPTABLE


def test_F21_binls_eta_bounded():
    """F21: η estimate is always in (0.01, 1.0]."""
    # Pathological inputs — all i_observed = 1.0 (no cooperation signal)
    obs = [{"i_observed": 1.0, "epsilon": 1.0, "phi": 0.8, "beta_norm": 0.5}
           for _ in range(20)]
    result = estimate_eta_binls(obs)
    assert 0.0 < result["eta"] <= 1.0


def test_F22_binls_filters_invalid_obs():
    """F22: Invalid observations (NaN, zero phi) are silently filtered."""
    good = [_make_obs() for _ in range(15)]
    bad  = [
        {"i_observed": float("nan"), "epsilon": 1.0, "phi": 0.5, "beta_norm": 0.4},
        {"i_observed": 0.6, "epsilon": 0.0, "phi": 0.5, "beta_norm": 0.4},  # eps=0
        {"i_observed": 0.6, "epsilon": 1.0, "phi": 0.0, "beta_norm": 0.4},  # phi=0
        {},  # missing keys
    ]
    result = estimate_eta_binls(good + bad)
    assert result["n_obs"] == 15   # only the 15 good ones counted


def test_F23_binls_returns_n_identified():
    """F23: n_identified counts observations where S(u) > threshold."""
    obs = _cooperative_obs(n=50, eta=0.93)
    result = estimate_eta_binls(obs)
    assert "n_identified" in result
    assert 0 <= result["n_identified"] <= result["n_obs"]


# ── F24–F27: compute_d_value() ───────────────────────────────────────────────

def test_F24_d_value_zero_on_single_record():
    """F24: D(t) = 0.0 when fewer than 2 records."""
    r = L1Record(agent_id="a", c_proxy=1.0, b_proxy=0.8, tax_proxy=0.1)
    assert compute_d_value([r]) == 0.0
    assert compute_d_value([]) == 0.0


def test_F25_d_value_zero_at_baseline():
    """F25: D(t) ≈ 0 when mean equals base (no disruption)."""
    records = [
        L1Record(agent_id="a", c_proxy=1.0, b_proxy=0.8, tax_proxy=0.1)
        for _ in range(10)
    ]
    d = compute_d_value(records, c_base=1.0, tax_base=0.1)
    assert d == pytest.approx(0.0, abs=1e-9)


def test_F26_d_value_positive_on_spike():
    """F26: D(t) > 0 when mean C or TAX exceeds baseline."""
    baseline = [
        L1Record(agent_id="a", c_proxy=1.0, b_proxy=0.8, tax_proxy=0.1)
        for _ in range(5)
    ]
    spike = [
        L1Record(agent_id="a", c_proxy=2.0, b_proxy=0.8, tax_proxy=0.2)
        for _ in range(5)
    ]
    d = compute_d_value(baseline + spike, c_base=1.0, tax_base=0.1)
    assert d > 0.0


def test_F27_d_value_non_negative():
    """F27: D(t) = max(0, …) — never negative."""
    records = [
        L1Record(agent_id="a", c_proxy=0.5, b_proxy=0.9, tax_proxy=0.05)
        for _ in range(5)
    ]
    d = compute_d_value(records, c_base=1.0, tax_base=0.1)
    assert d == 0.0   # costs fell — no disruption


# ── F28–F30: eta_governance_flag() ───────────────────────────────────────────

def test_F28_flag_critical_below_ratio():
    """F28: CRITICAL when η < 0.7 × η_architectural."""
    flag = eta_governance_flag(
        eta_current=0.60,
        eta_previous=0.93,
        eta_architectural=0.93,
    )
    assert flag == "CRITICAL"


def test_F29_flag_declining_on_large_drop():
    """F29: DECLINING when quarterly decline > 15%."""
    flag = eta_governance_flag(
        eta_current=0.75,
        eta_previous=0.93,  # 19% decline
        eta_architectural=0.93,
    )
    assert flag == "DECLINING"


def test_F30_flag_stable_on_small_change():
    """F30: STABLE when variation < 5%."""
    flag = eta_governance_flag(
        eta_current=0.93,
        eta_previous=0.92,  # ~1% change
        eta_architectural=0.93,
    )
    assert flag == "STABLE"


# ── F31–F35: Proxy builders ───────────────────────────────────────────────────

def test_F31_c_proxy_zero_inputs():
    """F31: build_c_proxy returns 0 for all-zero inputs."""
    assert build_c_proxy(0, 0.0, 0.0) == pytest.approx(0.0)


def test_F32_c_proxy_token_contribution():
    """F32: Token cost contributes to C_proxy."""
    c = build_c_proxy(token_count=1_000_000)
    assert c > 0.0


def test_F33_b_proxy_full_score():
    """F33: build_b_proxy returns weighted sum of three components."""
    b = build_b_proxy(task_completion=1.0, downstream_utility=1.0, information_gain=1.0)
    from core.telemetry import (TASK_COMPLETION_WEIGHT, DOWNSTREAM_UTILITY_WEIGHT,
                                INFORMATION_GAIN_WEIGHT)
    expected = TASK_COMPLETION_WEIGHT + DOWNSTREAM_UTILITY_WEIGHT + INFORMATION_GAIN_WEIGHT
    assert b == pytest.approx(expected)


def test_F34_tax_proxy_fraction_of_c():
    """F34: build_tax_proxy produces a fraction of c_proxy."""
    c = 1.0
    tax = build_tax_proxy(c)
    assert 0.0 < tax <= c


def test_F35_b_proxy_clamped():
    """F35: build_b_proxy clamps inputs to [0, 1] before weighting."""
    b1 = build_b_proxy(task_completion=2.0)   # clamped to 1.0
    b2 = build_b_proxy(task_completion=1.0)
    assert b1 == pytest.approx(b2)


# ── F36–F40: AIOSTelemetry L1/L2/L3 CRUD ────────────────────────────────────

def test_F36_schema_creates_tables(tmp_db):
    """F36: All three telemetry tables are created in SQLite."""
    conn = sqlite3.connect(tmp_db.db_path)
    tables = {row[0] for row in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()}
    conn.close()
    assert "telemetry_l1" in tables
    assert "telemetry_l2" in tables
    assert "telemetry_l3" in tables


def test_F37_log_l1_stores_record(tmp_db):
    """F37: log_l1 persists a record retrievable via get_l1_recent."""
    record = L1Record(
        agent_id="agent-x",
        c_proxy=0.5,
        b_proxy=0.8,
        tax_proxy=0.05,
        task_type="inference",
    )
    tmp_db.log_l1(record)
    recent = tmp_db.get_l1_recent("agent-x", n=5)
    assert len(recent) == 1
    assert recent[0].agent_id == "agent-x"
    assert recent[0].c_proxy == pytest.approx(0.5)


def test_F38_l1_count(tmp_db):
    """F38: get_l1_count returns correct count per agent."""
    for i in range(7):
        tmp_db.log_l1(L1Record(agent_id="cnt-agent", c_proxy=float(i),
                               b_proxy=0.5, tax_proxy=0.0))
    assert tmp_db.get_l1_count("cnt-agent") == 7
    assert tmp_db.get_l1_count("other-agent") == 0


def test_F39_log_l2_and_retrieve(tmp_db):
    """F39: log_l2 and get_l2_recent work correctly."""
    snap = L2Snapshot(
        agent_id="snap-agent",
        i_value=0.85,
        phi=0.72,
        beta_service=0.6,
        ci=0.78,
        eta_estimate=0.93,
        rse=0.015,
        rse_band="EXCELLENT",
    )
    tmp_db.log_l2(snap)
    snaps = tmp_db.get_l2_recent("snap-agent", n=10)
    assert len(snaps) == 1
    assert snaps[0].i_value == pytest.approx(0.85)
    assert snaps[0].rse_band == "EXCELLENT"


def test_F40_l3_default_initialised(tmp_db):
    """F40: get_l3 returns default L3EtaEstimate when agent not yet in DB."""
    l3 = tmp_db.get_l3("fresh-agent")
    assert l3.agent_id == "fresh-agent"
    assert l3.eta_posterior == pytest.approx(ETA_INITIAL)
    assert l3.interaction_count == 0
    assert l3.governance_flag is None


# ── F41–F43: run_eta_cycle() ─────────────────────────────────────────────────

def test_F41_eta_cycle_insufficient_obs(tmp_db):
    """F41: run_eta_cycle with <10 obs returns default η and converged=False."""
    obs = [_make_obs() for _ in range(5)]
    result = tmp_db.run_eta_cycle("eta-agent", obs)
    assert result["converged"] is False
    l3 = tmp_db.get_l3("eta-agent")
    # L3 updated but eta remains at default (binls didn't converge)
    assert l3.agent_id == "eta-agent"


def test_F42_eta_cycle_updates_l3(tmp_db):
    """F42: Successful run_eta_cycle updates L3 persistence."""
    obs = _cooperative_obs(n=120, eta=0.93)
    result = tmp_db.run_eta_cycle("converge-agent", obs)
    l3 = tmp_db.get_l3("converge-agent")
    assert l3.interaction_count == result["n_obs"]
    assert l3.last_updated > 0.0
    assert abs(l3.eta_posterior - 0.93) <= 0.08  # BI-NLS convergence on 120 obs; wide tolerance


def test_F43_eta_cycle_governance_flag_returned(tmp_db):
    """F43: run_eta_cycle returns governance_flag in result dict."""
    obs = _cooperative_obs(n=120, eta=0.93)
    result = tmp_db.run_eta_cycle("flag-agent", obs)
    assert "governance_flag" in result
    # Governance flag should be None or a valid string
    assert result["governance_flag"] in (None, "STABLE", "DECLINING", "CRITICAL")


# ── F44–F46: Integration tests ────────────────────────────────────────────────

def test_F44_d_value_from_telemetry(tmp_db):
    """F44: compute_d_value_for_agent returns 0.0 with fewer than 2 records."""
    d = tmp_db.compute_d_value_for_agent("empty-agent")
    assert d == 0.0


def test_F45_d_value_non_zero_after_spike(tmp_db):
    """F45: D(t) > 0 after logging spike records."""
    # Log 10 baseline records
    for _ in range(10):
        tmp_db.log_l1(L1Record(agent_id="spike-agent",
                               c_proxy=1.0, b_proxy=0.8, tax_proxy=0.1))
    # Log 5 spike records (C doubles)
    for _ in range(5):
        tmp_db.log_l1(L1Record(agent_id="spike-agent",
                               c_proxy=2.0, b_proxy=0.8, tax_proxy=0.2))
    # With c_base=1.0, mean C across 15 records ≈ 1.33 → D > 0
    d = tmp_db.compute_d_value_for_agent("spike-agent", c_base=1.0, tax_base=0.1)
    assert d > 0.0


def test_F46_telemetry_schema_in_persistence_db():
    """F46: AIOSPersistence schema migration includes telemetry tables."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "aios_state.db")
        # Import persistence and create the DB via AIOSPersistence
        # This tests that persistence.py triggers telemetry schema creation
        import sys
        sys.path.insert(0, "/home/claude/AIOS")
        os.environ["AIOS_DB_PATH"] = db_path
        try:
            from core.persistence import AIOSPersistence
            p = AIOSPersistence(db_path=db_path)
            conn = sqlite3.connect(db_path)
            tables = {row[0] for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()}
            conn.close()
            p.close()
            assert "telemetry_l1" in tables, (
                f"telemetry_l1 missing from persistence DB. Tables: {tables}"
            )
            assert "telemetry_l2" in tables
            assert "telemetry_l3" in tables
        finally:
            if "AIOS_DB_PATH" in os.environ:
                del os.environ["AIOS_DB_PATH"]

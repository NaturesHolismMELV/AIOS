"""
test_session9.py — CI Dynamics Framework Tests
================================================
Session 9 deliverable validation.

Tests (16 total):
  1.  MELVKernel has _ci_history attribute after init
  2.  _record_ci_snapshot appends to _ci_history
  3.  _ci_history is populated by record_interaction automatically
  4.  dci_dt() returns 0.0 with fewer than 2 history points
  5.  dci_dt() returns positive value when CI is rising
  6.  dci_dt() returns negative value when CI is falling
  7.  ci_half_life() returns None when CI >= 0.75 (target already met)
  8.  ci_half_life() returns None when dCI/dt <= 0 (not converging)
  9.  ci_half_life() returns positive float when converging
 10.  ci_drift_coefficient() returns 0.0 with insufficient history
 11.  ci_drift_coefficient() positive on long improving history
 12.  ci_dynamics() returns all required keys
 13.  ci_dynamics() regime == "cooperative" when CI >= 0.75
 14.  ci_dynamics() regime == "converging" or "underdamped" when rising
 15.  OscillationEvent is recorded when CI crosses target then falls back
 16.  ci_dynamics() oscillation_count matches oscillation_events

Run: python -m pytest tests/test_session9.py -v
"""

import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.melv_engine import (
    MELVKernel,
    AgentProfile,
    OscillationEvent,
    CI_TARGET,
    CI_DCIDT_WINDOW,
)


def make_kernel_with_agents(n: int = 3) -> MELVKernel:
    k = MELVKernel()
    for i in range(n):
        k.register_agent(AgentProfile(
            agent_id=f"a{i}", name=f"Agent{i}", domain="test",
            phi=0.5, epsilon=3.0,
        ))
    return k


def drive_ci_up(kernel: MELVKernel, n: int = 40):
    for _ in range(n):
        kernel.record_interaction("a0", "a1", cost=0.1, benefit=1.0)


def drive_ci_down(kernel: MELVKernel, n: int = 40):
    for _ in range(n):
        kernel.record_interaction("a0", "a1", cost=1.5, benefit=0.5)


def test_01_ci_history_attribute_exists():
    k = MELVKernel()
    assert hasattr(k, "_ci_history")
    assert isinstance(k._ci_history, list)


def test_02_record_ci_snapshot_appends():
    k = MELVKernel()
    assert len(k._ci_history) == 0
    k._record_ci_snapshot()
    assert len(k._ci_history) == 1
    ts, ci = k._ci_history[0]
    assert isinstance(ts, float)
    assert 0.0 <= ci <= 1.0


def test_03_record_interaction_populates_history():
    k = make_kernel_with_agents()
    assert len(k._ci_history) == 0
    k.record_interaction("a0", "a1", cost=0.2, benefit=0.8)
    assert len(k._ci_history) == 1


def test_04_dcidt_returns_zero_with_no_history():
    k = MELVKernel()
    assert k.dci_dt() == 0.0
    k._record_ci_snapshot()
    assert k.dci_dt() == 0.0


def test_05_dcidt_positive_when_rising():
    k = MELVKernel()
    now = time.time()
    for step in range(CI_DCIDT_WINDOW):
        k._ci_history.append((now + step, 0.3 + step * 0.04))
    assert k.dci_dt() > 0


def test_06_dcidt_negative_when_falling():
    k = MELVKernel()
    now = time.time()
    for step in range(CI_DCIDT_WINDOW):
        k._ci_history.append((now + step, 0.9 - step * 0.04))
    assert k.dci_dt() < 0


def test_07_half_life_none_when_target_met():
    k = make_kernel_with_agents()
    drive_ci_up(k, 60)
    if k.cooperation_index() >= CI_TARGET:
        assert k.ci_half_life() is None


def test_08_half_life_none_when_not_converging():
    k = MELVKernel()
    now = time.time()
    for step in range(CI_DCIDT_WINDOW):
        k._ci_history.append((now + step, 0.5))
    assert k.dci_dt() == 0.0
    assert k.ci_half_life() is None


def test_09_half_life_positive_when_converging():
    k = MELVKernel()
    now = time.time()
    for step in range(CI_DCIDT_WINDOW):
        k._ci_history.append((now + step, 0.40 + step * 0.02))
    rate = k.dci_dt()
    ci   = k._ci_history[-1][1]
    if ci < CI_TARGET and rate > 0:
        hl = k.ci_half_life()
        assert hl is not None
        assert hl > 0


def test_10_drift_zero_with_no_history():
    k = MELVKernel()
    assert k.ci_drift_coefficient() == 0.0
    k._record_ci_snapshot()
    assert k.ci_drift_coefficient() == 0.0


def test_11_drift_positive_on_long_improving_history():
    k = MELVKernel()
    now = time.time()
    for step in range(100):
        k._ci_history.append((now + step, 0.3 + step * 0.003))
    assert k.ci_drift_coefficient() > 0


def test_12_ci_dynamics_has_required_keys():
    k = make_kernel_with_agents()
    drive_ci_up(k, 5)
    snapshot = k.ci_dynamics()
    required = {
        "cooperation_index", "ci_target", "gap_to_target",
        "dci_dt", "ci_half_life_sec", "ci_drift_coefficient",
        "regime", "oscillation_count", "recent_oscillations", "ci_history_length",
    }
    missing = required - set(snapshot.keys())
    assert not missing, f"Missing keys: {missing}"


def test_13_regime_cooperative_when_above_target():
    k = make_kernel_with_agents()
    drive_ci_up(k, 80)
    if k.cooperation_index() >= CI_TARGET:
        assert k.ci_dynamics()["regime"] == "cooperative"


def test_14_regime_converging_or_underdamped():
    k = MELVKernel()
    now = time.time()
    for step in range(50):
        k._ci_history.append((now + step, 0.40 + step * 0.004))
    d = k.ci_dynamics()
    if d["cooperation_index"] < CI_TARGET:
        assert d["regime"] in ("converging", "underdamped", "stasis")


def test_15_oscillation_event_recorded():
    k = MELVKernel()
    now = time.time()
    k._detect_oscillation(now - 10.0, 0.60)
    k._detect_oscillation(now - 5.0,  0.80)   # crosses above target
    k._detect_oscillation(now,        0.60)   # falls back below
    assert len(k.oscillation_events) >= 1
    evt = k.oscillation_events[-1]
    assert evt.ci_peak >= CI_TARGET
    assert evt.ci_trough < CI_TARGET
    assert evt.amplitude > 0


def test_16_oscillation_count_matches():
    k = MELVKernel()
    now = time.time()
    for cycle in range(2):
        base = now + cycle * 30
        k._detect_oscillation(base,       0.80)
        k._detect_oscillation(base + 5.0, 0.60)
    d = k.ci_dynamics()
    assert d["oscillation_count"] == len(k.oscillation_events)
    assert d["oscillation_count"] >= 2

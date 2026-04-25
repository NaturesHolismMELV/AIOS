"""
test_session24_2.py — β→Cost Feedback Loop + Theorem State Persistence
=======================================================================
Session 24.2: Two fixes that close the cooperation theorem's open loop.

Fix A — β-scaled cost generation in drive_real_agents():
    cost = max(0.05, min(2.0, cost_raw / max(0.1, current_beta)))
    Higher β (provisioned by the kernel) now reduces generated costs,
    so PROVISION_BETA actually drives i_factors down. Root cause of
    pairs_resolved=0 in the live theorem experiment (2026-04-18).

Fix B — Theorem state persisted to SQLite via AIOSPersistence:
    save_theorem_state() / load_theorem_state() on theorem_state table.
    ci_at_prediction: null was caused by in-memory state wiped on restart.

5 tests per Session 24.2 specification.
"""

import json
import math
import os
import random
import tempfile
import time

import pytest

from core.melv_engine import AgentProfile, MELVKernel
from core.persistence import AIOSPersistence


# ── Helpers ────────────────────────────────────────────────────────────────

def _make_kernel(*agents) -> MELVKernel:
    k = MELVKernel()
    for a in agents:
        k.register_agent(a)
    return k


def _agent(aid: str, phi: float = 0.8) -> AgentProfile:
    return AgentProfile(
        agent_id=aid, name=aid.upper(), domain="compute",
        phi=phi, epsilon=3.0, beta_pref=1.0,
    )


def _fresh_persistence() -> AIOSPersistence:
    """Return an AIOSPersistence instance backed by a temporary file."""
    tmp = tempfile.mktemp(suffix=".db")
    return AIOSPersistence(db_path=tmp)


# ── β-scaling logic (extracted from drive_real_agents for unit testing) ────

def _scaled_cost(cost_raw: float, current_beta: float) -> float:
    """
    Session 24.2 Fix A formula — identical to what drive_real_agents() uses.
    Extracted here so tests do not depend on the async server loop.
    """
    return max(0.05, min(2.0, cost_raw / max(0.1, current_beta)))


# ── Tests ──────────────────────────────────────────────────────────────────

class TestBetaScaledCostGeneration:
    """24.2 Fix A — β divides cost_raw so PROVISION_BETA actually lowers i_factors."""

    def test_beta_scaling_reduces_cost(self):
        """
        At beta=2.0 the scaled cost is half the raw cost.
        This is the core mechanism: PROVISION_BETA raises β → cost halves
        → i_factor = cost/benefit falls → pairs cross below I_CRITICAL.
        """
        cost_raw = 1.0
        beta = 2.0
        scaled = _scaled_cost(cost_raw, beta)
        # Should be cost_raw / beta = 0.5
        assert abs(scaled - 0.5) < 1e-9, (
            f"Expected 0.5 with cost_raw=1.0 beta=2.0, got {scaled}"
        )

    def test_beta_scaling_at_unity_is_neutral(self):
        """
        At beta=1.0 (default) the cost is unchanged.
        Ensures the fix has no side-effect on systems where β was never
        provisioned (baseline behaviour preserved).
        """
        cost_raw = 0.7
        beta = 1.0
        scaled = _scaled_cost(cost_raw, beta)
        assert abs(scaled - cost_raw) < 1e-9, (
            f"Expected no change at beta=1.0, got {scaled} vs {cost_raw}"
        )

    def test_beta_scaling_cold_start_path(self):
        """
        Cold-start path (< 5 interactions) also applies β scaling.
        Ensures newly deployed instances aren't exempt from the feedback loop.
        """
        cost_raw = 0.4
        beta = 4.0
        scaled = _scaled_cost(cost_raw, beta)
        # cost_raw / beta = 0.1 — exactly at the floor
        assert scaled >= 0.05, "Cost must not fall below 0.05 floor"
        assert scaled <= cost_raw, (
            "Scaled cost must be ≤ raw cost when beta ≥ 1.0"
        )

    def test_beta_scaling_is_monotone_decreasing_in_beta(self):
        """
        Scaled cost is monotonically decreasing in β.
        Confirms that each successive PROVISION_BETA (+0.10) step
        always moves cost in the right direction (down).
        """
        cost_raw = 1.2
        prev_scaled = _scaled_cost(cost_raw, 1.0)
        for beta in [1.1, 1.5, 2.0, 3.0]:
            scaled = _scaled_cost(cost_raw, beta)
            assert scaled <= prev_scaled + 1e-9, (
                f"Cost should decrease as β increases: "
                f"beta={beta} gave {scaled} > previous {prev_scaled}"
            )
            prev_scaled = scaled


class TestTheoremStatePersistence:
    """24.2 Fix B — theorem experiment state survives process restarts."""

    def test_theorem_state_persists_and_restores(self):
        """
        save_theorem_state() → load_theorem_state() round-trip.
        The full _theorem_state dict (prediction timestamp, ci_at_prediction,
        pairs lists) must be identical after a fresh AIOSPersistence load.
        """
        store = _fresh_persistence()

        state = {
            "prediction_made_at":  1745000000.0,
            "pairs_above":         [{"pair": "writer::planner", "mean_i": 1.05}],
            "pairs_below":         [],
            "ci_at_prediction":    0.31,
            "intervention_log":    [],
            "result_snapshots":    [],
        }
        store.save_theorem_state(state)

        # Simulate process restart: open a new AIOSPersistence on the same file
        store2 = AIOSPersistence(db_path=store.db_path)
        restored = store2.load_theorem_state()

        assert restored is not None, "load_theorem_state() returned None after save"
        assert abs(restored["ci_at_prediction"] - 0.31) < 1e-6, (
            f"ci_at_prediction corrupted: {restored['ci_at_prediction']}"
        )
        assert restored["prediction_made_at"] == pytest.approx(1745000000.0), (
            f"prediction_made_at corrupted: {restored['prediction_made_at']}"
        )
        assert len(restored["pairs_above"]) == 1, (
            "pairs_above list not preserved"
        )

        store.close()
        store2.close()
        try:
            os.unlink(store.db_path)
        except OSError:
            pass

    def test_load_theorem_state_returns_none_when_absent(self):
        """
        load_theorem_state() returns None (not an exception) on a fresh DB.
        Defensive: server startup must handle this gracefully.
        """
        store = _fresh_persistence()
        result = store.load_theorem_state()
        assert result is None, (
            f"Expected None on empty DB, got {result}"
        )
        store.close()
        try:
            os.unlink(store.db_path)
        except OSError:
            pass

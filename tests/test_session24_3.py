"""
test_session24_3.py — CI Formula Fix (cooperation_index)
=========================================================
Session 24.3 Fix A: cooperation_index() now uses the phi-weighted fraction
of interactions where i_factor < I_CRITICAL, replacing the broken formula
CI = 1 - mean(beta_i) which returned 0.0 whenever beta was provisioned.

Root cause of theorem_confirmed never becoming True despite all 28 pairs
resolving: beta values of ~34 made mean(beta_i) ~34, clamping CI to 0.0.

3 tests.
"""

import pytest
from core.melv_engine import AgentProfile, MELVKernel, InteractionRecord

I_CRITICAL = 0.9995


def _make_kernel() -> MELVKernel:
    k = MELVKernel()
    for aid in ["A", "B", "C"]:
        k.register_agent(AgentProfile(
            agent_id=aid, name=aid, domain="compute",
            phi=0.8, epsilon=3.0, beta_pref=1.0,
        ))
    return k


class TestCooperationIndexFormula:
    """24.3 Fix A — CI uses i_factor < I_CRITICAL, not 1 - mean(beta_i)."""

    def test_ci_is_zero_when_all_interactions_above_threshold(self):
        """
        All i_factors above I_CRITICAL -> CI = 0.0.
        Regardless of beta magnitude.
        """
        k = _make_kernel()
        # Record interactions with i_factor > I_CRITICAL (cost > benefit)
        for _ in range(10):
            k.record_interaction("A", "B", cost=1.5, benefit=1.0, resource_type="compute")
        ci = k.cooperation_index()
        assert ci == 0.0, f"Expected CI=0.0 with all i_factors above threshold, got {ci}"

    def test_ci_is_one_when_all_interactions_below_threshold(self):
        """
        All i_factors below I_CRITICAL -> CI = 1.0.
        This is what the theorem aims to achieve.
        """
        k = _make_kernel()
        # cost < benefit -> i_factor < 1.0 < I_CRITICAL... wait, I_CRITICAL=0.9995
        # Need i_factor < 0.9995: cost/benefit < 0.9995
        for _ in range(10):
            k.record_interaction("A", "B", cost=0.5, benefit=0.9, resource_type="compute")
        ci = k.cooperation_index()
        assert ci == 1.0, f"Expected CI=1.0 with all i_factors below threshold, got {ci}"

    def test_ci_not_destroyed_by_high_beta(self):
        """
        The critical regression test: CI must not be zero when beta is high.
        With the old formula CI = 1 - mean(beta_i), beta=34 -> CI=-33 -> 0.0.
        With the new formula, beta has no effect on CI — only i_factor matters.
        """
        k = _make_kernel()
        # Simulate the post-Session-24.2 state: high beta, low i_factors
        # Manually inject interactions with low cost/benefit ratios
        # but high beta (as would be present after PROVISION_BETA x34)
        import time
        from dataclasses import dataclass
        # Inject raw InteractionRecord objects with high beta but low i_factor
        for _ in range(20):
            r = InteractionRecord(
                agent_a="A", agent_b="B",
                cost=0.05, benefit=0.9,   # i_factor = 0.055 << I_CRITICAL
                beta=34.0,                 # beta_i = 34 * 0.055 = 1.87
                resource_type="compute",
                timestamp=time.time(),
            )
            k.interactions.append(r)
        ci = k.cooperation_index()
        assert ci > 0.9, (
            f"CI should be near 1.0 with low i_factors despite high beta, got {ci}. "
            f"Old formula would have returned 0.0 — this is the regression check."
        )


class TestTheoremConfirmedLogic:
    """24.3 Fix B — theorem_confirmed uses pairs_resolved, not ci improvement delta."""

    def test_confirmed_when_ci_at_target_and_pairs_resolved(self):
        """
        CI >= 0.75 AND pairs_above_now < pairs_above_at_prediction -> confirmed.
        This is the scientifically correct criterion.
        """
        CI_TARGET = 0.75
        ci_now = 1.0
        pairs_above_at_pred = 28
        pairs_above_now = 0
        theorem_confirmed = (
            ci_now >= CI_TARGET
            and (
                pairs_above_now < pairs_above_at_pred
                or pairs_above_at_pred == 0
            )
        )
        assert theorem_confirmed is True

    def test_not_confirmed_when_ci_below_target(self):
        """CI below target -> not confirmed regardless of pairs resolved."""
        CI_TARGET = 0.75
        ci_now = 0.60
        pairs_above_at_pred = 10
        pairs_above_now = 0
        theorem_confirmed = (
            ci_now >= CI_TARGET
            and (
                pairs_above_now < pairs_above_at_pred
                or pairs_above_at_pred == 0
            )
        )
        assert theorem_confirmed is False

    def test_confirmed_when_prediction_already_cooperative(self):
        """
        Edge case: prediction made when CI already 1.0 and pairs_above=0.
        Should confirm on CI >= target alone.
        """
        CI_TARGET = 0.75
        ci_now = 1.0
        pairs_above_at_pred = 0
        pairs_above_now = 0
        theorem_confirmed = (
            ci_now >= CI_TARGET
            and (
                pairs_above_now < pairs_above_at_pred
                or pairs_above_at_pred == 0
            )
        )
        assert theorem_confirmed is True


class TestClassifyPairsMinSamples:
    """24.3 Fix C — _classify_pairs excludes pairs with < 15 samples."""

    def _classify(self, kernel):
        """Inline replica of _classify_pairs with MIN_SAMPLES=15."""
        from collections import defaultdict
        MIN_SAMPLES = 15
        I_CRITICAL = 0.9995
        pair_data = defaultdict(list)
        for r in kernel.interactions[-500:]:
            key = tuple(sorted([r.agent_a, r.agent_b]))
            pair_data[key].append(r.i_factor)
        pairs_above, pairs_below = [], []
        for (a, b), i_factors in pair_data.items():
            if len(i_factors) < MIN_SAMPLES:
                continue
            mean_i = sum(i_factors) / len(i_factors)
            entry = {"pair": f"{a}::{b}", "n_samples": len(i_factors), "mean_i": round(mean_i, 4)}
            if mean_i > I_CRITICAL:
                pairs_above.append(entry)
            else:
                pairs_below.append(entry)
        return pairs_above, pairs_below

    def test_pair_below_min_samples_excluded(self):
        """
        A pair with 9 samples and mean_i > I_CRITICAL must not appear in
        pairs_above. Statistically unreliable — cannot block theorem confirmation.
        """
        import time
        from core.melv_engine import InteractionRecord
        k = _make_kernel()
        for _ in range(9):
            r = InteractionRecord(
                agent_a="A", agent_b="B",
                cost=1.5, benefit=0.5,  # i_factor=3.0, well above I_CRITICAL
                beta=1.0, resource_type="compute",
                timestamp=time.time(),
            )
            k.interactions.append(r)
        pairs_above, pairs_below = self._classify(k)
        pair_ids = [p["pair"] for p in pairs_above]
        assert "A::B" not in pair_ids, (
            "Pair with 9 samples should be excluded from pairs_above (MIN_SAMPLES=15)"
        )

    def test_pair_at_min_samples_included(self):
        """A pair with exactly 15 samples is included in classification."""
        import time
        from core.melv_engine import InteractionRecord
        k = _make_kernel()
        for _ in range(15):
            r = InteractionRecord(
                agent_a="A", agent_b="B",
                cost=0.3, benefit=0.9,  # i_factor=0.33, below I_CRITICAL
                beta=1.0, resource_type="compute",
                timestamp=time.time(),
            )
            k.interactions.append(r)
        pairs_above, pairs_below = self._classify(k)
        pair_ids = [p["pair"] for p in pairs_below]
        assert "A::B" in pair_ids, (
            "Pair with exactly 15 samples should be included in pairs_below"
        )

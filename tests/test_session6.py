"""
test_session6.py — CostCalculator test suite
=============================================
6 offline tests covering:
  1. Weight profile correctness for each task type
  2. Default / unknown task type falls back to balanced
  3. Zero-token edge case (latency-only cost)
  4. Cost cap enforcement (max 2.0)
  5. Refactored agent integration — agents call CostCalculator, not inline code
  6. Cost breakdown endpoint structure

All tests are offline (no API key required).

Run: python -m pytest tests/test_session6.py -v
"""

import pytest
import sys
import os

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.cost_calculator import (
    CostCalculator,
    CostProfile,
    COST_PROFILES,
    get_calculator,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def calc():
    """Fresh CostCalculator for each test — avoids shared history contamination."""
    return CostCalculator()


# ── Test 1: Weight profile correctness ───────────────────────────────────────

class TestWeightProfiles:
    """All four task-type profiles must have correct weights and labels."""

    def test_analysis_is_token_heavy(self):
        p = COST_PROFILES["ANALYSIS"]
        assert p.token_weight > p.latency_weight, \
            "ANALYSIS should be token_heavy: token_weight > latency_weight"
        assert p.label() == "token_heavy"
        assert p.token_weight == 1.4
        assert p.latency_weight == 0.6

    def test_writer_is_balanced(self):
        p = COST_PROFILES["WRITER"]
        assert p.token_weight == p.latency_weight, \
            "WRITER should be balanced: token_weight == latency_weight"
        assert p.label() == "balanced"
        assert p.token_weight == 1.0

    def test_planner_is_token_heavy(self):
        p = COST_PROFILES["PLANNER"]
        assert p.token_weight > p.latency_weight, \
            "PLANNER should be token_heavy: token_weight > latency_weight"
        assert p.label() == "token_heavy"
        assert p.token_weight == 1.4

    def test_research_is_latency_heavy(self):
        p = COST_PROFILES["RESEARCH"]
        assert p.latency_weight > p.token_weight, \
            "RESEARCH should be latency_heavy: latency_weight > token_weight"
        assert p.label() == "latency_heavy"
        assert p.latency_weight == 1.6

    def test_all_four_profiles_present(self):
        required = {"ANALYSIS", "WRITER", "PLANNER", "RESEARCH"}
        assert required.issubset(COST_PROFILES.keys()), \
            f"Missing profiles: {required - set(COST_PROFILES.keys())}"


# ── Test 2: Default / unknown task type ──────────────────────────────────────

class TestDefaultFallback:
    """Unknown task types must fall back to balanced (1.0 / 1.0) profile."""

    def test_unknown_type_returns_balanced_cost(self, calc):
        cost_unknown  = calc.compute_cost(100, 50, 1.0, task_type="UNKNOWN_AGENT")
        cost_balanced = calc.compute_cost(100, 50, 1.0, task_type="WRITER")
        assert cost_unknown == cost_balanced, \
            "Unknown task type should produce same cost as balanced WRITER profile"

    def test_empty_string_falls_back(self, calc):
        # Should not raise — falls back gracefully
        cost = calc.compute_cost(100, 50, 1.0, task_type="")
        assert 0.0 < cost <= 2.0

    def test_get_profile_returns_default_for_unknown(self, calc):
        profile = calc.get_profile("TOTALLY_UNKNOWN")
        assert profile.token_weight == 1.0
        assert profile.latency_weight == 1.0


# ── Test 3: Zero-token edge case ─────────────────────────────────────────────

class TestZeroTokens:
    """Zero tokens — cost should be latency-only, not zero, not negative."""

    def test_zero_tokens_nonzero_cost(self, calc):
        cost = calc.compute_cost(in_tok=0, out_tok=0, latency_s=1.0, task_type="WRITER")
        # latency component: 1.0 * 0.1 * 1.0 = 0.1
        assert cost == pytest.approx(0.1, abs=1e-4), \
            "Zero tokens with 1s latency should give cost ≈ 0.1"

    def test_zero_tokens_zero_latency_gives_zero(self, calc):
        cost = calc.compute_cost(in_tok=0, out_tok=0, latency_s=0.0, task_type="ANALYSIS")
        assert cost == 0.0, "Zero tokens + zero latency should be zero cost"

    def test_latency_heavy_research_zero_tokens(self, calc):
        # RESEARCH: latency_weight=1.6, token_weight=0.7
        # cost = 0 + 2.0 * 0.1 * 1.6 = 0.32
        cost = calc.compute_cost(in_tok=0, out_tok=0, latency_s=2.0, task_type="RESEARCH")
        assert cost == pytest.approx(0.32, abs=1e-4)


# ── Test 4: Cost cap enforcement ─────────────────────────────────────────────

class TestCostCap:
    """Cost must never exceed 2.0 regardless of token count or latency."""

    def test_massive_tokens_capped(self, calc):
        # 1,000,000 input tokens at Haiku pricing would be huge
        cost = calc.compute_cost(
            in_tok=1_000_000, out_tok=500_000, latency_s=5.0, task_type="ANALYSIS"
        )
        assert cost == 2.0, "Cost must be capped at 2.0"

    def test_record_marks_capped_correctly(self, calc):
        calc.compute_cost(in_tok=1_000_000, out_tok=500_000, latency_s=5.0, task_type="WRITER")
        records = calc.recent_breakdown(n=1)
        assert records[0]["capped"] is True

    def test_normal_haiku_call_not_capped(self, calc):
        # Typical Haiku call: 300 in, 150 out, 1s latency
        cost = calc.compute_cost(in_tok=300, out_tok=150, latency_s=1.0, task_type="ANALYSIS")
        assert cost < 2.0, "Typical Haiku call should not hit the cap"
        records = calc.recent_breakdown(n=1)
        assert records[0]["capped"] is False

    def test_cap_applies_across_all_profiles(self, calc):
        for task_type in ["ANALYSIS", "WRITER", "PLANNER", "RESEARCH"]:
            cost = calc.compute_cost(
                in_tok=2_000_000, out_tok=2_000_000, latency_s=100.0,
                task_type=task_type
            )
            assert cost == 2.0, f"{task_type}: cost exceeded cap"


# ── Test 5: Agent integration — no inline normalisation ──────────────────────

class TestAgentIntegration:
    """
    Verify that the three LLM agents import and use CostCalculator,
    not inline token_cost arithmetic.
    """

    def test_analysis_agent_imports_cost_calculator(self):
        import inspect
        import agents.implementations as impl_module
        source = inspect.getsource(impl_module)
        assert "get_calculator" in source, \
            "AnalysisAgent must call get_calculator() — inline normalisation removed"
        assert "token_cost = in_tok * 0.0000008" not in source, \
            "Inline token_cost formula must be removed from AnalysisAgent"

    def test_writer_agent_imports_cost_calculator(self):
        import inspect
        import agents.writer_agent as writer_module
        source = inspect.getsource(writer_module)
        assert "get_calculator" in source, \
            "WriterAgent must call get_calculator()"
        assert "token_cost = in_tok * 0.0000008" not in source, \
            "Inline token_cost formula must be removed from WriterAgent"

    def test_planner_agent_imports_cost_calculator(self):
        import inspect
        import agents.planner_agent as planner_module
        source = inspect.getsource(planner_module)
        assert "get_calculator" in source, \
            "PlannerAgent must call get_calculator()"
        assert "token_cost = in_tok * 0.0000008" not in source, \
            "Inline token_cost formula must be removed from PlannerAgent"


# ── Test 6: Cost breakdown endpoint structure ─────────────────────────────────

class TestCostBreakdownStructure:
    """
    Verify CostCalculator exposes the correct data structure for /melv/costs.
    Tests the Python methods directly — no live server needed.
    """

    def test_all_profiles_in_output(self, calc):
        profiles = calc.all_profiles()
        for key in ["ANALYSIS", "WRITER", "PLANNER", "RESEARCH"]:
            assert key in profiles, f"{key} missing from all_profiles()"

    def test_profile_output_has_required_fields(self, calc):
        profiles = calc.all_profiles()
        required = {"token_weight", "latency_weight", "profile_label", "description"}
        for key, p in profiles.items():
            missing = required - set(p.keys())
            assert not missing, f"{key} profile missing fields: {missing}"

    def test_recent_breakdown_populates_on_compute(self, calc):
        assert calc.recent_breakdown() == [], "History should start empty"
        calc.compute_cost(250, 120, 0.85, "WRITER")
        records = calc.recent_breakdown()
        assert len(records) == 1
        r = records[0]
        assert r["task_type"] == "WRITER"
        assert "token_component" in r
        assert "latency_component" in r
        assert "cost" in r
        assert "capped" in r

    def test_summary_by_type_aggregates_correctly(self, calc):
        calc.compute_cost(300, 150, 1.0, "ANALYSIS")
        calc.compute_cost(300, 150, 1.0, "ANALYSIS")
        calc.compute_cost(200, 100, 0.5, "WRITER")
        summary = calc.summary_by_type()
        assert "ANALYSIS" in summary
        assert summary["ANALYSIS"]["count"] == 2
        assert "WRITER" in summary
        assert summary["WRITER"]["count"] == 1

    def test_singleton_is_shared(self):
        """get_calculator() must return the same instance each call."""
        c1 = get_calculator()
        c2 = get_calculator()
        assert c1 is c2, "get_calculator() must return a singleton"

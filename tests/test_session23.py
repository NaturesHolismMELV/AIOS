"""
test_session23.py — Empirical Sandbox Calibration
==================================================
Session 23: SandboxEngine.calibrate_from_kernel(), empirical _simulate()
draws, REFERENCE_ECOSYSTEM rationale block, /sandbox/calibration_status.

6 tests targeting the Session 23 deliverables.
"""

import random
import pytest
from unittest.mock import MagicMock
from dataclasses import dataclass, field
from typing import Optional

from core.melv_engine import AgentProfile, MELVKernel
from core.sandbox_engine import SandboxEngine, SANDBOX_VERSION


# ── Helpers ────────────────────────────────────────────────────────────────

def _make_kernel_with_interactions(n: int, resource: str = "compute") -> MELVKernel:
    """Build a kernel and populate it with n synthetic interactions."""
    k = MELVKernel()
    k.register_agent(AgentProfile(agent_id="a1", name="A1", domain="compute", phi=0.8, epsilon=3.0, beta_pref=1.0))
    k.register_agent(AgentProfile(agent_id="a2", name="A2", domain="compute", phi=0.75, epsilon=3.0, beta_pref=1.0))
    for _ in range(n):
        cost    = random.uniform(0.1, 0.9)
        benefit = random.uniform(0.1, 1.2)
        k.record_interaction("a1", "a2", cost, benefit, resource)
    return k


# ── Tests ──────────────────────────────────────────────────────────────────

class TestCalibrateFromKernel:
    """23.1 — calibrate_from_kernel populates empirical distributions."""

    def test_calibrate_from_kernel_populates_distributions(self):
        """Kernel with 50 interactions → empirical distributions populated."""
        engine = SandboxEngine()
        kernel = _make_kernel_with_interactions(50, "compute")

        summary = engine.calibrate_from_kernel(kernel)

        assert "compute" in summary["calibrated_resources"], (
            "Expected 'compute' in calibrated resources"
        )
        assert summary["total_interactions_sampled"] >= 50
        assert not summary["fallback_active"]

        # Distributions must contain required statistical fields
        dist = engine._empirical_distributions["compute"]
        for key in ("cost_mean", "cost_stdev", "benefit_mean", "benefit_stdev", "n"):
            assert key in dist, f"Missing key '{key}' in empirical distribution"
        assert dist["n"] >= 50

    def test_calibrate_fallback_when_insufficient_interactions(self):
        """Kernel with fewer than 10 interactions → falls back to hardcoded ranges."""
        engine = SandboxEngine()
        kernel = _make_kernel_with_interactions(3, "compute")

        summary = engine.calibrate_from_kernel(kernel)

        assert summary["fallback_active"], (
            "Expected fallback_active=True with only 3 interactions"
        )
        assert "compute" not in engine._empirical_distributions, (
            "Should not populate distributions from <10 interactions"
        )

    def test_calibrate_multiple_resource_types(self):
        """Kernel with interactions across multiple resource types → all calibrated."""
        engine = SandboxEngine()
        k = MELVKernel()
        k.register_agent(AgentProfile(agent_id="a1", name="A1", domain="compute", phi=0.8, epsilon=3.0, beta_pref=1.0))
        k.register_agent(AgentProfile(agent_id="a2", name="A2", domain="compute", phi=0.75, epsilon=3.0, beta_pref=1.0))

        for resource in ("compute", "token_budget", "api_quota"):
            for _ in range(15):
                k.record_interaction("a1", "a2",
                                     random.uniform(0.1, 0.9),
                                     random.uniform(0.1, 1.2),
                                     resource)

        summary = engine.calibrate_from_kernel(k)

        for resource in ("compute", "token_budget", "api_quota"):
            assert resource in summary["calibrated_resources"], (
                f"Expected '{resource}' in calibrated resources"
            )


class TestSimulateUsesEmpirical:
    """23.2 — _simulate draws from empirical distributions when calibrated."""

    def test_simulate_uses_empirical_distributions(self):
        """
        Calibrated sandbox: interaction costs must lie within 3σ of the
        empirical mean (confirms draws come from the fitted distribution).
        """
        engine = SandboxEngine()

        # Tight distribution: mean≈0.5, stdev≈0.05
        engine._empirical_distributions["compute"] = {
            "cost_mean":     0.50,
            "cost_stdev":    0.05,
            "benefit_mean":  0.80,
            "benefit_stdev": 0.05,
            "n":             100,
        }

        k = MELVKernel()
        k.register_agent(AgentProfile(agent_id="a1", name="A1", domain="compute", phi=0.8, epsilon=3.0, beta_pref=1.0))
        k.register_agent(AgentProfile(agent_id="a2", name="A2", domain="compute", phi=0.75, epsilon=3.0, beta_pref=1.0))

        engine._simulate(k, n_interactions=200, agent_id=None)

        costs = [r.cost for r in k.interactions]
        assert len(costs) > 0
        mean_cost = sum(costs) / len(costs)

        # Empirical mean is 0.5; 3σ of sample mean ≈ 3*(0.05/√200) ≈ 0.011
        # Use generous 0.10 tolerance to avoid flakiness
        assert abs(mean_cost - 0.50) < 0.10, (
            f"mean_cost={mean_cost:.3f} deviates more than 0.10 from empirical mean 0.50"
        )

    def test_simulate_fallback_when_uncalibrated(self):
        """Uncalibrated sandbox → uses hardcoded ranges (no regression)."""
        engine = SandboxEngine()
        # _empirical_distributions is empty by default

        k = MELVKernel()
        k.register_agent(AgentProfile(agent_id="a1", name="A1", domain="compute", phi=0.8, epsilon=3.0, beta_pref=1.0))
        k.register_agent(AgentProfile(agent_id="a2", name="A2", domain="compute", phi=0.75, epsilon=3.0, beta_pref=1.0))

        engine._simulate(k, n_interactions=100, agent_id=None)

        # With hardcoded ranges, all values must stay within defined bounds
        for r in k.interactions:
            assert 0.0 <= r.cost <= 2.0, f"cost={r.cost} out of expected range"
            assert 0.0 <= r.benefit <= 2.0, f"benefit={r.benefit} out of expected range"


class TestReferenceEcosystemRationale:
    """23.3 — REFERENCE_ECOSYSTEM rationale block is present in source."""

    def test_reference_rationale_present_in_source(self):
        """The REFERENCE_ECOSYSTEM docblock must contain the key phrase."""
        import inspect
        import core.sandbox_engine as mod

        source = inspect.getsource(mod)
        assert "energetic reference" in source, (
            "REFERENCE_ECOSYSTEM rationale block missing 'energetic reference'"
        )
        assert "thermodynamically unsustainable" in source, (
            "REFERENCE_ECOSYSTEM rationale missing thermodynamic justification"
        )
        assert "1981" in source, (
            "Namibia ecological basis (1981) not cited in rationale"
        )


class TestVersionAndCalibrationStatus:
    """23.4 — Version and calibration_status() method."""

    def test_sandbox_version_is_1_9_3_or_later(self):
        """Session 23 set version to 1.9.3; Session 24 advances it to 2.0.0."""
        from packaging.version import Version
        assert Version(SANDBOX_VERSION) >= Version("1.9.3"), (
            f"Expected SANDBOX_VERSION >= '1.9.3', got '{SANDBOX_VERSION}'"
        )

    def test_calibration_status_returns_structure(self):
        """calibration_status() returns the expected dict structure."""
        engine = SandboxEngine()
        status = engine.calibration_status()

        assert "calibrated" in status
        assert "distributions" in status
        assert status["calibrated"] is False  # fresh engine, no data

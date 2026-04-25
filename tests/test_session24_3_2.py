"""
test_session24_3_2.py — Macro MELV Validation + OpenAPI Security
=================================================================
Session 24.3.2: Fixes the two remaining ① stubs from the Master Roadmap:
  1. melv_metadata cooperation_index: 0.92 hardcoded → real computation removed
  2. /data/melv/{country_code} → real φ/β/i_factor from World Bank indicators
  3. /data/melv/energy-cooperation → MELV label with MELV computation available
  4. OpenAPI securitySchemes → Swagger Authorize button

10 tests across 4 classes.

All tests are offline (no World Bank API calls) — parameter derivation
functions are pure Python math, tested with synthetic indicator values.
"""

import math
import pytest

# ── imports under test ─────────────────────────────────────────────────────
from agents.data_agent import (
    _derive_phi,
    _derive_beta,
    _derive_i_factor,
    BETA_ENERGY_REF,
    PHI_GDP_REF,
    I_CRITICAL,
    DataAgent,
)


# ===========================================================================
# Class 1 — β derivation from energy use
# ===========================================================================

class TestDeriveBeta:
    """β = energy_use / 1900 reference."""

    def test_global_mean_gives_beta_one(self):
        """Energy use equal to the reference → β = 1.0 exactly."""
        beta, basis = _derive_beta(BETA_ENERGY_REF)
        assert beta == pytest.approx(1.0, abs=1e-6)
        assert "1900" in basis

    def test_energy_scarce_below_one(self):
        """Low energy use (e.g. 950 kg/cap) → β < 1 (resource-scarce)."""
        beta, _ = _derive_beta(950.0)
        assert beta == pytest.approx(0.5, abs=1e-4)
        assert beta < 1.0

    def test_energy_rich_above_one(self):
        """High energy use (e.g. 3800 kg/cap) → β = 2.0 (resource-abundant)."""
        beta, _ = _derive_beta(3800.0)
        assert beta == pytest.approx(2.0, abs=1e-4)

    def test_none_energy_returns_none(self):
        """Missing energy data → β = None, honest unavailable signal."""
        beta, basis = _derive_beta(None)
        assert beta is None
        assert "unavailable" in basis


# ===========================================================================
# Class 2 — φ derivation from three indicators
# ===========================================================================

class TestDerivePhi:
    """φ weighted composite: GDP 0.5 + employment 0.3 + governance 0.2."""

    def test_sa_profile_suppressed_by_unemployment(self):
        """
        South Africa scenario: mid-range GDP, high unemployment (32.4%),
        moderate governance. φ should be clearly below 0.5 (developing/novice).
        """
        phi, basis = _derive_phi(
            gdp_per_capita=6000.0,
            unemployment_rate=32.4,
            governance_est=-0.2,
        )
        assert phi < 0.5, f"SA φ should be below 0.5, got {phi}"
        assert "32.4" in basis

    def test_norway_profile_high_phi(self):
        """Norway: high GDP/cap, low unemployment, strong governance → φ near 1."""
        phi, basis = _derive_phi(
            gdp_per_capita=100000.0,
            unemployment_rate=3.5,
            governance_est=1.8,
        )
        assert phi > 0.75, f"Norway φ should be above 0.75, got {phi}"

    def test_zero_unemployment_not_above_one(self):
        """φ is clamped to [0, 1] — perfect indicators must not exceed 1."""
        phi, _ = _derive_phi(
            gdp_per_capita=500000.0,
            unemployment_rate=0.0,
            governance_est=2.5,
        )
        assert 0.0 <= phi <= 1.0

    def test_missing_governance_falls_back_neutral(self):
        """Missing governance indicator → 0.5 neutral stub, basis notes ①."""
        phi_with, _  = _derive_phi(
            gdp_per_capita=10000.0,
            unemployment_rate=10.0,
            governance_est=0.5,
        )
        phi_without, basis = _derive_phi(
            gdp_per_capita=10000.0,
            unemployment_rate=10.0,
            governance_est=None,
        )
        assert "unavailable" in basis or "stub" in basis
        # Without governance the φ should differ from the full computation
        assert phi_with != phi_without


# ===========================================================================
# Class 3 — i_factor and cooperation prediction
# ===========================================================================

class TestDeriveIFactor:
    """i_factor = 1 / (φ × β); cooperation_prediction vs i_critical=0.9995."""

    def test_cooperative_prediction_when_phi_beta_product_high(self):
        """
        φ=0.8, β=2.0 → φ·β = 1.6 → i_factor = 0.625 < I_CRITICAL → COOPERATIVE.
        """
        i_est, basis, prediction = _derive_i_factor(phi=0.8, beta=2.0)
        assert i_est == pytest.approx(0.625, abs=1e-3)
        assert prediction == "COOPERATIVE"
        assert i_est < I_CRITICAL

    def test_not_cooperative_when_phi_beta_product_low(self):
        """
        φ=0.4, β=0.8 → φ·β = 0.32 → i_factor ≈ 3.125 → NOT_COOPERATIVE.
        """
        i_est, _, prediction = _derive_i_factor(phi=0.4, beta=0.8)
        assert i_est > I_CRITICAL
        assert prediction == "NOT_COOPERATIVE"

    def test_none_beta_returns_none_i_factor(self):
        """No β available → i_factor is None, prediction is 'unknown'."""
        i_est, basis, prediction = _derive_i_factor(phi=0.7, beta=None)
        assert i_est is None
        assert prediction == "unknown"


# ===========================================================================
# Class 4 — DataAgent class-level stub removal and action dispatch
# ===========================================================================

class TestDataAgentStubRemoval:
    """Verify the hardcoded cooperation_index: 0.92 is gone and melv action exists."""

    def test_hardcoded_cooperation_index_removed(self):
        """
        melv_metadata must NOT contain a hardcoded cooperation_index.
        This was the ① stub that returned 0.92 for every country regardless
        of actual indicators — the primary defect Session 24.3.2 fixes.
        """
        agent = DataAgent()
        assert "cooperation_index" not in agent.melv_metadata, (
            "cooperation_index: 0.92 hardcoded stub must be removed from melv_metadata. "
            "Use action='melv' to compute real values per country."
        )

    def test_melv_action_recognised(self):
        """
        DataAgent.execute() must not return 'Unknown action' for action='melv'.
        We test the dispatch logic directly without making a real API call
        by patching compute_melv_for_country to return a minimal sentinel.
        """
        import asyncio
        from unittest.mock import AsyncMock, patch

        agent = DataAgent()
        sentinel = {"country_code": "ZA", "phi": 0.42, "beta": 0.94}

        with patch("agents.data_agent.compute_melv_for_country", new=AsyncMock(return_value=sentinel)):
            result = asyncio.run(
                agent.execute({"action": "melv", "country": "ZA"})
            )

        assert result["status"] == "success", f"Expected success, got: {result}"
        assert result["result"]["phi"] == 0.42
        assert result["result"]["beta"] == 0.94

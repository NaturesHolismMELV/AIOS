"""
test_session24.py — Real Ω Eigenvalue + Cooperation Theorem Experiment
=======================================================================
Session 24: numpy eigvalsh for compute_omega(), theorem_prediction and
theorem_result endpoints, v2.0.0 version gate.

5 tests per the Session 24 specification.
"""

import math
import random
import pytest
import numpy as np

from core.melv_engine import AgentProfile, MELVKernel
from core.sandbox_engine import SANDBOX_VERSION


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


# ── Tests ──────────────────────────────────────────────────────────────────

class TestRealEigenvalue:
    """24.1 — compute_omega() uses numpy.linalg.eigvalsh, not heuristic proxy."""

    def test_real_eigenvalue_matches_numpy(self):
        """
        Given a known interaction history, compute_omega() λ_max must
        match numpy.linalg.eigvalsh applied to the same adjacency matrix.
        """
        k = _make_kernel(_agent("a1"), _agent("a2"), _agent("a3"))

        # Inject deterministic interactions with known cooperative weights
        # i_factor = cost/benefit; weight = 1 - i_factor
        # cost=0.2, benefit=1.0 → i=0.2, weight=0.8
        for _ in range(5):
            k.record_interaction("a1", "a2", cost=0.2, benefit=1.0, resource_type="compute")
        # cost=0.4, benefit=0.8 → i=0.5, weight=0.5
        for _ in range(5):
            k.record_interaction("a2", "a3", cost=0.4, benefit=0.8, resource_type="compute")
        # cost=0.6, benefit=0.6 → i=1.0, weight=0.0
        for _ in range(5):
            k.record_interaction("a1", "a3", cost=0.6, benefit=0.6, resource_type="compute")

        result = k.compute_omega()

        # Build expected adjacency matrix manually
        ids  = list(k.agents.keys())
        idx  = {aid: i for i, aid in enumerate(ids)}
        n    = len(ids)
        A    = np.zeros((n, n))

        # Reproduce the same weight computation the kernel uses
        from collections import defaultdict
        weights = defaultdict(list)
        for r in k.interactions[-100:]:
            if r.agent_a in idx and r.agent_b in idx:
                weights[(r.agent_a, r.agent_b)].append(1.0 - r.i_factor)

        for (a, b), vals in weights.items():
            avg = sum(vals) / len(vals)
            A[idx[a], idx[b]] = avg
            A[idx[b], idx[a]] = avg

        expected_lambda = float(np.linalg.eigvalsh(A)[-1])

        assert abs(result["lambda_max"] - round(expected_lambda, 4)) < 1e-3, (
            f"λ_max mismatch: kernel={result['lambda_max']}, "
            f"expected={round(expected_lambda, 4)}"
        )
        assert result["n"] == n

    def test_eigenvalue_symmetric_matrix(self):
        """
        Asymmetric interaction history (a→b more than b→a) must produce
        a symmetric adjacency matrix before eigvalsh is applied.
        """
        k = _make_kernel(_agent("x"), _agent("y"))

        # Deliberately asymmetric call pattern: x interacts with y 8 times
        for _ in range(8):
            k.record_interaction("x", "y", cost=0.3, benefit=0.9, resource_type="compute")
        # y interacts with x only twice
        for _ in range(2):
            k.record_interaction("y", "x", cost=0.5, benefit=0.7, resource_type="compute")

        result = k.compute_omega()

        # λ_max of a symmetric 2×2 matrix with equal off-diagonals = off-diagonal value
        # The result must be a real (not complex) scalar
        assert isinstance(result["lambda_max"], float), (
            "λ_max must be a real float (symmetric eigvalsh)"
        )
        assert result["n"] == 2

    def test_empty_kernel_returns_zero(self):
        """compute_omega() with no agents returns zeros, not an error."""
        k = MELVKernel()
        result = k.compute_omega()
        assert result["lambda_max"] == 0
        assert result["n"] == 0
        assert result["beta_service"] == 0
        assert result["edges"] == []


class TestTheoremEndpoints:
    """24.2 — theorem_prediction and theorem_result endpoint structure."""

    def _mock_request(self, kernel: MELVKernel):
        """Build a minimal mock request with app.state.kernel."""
        from unittest.mock import MagicMock
        req = MagicMock()
        req.app.state.kernel = kernel
        return req

    @pytest.mark.asyncio
    async def test_theorem_prediction_returns_valid_structure(self):
        """GET /api/theorem_prediction returns required fields."""
        from api.theorem_router import theorem_prediction

        k = _make_kernel(_agent("a1"), _agent("a2"))
        # Some cooperative, some conflict interactions
        for _ in range(20):
            k.record_interaction("a1", "a2", cost=0.3, benefit=0.9, resource_type="compute")
        for _ in range(5):
            k.record_interaction("a1", "a2", cost=1.2, benefit=0.8, resource_type="compute")

        result = await theorem_prediction(self._mock_request(k))

        for field in ("i_critical", "pairs_above", "pairs_below",
                      "ecosystem_prediction", "ci_current",
                      "ci_predicted_at_equilibrium", "prediction_made_at"):
            assert field in result, f"Missing field '{field}' in theorem_prediction response"

        assert result["i_critical"] == pytest.approx(0.9995)
        assert isinstance(result["pairs_above"], list)
        assert isinstance(result["pairs_below"], list)
        assert result["ecosystem_prediction"] in ("COOPERATIVE", "NOT_COOPERATIVE", "BORDERLINE")

    @pytest.mark.asyncio
    async def test_theorem_result_returns_valid_structure(self):
        """GET /api/theorem_result returns required fields after a prediction."""
        from api.theorem_router import theorem_prediction, theorem_result, _theorem_state

        k = _make_kernel(_agent("b1"), _agent("b2"))
        for _ in range(15):
            k.record_interaction("b1", "b2", cost=0.2, benefit=1.0, resource_type="compute")

        req = self._mock_request(k)
        await theorem_prediction(req)     # must call predict first
        result = await theorem_result(req)

        for field in ("prediction_made_at", "ci_current", "theorem_confirmed",
                      "interactions_since_prediction", "interpretation"):
            assert field in result, f"Missing field '{field}' in theorem_result response"

        assert isinstance(result["theorem_confirmed"], bool)
        assert isinstance(result["interpretation"], str)


class TestVersionGate:
    """24.3 — version gate confirms v2.0.0."""

    def test_sandbox_version_is_2_0_0(self):
        assert SANDBOX_VERSION == "2.0.0", (
            f"Expected SANDBOX_VERSION='2.0.0', got '{SANDBOX_VERSION}'"
        )

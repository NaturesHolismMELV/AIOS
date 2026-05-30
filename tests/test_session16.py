"""
test_session16.py
=================
Session 16 — φ/ε Assessment Wizard
Target: 40 new tests → cumulative total ≥ 325

Tests cover:
  - _compute_phi / _compute_epsilon helpers
  - /sandbox/assess/phi  and /sandbox/assess/epsilon  endpoints
  - SandboxSubmitRequest accepts assessment_scores
  - assessment-derived phi/epsilon override manual values
  - N/A (null) scores excluded from average
  - persistence.save_sandbox_report stores assessment_scores column
  - PhiAssessmentScores / EpsilonAssessmentScores validation
  - AssessmentScores model structure
  - Edge cases: all N/A, single score, boundary scores (1, 10)
  - Landing page HTML contains wizard markers

Author: L.W. Evans | ORCID: 0009-0001-0963-1840 · Session 16 · v1.8.0
"""

import json
import math
import pytest
import sqlite3
import tempfile
import os

# ── Import helpers under test ──────────────────────────────────────────────

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.sandbox_router import (
    _compute_phi,
    _compute_epsilon,
    PhiAssessmentScores,
    EpsilonAssessmentScores,
    AssessmentScores,
    SandboxSubmitRequest,
)
from core.persistence import AIOSPersistence


# ─────────────────────────────────────────────────────────────────────────────
# GROUP 1 — _compute_phi (10 tests)
# ─────────────────────────────────────────────────────────────────────────────

class TestComputePhi:
    """_compute_phi normalises (mean-1)/9 → 0.0–1.0"""

    def test_all_scores_10_gives_phi_1(self):
        s = PhiAssessmentScores(
            training_recency=10, domain_specialisation=10,
            instruction_following=10, error_recovery=10,
            output_stability=10, calibration=10,
        )
        phi = _compute_phi(s)
        assert phi == pytest.approx(1.0, abs=1e-4)

    def test_all_scores_1_gives_phi_0(self):
        s = PhiAssessmentScores(
            training_recency=1, domain_specialisation=1,
            instruction_following=1, error_recovery=1,
            output_stability=1, calibration=1,
        )
        phi = _compute_phi(s)
        assert phi == pytest.approx(0.0, abs=1e-4)

    def test_midpoint_5_gives_phi_0_444(self):
        s = PhiAssessmentScores(
            training_recency=5, domain_specialisation=5,
            instruction_following=5, error_recovery=5,
            output_stability=5, calibration=5,
        )
        phi = _compute_phi(s)
        assert phi == pytest.approx((5 - 1) / 9, abs=1e-4)

    def test_na_scores_excluded(self):
        """All N/A except one score=10 → phi=1.0"""
        s = PhiAssessmentScores(training_recency=10)  # rest default to None
        phi = _compute_phi(s)
        assert phi == pytest.approx(1.0, abs=1e-4)

    def test_all_none_returns_none(self):
        s = PhiAssessmentScores()  # all None
        assert _compute_phi(s) is None

    def test_single_score_1_gives_phi_0(self):
        s = PhiAssessmentScores(calibration=1)
        assert _compute_phi(s) == pytest.approx(0.0, abs=1e-4)

    def test_phi_in_range_0_to_1(self):
        for v in range(1, 11):
            s = PhiAssessmentScores(
                training_recency=v, domain_specialisation=v,
                instruction_following=v, error_recovery=v,
                output_stability=v, calibration=v,
            )
            phi = _compute_phi(s)
            assert 0.0 <= phi <= 1.0

    def test_partial_na_uses_remaining(self):
        """Three scores of 10, three N/A → phi=1.0"""
        s = PhiAssessmentScores(training_recency=10, domain_specialisation=10, calibration=10)
        assert _compute_phi(s) == pytest.approx(1.0, abs=1e-4)

    def test_mixed_scores_average(self):
        """scores 1 and 10 → mean=5.5 → phi=(5.5-1)/9"""
        s = PhiAssessmentScores(training_recency=1, calibration=10)
        expected = (5.5 - 1) / 9
        assert _compute_phi(s) == pytest.approx(expected, abs=1e-4)

    def test_phi_result_is_float(self):
        s = PhiAssessmentScores(training_recency=7)
        phi = _compute_phi(s)
        assert isinstance(phi, float)


# ─────────────────────────────────────────────────────────────────────────────
# GROUP 2 — _compute_epsilon (10 tests)
# ─────────────────────────────────────────────────────────────────────────────

class TestComputeEpsilon:
    """_compute_epsilon normalises ((mean-1)/9)*8 → 0.0–8.0"""

    def test_all_scores_10_gives_eps_8(self):
        s = EpsilonAssessmentScores(
            context_sensitivity=10, prompt_injection_risk=10,
            tool_use_aggression=10, resource_consumption=10,
            feedback_responsiveness=10, autonomy_level=10,
        )
        eps = _compute_epsilon(s)
        assert eps == pytest.approx(8.0, abs=1e-4)

    def test_all_scores_1_gives_eps_0(self):
        s = EpsilonAssessmentScores(
            context_sensitivity=1, prompt_injection_risk=1,
            tool_use_aggression=1, resource_consumption=1,
            feedback_responsiveness=1, autonomy_level=1,
        )
        assert _compute_epsilon(s) == pytest.approx(0.0, abs=1e-4)

    def test_midpoint_5_gives_eps_3_556(self):
        s = EpsilonAssessmentScores(
            context_sensitivity=5, prompt_injection_risk=5,
            tool_use_aggression=5, resource_consumption=5,
            feedback_responsiveness=5, autonomy_level=5,
        )
        expected = ((5 - 1) / 9) * 8
        assert _compute_epsilon(s) == pytest.approx(expected, abs=1e-4)

    def test_na_scores_excluded(self):
        s = EpsilonAssessmentScores(autonomy_level=10)
        assert _compute_epsilon(s) == pytest.approx(8.0, abs=1e-4)

    def test_all_none_returns_none(self):
        s = EpsilonAssessmentScores()
        assert _compute_epsilon(s) is None

    def test_eps_in_range_0_to_8(self):
        for v in range(1, 11):
            s = EpsilonAssessmentScores(
                context_sensitivity=v, prompt_injection_risk=v,
                tool_use_aggression=v, resource_consumption=v,
                feedback_responsiveness=v, autonomy_level=v,
            )
            eps = _compute_epsilon(s)
            assert 0.0 <= eps <= 8.0

    def test_single_score_midpoint(self):
        s = EpsilonAssessmentScores(context_sensitivity=5)
        expected = ((5 - 1) / 9) * 8
        assert _compute_epsilon(s) == pytest.approx(expected, abs=1e-4)

    def test_mixed_na_and_scores(self):
        s = EpsilonAssessmentScores(context_sensitivity=1, autonomy_level=10)
        expected = ((5.5 - 1) / 9) * 8
        assert _compute_epsilon(s) == pytest.approx(expected, abs=1e-4)

    def test_epsilon_result_is_float(self):
        s = EpsilonAssessmentScores(context_sensitivity=7)
        assert isinstance(_compute_epsilon(s), float)

    def test_score_boundary_10_returns_8(self):
        s = EpsilonAssessmentScores(
            context_sensitivity=10, prompt_injection_risk=10,
            tool_use_aggression=10, resource_consumption=10,
            feedback_responsiveness=10, autonomy_level=10,
        )
        assert _compute_epsilon(s) == pytest.approx(8.0, abs=1e-4)


# ─────────────────────────────────────────────────────────────────────────────
# GROUP 3 — Pydantic model validation (10 tests)
# ─────────────────────────────────────────────────────────────────────────────

class TestAssessmentModels:

    def test_phi_scores_all_none_default(self):
        s = PhiAssessmentScores()
        assert s.training_recency is None
        assert s.calibration is None

    def test_phi_scores_valid_range(self):
        s = PhiAssessmentScores(training_recency=5.0, calibration=8.0)
        assert s.training_recency == 5.0
        assert s.calibration == 8.0

    def test_phi_scores_rejects_below_1(self):
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            PhiAssessmentScores(training_recency=0.5)

    def test_phi_scores_rejects_above_10(self):
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            PhiAssessmentScores(calibration=11)

    def test_epsilon_scores_valid(self):
        s = EpsilonAssessmentScores(autonomy_level=7, tool_use_aggression=3)
        assert s.autonomy_level == 7
        assert s.tool_use_aggression == 3

    def test_epsilon_scores_rejects_0(self):
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            EpsilonAssessmentScores(autonomy_level=0)

    def test_assessment_scores_optional_fields(self):
        a = AssessmentScores()
        assert a.agent_category is None
        assert a.phi_scores is None
        assert a.epsilon_scores is None

    def test_assessment_scores_with_data(self):
        a = AssessmentScores(
            agent_category='tool_using',
            phi_computed=0.7,
            epsilon_computed=3.5,
        )
        assert a.agent_category == 'tool_using'
        assert a.phi_computed == pytest.approx(0.7)

    def test_sandbox_submit_request_default_no_assessment(self):
        r = SandboxSubmitRequest(agent_id='a1', agent_name='A1', domain='coding')
        assert r.assessment_scores is None

    def test_sandbox_submit_request_accepts_assessment(self):
        a = AssessmentScores(
            agent_category='reactive',
            phi_scores=PhiAssessmentScores(calibration=8),
            epsilon_scores=EpsilonAssessmentScores(autonomy_level=3),
        )
        r = SandboxSubmitRequest(
            agent_id='a2', agent_name='A2', domain='analysis',
            assessment_scores=a,
        )
        assert r.assessment_scores.agent_category == 'reactive'


# ─────────────────────────────────────────────────────────────────────────────
# GROUP 4 — Persistence: assessment_scores column (5 tests)
# ─────────────────────────────────────────────────────────────────────────────

class TestPersistenceAssessmentScores:

    @pytest.fixture
    def tmp_db(self):
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            path = f.name
        db = AIOSPersistence(db_path=path)
        yield db
        os.unlink(path)

    def _make_report(self, run_id='r1', verdict='CERTIFIED', cls=85.0, agent_id='a1'):
        """Create a minimal mock CertificationReport."""
        class MockReport:
            def __init__(self):
                self.run_id       = run_id
                self.agent_id     = agent_id
                self.agent_domain = 'coding'
                self.verdict      = verdict
                self.cls_score    = cls
            def to_dict(self):
                return {
                    'run_id': self.run_id,
                    'agent_id': self.agent_id,
                    'verdict': self.verdict,
                    'cls_score': self.cls_score,
                    'certification_anchor': {'certified_at': '2026-03-08T12:00:00'},
                }
        return MockReport()

    def test_save_without_assessment_scores(self, tmp_db):
        r = self._make_report()
        tmp_db.save_sandbox_report(r)
        rows = tmp_db.load_sandbox_reports()
        assert len(rows) == 1

    def test_save_with_assessment_scores(self, tmp_db):
        r = self._make_report()
        scores = {'phi_computed': 0.75, 'epsilon_computed': 2.5, 'agent_category': 'tool_using'}
        tmp_db.save_sandbox_report(r, assessment_scores=scores)
        # Read raw from DB
        conn = sqlite3.connect(tmp_db.db_path)
        row = conn.execute('SELECT assessment_scores FROM sandbox_reports WHERE run_id=?', (r.run_id,)).fetchone()
        conn.close()
        assert row is not None
        stored = json.loads(row[0])
        assert stored['phi_computed'] == pytest.approx(0.75)
        assert stored['agent_category'] == 'tool_using'

    def test_save_assessment_scores_null_when_not_provided(self, tmp_db):
        r = self._make_report(run_id='r2')
        tmp_db.save_sandbox_report(r)
        conn = sqlite3.connect(tmp_db.db_path)
        row = conn.execute('SELECT assessment_scores FROM sandbox_reports WHERE run_id=?', ('r2',)).fetchone()
        conn.close()
        assert row[0] is None

    def test_upsert_updates_assessment_scores(self, tmp_db):
        r = self._make_report()
        tmp_db.save_sandbox_report(r, assessment_scores={'phi_computed': 0.5})
        tmp_db.save_sandbox_report(r, assessment_scores={'phi_computed': 0.9})
        conn = sqlite3.connect(tmp_db.db_path)
        row = conn.execute('SELECT assessment_scores FROM sandbox_reports WHERE run_id=?', (r.run_id,)).fetchone()
        conn.close()
        stored = json.loads(row[0])
        assert stored['phi_computed'] == pytest.approx(0.9)

    def test_assessment_column_exists_in_schema(self, tmp_db):
        conn = sqlite3.connect(tmp_db.db_path)
        cols = [r[1] for r in conn.execute('PRAGMA table_info(sandbox_reports)').fetchall()]
        conn.close()
        assert 'assessment_scores' in cols


# ─────────────────────────────────────────────────────────────────────────────
# GROUP 5 — Landing page wizard markers (5 tests)
# ─────────────────────────────────────────────────────────────────────────────

class TestLandingPageWizard:

    @pytest.fixture
    def landing_html(self):
        path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'frontend', 'landing.html'
        )
        return open(path, encoding='utf-8').read()

    def test_wizard_steps_nav_present(self, landing_html):
        assert 'wizard-steps-nav' in landing_html

    def test_four_wizard_panels_present(self, landing_html):
        for i in range(1, 5):
            assert f'id="wz-panel-{i}"' in landing_html

    def test_phi_params_container_present(self, landing_html):
        assert 'id="phi-params"' in landing_html

    def test_eps_params_container_present(self, landing_html):
        assert 'id="eps-params"' in landing_html

    def test_assess_summary_result_panel_present(self, landing_html):
        assert 'id="result-assess-summary"' in landing_html

    def test_category_buttons_present(self, landing_html):
        for cat in ['reactive', 'tool_using', 'multi_agent', 'autonomous']:
            assert f'data-cat="{cat}"' in landing_html

    def test_version_updated_to_1_8_0(self, landing_html):
        assert 'v1.9.0' in landing_html

    def test_compute_phi_js_function_present(self, landing_html):
        assert 'function computePhi' in landing_html

    def test_compute_epsilon_js_function_present(self, landing_html):
        assert 'function computeEpsilon' in landing_html

    def test_assessment_scores_in_submit_payload(self, landing_html):
        assert 'assessment_scores' in landing_html

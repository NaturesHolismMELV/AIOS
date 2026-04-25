"""
test_session17.py — Session 17 · v1.9.0

Tests for:
  - Parameter-aware advisory text (sandbox_engine._build_advisory)
  - Coordination Overhead Score (compute_coordination_overhead_score)
  - φ Lifecycle Classification (classify_phi_lifecycle)
  - Iterative Loop category (landing.html)
  - Operation mode, tool count, shared state (landing.html + router)
  - New router endpoints (/assess/coordination-overhead, /assess/phi-lifecycle)
  - CertificationReport new fields (coordination_overhead, phi_lifecycle)
  - Version bump to 1.9.0
  - CertificationRun new fields (tool_count, operation_mode, shared_state)
  - Continuous ε penalty in submit
"""

import json
import os
import sys
import pytest
import math

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from core.sandbox_engine import SandboxEngine, CertificationRun, CertificationReport, CISnapshot
from core.melv_engine import AgentProfile, AgentStatus


# ══════════════════════════════════════════════════════════════════
# SECTION 1: Coordination Overhead Score (10 tests)
# ══════════════════════════════════════════════════════════════════

class TestCoordinationOverheadScore:
    """SandboxEngine.compute_coordination_overhead_score — 10 tests."""

    def test_low_band_small_product(self):
        result = SandboxEngine.compute_coordination_overhead_score(1.0, 1)
        assert result["band"] == "LOW"
        assert result["score"] == 1.0
        assert result["advisory"] is None

    def test_low_band_zero_tools(self):
        result = SandboxEngine.compute_coordination_overhead_score(3.0, 0)
        assert result["band"] == "LOW"
        assert result["score"] == 0.0

    def test_moderate_band_boundary(self):
        # 3.0 * 1 = 3.0 → MODERATE
        result = SandboxEngine.compute_coordination_overhead_score(3.0, 1)
        assert result["band"] == "MODERATE"
        assert result["advisory"] is not None

    def test_moderate_band_upper_boundary(self):
        # 2.0 * 1 = 2.0 → MODERATE (boundary)
        result = SandboxEngine.compute_coordination_overhead_score(2.0, 1)
        assert result["band"] == "MODERATE"

    def test_high_band_triggered(self):
        # 4.0 * 2 = 8.0 → HIGH
        result = SandboxEngine.compute_coordination_overhead_score(4.0, 2)
        assert result["band"] == "HIGH"
        assert "Jones" in result["advisory"]

    def test_high_band_advisory_mentions_tools(self):
        result = SandboxEngine.compute_coordination_overhead_score(5.0, 3)
        assert "tool" in result["advisory"].lower()

    def test_high_band_score_correct(self):
        result = SandboxEngine.compute_coordination_overhead_score(3.5, 10)
        assert result["score"] == pytest.approx(35.0, rel=1e-4)
        assert result["band"] == "HIGH"

    def test_score_exact_4_is_high(self):
        # score == 4.0 → strictly above threshold → HIGH
        result = SandboxEngine.compute_coordination_overhead_score(2.0, 2)
        assert result["band"] == "HIGH"

    def test_score_just_below_4_is_moderate(self):
        result = SandboxEngine.compute_coordination_overhead_score(1.99, 2)
        assert result["band"] == "MODERATE"

    def test_returns_dict_with_required_keys(self):
        result = SandboxEngine.compute_coordination_overhead_score(3.0, 5)
        assert "score" in result
        assert "band" in result
        assert "advisory" in result


# ══════════════════════════════════════════════════════════════════
# SECTION 2: φ Lifecycle Classification (10 tests)
# ══════════════════════════════════════════════════════════════════

class TestPhiLifecycleClassification:
    """SandboxEngine.classify_phi_lifecycle — 10 tests."""

    def test_permanent_tier_at_085(self):
        result = SandboxEngine.classify_phi_lifecycle(0.85)
        assert result["tier"] == "Permanent"

    def test_permanent_tier_at_1(self):
        result = SandboxEngine.classify_phi_lifecycle(1.0)
        assert result["tier"] == "Permanent"

    def test_permanent_has_no_advisory(self):
        result = SandboxEngine.classify_phi_lifecycle(0.90)
        assert result["advisory"] is None

    def test_working_tier_midrange(self):
        result = SandboxEngine.classify_phi_lifecycle(0.70)
        assert result["tier"] == "Working"

    def test_working_tier_at_050(self):
        result = SandboxEngine.classify_phi_lifecycle(0.50)
        assert result["tier"] == "Working"

    def test_working_advisory_present(self):
        result = SandboxEngine.classify_phi_lifecycle(0.65)
        assert result["advisory"] is not None
        assert "record_interaction" in result["advisory"].lower() or "persist" in result["advisory"].lower()

    def test_ephemeral_tier_below_050(self):
        result = SandboxEngine.classify_phi_lifecycle(0.30)
        assert result["tier"] == "Ephemeral"

    def test_ephemeral_tier_at_zero(self):
        result = SandboxEngine.classify_phi_lifecycle(0.0)
        assert result["tier"] == "Ephemeral"

    def test_ephemeral_advisory_mentions_jones(self):
        result = SandboxEngine.classify_phi_lifecycle(0.20)
        assert "Jones" in result["advisory"]

    def test_classification_returns_all_keys(self):
        result = SandboxEngine.classify_phi_lifecycle(0.75)
        assert "tier" in result
        assert "label" in result
        assert "advisory" in result


# ══════════════════════════════════════════════════════════════════
# SECTION 3: Parameter-aware Advisory Text (10 tests)
# ══════════════════════════════════════════════════════════════════

class TestParameterAwareAdvisory:
    """SandboxEngine._build_advisory with assessment_scores — 10 tests."""

    def _make_profile(self, phi=0.5, epsilon=5.0):
        return AgentProfile(
            agent_id="test-agent", name="Test", domain="test",
            phi=phi, epsilon=epsilon, beta_pref=1.0,
            status=AgentStatus.MATURING,
        )

    def test_high_prompt_injection_risk_triggers_mitigation(self):
        profile = self._make_profile(epsilon=5.5)
        assessment = {
            "agent_category": "tool_using",
            "epsilon_scores": {"prompt_injection_risk": 9.0, "autonomy_level": 5.0,
                                "context_sensitivity": None, "tool_use_aggression": None,
                                "resource_consumption": None, "feedback_responsiveness": None},
            "phi_scores": {},
        }
        engine = SandboxEngine()
        advisory = engine._build_advisory(
            "CERTIFIED_WITH_ADVISORY", 0.0, 0.0, 0.0, profile, assessment
        )
        assert advisory is not None
        assert "prompt" in advisory.lower() or "injection" in advisory.lower()

    def test_high_tool_use_aggression_triggers_mitigation(self):
        profile = self._make_profile(epsilon=5.0)
        assessment = {
            "agent_category": "tool_using",
            "epsilon_scores": {"tool_use_aggression": 8.0, "prompt_injection_risk": None,
                                "autonomy_level": None, "context_sensitivity": None,
                                "resource_consumption": None, "feedback_responsiveness": None},
            "phi_scores": {},
        }
        engine = SandboxEngine()
        advisory = engine._build_advisory(
            "CERTIFIED_WITH_ADVISORY", 0.0, 0.0, 0.0, profile, assessment
        )
        assert "tool" in advisory.lower() or "rate limit" in advisory.lower() or "budget" in advisory.lower()

    def test_category_specific_mitigation_iterative_loop(self):
        """autonomy_level high + iterative_loop category → loop-specific advice."""
        profile = self._make_profile(epsilon=5.5)
        assessment = {
            "agent_category": "iterative_loop",
            "epsilon_scores": {"autonomy_level": 9.0, "prompt_injection_risk": None,
                                "tool_use_aggression": None, "context_sensitivity": None,
                                "resource_consumption": None, "feedback_responsiveness": None},
            "phi_scores": {},
        }
        engine = SandboxEngine()
        advisory = engine._build_advisory(
            "CERTIFIED_WITH_ADVISORY", 0.0, 0.0, 0.0, profile, assessment
        )
        assert "loop" in advisory.lower() or "iteration" in advisory.lower() or "operator" in advisory.lower()

    def test_low_phi_parameter_generates_improvement_suggestion(self):
        profile = self._make_profile(phi=0.3, epsilon=2.0)
        assessment = {
            "agent_category": "reactive",
            "epsilon_scores": {},
            "phi_scores": {"instruction_following": 2.0, "training_recency": None,
                            "domain_specialisation": None, "error_recovery": None,
                            "output_stability": None, "calibration": None},
        }
        engine = SandboxEngine()
        advisory = engine._build_advisory(
            "CERTIFIED_WITH_ADVISORY", 0.0, 0.0, 0.0, profile, assessment
        )
        # Should mention the worst-scoring phi parameter
        assert "instruction" in advisory.lower() or "rlhf" in advisory.lower() or "chain" in advisory.lower()

    def test_certified_verdict_returns_none_even_with_scores(self):
        profile = self._make_profile()
        assessment = {
            "agent_category": "reactive",
            "epsilon_scores": {"prompt_injection_risk": 9.0},
            "phi_scores": {},
        }
        engine = SandboxEngine()
        advisory = engine._build_advisory("CERTIFIED", 0.0, 0.0, 0.0, profile, assessment)
        assert advisory is None

    def test_no_assessment_scores_falls_back_to_generic(self):
        """Without assessment, epsilon > 4.5 triggers generic advisory."""
        profile = self._make_profile(epsilon=5.0)
        engine = SandboxEngine()
        advisory = engine._build_advisory(
            "CERTIFIED_WITH_ADVISORY", 0.0, 0.0, 0.0, profile, None
        )
        assert advisory is not None
        assert "bifurcation" in advisory.lower() or "plasticity" in advisory.lower()

    def test_ois_triggers_oscillation_advisory(self):
        profile = self._make_profile()
        engine = SandboxEngine()
        advisory = engine._build_advisory(
            "CERTIFIED_WITH_ADVISORY", 0.5, 0.0, 0.0, profile, None
        )
        assert "oscillation" in advisory.lower() or "ois" in advisory.lower()

    def test_ddc_triggers_drift_advisory(self):
        profile = self._make_profile()
        engine = SandboxEngine()
        advisory = engine._build_advisory(
            "CERTIFIED_WITH_ADVISORY", 0.0, -1e-4, 0.0, profile, None
        )
        assert "drift" in advisory.lower() or "degradation" in advisory.lower()

    def test_high_delta_hl_triggers_halflife_advisory(self):
        profile = self._make_profile()
        engine = SandboxEngine()
        advisory = engine._build_advisory(
            "CERTIFIED_WITH_ADVISORY", 0.0, 0.0, 10.0, profile, None
        )
        assert "half-life" in advisory.lower() or "10." in advisory

    def test_multiple_mitigations_in_advisory_for_multiple_high_scores(self):
        profile = self._make_profile(epsilon=6.0)
        assessment = {
            "agent_category": "autonomous",
            "epsilon_scores": {
                "prompt_injection_risk": 9.0,
                "tool_use_aggression": 8.0,
                "autonomy_level": 8.0,
                "context_sensitivity": None,
                "resource_consumption": None,
                "feedback_responsiveness": None,
            },
            "phi_scores": {},
        }
        engine = SandboxEngine()
        advisory = engine._build_advisory(
            "NOT_CERTIFIED", 0.0, 0.0, 0.0, profile, assessment
        )
        # Should have multiple bullet points of advice
        assert advisory is not None
        assert "•" in advisory or "mitigation" in advisory.lower()


# ══════════════════════════════════════════════════════════════════
# SECTION 4: CertificationReport New Fields (5 tests)
# ══════════════════════════════════════════════════════════════════

class TestCertificationReportNewFields:
    """CertificationReport coordination_overhead and phi_lifecycle fields."""

    def _make_report(self, phi=0.75, epsilon=3.0, tool_count=0):
        """Build a complete report via SandboxEngine for testing."""
        engine = SandboxEngine()
        profile = AgentProfile(
            agent_id="rpt-test", name="RptTest", domain="test",
            phi=phi, epsilon=epsilon, beta_pref=1.0,
            status=AgentStatus.MATURING,
        )
        run = engine.submit(profile)
        run.tool_count = tool_count
        run.baseline_metrics = CISnapshot(
            ci_half_life_sec=None, ci_drift_coefficient=0.0,
            oscillation_count=5, final_ci=0.80, regime="cooperative"
        )
        run.agent_metrics = CISnapshot(
            ci_half_life_sec=None, ci_drift_coefficient=0.0,
            oscillation_count=5, final_ci=0.80, regime="cooperative"
        )
        report = engine.compute_report(run.run_id)
        return report

    def test_phi_lifecycle_present_in_report(self):
        report = self._make_report(phi=0.75)
        assert report.phi_lifecycle is not None
        assert "tier" in report.phi_lifecycle

    def test_phi_lifecycle_tier_working_for_075(self):
        report = self._make_report(phi=0.75)
        assert report.phi_lifecycle["tier"] == "Working"

    def test_phi_lifecycle_tier_permanent_for_090(self):
        report = self._make_report(phi=0.90)
        assert report.phi_lifecycle["tier"] == "Permanent"

    def test_coordination_overhead_none_when_tool_count_zero(self):
        report = self._make_report(tool_count=0)
        assert report.coordination_overhead is None

    def test_coordination_overhead_present_when_tools_set(self):
        report = self._make_report(epsilon=3.0, tool_count=5)
        assert report.coordination_overhead is not None
        assert report.coordination_overhead["score"] == pytest.approx(15.0, rel=1e-4)


# ══════════════════════════════════════════════════════════════════
# SECTION 5: CertificationRun New Fields (5 tests)
# ══════════════════════════════════════════════════════════════════

class TestCertificationRunNewFields:
    """CertificationRun tool_count, operation_mode, shared_state."""

    def _make_run(self):
        engine = SandboxEngine()
        profile = AgentProfile(
            agent_id="run-test", name="RunTest", domain="test",
            phi=0.7, epsilon=3.0, beta_pref=1.0,
            status=AgentStatus.MATURING,
        )
        return engine.submit(profile)

    def test_default_tool_count_is_zero(self):
        run = self._make_run()
        assert run.tool_count == 0

    def test_default_operation_mode_is_episodic(self):
        run = self._make_run()
        assert run.operation_mode == "episodic"

    def test_default_shared_state_is_none(self):
        run = self._make_run()
        assert run.shared_state == "none"

    def test_to_dict_includes_tool_count(self):
        run = self._make_run()
        run.tool_count = 7
        d = run.to_dict()
        assert d["tool_count"] == 7

    def test_to_dict_includes_operation_mode(self):
        run = self._make_run()
        run.operation_mode = "continuous"
        d = run.to_dict()
        assert d["operation_mode"] == "continuous"


# ══════════════════════════════════════════════════════════════════
# SECTION 6: Landing Page Session 17 Enhancements (10 tests)
# ══════════════════════════════════════════════════════════════════

class TestLandingPageSession17:
    """Landing page wizard enhancements for Session 17."""

    LANDING = os.path.join(ROOT, "frontend", "landing.html")

    def _html(self):
        return open(self.LANDING, encoding="utf-8").read()

    def test_iterative_loop_category_present(self):
        assert 'data-cat="iterative_loop"' in self._html()

    def test_iterative_loop_label_present(self):
        assert "Iterative Loop" in self._html()

    def test_operation_mode_radio_present(self):
        html = self._html()
        assert "operation_mode" in html
        assert "episodic" in html
        assert "continuous" in html

    def test_tool_count_input_present(self):
        assert "wz-tool-count" in self._html()

    def test_shared_state_select_present(self):
        assert "wz-shared-state" in self._html()

    def test_continuous_warning_element_present(self):
        assert "continuous-warning" in self._html()

    def test_shared_state_warning_element_present(self):
        assert "shared-state-warning" in self._html()

    def test_on_operation_mode_change_function_present(self):
        assert "onOperationModeChange" in self._html()

    def test_on_shared_state_change_function_present(self):
        assert "onSharedStateChange" in self._html()

    def test_version_updated_to_1_9_0(self):
        assert "v1.9.0" in self._html() or "1.9.0" in self._html()


# ══════════════════════════════════════════════════════════════════
# SECTION 7: Version and Router (5 tests)
# ══════════════════════════════════════════════════════════════════

class TestSession17Version:
    """Version bump and router endpoints."""

    def test_melvcore_version_1_9_0(self):
        import melvcore
        assert melvcore.__version__ == "1.9.0"

    def test_mcp_json_version(self):
        path = os.path.join(ROOT, "mcp.json")
        d = json.loads(open(path).read())
        assert d["version"] == "1.9.0"

    def test_sandbox_router_has_coordination_overhead_endpoint(self):
        content = open(os.path.join(ROOT, "api", "sandbox_router.py"), encoding="utf-8").read()
        assert "coordination-overhead" in content

    def test_sandbox_router_has_phi_lifecycle_endpoint(self):
        content = open(os.path.join(ROOT, "api", "sandbox_router.py"), encoding="utf-8").read()
        assert "phi-lifecycle" in content

    def test_sandbox_router_session17_fields_present(self):
        content = open(os.path.join(ROOT, "api", "sandbox_router.py"), encoding="utf-8").read()
        assert "tool_count" in content
        assert "operation_mode" in content
        assert "shared_state" in content
        assert "continuous_penalty_applied" in content

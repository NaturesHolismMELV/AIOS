"""
test_session34.py — MELVcore Session 34: Enterprise Platform Adapters (v3.0.0)
===============================================================================

Tests for Session 34 deliverables:
  - Ecotao identity cleanup verification
  - AgentforceObservationBuilder
  - CopilotObservationBuilder
  - VertexObservationBuilder
  - ServiceNowObservationBuilder
  - adapters/__init__.py exports
  - Dashboard status-undefined fix (static check)
  - GitHub/Zenodo hygiene constants

Test groups
-----------
  D01–D04  Ecotao identity cleanup (grep checks)
  D05–D12  AgentforceObservationBuilder
  D13–D20  CopilotObservationBuilder
  D21–D28  VertexObservationBuilder
  D29–D36  ServiceNowObservationBuilder
  D37–D40  adapters/__init__.py exports
  D41–D44  Version string and hygiene checks

Author: Laurence W. Evans | ORCID: 0009-0001-0963-1840 | Cape Town, South Africa
Session: 34 · Version: 3.0.0
"""

import os
import glob

import pytest

from core.observe_schema import ResourcePolicy, ContentionEvent, LatencySample
from adapters.agentforce_adapter import AgentforceObservationBuilder
from adapters.copilot_adapter import CopilotObservationBuilder
from adapters.vertex_adapter import VertexObservationBuilder
from adapters.servicenow_adapter import ServiceNowObservationBuilder
import adapters as adapters_module


# ── Helpers ────────────────────────────────────────────────────────────────

def _make_agentforce(n_steps=5, success=True, branch=0, fault=0):
    b = AgentforceObservationBuilder(
        agent_id="af-test",
        task_domain="case_resolution",
        resource_policy=ResourcePolicy(
            token_budget_per_hour=100_000,
            api_quota_per_minute=100,
        ),
    )
    for i in range(n_steps):
        b.record_action_step(
            task_id=f"case-{i:04d}",
            success=success,
            branch_retries=branch,
            fault_handler_activations=fault,
            duration_seconds=3.0,
            downstream_accepted=success,
            latency_ms=300.0,
            task_type="case_update",
        )
    return b


def _make_copilot(n_turns=5, success=True, fallbacks=0, escalations=0):
    b = CopilotObservationBuilder(
        agent_id="copilot-test",
        task_domain="hr_query",
        resource_policy=ResourcePolicy(
            token_budget_per_hour=80_000,
            api_quota_per_minute=60,
        ),
    )
    for i in range(n_turns):
        b.record_topic_turn(
            session_id=f"sess-{i:04d}",
            topic_name="HRPolicyTopic",
            success=success,
            fallback_redirects=fallbacks,
            escalations=escalations,
            duration_seconds=4.0,
            downstream_accepted=success,
            latency_ms=400.0,
            task_type="hr_query",
        )
    return b


def _make_vertex(n_turns=5, success=True, fallbacks=0, webhook_retries=0):
    b = VertexObservationBuilder(
        agent_id="vertex-test",
        task_domain="product_support",
        resource_policy=ResourcePolicy(
            token_budget_per_hour=120_000,
            api_quota_per_minute=300,
        ),
    )
    for i in range(n_turns):
        b.record_agent_turn(
            session_id=f"vsess-{i:04d}",
            intent_name="ProductSupportIntent",
            success=success,
            fallback_activations=fallbacks,
            webhook_retries=webhook_retries,
            duration_seconds=2.8,
            downstream_accepted=success,
            latency_ms=280.0,
            task_type="product_support",
        )
    return b


def _make_snow(n_turns=5, success=True, redirects=0, fallbacks=0):
    b = ServiceNowObservationBuilder(
        agent_id="snow-test",
        task_domain="incident_resolution",
        resource_policy=ResourcePolicy(
            token_budget_per_hour=60_000,
            api_quota_per_minute=120,
        ),
    )
    for i in range(n_turns):
        b.record_conversation_turn(
            session_id=f"snow-{i:04d}",
            topic_name="IncidentReportTopic",
            success=success,
            topic_redirects=redirects,
            fallback_scripts=fallbacks,
            duration_seconds=5.0,
            downstream_accepted=success,
            latency_ms=500.0,
            task_type="incident_report",
        )
    return b


# ═══════════════════════════════════════════════════════════════════════════
# D01–D04  Ecotao identity cleanup
# ═══════════════════════════════════════════════════════════════════════════

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def _grep_ecotao():
    hits = []
    # Exclude test_session34.py itself — it contains "Ecotao" as string literals in test code
    exclude = os.path.abspath(__file__)
    patterns = ["**/*.py", "**/*.md", "**/*.txt", "**/*.cff"]
    for pat in patterns:
        for path in glob.glob(os.path.join(ROOT, pat), recursive=True):
            if os.path.abspath(path) == exclude:
                continue
            try:
                with open(path, "r", encoding="utf-8", errors="replace") as f:
                    for lineno, line in enumerate(f, 1):
                        if "Ecotao" in line:
                            hits.append((path, lineno, line.strip()))
            except Exception:
                pass
    return hits


def test_D01_no_ecotao_in_py_files():
    """D01 — No 'Ecotao' in any .py file."""
    hits = [(p, n, l) for p, n, l in _grep_ecotao() if p.endswith(".py")]
    assert hits == [], f"Ecotao found in .py files: {hits}"


def test_D02_no_ecotao_in_md_files():
    """D02 — No 'Ecotao' in any .md file."""
    hits = [(p, n, l) for p, n, l in _grep_ecotao() if p.endswith(".md")]
    assert hits == [], f"Ecotao found in .md files: {hits}"


def test_D03_melv_engine_correct_attribution():
    """D03 — melv_engine.py uses canonical ORCID attribution."""
    path = os.path.join(ROOT, "core", "melv_engine.py")
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        content = f.read()
    assert "0009-0001-0963-1840" in content, "ORCID missing from melv_engine.py"
    assert "Ecotao" not in content, "Ecotao still present in melv_engine.py"


def test_D04_governance_init_correct_attribution():
    """D04 — governance/__init__.py uses canonical attribution."""
    path = os.path.join(ROOT, "governance", "__init__.py")
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        content = f.read()
    assert "Ecotao" not in content


# ═══════════════════════════════════════════════════════════════════════════
# D05–D12  AgentforceObservationBuilder
# ═══════════════════════════════════════════════════════════════════════════

def test_D05_agentforce_builder_instantiates():
    """D05 — AgentforceObservationBuilder instantiates without error."""
    b = AgentforceObservationBuilder(
        agent_id="af-001",
        task_domain="case_resolution",
    )
    assert b.agent_id == "af-001"


def test_D06_agentforce_build_returns_payload():
    """D06 — build() returns a valid ObservationPayload."""
    b = _make_agentforce(n_steps=5)
    payload = b.build()
    assert payload.agent_id == "af-test"
    assert payload.task_domain == "case_resolution"
    assert len(payload.domain_success_history) == 5


def test_D07_agentforce_step_count():
    """D07 — step_count tracks recorded action steps."""
    b = _make_agentforce(n_steps=8)
    assert b.step_count == 8


def test_D08_agentforce_success_rate_all_pass():
    """D08 — success_rate = 1.0 when all steps succeed."""
    b = _make_agentforce(n_steps=6, success=True)
    assert b.success_rate == pytest.approx(1.0)


def test_D09_agentforce_success_rate_all_fail():
    """D09 — success_rate = 0.0 when all steps fail."""
    b = _make_agentforce(n_steps=4, success=False)
    assert b.success_rate == pytest.approx(0.0)


def test_D10_agentforce_branch_retries_produce_reconfigs():
    """D10 — branch_retries=2 per step → 2n ReconfigEvents(type=branching)."""
    b = _make_agentforce(n_steps=3, branch=2)
    payload = b.build()
    branching = [r for r in payload.reconfiguration_events if r.event_type == "branching"]
    assert len(branching) == 6


def test_D11_agentforce_governor_limit_produces_contention():
    """D11 — record_governor_limit_hit() appends ContentionEvent(origin=infra)."""
    b = _make_agentforce(n_steps=2)
    b.record_governor_limit_hit("SOQL_LIMIT_HIT")
    payload = b.build()
    infra = [c for c in payload.contention_events if c.origin == "infra"]
    assert len(infra) == 1
    # SOQL_LIMIT_HIT maps to resource_type='compute'
    assert infra[0].resource_type == "compute"


def test_D12_agentforce_reset_clears_buffers():
    """D12 — reset() clears all accumulated signals."""
    b = _make_agentforce(n_steps=5)
    b.reset()
    assert b.step_count == 0
    assert b.success_rate is None


# ═══════════════════════════════════════════════════════════════════════════
# D13–D20  CopilotObservationBuilder
# ═══════════════════════════════════════════════════════════════════════════

def test_D13_copilot_builder_instantiates():
    """D13 — CopilotObservationBuilder instantiates without error."""
    b = CopilotObservationBuilder(agent_id="cp-001", task_domain="hr_query")
    assert b.agent_id == "cp-001"


def test_D14_copilot_build_returns_payload():
    """D14 — build() returns a valid ObservationPayload."""
    b = _make_copilot(n_turns=4)
    payload = b.build()
    assert len(payload.domain_success_history) == 4


def test_D15_copilot_step_count():
    """D15 — step_count tracks recorded turns."""
    b = _make_copilot(n_turns=7)
    assert b.step_count == 7


def test_D16_copilot_success_rate():
    """D16 — success_rate = 1.0 for all-pass turns."""
    b = _make_copilot(n_turns=5, success=True)
    assert b.success_rate == pytest.approx(1.0)


def test_D17_copilot_fallback_redirects_produce_branching():
    """D17 — fallback_redirects=1 per turn → n branching ReconfigEvents."""
    b = _make_copilot(n_turns=4, fallbacks=1)
    payload = b.build()
    branching = [r for r in payload.reconfiguration_events if r.event_type == "branching"]
    assert len(branching) == 4


def test_D18_copilot_escalations_produce_repair():
    """D18 — escalations=1 per turn → n repair ReconfigEvents."""
    b = _make_copilot(n_turns=3, escalations=1)
    payload = b.build()
    repair = [r for r in payload.reconfiguration_events if r.event_type == "repair"]
    assert len(repair) == 3


def test_D19_copilot_connector_throttle_produces_contention():
    """D19 — record_connector_throttle() appends infra ContentionEvent."""
    b = _make_copilot(n_turns=2)
    b.record_connector_throttle("SharePoint")
    payload = b.build()
    infra = [c for c in payload.contention_events if c.origin == "infra"]
    assert len(infra) == 1
    assert infra[0].resource_type == "api_quota"


def test_D20_copilot_reset_clears_buffers():
    """D20 — reset() clears all accumulated signals."""
    b = _make_copilot(n_turns=4)
    b.reset()
    assert b.step_count == 0


# ═══════════════════════════════════════════════════════════════════════════
# D21–D28  VertexObservationBuilder
# ═══════════════════════════════════════════════════════════════════════════

def test_D21_vertex_builder_instantiates():
    """D21 — VertexObservationBuilder instantiates without error."""
    b = VertexObservationBuilder(agent_id="vx-001", task_domain="product_support")
    assert b.agent_id == "vx-001"


def test_D22_vertex_build_returns_payload():
    """D22 — build() returns a valid ObservationPayload."""
    b = _make_vertex(n_turns=5)
    payload = b.build()
    assert len(payload.domain_success_history) == 5


def test_D23_vertex_step_count():
    """D23 — step_count tracks recorded turns."""
    b = _make_vertex(n_turns=6)
    assert b.step_count == 6


def test_D24_vertex_success_rate():
    """D24 — success_rate = 1.0 for all-pass turns."""
    b = _make_vertex(n_turns=5, success=True)
    assert b.success_rate == pytest.approx(1.0)


def test_D25_vertex_fallback_activations_produce_branching():
    """D25 — fallback_activations=2 per turn → 2n branching ReconfigEvents."""
    b = _make_vertex(n_turns=3, fallbacks=2)
    payload = b.build()
    branching = [r for r in payload.reconfiguration_events if r.event_type == "branching"]
    assert len(branching) == 6


def test_D26_vertex_webhook_retries_produce_repair():
    """D26 — webhook_retries=1 per turn → n repair ReconfigEvents."""
    b = _make_vertex(n_turns=4, webhook_retries=1)
    payload = b.build()
    repair = [r for r in payload.reconfiguration_events if r.event_type == "repair"]
    assert len(repair) == 4


def test_D27_vertex_quota_event_produces_contention():
    """D27 — record_quota_event() appends infra ContentionEvent."""
    b = _make_vertex(n_turns=2)
    b.record_quota_event("TOKENS_PER_MINUTE", http_status=429)
    payload = b.build()
    infra = [c for c in payload.contention_events if c.origin == "infra"]
    assert len(infra) == 1
    assert infra[0].resource_type == "api_quota"  # 429 → api_quota


def test_D28_vertex_reset_clears_buffers():
    """D28 — reset() clears all accumulated signals."""
    b = _make_vertex(n_turns=5)
    b.reset()
    assert b.step_count == 0


# ═══════════════════════════════════════════════════════════════════════════
# D29–D36  ServiceNowObservationBuilder
# ═══════════════════════════════════════════════════════════════════════════

def test_D29_snow_builder_instantiates():
    """D29 — ServiceNowObservationBuilder instantiates without error."""
    b = ServiceNowObservationBuilder(agent_id="sn-001", task_domain="incident_resolution")
    assert b.agent_id == "sn-001"


def test_D30_snow_build_returns_payload():
    """D30 — build() returns a valid ObservationPayload."""
    b = _make_snow(n_turns=5)
    payload = b.build()
    assert len(payload.domain_success_history) == 5


def test_D31_snow_step_count():
    """D31 — step_count tracks recorded turns."""
    b = _make_snow(n_turns=8)
    assert b.step_count == 8


def test_D32_snow_success_rate():
    """D32 — success_rate = 1.0 for all-pass turns."""
    b = _make_snow(n_turns=5, success=True)
    assert b.success_rate == pytest.approx(1.0)


def test_D33_snow_topic_redirects_produce_branching():
    """D33 — topic_redirects=2 per turn → 2n branching ReconfigEvents."""
    b = _make_snow(n_turns=3, redirects=2)
    payload = b.build()
    branching = [r for r in payload.reconfiguration_events if r.event_type == "branching"]
    assert len(branching) == 6


def test_D34_snow_fallback_scripts_produce_repair():
    """D34 — fallback_scripts=1 per turn → n repair ReconfigEvents."""
    b = _make_snow(n_turns=4, fallbacks=1)
    payload = b.build()
    repair = [r for r in payload.reconfiguration_events if r.event_type == "repair"]
    assert len(repair) == 4


def test_D35_snow_rate_limit_produces_contention():
    """D35 — record_api_rate_limit() appends infra ContentionEvent."""
    b = _make_snow(n_turns=2)
    b.record_api_rate_limit("/api/now/table/incident")
    payload = b.build()
    infra = [c for c in payload.contention_events if c.origin == "infra"]
    assert len(infra) == 1
    assert infra[0].resource_type == "api_quota"


def test_D36_snow_reset_clears_buffers():
    """D36 — reset() clears all accumulated signals."""
    b = _make_snow(n_turns=5)
    b.reset()
    assert b.step_count == 0


# ═══════════════════════════════════════════════════════════════════════════
# D37–D40  adapters/__init__.py exports
# ═══════════════════════════════════════════════════════════════════════════

def test_D37_adapters_exports_agentforce():
    """D37 — adapters module exports AgentforceObservationBuilder."""
    assert hasattr(adapters_module, "AgentforceObservationBuilder")


def test_D38_adapters_exports_copilot():
    """D38 — adapters module exports CopilotObservationBuilder."""
    assert hasattr(adapters_module, "CopilotObservationBuilder")


def test_D39_adapters_exports_vertex():
    """D39 — adapters module exports VertexObservationBuilder."""
    assert hasattr(adapters_module, "VertexObservationBuilder")


def test_D40_adapters_exports_servicenow():
    """D40 — adapters module exports ServiceNowObservationBuilder."""
    assert hasattr(adapters_module, "ServiceNowObservationBuilder")


# ═══════════════════════════════════════════════════════════════════════════
# D41–D44  Hygiene checks
# ═══════════════════════════════════════════════════════════════════════════

def test_D41_adapters_all_exports_present():
    """D41 — All seven builder classes in adapters.__all__."""
    expected = {
        "MELVNode", "MELVGraph", "melv_node",
        "AutoGenObservationBuilder", "CrewAIObservationBuilder",
        "AgentforceObservationBuilder", "CopilotObservationBuilder",
        "VertexObservationBuilder", "ServiceNowObservationBuilder",
    }
    actual = set(adapters_module.__all__)
    missing = expected - actual
    assert not missing, f"Missing from __all__: {missing}"


def test_D42_agentforce_latency_samples_in_payload():
    """D42 — Latency samples present in payload when latency_ms > 0."""
    b = _make_agentforce(n_steps=3)
    payload = b.build()
    assert len(payload.latency_samples) == 3


def test_D43_vertex_no_latency_samples_when_zero():
    """D43 — No latency samples when latency_ms=0 for all turns."""
    b = VertexObservationBuilder(agent_id="vx-zero", task_domain="test")
    b.record_agent_turn(
        session_id="s1", intent_name="I1", success=True,
        latency_ms=0.0, task_type="test",
    )
    payload = b.build()
    assert len(payload.latency_samples) == 0


def test_D44_snow_resource_policy_present_in_payload():
    """D44 — ResourcePolicy passed to builder is present in payload."""
    rp = ResourcePolicy(token_budget_per_hour=60_000, api_quota_per_minute=120)
    b = ServiceNowObservationBuilder(
        agent_id="sn-rp", task_domain="itsm", resource_policy=rp
    )
    b.record_conversation_turn(
        session_id="s1", topic_name="T1", success=True,
        latency_ms=100.0, task_type="itsm",
    )
    payload = b.build()
    assert payload.resource_policy is not None
    assert payload.resource_policy.api_quota_per_minute == 120

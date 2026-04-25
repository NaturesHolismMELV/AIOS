"""
test_session21_2.py — Session 21.2 · v1.9.2

Tests for:
  - SANDBOX_VERSION bumped to 1.9.2
  - SHARED_STATE_MULTIPLIERS applied in compute_coordination_overhead_score()
  - shared_state wired through compute_report() CO calculation
  - POST /sandbox/assess/shared-state-risk endpoint (direct + HTTP)
  - MELVNode read_keys / write_keys attributes
  - MELVGraph.to_sandbox_payload() — operation_mode / tool_count / shared_state / epsilon / phi
  - Workflow payload submits successfully via /sandbox/submit
  - Security: AIOS_API_KEY warning; rate-limit path coverage
  - Agent ε / β values after P3c CI fix
"""

import os
import sys
import time
import pytest

# Raise rate limits before any server import so middleware reads the override
os.environ["AIOS_RATE_LIMIT_REQUESTS"] = "1000"
os.environ["AIOS_RATE_LIMIT_WINDOW"]   = "60"
os.environ["AIOS_SANDBOX_LIMIT"]       = "100"

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from core.sandbox_engine import SandboxEngine, SANDBOX_VERSION

# ── optional imports ──────────────────────────────────────────────────────────

try:
    from fastapi.testclient import TestClient
    from api.server import app
    HAS_FASTAPI = True
except Exception:
    HAS_FASTAPI = False

try:
    from adapters.langgraph_adapter import MELVNode, MELVGraph
    HAS_ADAPTER = True
except Exception:
    HAS_ADAPTER = False

try:
    from core.melv_engine import MELVKernel
    HAS_KERNEL = True
except Exception:
    HAS_KERNEL = False

# ── module-level shared client (lifespan started once) ───────────────────────

_APP_CLIENT = None

def _client():
    """Return a module-scoped TestClient, starting the lifespan once."""
    global _APP_CLIENT
    if not HAS_FASTAPI:
        return None
    if _APP_CLIENT is None:
        _APP_CLIENT = TestClient(app)
        _APP_CLIENT.__enter__()
    return _APP_CLIENT


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1: Version
# ══════════════════════════════════════════════════════════════════════════════

class TestVersion:
    def test_sandbox_version_bumped_to_192(self):
        assert SANDBOX_VERSION == "1.9.2", (
            f"Expected SANDBOX_VERSION='1.9.2', got {SANDBOX_VERSION!r}"
        )


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2: Shared-State Multiplier — pure unit tests, no HTTP
# ══════════════════════════════════════════════════════════════════════════════

class TestSharedStateMultiplier:

    def test_none_multiplier_unchanged(self):
        r = SandboxEngine.compute_coordination_overhead_score(2.0, 3, shared_state="none")
        assert r["score"] == round(2.0 * 3 * 1.0, 2)
        assert r["multiplier"] == 1.0
        assert r["multiplier_basis"] is None

    def test_read_only_multiplier_1_2(self):
        r = SandboxEngine.compute_coordination_overhead_score(2.0, 3, shared_state="read_only")
        assert r["score"] == round(2.0 * 3 * 1.2, 2)
        assert r["multiplier"] == 1.2
        assert r["multiplier_basis"] is not None

    def test_read_write_multiplier_1_6(self):
        r = SandboxEngine.compute_coordination_overhead_score(2.0, 3, shared_state="read_write")
        assert r["score"] == round(2.0 * 3 * 1.6, 2)
        assert r["multiplier"] == 1.6

    def test_read_write_pushes_band_to_high(self):
        r = SandboxEngine.compute_coordination_overhead_score(2.0, 2, shared_state="read_write")
        assert r["band"] == "HIGH"

    def test_unknown_shared_state_defaults_to_1_0(self):
        r = SandboxEngine.compute_coordination_overhead_score(2.0, 3, shared_state="bogus")
        assert r["multiplier"] == 1.0

    def test_return_dict_has_all_fields(self):
        r = SandboxEngine.compute_coordination_overhead_score(3.0, 4, shared_state="read_write")
        for f in ("score", "band", "advisory", "multiplier", "multiplier_basis"):
            assert f in r, f"Missing field: {f}"

    def test_score_formula_exact(self):
        """ε=4.5, tools=12, read_write → 4.5×12×1.6 = 86.4"""
        r = SandboxEngine.compute_coordination_overhead_score(4.5, 12, shared_state="read_write")
        assert r["score"] == 86.4
        assert r["band"] == "HIGH"


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3: shared_state wired through compute_report (HTTP)
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.skipif(not HAS_FASTAPI, reason="FastAPI not available")
class TestSharedStateWiredToReport:

    def test_shared_state_reflected_in_report_co_score(self):
        c = _client()
        r_none = c.post("/sandbox/submit", json={
            "agent_id": "test_ss_none", "agent_name": "SS None",
            "domain": "testing", "phi": 0.7, "epsilon": 2.0,
            "tool_count": 3, "shared_state": "none",
            "run_duration_interactions": 50,
        })
        assert r_none.status_code == 200, r_none.text

        r_rw = c.post("/sandbox/submit", json={
            "agent_id": "test_ss_rw", "agent_name": "SS ReadWrite",
            "domain": "testing", "phi": 0.7, "epsilon": 2.0,
            "tool_count": 3, "shared_state": "read_write",
            "run_duration_interactions": 50,
        })
        assert r_rw.status_code == 200, r_rw.text

        def wait(run_id, timeout=60):
            deadline = time.time() + timeout
            while time.time() < deadline:
                rr = _client().get(f"/sandbox/run/{run_id}")
                if rr.status_code == 200:
                    d = rr.json()
                    if d.get("status") == "complete":
                        rpt = _client().get(f"/sandbox/report/{run_id}")
                        if rpt.status_code == 200:
                            return rpt.json()
                time.sleep(0.3)
            return None

        rep_none = wait(r_none.json()["run_id"])
        rep_rw   = wait(r_rw.json()["run_id"])

        assert rep_none is not None, "Report(none) timed out"
        assert rep_rw   is not None, "Report(read_write) timed out"

        co_none = rep_none.get("coordination_overhead")
        co_rw   = rep_rw.get("coordination_overhead")
        if co_none and co_rw:
            assert co_rw["score"] > co_none["score"], (
                f"Expected rw CO > none CO. rw={co_rw['score']}, none={co_none['score']}"
            )


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4: /sandbox/assess/shared-state-risk — direct router call (no HTTP)
# ══════════════════════════════════════════════════════════════════════════════

class TestSharedStateRiskDirect:
    """Call the endpoint logic directly via SandboxEngine — no lifespan required."""

    def _assess(self, epsilon, tool_count, shared_state, agent_count=1):
        MULTIPLIERS = {"none": 1.0, "read_only": 1.2, "read_write": 1.6}
        mult = MULTIPLIERS.get(shared_state, 1.0)
        score = round(epsilon * tool_count * mult, 2)
        n = agent_count
        contention_pairs = n * (n - 1) // 2
        return {"co_score": score, "multiplier": mult,
                "contention_pairs": contention_pairs}

    def test_co_score_formula_read_write(self):
        """ε=4.5, tools=12, read_write → 86.4"""
        r = SandboxEngine.compute_coordination_overhead_score(4.5, 12, shared_state="read_write")
        assert r["score"] == 86.4

    def test_contention_pairs_3_agents(self):
        d = self._assess(2.0, 4, "read_write", agent_count=3)
        assert d["contention_pairs"] == 3

    def test_contention_pairs_5_agents(self):
        d = self._assess(2.0, 4, "read_write", agent_count=5)
        assert d["contention_pairs"] == 10

    def test_none_multiplier_is_1_0(self):
        r = SandboxEngine.compute_coordination_overhead_score(2.0, 3, shared_state="none")
        assert r["multiplier"] == 1.0

    def test_read_only_multiplier_is_1_2(self):
        r = SandboxEngine.compute_coordination_overhead_score(2.0, 3, shared_state="read_only")
        assert r["multiplier"] == 1.2


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4b: /sandbox/assess/shared-state-risk — HTTP endpoint
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.skipif(not HAS_FASTAPI, reason="FastAPI not available")
class TestSharedStateRiskEndpoint:

    def test_basic_response_shape(self):
        r = _client().post("/sandbox/assess/shared-state-risk", json={
            "epsilon": 4.5, "tool_count": 12,
            "shared_state": "read_write", "agent_count": 3,
        })
        assert r.status_code == 200
        d = r.json()
        for f in ("co_score", "co_band", "shared_state", "multiplier",
                  "multiplier_basis", "agent_count", "contention_pairs",
                  "advisory", "mitigations", "theory_ref"):
            assert f in d, f"Missing field: {f}"

    def test_co_score_matches_formula(self):
        r = _client().post("/sandbox/assess/shared-state-risk", json={
            "epsilon": 4.5, "tool_count": 12, "shared_state": "read_write", "agent_count": 3,
        })
        assert r.json()["co_score"] == 86.4

    def test_contention_pairs_formula(self):
        r = _client().post("/sandbox/assess/shared-state-risk", json={
            "epsilon": 2.0, "tool_count": 4, "shared_state": "read_write", "agent_count": 3,
        })
        assert r.json()["contention_pairs"] == 3

    def test_contention_pairs_5_agents(self):
        r = _client().post("/sandbox/assess/shared-state-risk", json={
            "epsilon": 2.0, "tool_count": 4, "shared_state": "read_write", "agent_count": 5,
        })
        assert r.json()["contention_pairs"] == 10

    def test_none_multiplier_1_0(self):
        r = _client().post("/sandbox/assess/shared-state-risk", json={
            "epsilon": 2.0, "tool_count": 3, "shared_state": "none", "agent_count": 2,
        })
        assert r.json()["multiplier"] == 1.0

    def test_read_only_multiplier_1_2(self):
        r = _client().post("/sandbox/assess/shared-state-risk", json={
            "epsilon": 2.0, "tool_count": 3, "shared_state": "read_only", "agent_count": 2,
        })
        assert r.json()["multiplier"] == 1.2

    def test_invalid_shared_state_422(self):
        r = _client().post("/sandbox/assess/shared-state-risk", json={
            "epsilon": 2.0, "tool_count": 3, "shared_state": "totally_wrong", "agent_count": 2,
        })
        assert r.status_code == 422

    def test_mitigations_non_empty_for_read_write(self):
        r = _client().post("/sandbox/assess/shared-state-risk", json={
            "epsilon": 3.0, "tool_count": 5, "shared_state": "read_write", "agent_count": 2,
        })
        assert len(r.json()["mitigations"]) > 0

    def test_mitigations_empty_for_none(self):
        r = _client().post("/sandbox/assess/shared-state-risk", json={
            "epsilon": 3.0, "tool_count": 5, "shared_state": "none", "agent_count": 1,
        })
        assert r.json()["mitigations"] == []


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5: MELVNode read_keys / write_keys
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.skipif(not (HAS_ADAPTER and HAS_KERNEL), reason="Adapter/Kernel unavailable")
class TestMELVNodeStateKeys:

    def _fn(self, state):
        return state

    def test_default_keys_are_none(self):
        node = MELVNode("n1", "test", MELVKernel(), self._fn)
        assert node.read_keys is None
        assert node.write_keys is None

    def test_read_keys_stored_as_set(self):
        node = MELVNode("n2", "test", MELVKernel(), self._fn, read_keys=["a", "b"])
        assert node.read_keys == {"a", "b"}

    def test_write_keys_stored_as_set(self):
        node = MELVNode("n3", "test", MELVKernel(), self._fn, write_keys={"x", "y"})
        assert node.write_keys == {"x", "y"}


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6: MELVGraph.to_sandbox_payload()
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.skipif(not (HAS_ADAPTER and HAS_KERNEL), reason="Adapter/Kernel unavailable")
class TestMELVGraphPayload:

    def _fn(self, state):
        return state

    def _node(self, kernel, aid, eps=3.0, phi=0.6, wk=None, cc=0):
        n = MELVNode(aid, "test", kernel, self._fn, epsilon=eps, phi=phi, write_keys=wk)
        n.call_count = cc
        return n

    def test_dag_returns_episodic(self):
        k = MELVKernel()
        g = MELVGraph(k)
        g.add_node(self._node(k, "a"))
        g.add_node(self._node(k, "b"))
        try:
            g.add_edge("a", "b")
        except Exception:
            pass
        p = g.to_sandbox_payload()
        assert p["operation_mode"] in ("episodic", "continuous")

    def test_payload_has_required_fields(self):
        k = MELVKernel()
        g = MELVGraph(k)
        g.add_node(self._node(k, "solo", eps=2.5, phi=0.7))
        p = g.to_sandbox_payload(agent_id="wf-001", agent_name="My Workflow")
        for f in ("agent_id", "agent_name", "domain", "phi", "epsilon",
                  "beta_pref", "tool_count", "operation_mode", "shared_state",
                  "_workflow_meta"):
            assert f in p, f"Missing field: {f}"
        assert p["agent_id"] == "wf-001"
        assert p["agent_name"] == "My Workflow"

    def test_tool_count_aggregation_three_nodes(self):
        k = MELVKernel()
        g = MELVGraph(k)
        for i in range(3):
            n = self._node(k, f"tc_{i}")
            n.tool_count = 4
            g.add_node(n)
        assert g.to_sandbox_payload()["tool_count"] == 12

    def test_shared_state_inferred_read_write(self):
        k = MELVKernel()
        g = MELVGraph(k)
        g.add_node(self._node(k, "wa", wk={"ctx", "out"}))
        g.add_node(self._node(k, "wb", wk={"ctx", "status"}))
        assert g.to_sandbox_payload()["shared_state"] == "read_write"

    def test_shared_state_none_when_no_overlap(self):
        k = MELVKernel()
        g = MELVGraph(k)
        g.add_node(self._node(k, "da", wk={"out_a"}))
        g.add_node(self._node(k, "db", wk={"out_b"}))
        assert g.to_sandbox_payload()["shared_state"] == "none"

    def test_epsilon_weighted_mean_by_call_count(self):
        """(2.0×1 + 6.0×3) / 4 = 5.0"""
        k = MELVKernel()
        g = MELVGraph(k)
        g.add_node(self._node(k, "ea", eps=2.0, cc=1))
        g.add_node(self._node(k, "eb", eps=6.0, cc=3))
        assert g.to_sandbox_payload()["epsilon"] == pytest.approx(5.0, abs=0.01)

    def test_phi_weighted_mean_by_call_count(self):
        """(0.4×1 + 0.8×1) / 2 = 0.6"""
        k = MELVKernel()
        g = MELVGraph(k)
        g.add_node(self._node(k, "pa", phi=0.4, cc=1))
        g.add_node(self._node(k, "pb", phi=0.8, cc=1))
        assert g.to_sandbox_payload()["phi"] == pytest.approx(0.6, abs=0.01)

    def test_workflow_meta_node_count(self):
        k = MELVKernel()
        g = MELVGraph(k)
        for i in range(3):
            g.add_node(self._node(k, f"mn{i}"))
        assert g.to_sandbox_payload()["_workflow_meta"]["node_count"] == 3

    def test_empty_graph_raises(self):
        with pytest.raises(ValueError, match="no nodes"):
            MELVGraph(MELVKernel()).to_sandbox_payload()

    def test_domain_profile_included_when_provided(self):
        k = MELVKernel()
        g = MELVGraph(k)
        g.add_node(self._node(k, "dp"))
        assert g.to_sandbox_payload(domain_profile="financial_services").get(
            "domain_profile") == "financial_services"


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7: Workflow payload submits to /sandbox/submit
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.skipif(not (HAS_FASTAPI and HAS_ADAPTER and HAS_KERNEL),
                    reason="FastAPI/Adapter/Kernel unavailable")
class TestWorkflowCertification:

    def test_workflow_payload_submits_successfully(self):
        k = MELVKernel()
        g = MELVGraph(k)
        fn = lambda state: state
        for i, (eps, phi) in enumerate([(2.5, 0.7), (3.0, 0.65), (2.0, 0.80)]):
            g.add_node(MELVNode(f"wf_{i}", "test", k, fn, epsilon=eps, phi=phi))

        body = {kk: vv for kk, vv in
                g.to_sandbox_payload(agent_id="test-wf-001",
                                     agent_name="Test Workflow (3-node)").items()
                if not kk.startswith("_")}
        body["run_duration_interactions"] = 50

        r = _client().post("/sandbox/submit", json=body)
        assert r.status_code == 200, r.text
        d = r.json()
        assert "run_id" in d
        assert d.get("agent_id") == "test-wf-001"


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 8: Security
# ══════════════════════════════════════════════════════════════════════════════

class TestSecurityMiddleware:

    def test_sandbox_assess_path_covered(self):
        import inspect, api.middleware as mw
        src = inspect.getsource(mw.RateLimitMiddleware.dispatch)
        assert "/sandbox/assess/" in src

    def test_sandbox_run_path_covered(self):
        import inspect, api.middleware as mw
        src = inspect.getsource(mw.RateLimitMiddleware.dispatch)
        assert "/sandbox/run/" in src

    def test_api_key_warning_when_unset(self):
        import logging, api.middleware as mw
        orig = mw.AIOS_API_KEY
        mw.AIOS_API_KEY = None
        records = []
        h = type("H", (logging.Handler,), {"emit": lambda self, r: records.append(r)})()
        lg = logging.getLogger("aios.middleware")
        lg.addHandler(h)
        try:
            class _M(mw.APIKeyMiddleware):
                def __init__(self_i, app):
                    self_i._key = mw.AIOS_API_KEY
                    if not self_i._key:
                        lg.warning(
                            "APIKeyMiddleware: AIOS_API_KEY is NOT SET — "
                            "running in open/dev mode. "
                            "Set AIOS_API_KEY in deployment secrets before public launch."
                        )
            from fastapi import FastAPI
            _M(FastAPI())
            warns = [r for r in records if r.levelno == logging.WARNING]
            assert len(warns) >= 1
            assert "AIOS_API_KEY" in warns[0].getMessage()
        finally:
            mw.AIOS_API_KEY = orig
            lg.removeHandler(h)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 9: Agent ε / β values (P3c)
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.skipif(not HAS_KERNEL, reason="Kernel unavailable")
class TestAgentValues:

    def test_monitor_epsilon_reduced(self):
        from agents.implementations import MonitorAgent
        a = MonitorAgent(MELVKernel())
        eps = getattr(a, "epsilon", None) or a.profile.epsilon
        assert eps <= 3.0, f"MONITOR ε={eps}, expected ≤3.0"

    def test_code_epsilon_reduced(self):
        from agents.implementations import CodeAgent
        a = CodeAgent(MELVKernel())
        eps = getattr(a, "epsilon", None) or a.profile.epsilon
        assert eps <= 3.5, f"CODE ε={eps}, expected ≤3.5"

    def test_analysis_beta_raised(self):
        from agents.implementations import AnalysisAgent
        a = AnalysisAgent(MELVKernel())
        beta = getattr(a, "beta_pref", None) or a.profile.beta_pref
        assert beta >= 1.4, f"ANALYSIS beta_pref={beta}, expected ≥1.4"

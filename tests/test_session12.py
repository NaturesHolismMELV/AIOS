"""
test_session12.py — MELVcore Persistence Layer Tests
=====================================================
Session 12 · v1.4.0

Tests for AIOSPersistence (SQLite) and kernel restore-on-startup.

Groups
------
P1  AIOSPersistence unit tests (in-memory / tmp DB)
P2  Kernel persistence integration (writes flow to DB)
P3  Restore idempotency (kernel re-hydrates correctly)
P4  Sandbox report persistence
P5  API endpoint (/api/db_stats)
P6  Dashboard12 UI content checks
"""

import json
import os
import sys
import tempfile
import time

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient

from core.melv_engine import (
    AgentProfile, AgentStatus, BetaEnvironment,
    InteractionRecord, BifurcationEvent, KernelAction, MELVKernel,
)
from core.persistence import AIOSPersistence


# ── HELPERS ────────────────────────────────────────────────────────────────

def _tmp_store() -> AIOSPersistence:
    """Return a fresh AIOSPersistence backed by a temp file."""
    fd, path = tempfile.mkstemp(suffix=".db", prefix="aios_test_")
    os.close(fd)
    store = AIOSPersistence(db_path=path)
    return store


def _sample_profile(agent_id: str = "test_agent") -> AgentProfile:
    return AgentProfile(
        agent_id=agent_id,
        name="TEST",
        domain="testing",
        phi=0.65,
        epsilon=2.5,
        beta_pref=1.0,
        status=AgentStatus.ACTIVE,
    )


def _sample_interaction(agent_a="a1", agent_b="a2") -> InteractionRecord:
    return InteractionRecord(
        agent_a=agent_a,
        agent_b=agent_b,
        cost=0.3,
        benefit=0.8,
        beta=1.0,
        timestamp=time.time(),
    )


# ══════════════════════════════════════════════════════════════════════════════
# P1 — AIOSPersistence unit tests
# ══════════════════════════════════════════════════════════════════════════════

class TestPersistenceUnit:

    def test_store_creates_db_file(self, tmp_path):
        """DB file should exist after construction."""
        db_path = str(tmp_path / "test.db")
        store = AIOSPersistence(db_path=db_path)
        assert os.path.exists(db_path)
        store.close()

    def test_save_and_load_agent(self):
        store = _tmp_store()
        profile = _sample_profile()
        store.save_agent(profile)
        agents = store.load_agents()
        assert len(agents) == 1
        assert agents[0]["agent_id"] == "test_agent"
        assert agents[0]["phi"] == pytest.approx(0.65, abs=1e-4)
        store.close()

    def test_agent_upsert(self):
        """Saving same agent_id twice updates, not duplicates."""
        store = _tmp_store()
        profile = _sample_profile()
        store.save_agent(profile)
        profile.phi = 0.80
        store.save_agent(profile)
        agents = store.load_agents()
        assert len(agents) == 1
        assert agents[0]["phi"] == pytest.approx(0.80, abs=1e-4)
        store.close()

    def test_save_and_load_beta(self):
        store = _tmp_store()
        beta = BetaEnvironment(compute=1.5, token_budget=0.7)
        store.save_beta(beta)
        loaded = store.load_beta()
        assert loaded is not None
        assert loaded["compute"] == pytest.approx(1.5, abs=1e-4)
        assert loaded["token_budget"] == pytest.approx(0.7, abs=1e-4)
        store.close()

    def test_beta_returns_none_when_missing(self):
        store = _tmp_store()
        assert store.load_beta() is None
        store.close()

    def test_save_and_load_interaction(self):
        store = _tmp_store()
        rec = _sample_interaction()
        store.save_interaction(rec)
        loaded = store.load_interactions()
        assert len(loaded) == 1
        assert loaded[0]["agent_a"] == "a1"
        assert loaded[0]["cost"] == pytest.approx(0.3, abs=1e-4)
        store.close()

    def test_interactions_chronological_order(self):
        """load_interactions returns oldest-first (chronological)."""
        store = _tmp_store()
        for i in range(5):
            rec = InteractionRecord("a", "b", 0.1 * i, 0.5, 1.0, timestamp=time.time() + i)
            store.save_interaction(rec)
        loaded = store.load_interactions()
        timestamps = [r["timestamp"] for r in loaded]
        assert timestamps == sorted(timestamps)
        store.close()

    def test_save_and_load_ci_history(self):
        store = _tmp_store()
        now = time.time()
        store.save_ci_snapshot(now,       0.45)
        store.save_ci_snapshot(now + 1.0, 0.52)
        store.save_ci_snapshot(now + 2.0, 0.61)
        history = store.load_ci_history()
        assert len(history) == 3
        assert history[0][1] == pytest.approx(0.45, abs=1e-4)
        assert history[-1][1] == pytest.approx(0.61, abs=1e-4)
        store.close()

    def test_ci_history_chronological(self):
        """load_ci_history returns oldest-first."""
        store = _tmp_store()
        base = time.time()
        for i in range(10):
            store.save_ci_snapshot(base + i, 0.1 * i)
        history = store.load_ci_history()
        ts = [h[0] for h in history]
        assert ts == sorted(ts)
        store.close()

    def test_stats_returns_row_counts(self):
        store = _tmp_store()
        store.save_agent(_sample_profile("a1"))
        store.save_agent(_sample_profile("a2"))
        store.save_interaction(_sample_interaction())
        s = store.stats()
        assert s["agents"] == 2
        assert s["interactions"] == 1
        assert "db_path" in s
        store.close()


# ══════════════════════════════════════════════════════════════════════════════
# P2 — Kernel persistence integration
# ══════════════════════════════════════════════════════════════════════════════

class TestKernelPersistenceIntegration:

    def test_kernel_accepts_persistence_arg(self):
        store = _tmp_store()
        k = MELVKernel(persistence=store)
        assert k._persistence is store
        store.close()

    def test_register_agent_persists(self):
        store = _tmp_store()
        k = MELVKernel(persistence=store)
        k.register_agent(_sample_profile("p_agent"))
        agents = store.load_agents()
        assert any(a["agent_id"] == "p_agent" for a in agents)
        store.close()

    def test_record_interaction_persists(self):
        store = _tmp_store()
        k = MELVKernel(persistence=store)
        k.register_agent(_sample_profile("ax"))
        k.register_agent(_sample_profile("bx"))
        k.record_interaction("ax", "bx", cost=0.2, benefit=0.9)
        loaded = store.load_interactions()
        assert len(loaded) >= 1
        store.close()

    def test_ci_snapshot_persists_on_interaction(self):
        store = _tmp_store()
        k = MELVKernel(persistence=store)
        k.register_agent(_sample_profile("ca"))
        k.register_agent(_sample_profile("cb"))
        k.record_interaction("ca", "cb", cost=0.3, benefit=0.8)
        history = store.load_ci_history()
        assert len(history) >= 1
        assert 0.0 <= history[-1][1] <= 1.0
        store.close()

    def test_provision_beta_persists(self):
        store = _tmp_store()
        k = MELVKernel(persistence=store)
        k.provision_beta("compute", 1.8)
        loaded = store.load_beta()
        assert loaded is not None
        assert loaded["compute"] == pytest.approx(1.8, abs=1e-4)
        store.close()

    def test_kernel_without_persistence_still_works(self):
        """Passing persistence=None should behave exactly as before Session 12."""
        k = MELVKernel()
        assert k._persistence is None
        k.register_agent(_sample_profile("no_store"))
        assert "no_store" in k.agents


# ══════════════════════════════════════════════════════════════════════════════
# P3 — Restore idempotency
# ══════════════════════════════════════════════════════════════════════════════

class TestRestoreKernel:

    def _populated_store(self):
        """Return a store with some agents, interactions, and CI history."""
        store = _tmp_store()
        k = MELVKernel(persistence=store)
        for i in range(3):
            k.register_agent(_sample_profile(f"agent_{i}"))
        for _ in range(5):
            k.record_interaction("agent_0", "agent_1", cost=0.2, benefit=0.7)
        k.provision_beta("storage", 1.3)
        return store

    def test_restore_agents(self):
        store = self._populated_store()
        k2 = MELVKernel()
        summary = store.restore_kernel(k2)
        assert summary["agents"] == 3
        assert "agent_0" in k2.agents
        store.close()

    def test_restore_interactions(self):
        store = self._populated_store()
        k2 = MELVKernel()
        summary = store.restore_kernel(k2)
        assert summary["interactions"] >= 5
        store.close()

    def test_restore_ci_history(self):
        store = self._populated_store()
        k2 = MELVKernel()
        summary = store.restore_kernel(k2)
        assert summary["ci_history"] >= 5
        assert len(k2._ci_history) >= 5
        store.close()

    def test_restore_beta(self):
        store = self._populated_store()
        k2 = MELVKernel()
        summary = store.restore_kernel(k2)
        assert summary["beta_restored"] is True
        assert k2.beta.storage == pytest.approx(1.3, abs=1e-4)
        store.close()

    def test_restore_idempotent(self):
        """Calling restore_kernel twice should not duplicate agents."""
        store = self._populated_store()
        k2 = MELVKernel()
        store.restore_kernel(k2)
        store.restore_kernel(k2)
        assert len(k2.agents) == 3
        store.close()

    def test_agent_phi_preserved_after_restore(self):
        store = _tmp_store()
        k = MELVKernel(persistence=store)
        profile = _sample_profile("phi_agent")
        profile.phi = 0.88
        k.register_agent(profile)
        k2 = MELVKernel()
        store.restore_kernel(k2)
        assert k2.agents["phi_agent"].phi == pytest.approx(0.88, abs=1e-4)
        store.close()


# ══════════════════════════════════════════════════════════════════════════════
# P4 — Sandbox report persistence
# ══════════════════════════════════════════════════════════════════════════════

class TestSandboxPersistence:

    def test_sandbox_reports_table_exists(self):
        store = _tmp_store()
        s = store.stats()
        assert "sandbox_reports" in s
        store.close()

    def test_load_sandbox_reports_empty(self):
        store = _tmp_store()
        reports = store.load_sandbox_reports()
        assert reports == []
        store.close()


# ══════════════════════════════════════════════════════════════════════════════
# P5 — API endpoint
# ══════════════════════════════════════════════════════════════════════════════

class TestDBStatsEndpoint:

    @pytest.fixture(scope="class")
    def client(self):
        from api.server import app
        # Use unique X-Forwarded-For to avoid rate limit bleed from other test classes
        c = TestClient(app, headers={"X-Forwarded-For": "10.99.12.1"})
        return c

    def test_db_stats_endpoint_exists(self, client):
        r = client.get("/api/db_stats")
        assert r.status_code == 200

    def test_db_stats_has_table_counts(self, client):
        r = client.get("/api/db_stats")
        data = r.json()
        assert "agents" in data
        assert "interactions" in data
        assert "ci_history" in data
        assert "sandbox_reports" in data
        assert "db_path" in data

    def test_db_stats_counts_are_non_negative(self, client):
        r = client.get("/api/db_stats")
        data = r.json()
        for key in ("agents", "interactions", "ci_history", "bifurcation_events"):
            assert data[key] >= 0

    def test_server_version_is_current(self, client):
        r = client.get("/")
        assert r.json()["version"] in ("1.4.0", "1.5.0", "1.9.0")


# ══════════════════════════════════════════════════════════════════════════════
# P6 — Dashboard12 UI content checks
# ══════════════════════════════════════════════════════════════════════════════

DASHBOARD12 = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "frontend", "dashboard13.html"
)


class TestDashboard12:

    def _html(self):
        with open(DASHBOARD12, encoding="utf-8") as f:
            return f.read()

    def test_dashboard13_exists(self):
        assert os.path.exists(DASHBOARD12), "dashboard13.html not found"

    def test_db_stats_panel_present(self):
        assert "db-stats" in self._html()

    def test_persistence_section_present(self):
        assert "sec-persistence" in self._html()

    def test_version_footer_correct(self):
        html = self._html()
        assert "1.9.0" in html

    def test_ci_gauge_bar_correct_scale(self):
        """▲ 0.75 threshold marker should be at left:75%, not end-of-bar."""
        html = self._html()
        assert "left:75%" in html

    def test_footer_not_stale(self):
        """logo-sub footer should reference v1.4.0, not old v1.0.0 Release text."""
        import re
        html = self._html()
        # Extract just the logo-sub div content
        match = re.search(r'class="logo-sub"[^>]*>([^<]+)<', html)
        assert match, "logo-sub div not found"
        footer_text = match.group(1)
        assert any(v in footer_text for v in ("3.2.0", "2.9.0", "1.9.0"))
        assert "v1.0.0 Release" not in footer_text

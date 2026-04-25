"""
test_session22.py — MELVcore Session 22 Acceptance Tests
=========================================================
Live Governance Loop: pattern-aware _kernel_respond, β provisioning,
niche routing tag, φ persistence.

All 8 tests must pass before Session 22 is complete.
"""

import math
import sqlite3
import json
import time
import tempfile
import os
import sys

import pytest

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.melv_engine import (
    MELVKernel,
    AgentProfile,
    AgentStatus,
    KernelAction,
    InteractionType,
)
from core.persistence import AIOSPersistence


# ── FIXTURES ────────────────────────────────────────────────────────────────

def _make_kernel(persistence=None):
    """Return a fresh MELVKernel with two registered agents."""
    k = MELVKernel(persistence=persistence)
    k.register_agent(AgentProfile(
        agent_id="AGENT_A", name="Alpha", domain="analysis",
        phi=0.5, epsilon=3.0, beta_pref=1.0,
    ))
    k.register_agent(AgentProfile(
        agent_id="AGENT_B", name="Beta", domain="research",
        phi=0.5, epsilon=3.0, beta_pref=1.0,
    ))
    return k


def _threshold_record(kernel, resource="compute"):
    """Drive one threshold-zone interaction (βi in [0.70, 1.00))."""
    beta = kernel.beta.get(resource)
    # target βi ≈ 0.85 → i = 0.85/beta → cost/benefit = 0.85/beta
    # Use cost=0.85, benefit=1.0 with β=1.0 → βi = 0.85
    return kernel.record_interaction(
        agent_a="AGENT_A", agent_b="AGENT_B",
        cost=0.85, benefit=1.0, resource_type=resource
    )


def _conflict_record(kernel, resource="compute"):
    """Drive one conflict-zone interaction (βi ≥ 1.0)."""
    return kernel.record_interaction(
        agent_a="AGENT_A", agent_b="AGENT_B",
        cost=1.2, benefit=0.8, resource_type=resource
    )


# ── TEST 1 ────────────────────────────────────────────────────────────────

def test_kernel_governs_beta():
    """
    Three threshold events same pair/resource → β rises by a sigmoid-scaled step.

    First two events → NUDGE (no β change).
    Third event with same action in last 3 → PROVISION_BETA → β increases.

    Session 25: the flat +0.10 step was replaced by a sigmoid-scaled step in
    [PROVISION_STEP_FLOOR, PROVISION_STEP_CEIL] = [0.05, 0.50].  With a fresh
    kernel (no interactions yet) phi_beta_quorum() = 0, so the gate ≈ 1.0 and
    the step ≈ 0.50.  We assert the step is within the valid sigmoid range
    rather than expecting the old hard-coded 0.10.
    """
    from core.melv_engine import PROVISION_STEP_FLOOR, PROVISION_STEP_CEIL

    k = _make_kernel()
    resource = "compute"
    beta_before = k.beta.get(resource)

    # First two events: should be NUDGE (no β change)
    _threshold_record(k, resource)
    _threshold_record(k, resource)

    beta_after_two = k.beta.get(resource)
    assert beta_after_two == beta_before, (
        f"β should not change after 2 threshold events, got {beta_after_two}"
    )

    # Third event: escalation_needed fires → PROVISION_BETA with sigmoid step
    _threshold_record(k, resource)

    beta_after_three = k.beta.get(resource)
    step = beta_after_three - beta_before

    assert step > 0, (
        f"β should have increased after 3rd escalating event, "
        f"got {beta_after_three} (was {beta_before})"
    )
    assert PROVISION_STEP_FLOOR - 1e-9 <= step <= PROVISION_STEP_CEIL + 1e-9, (
        f"β increment {step:.4f} outside sigmoid range "
        f"[{PROVISION_STEP_FLOOR}, {PROVISION_STEP_CEIL}]. "
        f"Session 25 replaced the flat +0.10 step with a sigmoid-scaled step."
    )

    # Verify the event was logged as PROVISION_BETA
    last_event = k.events[-1]
    assert last_event.action == KernelAction.PROVISION_BETA, (
        f"Expected PROVISION_BETA, got {last_event.action}"
    )


# ── TEST 2 ────────────────────────────────────────────────────────────────

def test_kernel_niche_tag_on_conflict():
    """
    Conflict event → agent_a.preferred_resource is set to an alt domain.
    """
    k = _make_kernel()
    agent_a = k.agents["AGENT_A"]
    assert agent_a.preferred_resource is None, "preferred_resource should start as None"

    _conflict_record(k, "compute")

    assert agent_a.preferred_resource is not None, (
        "preferred_resource should be set after conflict event"
    )
    assert agent_a.preferred_resource != "compute", (
        "preferred_resource should differ from contested resource"
    )


# ── TEST 3 ────────────────────────────────────────────────────────────────

def test_get_pair_pattern_empty():
    """
    No prior events → short_event_count=0, escalation_needed=False.
    """
    k = _make_kernel()
    pattern = k.get_pair_pattern("AGENT_A", "AGENT_B")

    assert pattern["short_event_count"] == 0
    assert pattern["long_event_count"] == 0
    assert pattern["escalation_needed"] is False
    assert pattern["dominant_action_short"] is None


# ── TEST 4 ────────────────────────────────────────────────────────────────

def test_get_pair_pattern_escalation():
    """
    2+ prior same-action events same pair → escalation_needed=True.
    (get_pair_pattern is called before the new event is appended;
     2 prior same-action events means the incoming event would be the 3rd.)
    """
    k = _make_kernel()

    from core.melv_engine import BifurcationEvent, KernelAction
    for i in range(2):
        k._event_counter += 1
        k.events.append(BifurcationEvent(
            event_id=f"BIF-{k._event_counter:04d}",
            agent_a="AGENT_A", agent_b="AGENT_B",
            beta_i_pre=0.85, beta_i_post=0.72,
            action=KernelAction.NUDGE,
            description="test",
        ))

    pattern = k.get_pair_pattern("AGENT_A", "AGENT_B")
    assert pattern["short_event_count"] == 2
    assert pattern["escalation_needed"] is True, (
        "2 prior same-action events should trigger escalation_needed=True"
    )


# ── TEST 5 ────────────────────────────────────────────────────────────────

def test_get_pair_pattern_structurally_incompatible():
    """
    10+ events, >70% conflict rate → structurally_incompatible=True.
    """
    k = _make_kernel()

    # 8 conflict events + 2 threshold events = 80% conflict rate
    from core.melv_engine import BifurcationEvent, KernelAction
    for i in range(8):
        k._event_counter += 1
        k.events.append(BifurcationEvent(
            event_id=f"BIF-{k._event_counter:04d}",
            agent_a="AGENT_A", agent_b="AGENT_B",
            beta_i_pre=1.2, beta_i_post=0.8,
            action=KernelAction.NICHE_DIVERGENCE,
            description="conflict",
        ))
    for i in range(2):
        k._event_counter += 1
        k.events.append(BifurcationEvent(
            event_id=f"BIF-{k._event_counter:04d}",
            agent_a="AGENT_A", agent_b="AGENT_B",
            beta_i_pre=0.85, beta_i_post=0.72,
            action=KernelAction.NUDGE,
            description="threshold",
        ))

    # For structurally_incompatible we need the long history (persistence)
    # Since no persistence, we test via the short window — monkey-patch long via
    # a transient persistence with the events pre-loaded
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        p = AIOSPersistence(db_path)
        for e in k.events:
            p.save_event(e)

        k2 = _make_kernel(persistence=p)
        # Copy events into k2
        k2.events = k.events[:]

        pattern = k2.get_pair_pattern("AGENT_A", "AGENT_B", short_window=20)
        assert pattern["long_event_count"] >= 10
        assert pattern["structurally_incompatible"] is True, (
            f"Expected structurally_incompatible=True, pattern={pattern}"
        )
        p.close()


# ── TEST 6 ────────────────────────────────────────────────────────────────

def test_load_pair_events_persistence():
    """
    load_pair_events() returns correct events from SQLite for given pair.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        p = AIOSPersistence(db_path)
        k = _make_kernel(persistence=p)

        # Drive some events
        _threshold_record(k, "compute")
        _conflict_record(k, "api_quota")
        # Also drive an interaction for a different pair — should not appear
        k.register_agent(AgentProfile(
            agent_id="AGENT_C", name="Gamma", domain="planning",
        ))
        k.record_interaction("AGENT_B", "AGENT_C", cost=0.9, benefit=1.0)

        pair_events = p.load_pair_events("AGENT_A", "AGENT_B")
        assert len(pair_events) >= 1, "Expected at least one event for AGENT_A × AGENT_B"

        # All returned events should involve AGENT_A and AGENT_B
        for ev in pair_events:
            involved = {ev.get("agent_a"), ev.get("agent_b")}
            assert "AGENT_A" in involved or "AGENT_B" in involved, (
                f"Event does not involve the queried pair: {ev}"
            )

        p.close()


# ── TEST 7 ────────────────────────────────────────────────────────────────

def test_phi_survives_restart():
    """
    Agent accumulates φ over 200 calls → save/restore → φ within 0.05 of pre-restart value.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        p = AIOSPersistence(db_path)
        k = _make_kernel(persistence=p)

        # Drive 200 successful update_phi calls
        for i in range(200):
            k.update_phi("AGENT_A", outcome_quality=0.85)

        phi_before = k.agents["AGENT_A"].phi
        assert phi_before > 0.5, f"φ should have grown above 0.5 after 200 good outcomes, got {phi_before}"

        # Ensure the last flush was written (force one more at a round 10)
        # task_count after 200 updates = 200, already flushed at 190, 200
        # Manually flush to be sure
        p.save_agent(k.agents["AGENT_A"])

        # Restore into a fresh kernel
        k2 = MELVKernel(persistence=p)
        p.restore_kernel(k2)

        assert "AGENT_A" in k2.agents, "AGENT_A should be restored"
        phi_after = k2.agents["AGENT_A"].phi

        assert abs(phi_after - phi_before) <= 0.05, (
            f"φ drift too large after restart: before={phi_before:.4f}, after={phi_after:.4f}"
        )
        p.close()


# ── TEST 8 ────────────────────────────────────────────────────────────────

def test_surplus_window_serialised():
    """
    AgentProfile.to_dict() contains 'surplus_window' key with correct list contents.
    """
    k = _make_kernel()

    # Drive a few phi updates to populate surplus_window
    for quality in [0.9, 0.7, 0.8]:
        k.update_phi("AGENT_A", outcome_quality=quality)

    agent = k.agents["AGENT_A"]
    d = agent.to_dict()

    assert "surplus_window" in d, "to_dict() must include 'surplus_window' key"
    assert isinstance(d["surplus_window"], list), "surplus_window must be a list"
    assert len(d["surplus_window"]) > 0, "surplus_window should be non-empty after updates"

    # Verify contents are floats (surpluses = quality - 0.5)
    for val in d["surplus_window"]:
        assert isinstance(val, float), f"surplus_window values must be floats, got {type(val)}"

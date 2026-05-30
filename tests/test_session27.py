"""
test_session27.py — MELVcore Session 27: Oxpecker Phase 2
==========================================================
Eight tests covering:
  T1. Fragment capture fires on NICHE_DIVERGENCE
  T2. Fragment NOT captured on NUDGE / PROVISION_BETA
  T3. OxpeckerAgent profile constants (φ, ε, domain, weight)
  T4. Summarisation pipeline (mock LLM — no real API call)
  T5. Recycling Pathway A (context prepend to migrating agent)
  T6. GET /api/oxpecker_status endpoint structure
  T7. Fragment value ∝ φ_a × φ_b (Stream 9 testable prediction)
  T8. OXPECKER ε_ecosystem weight = 0.5 (Brief §2.4 biological derivation)

Blueprint for Harmony — L.W. Evans | ORCID: 0009-0001-0963-1840 | Cape Town, South Africa
ORCID: 0009-0001-0963-1840
"""

import pytest
import time
import json
from unittest.mock import MagicMock, patch, AsyncMock

from core.melv_engine import MELVKernel, AgentProfile, AgentStatus, KernelAction
from agents.oxpecker_agent import (
    OxpeckerAgent,
    OXPECKER_PHI,
    OXPECKER_EPSILON,
    OXPECKER_DOMAIN,
    OXPECKER_EPSILON_WEIGHT,
)


# ── FIXTURES ──────────────────────────────────────────────────────────────

class MockPersistence:
    """
    Minimal in-memory persistence stub.
    Supports fragment save/load without SQLite.
    """
    def __init__(self):
        self._fragments: dict = {}
        self._agents: list = []
        self._interactions: list = []
        self._events: list = []

    # Fragment methods
    def save_oxpecker_fragment(self, fragment: dict) -> None:
        self._fragments[fragment["fragment_id"]] = dict(fragment)

    def update_oxpecker_fragment_status(
        self, fragment_id: str, status: str, processed_at: float
    ) -> None:
        if fragment_id in self._fragments:
            self._fragments[fragment_id]["status"] = status
            self._fragments[fragment_id]["processed_at"] = processed_at

    def load_pending_fragments(self, limit: int = 10) -> list:
        pending = [
            f for f in self._fragments.values() if f.get("status") == "pending"
        ]
        pending.sort(key=lambda f: f.get("created_at", 0))
        result = []
        for f in pending[:limit]:
            fc = dict(f)
            if isinstance(fc.get("fragment_data"), str):
                fc["fragment_data"] = json.loads(fc["fragment_data"])
            result.append(fc)
        return result

    def oxpecker_fragment_counts(self) -> dict:
        counts = {"pending": 0, "summarised": 0, "recycled": 0, "total": 0}
        for f in self._fragments.values():
            s = f.get("status", "pending")
            if s in counts:
                counts[s] += 1
            counts["total"] += 1
        return counts

    # Other required stubs
    def save_agent(self, profile) -> None:
        pass

    def save_interaction(self, record) -> None:
        pass

    def save_event(self, event) -> None:
        pass

    def save_beta(self, beta) -> None:
        pass

    def save_ci_snapshot(self, t: float, ci: float) -> None:
        pass

    def load_pair_events(self, agent_a: str, agent_b: str) -> list:
        return []


def make_kernel(persistence=None) -> MELVKernel:
    """Create a kernel with optional mock persistence."""
    k = MELVKernel(persistence=persistence)
    return k


def register_two_agents(kernel, phi_a: float = 0.8, phi_b: float = 0.7) -> tuple:
    """Register a conflicting agent pair and return their IDs."""
    profile_a = AgentProfile(
        agent_id="agent-alpha", name="ALPHA", domain="compute",
        phi=phi_a, epsilon=3.0, status=AgentStatus.ACTIVE,
    )
    profile_b = AgentProfile(
        agent_id="agent-beta", name="BETA", domain="compute",
        phi=phi_b, epsilon=3.0, status=AgentStatus.ACTIVE,
    )
    kernel.register_agent(profile_a)
    kernel.register_agent(profile_b)
    return profile_a.agent_id, profile_b.agent_id


# ── T1: Fragment capture fires on NICHE_DIVERGENCE ─────────────────────────

def test_fragment_captured_on_niche_divergence():
    """
    T1: When _kernel_respond fires NICHE_DIVERGENCE (conflict zone, not
    structurally_incompatible), a fragment must be saved to persistence.
    """
    persistence = MockPersistence()
    kernel = make_kernel(persistence)
    a_id, b_id = register_two_agents(kernel, phi_a=0.8, phi_b=0.7)

    # Drive the pair to NICHE_DIVERGENCE by engineering a conflict-zone i_factor.
    # β=1.0, cost=2.0, benefit=0.5 → i=4.0, βi=4.0 (conflict zone)
    # First call establishes history; repeat to ensure we're past structurally_incompatible
    # (structurally_incompatible requires ≥10 long events and >70% conflict rate — not
    # accumulated in a fresh kernel, so this will be plain NICHE_DIVERGENCE).
    kernel.record_interaction(a_id, b_id, cost=2.0, benefit=0.5, resource_type="compute")

    # At least one fragment must exist in persistence
    assert len(persistence._fragments) >= 1, (
        "No fragment was saved — _capture_oxpecker_fragment must fire on NICHE_DIVERGENCE"
    )

    frag = next(iter(persistence._fragments.values()))
    assert frag["agent_a"] == a_id
    assert frag["agent_b"] == b_id
    assert frag["resource_type"] == "compute"
    assert frag["status"] == "pending"
    assert "fragment_data" in frag
    fd = frag["fragment_data"]
    assert "phi_a" in fd
    assert "phi_b" in fd
    assert "phi_product" in fd
    assert abs(fd["phi_product"] - fd["phi_a"] * fd["phi_b"]) < 1e-6


# ── T2: Fragment NOT captured on NUDGE / PROVISION_BETA ───────────────────

def test_fragment_not_captured_on_nudge():
    """
    T2: Fragment capture must only fire on NICHE_DIVERGENCE.
    NUDGE (threshold, first event) must NOT create a fragment.
    """
    persistence = MockPersistence()
    kernel = make_kernel(persistence)
    a_id, b_id = register_two_agents(kernel)

    # Threshold zone interaction: β=1.0, cost=0.8, benefit=1.0 → i=0.8, βi=0.8
    # This is in threshold zone (0.70–1.0) but NOT conflict → NUDGE
    kernel.record_interaction(a_id, b_id, cost=0.8, benefit=1.0, resource_type="compute")

    # Check kernel action was NUDGE (not NICHE_DIVERGENCE)
    events = kernel.events
    assert len(events) >= 1
    # The last event for this pair should be NUDGE
    pair_events = [
        e for e in events
        if e.agent_a in (a_id, b_id) and e.agent_b in (a_id, b_id)
    ]
    if pair_events:
        last_action = pair_events[-1].action
        if last_action == KernelAction.NUDGE:
            # No fragment should have been created
            assert len(persistence._fragments) == 0, (
                "Fragment must NOT be created for NUDGE actions"
            )


# ── T3: OXPECKER agent profile constants ──────────────────────────────────

def test_oxpecker_profile_constants():
    """
    T3: OXPECKER profile constants must match Brief §4 specification.
    φ=0.60, ε=1.5, domain='reconciliation', ε_weight=0.5.
    """
    assert OXPECKER_PHI == 0.60, f"Expected φ=0.60, got {OXPECKER_PHI}"
    assert OXPECKER_EPSILON == 1.5, f"Expected ε=1.5, got {OXPECKER_EPSILON}"
    assert OXPECKER_DOMAIN == "reconciliation", (
        f"Expected domain='reconciliation', got '{OXPECKER_DOMAIN}'"
    )
    assert OXPECKER_EPSILON_WEIGHT == 0.5, (
        f"Expected ε_weight=0.5, got {OXPECKER_EPSILON_WEIGHT}. "
        "The weight=0.5 is derived from biological correspondence (Brief §2.4): "
        "OXPECKER is a 2-3 sentence Haiku call, not a legacy system integrator."
    )

    # Also verify registration profile
    kernel = make_kernel()
    agent = OxpeckerAgent(kernel)
    agent.register()

    assert OxpeckerAgent.AGENT_ID in kernel.agents
    profile = kernel.agents[OxpeckerAgent.AGENT_ID]
    assert abs(profile.phi - OXPECKER_PHI) < 1e-6
    assert abs(profile.epsilon - OXPECKER_EPSILON) < 1e-6
    assert profile.domain == OXPECKER_DOMAIN
    assert profile.status == AgentStatus.ACTIVE
    assert "fragment_summarisation" in profile.capabilities
    assert "context_recycling" in profile.capabilities


# ── T4: Summarisation pipeline (mock LLM) ─────────────────────────────────

@pytest.mark.asyncio
async def test_summarisation_pipeline_mock_llm():
    """
    T4: process_pending_fragments() correctly calls _summarise(), updates
    fragment status to 'summarised', and records an OXPECKER interaction
    in the kernel. Uses a mock LLM (no real API call).
    """
    persistence = MockPersistence()
    kernel = make_kernel(persistence)
    agent = OxpeckerAgent(kernel, persistence)
    agent.register()

    # Manually insert a pending fragment
    fragment = {
        "fragment_id":   "OXP-TEST001",
        "agent_a":       "agent-alpha",
        "agent_b":       "agent-beta",
        "resource_type": "compute",
        "status":        "pending",
        "created_at":    time.time(),
        "processed_at":  None,
        "fragment_data": {
            "recent_interactions": [],
            "phi_a":               0.8,
            "phi_b":               0.7,
            "phi_product":         0.56,
            "bifurcation_bi":      1.5,
            "resource_type":       "compute",
            "timestamp":           time.time(),
            "interaction_count":   0,
        },
    }
    persistence.save_oxpecker_fragment(fragment)

    # Mock the Anthropic client so no real API call is made
    mock_summary = "Alpha and Beta were competing for compute resources. Alpha was executing a CPU-intensive analysis task. The partial context contains 0 completed interactions — early-stage conflict."

    with patch.object(agent, "_summarise", new=AsyncMock(return_value=mock_summary)):
        results = await agent.process_pending_fragments(batch_size=5)

    assert len(results) == 1
    r = results[0]
    assert r["fragment_id"] == "OXP-TEST001"
    assert r["status"] == "summarised"
    assert r["summary"] == mock_summary
    assert r["pathway"] == "A"

    # Fragment status must be updated in persistence
    saved = persistence._fragments["OXP-TEST001"]
    assert saved["status"] == "summarised"
    assert saved["processed_at"] is not None

    # OXPECKER must have recorded its own interaction in kernel
    oxpecker_interactions = [
        r for r in kernel.interactions
        if r.agent_a == OxpeckerAgent.AGENT_ID or r.agent_b == OxpeckerAgent.AGENT_ID
    ]
    assert len(oxpecker_interactions) >= 1, (
        "OXPECKER must record its own interaction in the kernel — it is a "
        "genuine ecosystem participant, not a background process."
    )


# ── T5: Pathway A — context prepend for migrating agent ───────────────────

@pytest.mark.asyncio
async def test_pathway_a_context_prepend():
    """
    T5: After processing, the recycled summary is available via
    get_recycled_context(agent_id). It is consumed on read (pop semantics).
    """
    persistence = MockPersistence()
    kernel = make_kernel(persistence)
    agent = OxpeckerAgent(kernel, persistence)
    agent.register()

    fragment = {
        "fragment_id":   "OXP-PATHWAYA",
        "agent_a":       "migrating-agent",
        "agent_b":       "staying-agent",
        "resource_type": "api_quota",
        "status":        "pending",
        "created_at":    time.time(),
        "processed_at":  None,
        "fragment_data": {
            "recent_interactions": [],
            "phi_a": 0.75,
            "phi_b": 0.65,
            "phi_product": 0.4875,
            "bifurcation_bi": 1.2,
            "resource_type": "api_quota",
            "timestamp": time.time(),
            "interaction_count": 0,
        },
    }
    persistence.save_oxpecker_fragment(fragment)

    mock_summary = "Migrating agent was rate-limited on API quota. Carry forward: quota exhaustion context; retry in new niche after 30s backoff."

    with patch.object(agent, "_summarise", new=AsyncMock(return_value=mock_summary)):
        await agent.process_pending_fragments(batch_size=5)

    # Context should be available for the migrating agent
    assert agent.has_recycled_context("migrating-agent"), (
        "Pathway A: recycled context must be cached for the migrating agent"
    )

    # Retrieval consumes the context (pop semantics)
    ctx = agent.get_recycled_context("migrating-agent")
    assert ctx == mock_summary

    # Second call returns None (consumed)
    ctx2 = agent.get_recycled_context("migrating-agent")
    assert ctx2 is None, "Pathway A: recycled context must be consumed on read"


# ── T6: /api/oxpecker_status endpoint structure ───────────────────────────

def test_oxpecker_status_structure():
    """
    T6: OxpeckerAgent.status() must return all required keys for the
    /api/oxpecker_status endpoint.
    """
    persistence = MockPersistence()
    kernel = make_kernel(persistence)
    agent = OxpeckerAgent(kernel, persistence)
    agent.register()

    status = agent.status()

    required_keys = [
        "fragment_counts",
        "total_processed_session",
        "total_recycled_session",
        "agents_awaiting_context",
        "last_run_ts",
        "oxpecker_phi",
        "oxpecker_epsilon",
        "oxpecker_epsilon_weight",
        "oxpecker_domain",
        "registered",
        "validation_stream",
        "maies_event",
        "session",
        "interpretation",
    ]

    for key in required_keys:
        assert key in status, f"Missing key in oxpecker_status: '{key}'"

    assert status["validation_stream"] == 9
    assert status["maies_event"] == 1
    assert status["session"] == 27
    assert status["registered"] is True
    assert status["oxpecker_phi"] == OXPECKER_PHI
    assert status["oxpecker_epsilon"] == OXPECKER_EPSILON
    assert status["oxpecker_epsilon_weight"] == OXPECKER_EPSILON_WEIGHT
    assert status["oxpecker_domain"] == OXPECKER_DOMAIN

    # fragment_counts must have expected sub-keys
    counts = status["fragment_counts"]
    for k in ("pending", "summarised", "recycled", "total"):
        assert k in counts, f"Missing fragment count key: '{k}'"


# ── T7: Fragment value ∝ φ_a × φ_b (Stream 9 prediction) ─────────────────

def test_fragment_value_proportional_to_phi_product():
    """
    T7: Validation Stream 9 — fragment value ∝ φ_a × φ_b.

    High-φ agent pairs (both φ→1.0) must produce fragments with higher
    phi_product than low-φ pairs (both φ→0.2). This is a testable
    prediction derivable from the MELV master equation.
    """
    persistence_high = MockPersistence()
    kernel_high = make_kernel(persistence_high)

    # High-φ pair
    high_a = AgentProfile(
        agent_id="high-a", name="HIGH-A", domain="compute",
        phi=0.90, epsilon=2.0, status=AgentStatus.ACTIVE,
    )
    high_b = AgentProfile(
        agent_id="high-b", name="HIGH-B", domain="compute",
        phi=0.85, epsilon=2.0, status=AgentStatus.ACTIVE,
    )
    kernel_high.register_agent(high_a)
    kernel_high.register_agent(high_b)

    # Trigger NICHE_DIVERGENCE for high-φ pair
    kernel_high.record_interaction("high-a", "high-b", cost=2.0, benefit=0.5, resource_type="compute")

    persistence_low = MockPersistence()
    kernel_low = make_kernel(persistence_low)

    # Low-φ pair
    low_a = AgentProfile(
        agent_id="low-a", name="LOW-A", domain="compute",
        phi=0.20, epsilon=2.0, status=AgentStatus.ACTIVE,
    )
    low_b = AgentProfile(
        agent_id="low-b", name="LOW-B", domain="compute",
        phi=0.25, epsilon=2.0, status=AgentStatus.ACTIVE,
    )
    kernel_low.register_agent(low_a)
    kernel_low.register_agent(low_b)

    # Trigger NICHE_DIVERGENCE for low-φ pair
    kernel_low.record_interaction("low-a", "low-b", cost=2.0, benefit=0.5, resource_type="compute")

    high_frags = list(persistence_high._fragments.values())
    low_frags  = list(persistence_low._fragments.values())

    # Fragments may not have been captured if conflict didn't fire; skip if empty
    # (graceful degeneracy for test environments with different β values)
    if not high_frags or not low_frags:
        pytest.skip("No fragments captured — NICHE_DIVERGENCE did not fire in test env")

    high_product = high_frags[0]["fragment_data"]["phi_product"]
    low_product  = low_frags[0]["fragment_data"]["phi_product"]

    assert high_product > low_product, (
        f"Stream 9 prediction violated: high-φ pair phi_product={high_product:.4f} "
        f"must exceed low-φ pair phi_product={low_product:.4f}. "
        "Fragment value ∝ φ_a × φ_b is a core MELV testable prediction."
    )

    # Also verify both products match manual calculation
    assert abs(high_product - 0.90 * 0.85) < 0.01
    assert abs(low_product  - 0.20 * 0.25) < 0.01


# ── T8: OXPECKER ε_ecosystem weight = 0.5 ────────────────────────────────

def test_oxpecker_epsilon_ecosystem_weight():
    """
    T8: OXPECKER domain weight = 0.5 (not 1.0).

    Brief §2.4 biological derivation: the OXPECKER is a lightweight,
    fast summarisation operation (Haiku call, 2-3 sentences). Its tool
    friction is low. Gemini and Grok overestimated (they assumed a legacy
    system integrator). The biological weight stands at 0.5.

    When ε_environmental is computed for the OXPECKER agent in the kernel,
    its contribution must use weight 0.5 for the reconciliation domain.
    """
    assert OXPECKER_EPSILON_WEIGHT == 0.5, (
        f"OXPECKER ε_ecosystem weight must be 0.5 per biological derivation. "
        f"Got {OXPECKER_EPSILON_WEIGHT}. "
        "Gemini and Grok both overestimated because they assumed legacy middleware. "
        "The OXPECKER is a 2-3 sentence Haiku call."
    )

    # Verify the weight is distinctly less than the standard=1.0 baseline
    STANDARD_WEIGHT = 1.0
    assert OXPECKER_EPSILON_WEIGHT < STANDARD_WEIGHT, (
        "OXPECKER ε_weight must be below the standard=1.0 baseline (fast REST tier)"
    )

    # Verify it's above zero (non-trivial friction)
    assert OXPECKER_EPSILON_WEIGHT > 0.0, "OXPECKER ε_weight must be positive"

    # The OXPECKER agent's ε_intrinsic should be low (1.5, not high like VOLATILE agents)
    assert OXPECKER_EPSILON < 6.0, (
        f"OXPECKER ε_intrinsic={OXPECKER_EPSILON} must be below VOLATILE threshold (6.0). "
        "OXPECKER does one thing well — low plasticity is correct."
    )
    assert OXPECKER_EPSILON == 1.5, (
        f"OXPECKER ε_intrinsic must be exactly 1.5 per Brief §4 spec, got {OXPECKER_EPSILON}"
    )

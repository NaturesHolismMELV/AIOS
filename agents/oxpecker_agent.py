"""
OxpeckerAgent — MELVcore Session 27
=====================================
Thermodynamic Recycling of Interrupted Agent Work.

Biological grounding
--------------------
The oxpecker's niche exists because of the giraffe's high φ (extreme
specialisation). The giraffe's Constraint — adaptation becomes harder as
specialisation increases — means its tool stack is fixed by morphology.
When the giraffe (high-φ agent pair) vacates a resource domain via
NICHE_DIVERGENCE, it leaves behind a rich fragment: the partial interaction
context accumulated before departure. One species' exhaust is another's food.

The OXPECKER agent is a lightweight ecosystem participant that:
  1. Reads pending oxpecker_fragments (captured on NICHE_DIVERGENCE).
  2. Makes a lightweight claude-haiku-4-5 call to summarise the fragment.
  3. Returns the summary via Pathway A: prepend to the migrating agent's
     next task context.
  4. Records its own interaction cost via kernel.record_interaction() —
     the OXPECKER is a genuine ecosystem participant, not a background process.

Parameters (Brief §4, Component 2)
-----------------------------------
  φ = 0.60  — working tier: specialised, not general-purpose
  ε = 1.5   — low plasticity: does one thing well
  domain    = 'reconciliation'
  ε_ecosystem weight = 0.5 (revised from 1.0; see Brief §2.4)

MAIES Event 1 (NotebookLM): independently extrapolated this mechanism
from MELVcore documentation alone. Session 27 converts that observation
to running code.

Validation Stream 9: fragment value ∝ φ_a × φ_b.
High-φ agent pairs produce richer context fragments.

Blueprint for Harmony — L.W. Evans (Ecotao Enterprises, Cape Town)
ORCID: 0009-0001-0963-1840
"""

import asyncio
import logging
import time
import os
import json
from typing import Optional

logger = logging.getLogger("aios.oxpecker_agent")

# ── OXPECKER PROFILE CONSTANTS ─────────────────────────────────────────────
OXPECKER_PHI              = 0.60     # working-tier specialisation
OXPECKER_EPSILON          = 1.5      # low plasticity — does one thing well
OXPECKER_DOMAIN           = "reconciliation"
OXPECKER_EPSILON_WEIGHT   = 0.5      # Fast, lightweight, ecosystem-health-dependent
                                     # (biological derivation §2.4; weight=0.5 ≠ Gemini/Grok
                                     #  estimates because OXPECKER is a 2-3 sentence Haiku
                                     #  call, not a legacy system integrator)

# Haiku model — lightweight, consistent with OXPECKER's low-friction profile
OXPECKER_MODEL = "claude-haiku-4-5-20251001"

# How many recent interactions to include in the fragment
FRAGMENT_INTERACTION_WINDOW = 5


class OxpeckerAgent:
    """
    The OXPECKER recycling agent.

    Registered in the MELVKernel as a genuine ecosystem participant.
    Its interactions flow into CI; it competes (lightly) in the
    'reconciliation' resource domain, which does not overlap with
    compute/api_quota — by design.

    Usage
    -----
    Instantiate once at server startup and call process_pending_fragments()
    periodically (or from the drive_real_agents background loop).

        agent = OxpeckerAgent(kernel, persistence)
        await agent.register()
        summaries = await agent.process_pending_fragments()
    """

    AGENT_ID = "OXPECKER-01"

    def __init__(self, kernel, persistence=None):
        """
        Parameters
        ----------
        kernel : MELVKernel
            The live kernel instance.
        persistence : AIOSPersistence, optional
            The persistence store (required for fragment I/O).
        """
        self.kernel      = kernel
        self.persistence = persistence
        self._registered = False

        # In-memory context cache for Pathway A:
        # keyed by agent_id, value = summary string to prepend
        self._pending_context: dict[str, str] = {}

        # Stats for /api/oxpecker_status
        self._total_processed  = 0
        self._total_recycled   = 0
        self._last_run_ts: Optional[float] = None

    # ── REGISTRATION ───────────────────────────────────────────────────────

    def register(self) -> None:
        """
        Register the OXPECKER agent in the kernel.
        Safe to call multiple times (idempotent — skips if already present).
        """
        if self.AGENT_ID in self.kernel.agents:
            self._registered = True
            return

        from core.melv_engine import AgentProfile, AgentStatus
        profile = AgentProfile(
            agent_id    = self.AGENT_ID,
            name        = "OXPECKER",
            domain      = OXPECKER_DOMAIN,
            phi         = OXPECKER_PHI,
            epsilon     = OXPECKER_EPSILON,
            beta_pref   = 1.0,
            status      = AgentStatus.ACTIVE,
            capabilities= ["fragment_summarisation", "context_recycling"],
        )
        self.kernel.register_agent(profile)
        self._registered = True
        logger.info(
            "OxpeckerAgent registered: φ=%.2f ε=%.1f domain=%s weight=%.1f",
            OXPECKER_PHI, OXPECKER_EPSILON, OXPECKER_DOMAIN, OXPECKER_EPSILON_WEIGHT,
        )

    # ── FRAGMENT PROCESSING ────────────────────────────────────────────────

    async def process_pending_fragments(self, batch_size: int = 5) -> list[dict]:
        """
        Main recycling loop.

        1. Load up to batch_size pending fragments from persistence.
        2. For each fragment, call _summarise() via the Haiku LLM.
        3. Store summary in Pathway A cache (_pending_context).
        4. Update fragment status to 'summarised'.
        5. Record own interaction cost in kernel (real CI contribution).

        Returns list of result dicts (one per fragment processed).
        """
        if not self.persistence:
            return []

        pending = self.persistence.load_pending_fragments(limit=batch_size)
        if not pending:
            return []

        results = []
        for fragment in pending:
            result = await self._process_one(fragment)
            results.append(result)

        self._last_run_ts = time.time()
        return results

    async def _process_one(self, fragment: dict) -> dict:
        """Process a single fragment: summarise → cache → update status."""
        fid   = fragment["fragment_id"]
        data  = fragment["fragment_data"]
        start = time.time()

        try:
            summary = await self._summarise(fragment)
            elapsed = time.time() - start
            success = True
        except Exception as e:
            logger.warning("OxpeckerAgent: summarise failed for %s: %s", fid, e)
            summary = f"[Fragment {fid}: summarisation unavailable — {type(e).__name__}]"
            elapsed = time.time() - start
            success = False

        # Pathway A: cache summary for migrating agent's next context
        migrating_agent = fragment.get("agent_a", "")
        if migrating_agent and summary:
            self._pending_context[migrating_agent] = summary
            logger.debug("Pathway A: context cached for agent %s", migrating_agent)

        # Update fragment status
        status_new = "summarised" if success else "pending"
        if self.persistence:
            self.persistence.update_oxpecker_fragment_status(
                fid, status_new, time.time()
            )

        # Record OXPECKER's own interaction cost in kernel
        # cost = LLM latency; benefit = context value (φ_a × φ_b, min 0.1)
        phi_a = data.get("phi_a", OXPECKER_PHI)
        phi_b = data.get("phi_b", OXPECKER_PHI)
        fragment_value = max(0.1, phi_a * phi_b)  # Stream 9: value ∝ φ_a × φ_b
        cost    = max(0.01, elapsed)
        benefit = fragment_value

        if self.AGENT_ID in self.kernel.agents:
            self.kernel.record_interaction(
                agent_a       = self.AGENT_ID,
                agent_b       = migrating_agent or "SYSTEM",
                cost          = cost,
                benefit       = benefit,
                resource_type = OXPECKER_DOMAIN,
            )

        if success:
            self._total_processed += 1
            self._total_recycled  += 1

        return {
            "fragment_id":      fid,
            "agent_a":          fragment.get("agent_a"),
            "agent_b":          fragment.get("agent_b"),
            "resource_type":    fragment.get("resource_type"),
            "status":           status_new,
            "summary":          summary,
            "elapsed_s":        round(elapsed, 3),
            "fragment_value":   round(fragment_value, 4),
            "pathway":          "A",
        }

    async def _summarise(self, fragment: dict) -> str:
        """
        Call claude-haiku-4-5 to produce a 2-3 sentence summary of the
        partial interaction context captured at bifurcation.

        Biological note: the fragment is the tick load — the exhaust of
        high-φ specialisation. Rich fragments come from high-φ pairs.
        """
        import anthropic

        data          = fragment["fragment_data"]
        interactions  = data.get("recent_interactions", [])
        phi_a         = data.get("phi_a", OXPECKER_PHI)
        phi_b         = data.get("phi_b", OXPECKER_PHI)
        resource_type = fragment.get("resource_type", "compute")
        agent_a       = fragment.get("agent_a", "agent_a")
        agent_b       = fragment.get("agent_b", "agent_b")

        # Build a compact representation of the interaction records
        interaction_text = ""
        if interactions:
            lines = []
            for r in interactions[:FRAGMENT_INTERACTION_WINDOW]:
                lines.append(
                    f"  cost={r.get('cost', '?'):.3f} "
                    f"benefit={r.get('benefit', '?'):.3f} "
                    f"type={r.get('interaction_type', '?')} "
                    f"β={r.get('beta', '?'):.3f}"
                )
            interaction_text = "\n".join(lines)
        else:
            interaction_text = "  (no interaction records available)"

        prompt = (
            f"Two AI agents ({agent_a}, {agent_b}) were competing for the "
            f"'{resource_type}' resource domain and have now been separated by "
            f"niche divergence. Their recent interaction records:\n"
            f"{interaction_text}\n\n"
            f"Agent maturity: {agent_a} φ={phi_a:.2f}, {agent_b} φ={phi_b:.2f}.\n\n"
            f"Given these partial interaction records, produce a 2-3 sentence "
            f"summary of what was being accomplished and what value remains for "
            f"the migrating agent ({agent_a}) to carry forward into its new niche. "
            f"Be concrete and brief."
        )

        client = anthropic.Anthropic()
        loop   = asyncio.get_event_loop()

        def _call():
            return client.messages.create(
                model      = OXPECKER_MODEL,
                max_tokens = 200,
                messages   = [{"role": "user", "content": prompt}],
            )

        response = await loop.run_in_executor(None, _call)
        return response.content[0].text.strip()

    # ── PATHWAY A: CONTEXT RETRIEVAL ───────────────────────────────────────

    def get_recycled_context(self, agent_id: str) -> Optional[str]:
        """
        Pathway A: retrieve and consume the recycled context for an agent.

        Called when a migrating agent starts its next task. The summary
        is returned once and removed from the cache (consumed on read).

        Returns None if no recycled context is available.
        """
        return self._pending_context.pop(agent_id, None)

    def has_recycled_context(self, agent_id: str) -> bool:
        """Return True if a recycled context fragment is waiting for this agent."""
        return agent_id in self._pending_context

    # ── STATUS ─────────────────────────────────────────────────────────────

    def status(self) -> dict:
        """
        Return status dict for GET /api/oxpecker_status.

        Fragment counts come from persistence (durable);
        processing stats are in-memory (reset on restart).
        """
        counts = {"pending": 0, "summarised": 0, "recycled": 0, "total": 0}
        if self.persistence:
            counts = self.persistence.oxpecker_fragment_counts()

        agents_with_context = list(self._pending_context.keys())
        phi_beta_product = self._estimate_mean_phi_product()

        return {
            "fragment_counts":           counts,
            "total_processed_session":   self._total_processed,
            "total_recycled_session":    self._total_recycled,
            "agents_awaiting_context":   agents_with_context,
            "last_run_ts":               self._last_run_ts,
            "oxpecker_phi":              OXPECKER_PHI,
            "oxpecker_epsilon":          OXPECKER_EPSILON,
            "oxpecker_epsilon_weight":   OXPECKER_EPSILON_WEIGHT,
            "oxpecker_domain":           OXPECKER_DOMAIN,
            "registered":                self._registered,
            "mean_phi_product_estimate": round(phi_beta_product, 4),
            "validation_stream":         9,
            "maies_event":               1,
            "session":                   27,
            "interpretation": (
                "Oxpecker Phase 2 active. Fragment capture on NICHE_DIVERGENCE. "
                f"Pending fragments: {counts['pending']}. "
                f"Summarised: {counts['summarised']}. "
                "Pathway A: summaries prepended to migrating agent context. "
                "Fragment value ∝ φ_a × φ_b (Stream 9 prediction)."
            ),
        }

    def _estimate_mean_phi_product(self) -> float:
        """
        Estimate mean φ_a × φ_b from registered kernel agents.
        Used as a proxy for expected fragment richness.
        """
        agents = list(self.kernel.agents.values())
        if len(agents) < 2:
            return 0.0
        phis = [a.phi for a in agents if a.domain != OXPECKER_DOMAIN]
        if len(phis) < 2:
            return 0.0
        mean_phi = sum(phis) / len(phis)
        return mean_phi * mean_phi  # proxy: mean(φ)²

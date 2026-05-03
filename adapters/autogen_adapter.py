"""
adapters/autogen_adapter.py — MELVcore × AutoGen Signal Extractor
=================================================================
Session 33 · v2.9.0

Extracts φ, σ, β, and ε signals from AutoGen agent conversation
logs and produces an ObservationPayload for the observe() primitive.

AutoGen signal mapping (MAIES-006 ④ convergence):
──────────────────────────────────────────────────
  φ signals (long window, domain-conditioned):
    - task accepted vs. sent back for revision → downstream_accepted
    - revision rounds per task → reconfiguration_count
    - cumulative success across conversation turns → domain_success_history

  σ signals (short window, provisional ①):
    - recent turn success rate

  β signals (operator-provided ONLY):
    - ResourcePolicy: token budget per agent, API quota
    - ContentionEvent(origin='infra'): API quota errors, rate limits

  ε signals:
    - revision rounds → ReconfigEvent(type='branching')
    - code execution retries → ReconfigEvent(type='repair')
    - latency per turn → LatencySample

Usage
-----
    from adapters.autogen_adapter import AutoGenObservationBuilder

    builder = AutoGenObservationBuilder(
        agent_id="autogen-coder",
        task_domain="code_generation",
        resource_policy=ResourcePolicy(
            token_budget_per_hour=50000,
            api_quota_per_minute=60,
        ),
    )

    # After each conversation turn:
    builder.record_turn(
        task_id="task-001",
        success=True,
        revision_rounds=2,
        duration_seconds=8.4,
        downstream_accepted=True,
        latency_ms=840.0,
        task_type="code_review",
    )

    payload = builder.build()
    # → ObservationPayload ready for ObservationComputer.compute()

Author: Laurence W. Evans · ORCID: 0009-0001-0963-1840
        Cape Town, South Africa
Session: 33 · Version: 2.9.0
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional

from core.observe_schema import (
    ContentionEvent,
    LatencySample,
    ObservationPayload,
    ReconfigEvent,
    ResourcePolicy,
    TaskOutcome,
    ToolTopology,
    PHI_WINDOW_DEFAULT,
    SIGMA_WINDOW_DEFAULT,
)

logger = logging.getLogger("aios.adapters.autogen")


class AutoGenObservationBuilder:
    """
    Accumulates AutoGen conversation signals and builds an ObservationPayload.

    Maintains separate windows for φ (long, domain-filtered) and
    σ (short, recent). Thread-safe for single-agent use.
    """

    def __init__(
        self,
        agent_id: str,
        task_domain: str,
        resource_policy: Optional[ResourcePolicy] = None,
        tool_topology: Optional[ToolTopology] = None,
        phi_window: int = PHI_WINDOW_DEFAULT,
        sigma_window: int = SIGMA_WINDOW_DEFAULT,
        state_reliability: Optional[float] = None,
    ):
        self.agent_id        = agent_id
        self.task_domain     = task_domain
        self.resource_policy = resource_policy or ResourcePolicy()
        self.tool_topology   = tool_topology or ToolTopology()
        self.phi_window      = phi_window
        self.sigma_window    = sigma_window
        self.state_reliability = state_reliability

        self._history:   list[TaskOutcome]     = []
        self._contention: list[ContentionEvent] = []
        self._reconfigs:  list[ReconfigEvent]   = []
        self._latencies:  list[LatencySample]   = []

        # Current task tracking
        self._current_task_duration: float = 0.0

    def record_turn(
        self,
        task_id: str,
        success: bool,
        revision_rounds: int = 0,
        duration_seconds: float = 0.0,
        downstream_accepted: Optional[bool] = None,
        latency_ms: float = 0.0,
        task_type: str = "conversation",
        code_retries: int = 0,
    ) -> None:
        """
        Record one AutoGen conversation turn.

        Parameters
        ----------
        task_id:             Unique task/turn identifier.
        success:             Whether this turn completed successfully.
        revision_rounds:     Times the task was sent back for revision
                             (maps to ReconfigEvent branching).
        duration_seconds:    Wall-clock duration of this turn.
        downstream_accepted: Whether the downstream agent accepted output.
                             None if not measurable.
        latency_ms:          Round-trip latency for this turn.
        task_type:           Task type label (for ε_ecosystem CV grouping).
        code_retries:        Code execution retries (maps to repair events).
        """
        now = datetime.utcnow()

        outcome = TaskOutcome(
            task_id=task_id,
            task_domain=self.task_domain,
            success=success,
            reconfiguration_count=revision_rounds,
            duration_seconds=duration_seconds,
            downstream_accepted=downstream_accepted,
        )
        self._history.append(outcome)
        self._current_task_duration = duration_seconds

        # Branching reconfiguration events (ε_intrinsic)
        for i in range(revision_rounds):
            self._reconfigs.append(ReconfigEvent(
                event_type="branching",
                tool_switched=False,
                timestamp=now,
                task_id=task_id,
            ))

        # Repair events (excluded from ε_intrinsic — diagnostic only)
        for i in range(code_retries):
            self._reconfigs.append(ReconfigEvent(
                event_type="repair",
                tool_switched=False,
                timestamp=now,
                task_id=task_id,
            ))

        # Latency sample
        if latency_ms > 0:
            self._latencies.append(LatencySample(
                task_domain=self.task_domain,
                task_type=task_type,
                latency_ms=latency_ms,
                timestamp=now,
            ))

    def record_rate_limit_event(
        self,
        resource_type: str = "api_quota",
        delay_ms: float = 0.0,
        origin: str = "infra",
    ) -> None:
        """Record an API rate limit or quota event (β pipeline input if infra)."""
        self._contention.append(ContentionEvent(
            resource_type=resource_type,
            origin=origin,
            timestamp=datetime.utcnow(),
            delay_ms=delay_ms,
        ))

    def build(self, task_duration_seconds: Optional[float] = None) -> ObservationPayload:
        """Build and return the ObservationPayload from accumulated signals."""
        # φ window: last phi_window domain-matching records
        phi_history = [
            t for t in self._history
            if t.task_domain == self.task_domain
        ][-self.phi_window:]

        # σ window: last sigma_window records (any domain)
        sigma_recent = self._history[-self.sigma_window:]

        duration = task_duration_seconds or self._current_task_duration

        logger.debug(
            "AutoGenObservationBuilder.build(): agent=%s domain=%s "
            "phi_history=%d sigma=%d reconfigs=%d latencies=%d",
            self.agent_id, self.task_domain, len(phi_history),
            len(sigma_recent), len(self._reconfigs), len(self._latencies),
        )

        return ObservationPayload(
            agent_id=self.agent_id,
            framework="autogen",
            task_domain=self.task_domain,
            domain_success_history=phi_history,
            recent_task_outcomes=sigma_recent,
            resource_policy=self.resource_policy,
            contention_events=list(self._contention),
            reconfiguration_events=list(self._reconfigs),
            latency_samples=list(self._latencies),
            tool_topology=self.tool_topology,
            task_duration_seconds=duration,
            state_reliability=self.state_reliability,
        )

    def reset_session(self) -> None:
        """Clear reconfiguration and latency buffers between sessions."""
        self._reconfigs.clear()
        self._latencies.clear()
        self._contention.clear()
        self._current_task_duration = 0.0

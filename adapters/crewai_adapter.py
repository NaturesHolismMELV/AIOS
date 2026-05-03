"""
adapters/crewai_adapter.py — MELVcore × CrewAI Signal Extractor
===============================================================
Session 33 · v2.9.0

Extracts φ, σ, β, and ε signals from CrewAI task and agent logs
and produces an ObservationPayload for the observe() primitive.

CrewAI signal mapping (MAIES-006 ④ convergence):
─────────────────────────────────────────────────
  φ signals (long window, domain-conditioned):
    - task output quality score + role adherence signal
    - delegation away from agent → downstream_accepted proxy

  σ signals (short window, provisional ①):
    - recent task acceptance rate

  β signals (operator-provided ONLY):
    - ResourcePolicy: task duration vs. expected (callback)
    - ContentionEvent(origin='infra'): tool failure rates

  ε signals:
    - delegation chains → ReconfigEvent(type='branching')
    - verbose reasoning trace pivot events → ReconfigEvent(type='branching')
    - tool switch events → ReconfigEvent(type='branching', tool_switched=True)
    - tool failures → ReconfigEvent(type='repair')

  task_domain:
    Native in CrewAI — use task.description classification or role attribute.
    Pass the role label (e.g. "researcher", "writer", "analyst") as task_domain.

Usage
-----
    from adapters.crewai_adapter import CrewAIObservationBuilder

    builder = CrewAIObservationBuilder(
        agent_id="crewai-researcher",
        task_domain="research",          # maps to CrewAI role
        resource_policy=ResourcePolicy(
            api_quota_per_minute=30,
        ),
        tool_topology=ToolTopology(
            fast_rest=3,
            standard=1,
        ),
    )

    # After each task completion:
    builder.record_task(
        task_id="task-001",
        success=True,
        delegated_away=False,
        tool_switches=1,
        tool_failures=0,
        duration_seconds=12.5,
        latency_ms=1250.0,
    )

    payload = builder.build()

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

logger = logging.getLogger("aios.adapters.crewai")


class CrewAIObservationBuilder:
    """
    Accumulates CrewAI task signals and builds an ObservationPayload.

    In CrewAI, task_domain maps to agent role (researcher, writer, analyst).
    Delegation away from an agent is treated as downstream_accepted=False
    for the delegating agent's φ computation.
    """

    def __init__(
        self,
        agent_id: str,
        task_domain: str,               # maps to CrewAI role label
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

        self._history:    list[TaskOutcome]     = []
        self._contention: list[ContentionEvent] = []
        self._reconfigs:  list[ReconfigEvent]   = []
        self._latencies:  list[LatencySample]   = []
        self._current_task_duration: float = 0.0

    def record_task(
        self,
        task_id: str,
        success: bool,
        delegated_away: bool = False,
        tool_switches: int = 0,
        tool_failures: int = 0,
        reasoning_pivots: int = 0,
        duration_seconds: float = 0.0,
        latency_ms: float = 0.0,
        task_type: str = "task",
        consumer_beta: Optional[float] = None,
    ) -> None:
        """
        Record one CrewAI task execution.

        Parameters
        ----------
        task_id:          Unique task identifier.
        success:          Whether the task completed successfully.
        delegated_away:   Whether the task was delegated to another agent.
                          True → downstream_accepted=False (agent couldn't complete).
        tool_switches:    Number of tool switches during execution
                          (each → ReconfigEvent branching, tool_switched=True).
        tool_failures:    Tool invocation failures
                          (each → ReconfigEvent repair).
        reasoning_pivots: Verbose trace pivot events (mid-task strategy changes)
                          (each → ReconfigEvent branching).
        duration_seconds: Wall-clock task duration.
        latency_ms:       Round-trip latency for this task.
        task_type:        Task type label for ε_ecosystem CV grouping.
        consumer_beta:    β of the consuming agent if known.
        """
        now = datetime.utcnow()

        # Downstream acceptance: delegation away = the agent failed to complete
        downstream_accepted: Optional[bool] = None
        if delegated_away:
            downstream_accepted = False
        elif success:
            downstream_accepted = True

        reconfig_count = tool_switches + reasoning_pivots

        outcome = TaskOutcome(
            task_id=task_id,
            task_domain=self.task_domain,
            success=success,
            reconfiguration_count=reconfig_count,
            duration_seconds=duration_seconds,
            downstream_accepted=downstream_accepted,
            consumer_beta=consumer_beta,
        )
        self._history.append(outcome)
        self._current_task_duration = duration_seconds

        # Tool switch events → branching (ε_intrinsic)
        for i in range(tool_switches):
            self._reconfigs.append(ReconfigEvent(
                event_type="branching",
                tool_switched=True,
                timestamp=now,
                task_id=task_id,
            ))

        # Reasoning pivots → branching (ε_intrinsic)
        for i in range(reasoning_pivots):
            self._reconfigs.append(ReconfigEvent(
                event_type="branching",
                tool_switched=False,
                timestamp=now,
                task_id=task_id,
            ))

        # Tool failures → repair (diagnostic only, not ε_intrinsic)
        for i in range(tool_failures):
            self._reconfigs.append(ReconfigEvent(
                event_type="repair",
                tool_switched=False,
                timestamp=now,
                task_id=task_id,
            ))

        if latency_ms > 0:
            self._latencies.append(LatencySample(
                task_domain=self.task_domain,
                task_type=task_type,
                latency_ms=latency_ms,
                timestamp=now,
            ))

    def record_tool_failure_event(
        self,
        resource_type: str = "api_quota",
        delay_ms: float = 0.0,
    ) -> None:
        """Record a tool infrastructure failure as an infra ContentionEvent."""
        self._contention.append(ContentionEvent(
            resource_type=resource_type,
            origin="infra",
            timestamp=datetime.utcnow(),
            delay_ms=delay_ms,
        ))

    def build(self, task_duration_seconds: Optional[float] = None) -> ObservationPayload:
        """Build and return the ObservationPayload."""
        phi_history = [
            t for t in self._history
            if t.task_domain == self.task_domain
        ][-self.phi_window:]

        sigma_recent = self._history[-self.sigma_window:]

        duration = task_duration_seconds or self._current_task_duration

        logger.debug(
            "CrewAIObservationBuilder.build(): agent=%s role=%s "
            "phi_history=%d sigma=%d reconfigs=%d latencies=%d",
            self.agent_id, self.task_domain, len(phi_history),
            len(sigma_recent), len(self._reconfigs), len(self._latencies),
        )

        return ObservationPayload(
            agent_id=self.agent_id,
            framework="crewai",
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
        """Clear per-session buffers."""
        self._reconfigs.clear()
        self._latencies.clear()
        self._contention.clear()
        self._current_task_duration = 0.0

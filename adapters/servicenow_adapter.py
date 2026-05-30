"""
adapters/servicenow_adapter.py — MELVcore × ServiceNow Virtual Agent Governance Bridge
========================================================================================
Session 34 · v3.0.0

Extracts φ, σ, β, and ε signals from ServiceNow Virtual Agent and
Now Assist execution logs and produces an ObservationPayload for the
observe() primitive.

ServiceNow signal mapping (A2A v1.0 compatible):
  φ: incident/request resolved without re-open, downstream ITSM workflow acceptance
  σ: recent conversation topic resolution rate
  β: role-scoped table access (ACL), REST API rate limits
  ε: topic redirections (branching), scripted fallbacks (repair), API latency

Five-question substrate diagnostic (all yes — native agent substrate):
  1. Records?       → Incident, Request, Change, CMDB items (GlideRecord)
  2. State machine? → task.state, incident.state, change_request.state
  3. Ownership?     → assigned_to, opened_by, caller_id on every task
  4. Structural verbs? → createRecord, updateRecord, lookupRecords, runScript
  5. Queryable history? → sys_audit, sys_log, syslog_transaction, Performance Analytics

Author: Laurence W. Evans | ORCID: 0009-0001-0963-1840 | Cape Town, South Africa
Session: 34 · Version: 3.0.0
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

logger = logging.getLogger("aios.adapters.servicenow")


class ServiceNowObservationBuilder:
    """
    Accumulates ServiceNow Virtual Agent / Now Assist conversation signals
    and builds an ObservationPayload.

    β reconstructed from ResourcePolicy + ContentionEvents(origin='infra'):
      REST API rate limit → resource_type='api_quota'
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
        snow_instance: Optional[str] = None,
        a2a_enabled: bool = False,
    ):
        self.agent_id          = agent_id
        self.task_domain       = task_domain
        self.resource_policy   = resource_policy or ResourcePolicy()
        self.tool_topology     = tool_topology or ToolTopology()
        self.phi_window        = phi_window
        self.sigma_window      = sigma_window
        self.state_reliability = state_reliability
        self.snow_instance     = snow_instance
        self.a2a_enabled       = a2a_enabled

        self._history:    list[TaskOutcome]     = []
        self._reconfigs:  list[ReconfigEvent]   = []
        self._latencies:  list[LatencySample]   = []
        self._contention: list[ContentionEvent] = []
        self._current_task_duration: float = 0.0

    def record_conversation_turn(
        self,
        session_id: str,
        topic_name: str,
        success: bool,
        topic_redirects: int = 0,
        fallback_scripts: int = 0,
        duration_seconds: float = 0.0,
        downstream_accepted: Optional[bool] = None,
        latency_ms: float = 0.0,
        task_type: str = "conversation_turn",
    ) -> None:
        """
        Record one ServiceNow Virtual Agent conversation turn.

        topic_redirects  → ReconfigEvent(event_type='branching') → ε_intrinsic
        fallback_scripts → ReconfigEvent(event_type='repair')    → diagnostic
        latency_ms       → LatencySample → ε_ecosystem CV
        """
        now = datetime.utcnow()
        task_id = f"{session_id}:{topic_name}"

        outcome = TaskOutcome(
            task_id=task_id,
            task_domain=self.task_domain,
            success=success,
            reconfiguration_count=topic_redirects,
            duration_seconds=duration_seconds,
            downstream_accepted=downstream_accepted,
        )
        self._history.append(outcome)
        self._current_task_duration = duration_seconds

        for _ in range(topic_redirects):
            self._reconfigs.append(ReconfigEvent(
                event_type="branching",
                tool_switched=False,
                timestamp=now,
                task_id=task_id,
            ))
        for _ in range(fallback_scripts):
            self._reconfigs.append(ReconfigEvent(
                event_type="repair",
                tool_switched=False,
                timestamp=now,
                task_id=task_id,
            ))

        if latency_ms > 0.0:
            self._latencies.append(LatencySample(
                task_domain=self.task_domain,
                task_type=task_type,
                latency_ms=latency_ms,
                timestamp=now,
            ))

        logger.debug(
            "snow_turn session=%s topic=%s success=%s redirects=%d fallbacks=%d",
            session_id, topic_name, success, topic_redirects, fallback_scripts,
        )

    def record_api_rate_limit(
        self,
        endpoint: str = "/api/now/table",
        delay_ms: float = 0.0,
        http_status: int = 429,
    ) -> None:
        """Record a ServiceNow REST API rate limit event as β ContentionEvent(origin='infra')."""
        self._contention.append(ContentionEvent(
            resource_type="api_quota",
            origin="infra",
            timestamp=datetime.utcnow(),
            delay_ms=delay_ms,
        ))
        logger.warning("snow rate limit: %s http=%d", endpoint, http_status)

    def build(self, task_duration_seconds: Optional[float] = None) -> ObservationPayload:
        phi_history  = [t for t in self._history if t.task_domain == self.task_domain][-self.phi_window:]
        sigma_recent = self._history[-self.sigma_window:]
        duration     = task_duration_seconds or self._current_task_duration

        return ObservationPayload(
            agent_id=self.agent_id,
            framework="servicenow",
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

    def reset(self) -> None:
        self._history.clear(); self._reconfigs.clear()
        self._latencies.clear(); self._contention.clear()
        self._current_task_duration = 0.0

    @property
    def step_count(self) -> int:
        return len(self._history)

    @property
    def success_rate(self) -> Optional[float]:
        if not self._history: return None
        return sum(1 for o in self._history if o.success) / len(self._history)

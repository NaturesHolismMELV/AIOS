"""
adapters/vertex_adapter.py — MELVcore × Google Vertex AI Agent Builder Governance Bridge
==========================================================================================
Session 34 · v3.0.0

Extracts φ, σ, β, and ε signals from Google Vertex AI Agent Builder
(Dialogflow CX / Agent Builder) execution logs and produces an
ObservationPayload for the observe() primitive.

Vertex AI signal mapping (A2A v1.0 compatible):
  φ: intent match + fulfilment success, downstream Cloud action acceptance
  σ: recent session intent-fulfilment rate
  β: IAM resource quotas, Cloud project capacity allocations
  ε: fallback intent activations (branching), webhook retries (repair), turn latency

Five-question substrate diagnostic (all yes — native agent substrate):
  1. Records?       → Cloud Firestore / Spanner entities, BigQuery tables
  2. State machine? → Dialogflow CX flow/page state, session parameters
  3. Ownership?     → Cloud IAM principal, resource labels
  4. Structural verbs? → matchIntent, fulfillIntent, callWebhook, invokeCloudFunction
  5. Queryable history? → Cloud Logging, Dialogflow CX history API, BigQuery export

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

logger = logging.getLogger("aios.adapters.vertex")


class VertexObservationBuilder:
    """
    Accumulates Vertex AI Agent Builder turn signals and builds an ObservationPayload.

    β reconstructed from ResourcePolicy + ContentionEvents(origin='infra'):
      RESOURCE_EXHAUSTED (429), SERVICE_UNAVAILABLE (503) → api_quota or compute
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
        gcp_project: Optional[str] = None,
        region: str = "us-central1",
        a2a_enabled: bool = False,
    ):
        self.agent_id          = agent_id
        self.task_domain       = task_domain
        self.resource_policy   = resource_policy or ResourcePolicy()
        self.tool_topology     = tool_topology or ToolTopology()
        self.phi_window        = phi_window
        self.sigma_window      = sigma_window
        self.state_reliability = state_reliability
        self.gcp_project       = gcp_project
        self.region            = region
        self.a2a_enabled       = a2a_enabled

        self._history:    list[TaskOutcome]     = []
        self._reconfigs:  list[ReconfigEvent]   = []
        self._latencies:  list[LatencySample]   = []
        self._contention: list[ContentionEvent] = []
        self._current_task_duration: float = 0.0

    def record_agent_turn(
        self,
        session_id: str,
        intent_name: str,
        success: bool,
        fallback_activations: int = 0,
        webhook_retries: int = 0,
        duration_seconds: float = 0.0,
        downstream_accepted: Optional[bool] = None,
        latency_ms: float = 0.0,
        task_type: str = "agent_turn",
    ) -> None:
        """
        Record one Vertex AI Agent Builder turn.

        fallback_activations → ReconfigEvent(event_type='branching') → ε_intrinsic
        webhook_retries      → ReconfigEvent(event_type='repair')    → diagnostic
        latency_ms           → LatencySample → ε_ecosystem CV
        """
        now = datetime.utcnow()
        task_id = f"{session_id}:{intent_name}"

        outcome = TaskOutcome(
            task_id=task_id,
            task_domain=self.task_domain,
            success=success,
            reconfiguration_count=fallback_activations,
            duration_seconds=duration_seconds,
            downstream_accepted=downstream_accepted,
        )
        self._history.append(outcome)
        self._current_task_duration = duration_seconds

        for _ in range(fallback_activations):
            self._reconfigs.append(ReconfigEvent(
                event_type="branching",
                tool_switched=False,
                timestamp=now,
                task_id=task_id,
            ))
        for _ in range(webhook_retries):
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
            "vertex_turn session=%s intent=%s success=%s fallbacks=%d retries=%d",
            session_id, intent_name, success, fallback_activations, webhook_retries,
        )

    def record_quota_event(
        self,
        quota_type: str = "REQUESTS_PER_MINUTE",
        delay_ms: float = 0.0,
        http_status: int = 429,
    ) -> None:
        """Record a GCP quota/capacity event as β ContentionEvent(origin='infra')."""
        resource_type = "api_quota" if http_status == 429 else "compute"
        self._contention.append(ContentionEvent(
            resource_type=resource_type,
            origin="infra",
            timestamp=datetime.utcnow(),
            delay_ms=delay_ms,
        ))
        logger.warning("vertex quota event: %s http=%d", quota_type, http_status)

    def build(self, task_duration_seconds: Optional[float] = None) -> ObservationPayload:
        phi_history  = [t for t in self._history if t.task_domain == self.task_domain][-self.phi_window:]
        sigma_recent = self._history[-self.sigma_window:]
        duration     = task_duration_seconds or self._current_task_duration

        return ObservationPayload(
            agent_id=self.agent_id,
            framework="vertex_ai",
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

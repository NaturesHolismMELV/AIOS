"""
adapters/agentforce_adapter.py — MELVcore × Salesforce Agentforce Governance Bridge
=====================================================================================
Session 34 · v3.0.0

Extracts φ, σ, β, and ε signals from Salesforce Agentforce agent execution
logs and produces an ObservationPayload for the observe() primitive.

Agentforce signal mapping (A2A v1.0 compatible):
─────────────────────────────────────────────────
  φ signals (long window, domain-conditioned):
    - Case / opportunity records closed successfully → domain_success_history
    - Downstream CRM record acceptance → downstream_accepted
    - Planner revision rounds per case → reconfiguration_count (branching ε)

  σ signals (short window, provisional ①):
    - Recent Agentforce action-step success rate

  β signals (operator-provided ONLY):
    - ResourcePolicy: object/field permissions (profiles, permission sets)
    - ContentionEvent(origin='infra'): API governor limit hits, SOQL limits
    - Reconstructed from Salesforce org governor limits + permission scope

  ε signals:
    - Planner branch retries → ReconfigEvent(event_type='branching') → ε_intrinsic
    - Flow fault handler activations → ReconfigEvent(event_type='repair') → diagnostic
    - Action-step latency → LatencySample → ε_ecosystem

Five-question substrate diagnostic (all yes — native agent substrate):
  1. Records?       → YES (SObject: Case, Opportunity, AgentWork)
  2. State machine? → YES (Case.Status, Opportunity.StageName, AgentWork.Status)
  3. Ownership?     → YES (OwnerId, CreatedById on every SObject)
  4. Structural verbs? → YES (createRecord, updateRecord, invokeAction, sendMessage)
  5. Queryable history? → YES (SOQL, SetupAuditTrail, AgentWorkItem logs)

A2A protocol v1.0 integration surface:
  Agentforce agents expose A2A-compatible task endpoints.
  β signals from org-level permission sets and governor quotas.

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

logger = logging.getLogger("aios.adapters.agentforce")


class AgentforceObservationBuilder:
    """
    Accumulates Salesforce Agentforce execution signals and builds an
    ObservationPayload.

    β is reconstructed from ResourcePolicy:
      - api_quota_per_minute: Salesforce API calls/minute limit
      - ContentionEvents(origin='infra'): governor limit hits
        (SOQL_LIMIT_HIT, DML_LIMIT_HIT, APEX_CPU_LIMIT, API_DAILY_LIMIT)
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
        org_edition: str = "Enterprise",
        a2a_enabled: bool = False,
    ):
        self.agent_id          = agent_id
        self.task_domain       = task_domain
        self.resource_policy   = resource_policy or ResourcePolicy()
        self.tool_topology     = tool_topology or ToolTopology()
        self.phi_window        = phi_window
        self.sigma_window      = sigma_window
        self.state_reliability = state_reliability
        self.org_edition       = org_edition
        self.a2a_enabled       = a2a_enabled

        self._history:    list[TaskOutcome]     = []
        self._reconfigs:  list[ReconfigEvent]   = []
        self._latencies:  list[LatencySample]   = []
        self._contention: list[ContentionEvent] = []
        self._current_task_duration: float = 0.0

    # ── PUBLIC API ─────────────────────────────────────────────────────────

    def record_action_step(
        self,
        task_id: str,
        success: bool,
        branch_retries: int = 0,
        fault_handler_activations: int = 0,
        duration_seconds: float = 0.0,
        downstream_accepted: Optional[bool] = None,
        latency_ms: float = 0.0,
        task_type: str = "case_action",
    ) -> None:
        """
        Record one Agentforce action-step execution.

        branch_retries        → ReconfigEvent(event_type='branching') → ε_intrinsic
        fault_handler_activations → ReconfigEvent(event_type='repair') → diagnostic
        latency_ms            → LatencySample → ε_ecosystem CV
        """
        now = datetime.utcnow()

        outcome = TaskOutcome(
            task_id=task_id,
            task_domain=self.task_domain,
            success=success,
            reconfiguration_count=branch_retries,
            duration_seconds=duration_seconds,
            downstream_accepted=downstream_accepted,
        )
        self._history.append(outcome)
        self._current_task_duration = duration_seconds

        for _ in range(branch_retries):
            self._reconfigs.append(ReconfigEvent(
                event_type="branching",
                tool_switched=False,
                timestamp=now,
                task_id=task_id,
            ))
        for _ in range(fault_handler_activations):
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
            "agentforce_step task=%s success=%s branch=%d fault=%d latency=%.1f",
            task_id, success, branch_retries, fault_handler_activations, latency_ms,
        )

    def record_governor_limit_hit(
        self,
        limit_type: str = "api_quota",
        delay_ms: float = 0.0,
    ) -> None:
        """
        Record a Salesforce governor limit hit as a β ContentionEvent(origin='infra').

        limit_type maps to resource_type:
          SOQL_LIMIT_HIT / DML_LIMIT_HIT → 'compute'
          API_DAILY_LIMIT               → 'api_quota'
          HEAP_LIMIT_HIT                → 'memory'
          APEX_CPU_LIMIT                → 'compute'
        """
        type_map = {
            "SOQL_LIMIT_HIT": "compute",
            "DML_LIMIT_HIT": "compute",
            "API_DAILY_LIMIT": "api_quota",
            "HEAP_LIMIT_HIT": "memory",
            "APEX_CPU_LIMIT": "compute",
        }
        resource_type = type_map.get(limit_type, "api_quota")
        self._contention.append(ContentionEvent(
            resource_type=resource_type,
            origin="infra",
            timestamp=datetime.utcnow(),
            delay_ms=delay_ms,
        ))
        logger.warning("agentforce governor limit: %s → resource_type=%s", limit_type, resource_type)

    def build(self, task_duration_seconds: Optional[float] = None) -> ObservationPayload:
        """Construct and return an ObservationPayload."""
        phi_history = [
            t for t in self._history if t.task_domain == self.task_domain
        ][-self.phi_window:]
        sigma_recent = self._history[-self.sigma_window:]
        duration = task_duration_seconds or self._current_task_duration

        return ObservationPayload(
            agent_id=self.agent_id,
            framework="agentforce",
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
        """Clear accumulated signal buffers."""
        self._history.clear()
        self._reconfigs.clear()
        self._latencies.clear()
        self._contention.clear()
        self._current_task_duration = 0.0

    @property
    def step_count(self) -> int:
        return len(self._history)

    @property
    def success_rate(self) -> Optional[float]:
        if not self._history:
            return None
        return sum(1 for o in self._history if o.success) / len(self._history)

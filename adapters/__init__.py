"""
adapters — MELVcore Framework Integration Adapters
===================================================
Session 34 · v3.0.0

Available adapters:
  langgraph_adapter   — MELVNode / MELVGraph / @melv_node decorator
                        Wrap LangGraph nodes with thermodynamic governance.

  autogen_adapter     — AutoGenObservationBuilder (Session 33)
                        Extract φ/σ/β/ε signals from AutoGen conversation logs.

  crewai_adapter      — CrewAIObservationBuilder (Session 33)
                        Extract φ/σ/β/ε signals from CrewAI task execution.

  agentforce_adapter  — AgentforceObservationBuilder (Session 34)
                        Salesforce Agentforce — SObject records, governor limits.

  copilot_adapter     — CopilotObservationBuilder (Session 34)
                        Microsoft Copilot Studio — Power Platform connectors,
                        role-scoped action permissions.

  vertex_adapter      — VertexObservationBuilder (Session 34)
                        Google Vertex AI Agent Builder — IAM quotas, Cloud limits.

  servicenow_adapter  — ServiceNowObservationBuilder (Session 34)
                        ServiceNow Virtual Agent / Now Assist — ACL table scope,
                        REST API rate limits.

All builders produce ObservationPayload for POST /api/observe/.
β is NEVER set by an adapter. It is reconstructed from ResourcePolicy
and ContentionEvents (operator-provided).
"""

from adapters.langgraph_adapter import MELVNode, MELVGraph, melv_node
from adapters.autogen_adapter import AutoGenObservationBuilder
from adapters.crewai_adapter import CrewAIObservationBuilder
from adapters.agentforce_adapter import AgentforceObservationBuilder
from adapters.copilot_adapter import CopilotObservationBuilder
from adapters.vertex_adapter import VertexObservationBuilder
from adapters.servicenow_adapter import ServiceNowObservationBuilder

__all__ = [
    # LangGraph
    "MELVNode",
    "MELVGraph",
    "melv_node",
    # Open-source frameworks (Session 33)
    "AutoGenObservationBuilder",
    "CrewAIObservationBuilder",
    # Enterprise platforms (Session 34)
    "AgentforceObservationBuilder",
    "CopilotObservationBuilder",
    "VertexObservationBuilder",
    "ServiceNowObservationBuilder",
]

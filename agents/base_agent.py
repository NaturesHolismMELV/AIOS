"""
AIOS Agent Base Class
=====================
All agents in the AIOS ecosystem inherit from BaseAgent.
Each agent has an EcoProfile (φ, ε, β) tracked by MELVcore.
"""

import time
import asyncio
import uuid
from abc import ABC, abstractmethod
from typing import Any, Optional
from core.melv_engine import MELVKernel, AgentProfile, AgentStatus


class BaseAgent(ABC):
    """
    Base class for all AIOS agents.

    Subclass this to create domain-specific agents.
    The kernel automatically tracks all interactions via
    the record_interaction() method.
    """

    def __init__(
        self,
        name:       str,
        domain:     str,
        kernel:     MELVKernel,
        phi:        float = 0.5,
        epsilon:    float = 3.0,
        beta_pref:  float = 1.0,
        capabilities: list = None
    ):
        self.agent_id = f"{name.upper().replace(' ','-')}-{str(uuid.uuid4())[:4]}"
        self.name     = name
        self.kernel   = kernel
        self._running = False

        # Register with MELVcore
        self.profile = AgentProfile(
            agent_id=self.agent_id,
            name=name,
            domain=domain,
            phi=phi,
            epsilon=epsilon,
            beta_pref=beta_pref,
            status=AgentStatus.ACTIVE if phi > 0.6 else AgentStatus.MATURING,
            capabilities=capabilities or [],
        )
        kernel.register_agent(self.profile)

    @abstractmethod
    async def execute(self, task: dict) -> dict:
        """
        Execute a task. Return result dict with at minimum:
        {
            "success": bool,
            "output": any,
            "cost": float,    # measurable resource cost
            "benefit": float, # measurable quality/value
        }
        """
        pass

    async def run_task(self, task: dict, peer_agent_id: Optional[str] = None) -> dict:
        """
        Execute a task and report the interaction to MELVcore.
        This is the method callers should use (not execute() directly).
        """
        start = time.time()
        try:
            result = await self.execute(task)
        except Exception as e:
            result = {"success": False, "output": None,
                      "cost": 1.0, "benefit": 0.1, "error": str(e)}

        elapsed = time.time() - start

        # Report interaction to kernel
        cost    = result.get("cost",    elapsed)
        benefit = result.get("benefit", 1.0 if result.get("success") else 0.1)

        if peer_agent_id:
            self.kernel.record_interaction(
                agent_a=self.agent_id,
                agent_b=peer_agent_id,
                cost=cost,
                benefit=benefit,
                resource_type=task.get("resource_type", "compute")
            )

        # Update φ based on outcome quality
        quality = benefit / max(cost, 0.01)
        quality = min(1.0, quality)
        self.kernel.update_phi(self.agent_id, quality)

        result["agent_id"]  = self.agent_id
        result["elapsed"]   = round(elapsed, 4)
        result["timestamp"] = time.time()
        return result

    @property
    def phi(self) -> float:
        return self.profile.phi

    @property
    def epsilon(self) -> float:
        return self.profile.epsilon

    def __repr__(self):
        return (f"<Agent {self.agent_id} φ={self.phi:.2f} "
                f"ε={self.epsilon:.1f} {self.profile.status.value}>")

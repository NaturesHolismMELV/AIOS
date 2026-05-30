"""
governance/kernel.py — MELVcore v1.0.0
========================================
Thin re-export and convenience layer. Ensures governance/ is independently
importable without the full AIOS server stack (no fastapi / uvicorn imports).

This file provides:
    create_kernel()     — factory function returning a configured MELVKernel
    integrate_agent()   — register an external agent profile with one call

The heavy lifting stays in core/melv_engine.py. This layer is the clean
public face of the governance package.

Blueprint for Harmony — L.W. Evans | ORCID: 0009-0001-0963-1840 | Cape Town, South Africa
ORCID: 0009-0001-0963-1840
"""

from __future__ import annotations

import sys
import os

_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.dirname(_here)
if _root not in sys.path:
    sys.path.insert(0, _root)

from core.melv_engine import (
    MELVKernel,
    AgentProfile,
    AgentStatus,
    BetaEnvironment,
    KernelAction,
)
from core.nudge_engine import NudgeEngine
from core.cost_calculator import CostCalculator


def create_kernel(
    compute: float = 1.0,
    api_quota: float = 0.9,
    token_budget: float = 1.1,
) -> MELVKernel:
    """
    Factory: create and return a configured MELVKernel.

    Parameters
    ----------
    compute      : initial beta for compute resource
    api_quota    : initial beta for API quota resource
    token_budget : initial beta for LLM token budget resource

    Returns
    -------
    MELVKernel
        Fully initialised kernel ready for agent registration.

    Example
    -------
        kernel = create_kernel(compute=1.0, token_budget=1.2)
        from governance import NudgeEngine, CostCalculator
        nudge = NudgeEngine()
        calc  = CostCalculator()
    """
    kernel = MELVKernel()
    # Override selected environment values if non-default provided
    kernel.beta.compute      = compute
    kernel.beta.api_quota    = api_quota
    kernel.beta.token_budget = token_budget
    return kernel


def integrate_agent(
    kernel: MELVKernel,
    agent_id: str,
    name: str,
    domain: str,
    phi: float = 0.5,
    epsilon: float = 3.0,
    capabilities: list | None = None,
) -> AgentProfile:
    """
    Register an external agent with MELVcore in a single call.

    MELV variable integrity enforced here:
      * phi (phi) — passed by caller; represents agent-internal maturity.
      * beta is NEVER accepted as a parameter — it is owned by BetaEnvironment.

    Parameters
    ----------
    kernel       : MELVKernel — the governing kernel instance
    agent_id     : str        — unique identifier for this agent
    name         : str        — human-readable name
    domain       : str        — specialisation domain (e.g. "research")
    phi          : float      — initial evolutionary maturity in [0, 1]
    epsilon      : float      — adaptive plasticity in [0, 8]
    capabilities : list       — optional capability tags (strings)

    Returns
    -------
    AgentProfile
        The registered profile (also stored in kernel.agents[agent_id]).

    Example
    -------
        profile = integrate_agent(kernel, "r01", "ResearchAgent",
                                  domain="research", phi=0.82)
        print(profile.maturity_label())   # "proficient"
    """
    profile = AgentProfile(
        agent_id=agent_id,
        name=name,
        domain=domain,
        phi=phi,
        epsilon=epsilon,
        status=AgentStatus.ACTIVE if phi >= 0.5 else AgentStatus.MATURING,
        capabilities=capabilities or [],
    )
    return kernel.register_agent(profile)

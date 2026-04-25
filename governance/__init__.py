"""
MELVcore Governance Module — v1.0.0
=====================================
The MELVcore thermodynamic kernel, importable as a standalone Python library
without running the AIOS server stack.

Public API
----------
    from governance import (
        MELVKernel,
        AgentProfile,
        BetaEnvironment,
        NudgeEngine,
        CostCalculator,
        KernelAction,
        InteractionRecord,
        NudgeResponse,
    )

    kernel = MELVKernel()
    nudge_engine = NudgeEngine()

    # Register an agent
    profile = AgentProfile(
        agent_id="research_01",
        name="ResearchAgent",
        domain="research",
        phi=0.82,
    )
    kernel.register_agent(profile)

    # Record an interaction
    rec = kernel.record_interaction("research_01", "writer_01", cost=0.8, benefit=2.0)
    print(rec.interaction_type)   # "cooperative"

    # Compute CI
    print(kernel.cooperation_index())  # > 0.75 → ecosystem in cooperative basin

Design principles
-----------------
  * This module does NOT pull in the web framework at module level.
    It is importable in any Python 3.11+ environment with only standard-library
    dependencies plus the melvcore core files.
  * Agents and API routes remain in agents/ and api/. governance/ exports
    the kernel only.
  * beta (environmental suitability) is NEVER set by agents. It is owned by
    BetaEnvironment and modified only by the kernel (oxpecker, provision_beta).

Analogy: MELVcore is to AIOS as Linux is to Ubuntu.
         This folder is the "kernel source" — AIOS wraps it into a platform.

Blueprint for Harmony — L.W. Evans (Ecotao Enterprises, Cape Town)
ORCID: 0009-0001-0963-1840
"""

import sys
import os

_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.dirname(_here)
if _root not in sys.path:
    sys.path.insert(0, _root)

# Core kernel classes
from core.melv_engine import (
    MELVKernel,
    AgentProfile,
    AgentStatus,
    InteractionRecord,
    BifurcationEvent,
    BetaEnvironment,
    InteractionType,
    KernelAction,
)

# Nudge v2 (Session 7)
from core.nudge_engine import (
    NudgeEngine,
    NudgeResponse,
)

# Cost calculator (Session 6)
from core.cost_calculator import (
    CostCalculator,
    CostProfile,
    CostRecord,
    get_calculator,
)

# Thin helpers from governance/kernel.py
from governance.kernel import create_kernel, integrate_agent

__version__ = "1.6.0"
__author__   = "L.W. Evans (Ecotao Enterprises)"
__license__  = "Apache-2.0"
__orcid__    = "0009-0001-0963-1840"
__doi__      = "10.5281/zenodo.17680563"

__all__ = [
    "MELVKernel", "AgentProfile", "AgentStatus", "InteractionRecord",
    "BifurcationEvent", "BetaEnvironment", "InteractionType", "KernelAction",
    "NudgeEngine", "NudgeResponse",
    "CostCalculator", "CostProfile", "CostRecord", "get_calculator",
    "create_kernel", "integrate_agent",
    "__version__", "__author__", "__license__", "__orcid__", "__doi__",
]

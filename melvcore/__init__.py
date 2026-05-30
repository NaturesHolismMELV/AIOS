"""
MELVcore — Thermodynamic Governance Kernel for the Agentic Web
===============================================================
Version 1.0.0

Built on the Modified Energetic Lotka-Volterra (MELV) framework.
Blueprint for Harmony — L.W. Evans | ORCID: 0009-0001-0963-1840 | Cape Town, South Africa
ORCID: 0009-0001-0963-1840

Nobody else has the physics.

Usage
-----
    from melvcore import MELVKernel, AgentProfile, NudgeEngine, CostCalculator

    kernel = MELVKernel()
    nudge  = NudgeEngine()
    calc   = CostCalculator()

    # Register an agent
    from melvcore import integrate_agent
    profile = integrate_agent(kernel, "r01", "ResearchAgent",
                              domain="research", phi=0.82)

    # Record an interaction
    rec = kernel.record_interaction("r01", "w01", cost=0.9, benefit=2.1,
                                    resource_type="token_budget")
    print(rec.interaction_type)      # "cooperative"
    print(kernel.cooperation_index()) # 0.57 → above 0.75 target

    # Nudge on contention
    nudge_resp = nudge.build_nudge_v2(
        action="nudge", beta_i=1.2, resource="token_budget",
        contention_depth=2, agent_phi=0.78,
    )
    print(nudge_resp.nudge_type)     # "niche_diverge" (high-phi advances early)

Installation
------------
    pip install melvcore

Hierarchy
---------
    MELVcore (this kernel) → AIOS (reference platform) → 8 agents + dashboard + API

Analogy: MELVcore is to AIOS as Linux is to Ubuntu.

PyPI      : https://pypi.org/project/melvcore/
GitHub    : github.com/NaturesHolismMELV/AIOS
Zenodo    : https://doi.org/10.5281/zenodo.17680563  (p < 10^-300)
Book      : Blueprint for Harmony, Cooperation Press 2026
            ISBN 978-969-8992-10-1
"""

__version__ = "2.5.0"
__author__    = "L.W. Evans | ORCID: 0009-0001-0963-1840"
__email__     = "web@ecotao.com"
__license__   = "Apache-2.0"
__orcid__     = "0009-0001-0963-1840"
__doi__       = "10.5281/zenodo.17680563"

import sys
import os

_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.dirname(_here)
if _root not in sys.path:
    sys.path.insert(0, _root)

# ── Public API surface ────────────────────────────────────────────────────
from core.melv_engine import (
    MELVKernel,
    AgentProfile,
    AgentStatus,
    InteractionRecord,
    BifurcationEvent,
    BetaEnvironment,
    InteractionType,
    KernelAction,
    # Session 26 + 29: ε decomposition
    EpsilonProfile,
    ARCH_CATEGORY_WEIGHTS,
    ARCH_RECOMMENDATION_THRESHOLD,
    ARCH_BETA_MULTIPLIER_CAP,
    OXPECKER_ARCH_EPSILON_LOW,
    OXPECKER_ARCH_EPSILON_HIGH,
    OXPECKER_ECOSYSTEM_WEIGHT,
)

from core.nudge_engine import (
    NudgeEngine,
    NudgeResponse,
)

from core.cost_calculator import (
    CostCalculator,
    CostProfile,
    CostRecord,
    get_calculator,
)

from governance.kernel import create_kernel, integrate_agent

__all__ = [
    # Core kernel
    "MELVKernel",
    "AgentProfile",
    "AgentStatus",
    "InteractionRecord",
    "BifurcationEvent",
    "BetaEnvironment",
    "InteractionType",
    "KernelAction",
    # Nudge v2 (Session 7)
    "NudgeEngine",
    "NudgeResponse",
    # Cost (Session 6)
    "CostCalculator",
    "CostProfile",
    "CostRecord",
    "get_calculator",
    # Session 26 + 29: ε decomposition
    "EpsilonProfile",
    "ARCH_CATEGORY_WEIGHTS",
    "ARCH_RECOMMENDATION_THRESHOLD",
    "ARCH_BETA_MULTIPLIER_CAP",
    "OXPECKER_ARCH_EPSILON_LOW",
    "OXPECKER_ARCH_EPSILON_HIGH",
    "OXPECKER_ECOSYSTEM_WEIGHT",
    # Convenience helpers
    "create_kernel",
    "integrate_agent",
    # Metadata
    "__version__",
    "__author__",
    "__email__",
    "__license__",
    "__orcid__",
    "__doi__",
]

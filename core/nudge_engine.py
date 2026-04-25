"""
NudgeEngine — MELVcore Nudge v2
================================
Full structured bifurcation response system for the MELVcore thermodynamic kernel.

Implements a four-stage escalation sequence derived from MELV theory:
  Depth 1 → retry_with_jitter   (stochastic perturbation, break symmetry)
  Depth 2 → rephrase            (agent substitution signal, +0.2 temperature)
  Depth 3 → yield               (cooperative turn-taking, explicit hold)
  Depth 4+ → niche_diverge      (structural adaptation, new resource domain)

The oxpecker / Channel 2 mechanism:
  When niche_diverge fires, apply_oxpecker_effect() is called immediately.
  Adjacent agents (same resource_type) receive a small positive β adjustment
  (+0.05 to +0.10) via BetaEnvironment — never by the agent itself.
  This is the first computational implementation of Channel 2 cooperation
  in MELVcore: indirect environmental mediation as a thermodynamic side effect.

  Named for the mutualistic behaviour Zaid observed during his 1981–1983
  cavalry veterinary service in Namibia: hornbills and bee-eaters, oxpeckers
  and large mammals — cooperation emerging not from intent but from
  thermodynamic niche structure.

Blueprint for Harmony — L.W. Evans (Ecotao Enterprises, Cape Town)
ORCID: 0009-0001-0963-1840
"""

import random
from dataclasses import dataclass, field, asdict
from typing import Optional

from core.cost_calculator import get_calculator


# ── NUDGE TYPES ────────────────────────────────────────────────────────────

NUDGE_TYPES = ("retry_with_jitter", "rephrase", "yield", "niche_diverge")

# Depth → nudge type escalation sequence (1-indexed)
DEPTH_SEQUENCE = {
    1: "retry_with_jitter",
    2: "rephrase",
    3: "yield",
}
# depth 4+ → niche_diverge

# Alternative resource domains for niche routing
_ALT_DOMAIN_MAP = {
    "token_budget":    "storage",
    "api_quota":       "vector_db",
    "compute":         "data_retrieval",
    "vector_db":       "token_budget",
    "storage":         "api_quota",
    "context_window":  "api_quota",
}

def _suggest_alt_domain(resource: str) -> str:
    return _ALT_DOMAIN_MAP.get(resource, "data_retrieval")


# ── DATACLASS ──────────────────────────────────────────────────────────────

@dataclass
class NudgeResponse:
    """
    Structured governance instruction returned by NudgeEngine.build_nudge_v2().

    Fields:
        nudge_type      — one of retry_with_jitter | rephrase | yield | niche_diverge
        contention_depth — consecutive threshold/conflict events for this agent pair
        params          — typed instruction parameters (varies by nudge_type)
        rationale       — human-readable MELV-theoretic explanation
        phi_delta       — suggested φ adjustment for the receiving agent (0.0 if none)
        niche_suggestion — suggested alt domain (non-empty for niche_diverge only)
        beta_i          — the βi value that triggered this nudge
        resource        — resource type contested
        cost_threshold  — cost cap from CostCalculator (for context)
    """
    nudge_type:        str
    contention_depth:  int
    params:            dict
    rationale:         str
    phi_delta:         float
    niche_suggestion:  str
    beta_i:            float
    resource:          str
    cost_threshold:    float = field(default_factory=lambda: get_calculator().COST_CAP)

    def to_dict(self) -> dict:
        return asdict(self)


# ── NUDGE ENGINE ───────────────────────────────────────────────────────────

class NudgeEngine:
    """
    MELVcore bifurcation response engine — Nudge v2.

    Produces a structured NudgeResponse (not a bare dict) based on:
      - contention_depth: how many consecutive threshold/conflict events for this pair
      - beta_i:           the current interaction cost ratio × environmental suitability
      - resource:         the contested resource domain
      - agent_phi:        the receiving agent's evolutionary maturity

    φ integration:
      High-φ agents can exploit new niches more effectively and therefore receive
      niche_diverge one depth earlier than the standard sequence.
      Low-φ agents (φ < 0.5) retry longer before being redirected.
    """

    PHI_HIGH_THRESHOLD = 0.75   # above this → early niche_diverge eligible
    PHI_LOW_THRESHOLD  = 0.50   # below this → extra retry before rephrase

    def __init__(self):
        # Cost thresholds from CostCalculator singleton (do not duplicate logic)
        calc = get_calculator()
        self._cost_cap = calc.COST_CAP

    def _effective_depth(self, contention_depth: int, agent_phi: float) -> int:
        """
        Adjust the effective escalation depth based on agent φ.

        High-φ agents (≥ 0.75) advance one depth earlier — they are mature enough
        to exploit an alternative niche successfully.
        Low-φ agents (< 0.50) are held one depth lower — they need more perturbation
        time before structural adaptation is warranted.
        """
        depth = contention_depth
        if agent_phi >= self.PHI_HIGH_THRESHOLD:
            depth += 1   # advance: mature agent benefits from early niche routing
        elif agent_phi < self.PHI_LOW_THRESHOLD:
            depth = max(1, depth - 1)  # slow: immature agent retries longer
        return depth

    def build_nudge_v2(
        self,
        action: str,
        beta_i: float,
        resource: str,
        contention_depth: int,
        agent_phi: float = 0.5,
    ) -> NudgeResponse:
        """
        Build a full structured NudgeResponse for a bifurcation event.

        Parameters
        ----------
        action : str
            Kernel action string (from KernelAction enum value).
        beta_i : float
            Current βi value (β × i-factor).
        resource : str
            Contested resource type.
        contention_depth : int
            Consecutive threshold/conflict events for this agent pair.
        agent_phi : float
            Receiving agent's evolutionary maturity φ ∈ [0, 1].

        Returns
        -------
        NudgeResponse
            Fully typed, structured instruction for the agent.
        """
        eff_depth = self._effective_depth(contention_depth, agent_phi)

        # Determine nudge type from effective depth
        if eff_depth >= 4:
            nudge_type = "niche_diverge"
        else:
            nudge_type = DEPTH_SEQUENCE.get(eff_depth, "retry_with_jitter")

        # Build typed params and rationale per nudge_type
        if nudge_type == "retry_with_jitter":
            delay_ms = int(random.uniform(200, 800))
            # Contention depth modulates jitter range
            if contention_depth > 1:
                delay_ms = int(random.uniform(400, 800))
            params = {
                "delay_ms":         delay_ms,
                "resource_type":    resource,
                "contention_depth": contention_depth,
            }
            rationale = (
                f"βi={beta_i:.3f} in threshold zone (depth={contention_depth}). "
                f"Stochastic perturbation: retry after {delay_ms}ms to break symmetry. "
                f"Jitter range widens with contention depth."
            )
            phi_delta = 0.0
            niche_suggestion = ""

        elif nudge_type == "rephrase":
            temp_delta = +0.2
            params = {
                "temperature_delta": temp_delta,
                "contention_depth":  contention_depth,
                "resource_type":     resource,
            }
            rationale = (
                f"βi={beta_i:.3f}, contention depth={contention_depth}. "
                f"Same approach is failing repeatedly. "
                f"Increase LLM temperature by +{temp_delta} to explore alternative "
                f"task framings. Agent substitution signal."
            )
            phi_delta = 0.01   # small φ bump for attempting adaptation
            niche_suggestion = ""

        elif nudge_type == "yield":
            duration_ms = 1000 + (contention_depth - 3) * 200
            params = {
                "duration_ms":     max(1000, duration_ms),
                "resource":        resource,
                "contention_depth": contention_depth,
                "hold_note":       "Kernel tracks yield state; hold released automatically.",
            }
            rationale = (
                f"βi={beta_i:.3f}, contention depth={contention_depth}. "
                f"Explicit cooperative turn-taking: yield '{resource}' for "
                f"{max(1000, duration_ms)}ms while the other agent completes. "
                f"Kernel holds and releases automatically."
            )
            phi_delta = 0.02   # cooperation event raises φ
            niche_suggestion = ""

        else:  # niche_diverge
            alt_domain = _suggest_alt_domain(resource)
            params = {
                "suggested_domain": alt_domain,
                "current_resource": resource,
                "contention_depth": contention_depth,
                "phi_requirement":  "φ updated by kernel on success in new niche",
            }
            rationale = (
                f"βi={beta_i:.3f}, contention depth={contention_depth} (φ={agent_phi:.2f}). "
                f"Long-term structural adaptation required. "
                f"Route toward '{alt_domain}' — lower contention, higher β. "
                f"Thermodynamic equivalent: giraffe evolving a longer neck to access "
                f"the uncontested acacia crown. Agent φ updated by kernel on success."
            )
            phi_delta = 0.05   # significant φ increase for successful niche specialisation
            niche_suggestion = alt_domain

        return NudgeResponse(
            nudge_type=nudge_type,
            contention_depth=contention_depth,
            params=params,
            rationale=rationale,
            phi_delta=phi_delta,
            niche_suggestion=niche_suggestion,
            beta_i=beta_i,
            resource=resource,
            cost_threshold=self._cost_cap,
        )

    # ── OXPECKER / CHANNEL 2 ───────────────────────────────────────────────

    def apply_oxpecker_effect(
        self,
        vacating_agent: str,
        resource_type: str,
        environment,                # BetaEnvironment instance
        adjacent_resource_types: Optional[list] = None,
    ) -> dict:
        """
        Apply the Channel 2 (oxpecker) β lift to adjacent resource domains
        when an agent vacates a contested niche via niche_diverge.

        When one agent specialises into an alternative niche, contention in the
        original resource domain decreases. The kernel applies a small positive
        β adjustment (+0.05 to +0.10) to the BetaEnvironment for that resource —
        passively raising environmental suitability for the agents that remain.

        Named for the oxpecker–mammal and hornbill–bee-eater mutualistic relationships
        observed in Namibia (1981–1983): cooperation as thermodynamic side effect,
        not deliberate intent.

        MELV integrity: β is ALWAYS adjusted by the KERNEL via BetaEnvironment.
        The vacating agent never sets its own β. The adjacent agents never set β.
        This method is called by the kernel after niche_diverge fires.

        Parameters
        ----------
        vacating_agent : str
            The agent_id of the agent that is vacating the niche.
        resource_type : str
            The resource domain being vacated.
        environment : BetaEnvironment
            The kernel's BetaEnvironment instance (modified in-place).
        adjacent_resource_types : list, optional
            Additional resource types to receive β lift (e.g. overlapping capabilities).
            If None, only the primary vacated resource_type is adjusted.

        Returns
        -------
        dict
            Report of the β adjustments applied.
        """
        beta_delta = round(random.uniform(0.05, 0.10), 3)
        adjustments = {}

        # Primary: the vacated resource domain itself gets a β lift
        current_beta = environment.get(resource_type)
        new_beta = round(min(3.0, current_beta + beta_delta), 3)
        environment.set(resource_type, new_beta)
        adjustments[resource_type] = {
            "before":    current_beta,
            "after":     new_beta,
            "delta":     beta_delta,
        }

        # Adjacent: any declared overlapping resource types
        if adjacent_resource_types:
            adj_delta = round(beta_delta * 0.5, 3)  # smaller lift for adjacent domains
            for res in adjacent_resource_types:
                if res == resource_type:
                    continue
                cur = environment.get(res)
                nw = round(min(3.0, cur + adj_delta), 3)
                environment.set(res, nw)
                adjustments[res] = {
                    "before":  cur,
                    "after":   nw,
                    "delta":   adj_delta,
                }

        return {
            "oxpecker_effect": True,
            "vacating_agent":  vacating_agent,
            "resource_type":   resource_type,
            "beta_adjustments": adjustments,
            "mechanism": (
                "Channel 2: indirect β lift from niche vacating. "
                "Contention decreased; remaining agents benefit passively. "
                "β set by kernel (BetaEnvironment), never by agents."
            ),
            "named_for": (
                "Oxpecker–mammal mutualism observed in Namibia (L.W. Evans, 1981–1983). "
                "Cooperation as thermodynamic side effect, not deliberate intent."
            ),
        }

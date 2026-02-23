"""
AIOS MELVcore Engine
====================
Thermodynamic governance layer for agent ecosystems.
Based on the Modified Energetic Lotka-Volterra (MELV) framework.
Blueprint for Harmony — L.W. Evans (Ecotao Enterprises)

Core principle: cooperation emerges thermodynamically when βi < 1.0
"""

import math
import time
import random
import json
from dataclasses import dataclass, field, asdict
from typing import Optional
from enum import Enum


# ── ENUMS ──────────────────────────────────────────────────────────────────

class AgentStatus(str, Enum):
    ACTIVE    = "active"
    MATURING  = "maturing"
    THRESHOLD = "threshold"
    SUSPENDED = "suspended"
    RETIRED   = "retired"

class InteractionType(str, Enum):
    COOPERATIVE = "cooperative"   # βi < 0.70
    THRESHOLD   = "threshold"     # 0.70 ≤ βi < 1.0
    CONFLICT    = "conflict"      # βi ≥ 1.0

class KernelAction(str, Enum):
    NONE              = "none"
    NUDGE             = "nudge"              # stochastic perturbation (Axiom 8)
    NICHE_DIVERGENCE  = "niche_divergence"   # partition resource
    ROUTE_SERVICE     = "route_service"      # route through Omega mesh
    AGENT_SUBSTITUTE  = "agent_substitute"   # replace agent
    PROVISION_BETA    = "provision_beta"     # increase environmental capacity


# ── DATA STRUCTURES ────────────────────────────────────────────────────────

@dataclass
class AgentProfile:
    """
    MELV EcoProfile for a single agent.

    φ (phi)     — evolutionary maturity / domain optimization [0.0–1.0]
    ε (epsilon) — adaptive plasticity / learning rate [0.0–8.0]
    β_pref      — preferred environmental compatibility [0.0–2.0]
    """
    agent_id:    str
    name:        str
    domain:      str
    phi:         float = 0.5       # evolutionary maturity
    epsilon:     float = 3.0       # adaptive plasticity
    beta_pref:   float = 1.0       # preferred beta
    status:      AgentStatus = AgentStatus.MATURING
    capabilities: list = field(default_factory=list)
    created_at:  float = field(default_factory=time.time)
    task_count:  int = 0
    success_rate: float = 0.0

    def maturity_label(self) -> str:
        if self.phi >= 0.85: return "expert"
        if self.phi >= 0.65: return "proficient"
        if self.phi >= 0.40: return "developing"
        return "novice"

    def to_dict(self) -> dict:
        d = asdict(self)
        d['status'] = self.status.value
        d['maturity_label'] = self.maturity_label()
        return d


@dataclass
class InteractionRecord:
    """
    Measured interaction between two agents.

    i = C/B  (cost / benefit ratio)
    βi       (modulated by environmental suitability)
    """
    agent_a:      str
    agent_b:      str
    cost:         float          # measurable interaction cost
    benefit:      float          # measurable benefit
    beta:         float          # environmental suitability at time of interaction
    timestamp:    float = field(default_factory=time.time)

    @property
    def i_factor(self) -> float:
        """i = C/B — the core MELV interaction coefficient"""
        if self.benefit <= 0:
            return 2.0  # degenerate: no benefit
        return self.cost / self.benefit

    @property
    def beta_i(self) -> float:
        """β·i — the modulated threshold value"""
        return self.beta * self.i_factor

    @property
    def interaction_type(self) -> InteractionType:
        bi = self.beta_i
        if bi < 0.70:  return InteractionType.COOPERATIVE
        if bi < 1.00:  return InteractionType.THRESHOLD
        return InteractionType.CONFLICT

    def to_dict(self) -> dict:
        return {
            "agent_a":          self.agent_a,
            "agent_b":          self.agent_b,
            "cost":             round(self.cost, 4),
            "benefit":          round(self.benefit, 4),
            "beta":             round(self.beta, 4),
            "i_factor":         round(self.i_factor, 4),
            "beta_i":           round(self.beta_i, 4),
            "interaction_type": self.interaction_type.value,
            "timestamp":        self.timestamp,
        }


@dataclass
class BifurcationEvent:
    """
    Recorded event when the kernel intervenes to drive the ecosystem
    away from the threshold zone toward the cooperative basin.
    """
    event_id:    str
    agent_a:     str
    agent_b:     str
    beta_i_pre:  float
    beta_i_post: float
    action:      KernelAction
    description: str
    timestamp:   float = field(default_factory=time.time)
    resolved:    bool = False

    def to_dict(self) -> dict:
        return {
            "event_id":    self.event_id,
            "agent_a":     self.agent_a,
            "agent_b":     self.agent_b,
            "beta_i_pre":  round(self.beta_i_pre, 4),
            "beta_i_post": round(self.beta_i_post, 4),
            "action":      self.action.value,
            "description": self.description,
            "timestamp":   self.timestamp,
            "resolved":    self.resolved,
        }


@dataclass
class BetaEnvironment:
    """
    Environmental suitability (β) for different resource types.
    β modulates how effectively agents can interact.
    High β = abundant niches = lower effective i-factor.
    """
    compute:      float = 1.0   # CPU/GPU availability
    api_quota:    float = 0.9   # External API bandwidth
    vector_db:    float = 1.2   # Vector DB I/O
    storage:      float = 0.8   # File/blob storage
    token_budget: float = 1.1   # LLM token allocation

    def get(self, resource: str) -> float:
        return getattr(self, resource, 1.0)

    def mean(self) -> float:
        vals = [self.compute, self.api_quota, self.vector_db,
                self.storage, self.token_budget]
        return sum(vals) / len(vals)

    def to_dict(self) -> dict:
        return asdict(self)


# ── MELV KERNEL ────────────────────────────────────────────────────────────

class MELVKernel:
    """
    The thermodynamic watchdog.

    Continuously monitors i-factors across all agent pairs,
    detects threshold-zone interactions, and applies bifurcation
    nudges to drive the ecosystem toward cooperative basin (βi < 1.0).

    MELV equations implemented:
      i = C/B
      βi < 1.0  →  cooperative equilibrium
      di/dt = ε(i - i_target) + η   [adaptive dynamics]
      β_service = λ_max(Ω) / n      [service coupling]
    """

    COOPERATIVE_THRESHOLD = 0.70
    CONFLICT_THRESHOLD    = 1.00
    NUDGE_NOISE_SIGMA     = 0.05    # η — stochastic perturbation (Axiom 8)

    def __init__(self):
        self.agents:       dict[str, AgentProfile]    = {}
        self.interactions: list[InteractionRecord]     = []
        self.events:       list[BifurcationEvent]      = []
        self.beta:         BetaEnvironment             = BetaEnvironment()
        self._event_counter = 0

    # ── AGENT MANAGEMENT ──────────────────────────────────────────────────

    def register_agent(self, profile: AgentProfile) -> AgentProfile:
        """Register a new agent in the ecosystem."""
        self.agents[profile.agent_id] = profile
        return profile

    def get_agent(self, agent_id: str) -> Optional[AgentProfile]:
        return self.agents.get(agent_id)

    def update_phi(self, agent_id: str, outcome_quality: float):
        """
        Update agent's φ (evolutionary maturity) based on task outcome.
        φ increases with successful, efficient interactions.
        di/dt = ε(i - i_target) + η
        """
        agent = self.agents.get(agent_id)
        if not agent: return

        # φ adapts slowly (bounded by ε)
        delta = agent.epsilon * 0.01 * (outcome_quality - 0.5)
        agent.phi = max(0.0, min(1.0, agent.phi + delta))
        agent.task_count += 1

        # Update success rate rolling average
        agent.success_rate = (
            (agent.success_rate * (agent.task_count - 1) + outcome_quality)
            / agent.task_count
        )

        # Promote status based on maturity
        if agent.phi >= 0.75 and agent.status == AgentStatus.MATURING:
            agent.status = AgentStatus.ACTIVE
        elif agent.phi >= 0.90:
            agent.status = AgentStatus.ACTIVE

    # ── i-FACTOR MONITORING ───────────────────────────────────────────────

    def record_interaction(
        self,
        agent_a: str,
        agent_b: str,
        cost: float,
        benefit: float,
        resource_type: str = "compute"
    ) -> InteractionRecord:
        """
        Record an interaction and measure its i-factor.
        Triggers kernel response if βi approaches threshold.
        """
        beta = self.beta.get(resource_type)
        record = InteractionRecord(
            agent_a=agent_a,
            agent_b=agent_b,
            cost=cost,
            benefit=benefit,
            beta=beta
        )
        self.interactions.append(record)

        # Kernel responds to threshold and conflict states
        if record.interaction_type in (InteractionType.THRESHOLD,
                                        InteractionType.CONFLICT):
            self._kernel_respond(record)

        return record

    def _kernel_respond(self, record: InteractionRecord):
        """
        Bifurcation response logic.
        Detects which intervention will most efficiently drive βi < 1.0.
        Applies Axiom 8 (necessary heterogeneity) via stochastic nudge.
        """
        bi = record.beta_i

        if record.interaction_type == InteractionType.THRESHOLD:
            # Apply stochastic perturbation to break symmetry
            # η term from adaptive dynamics equation
            eta = random.gauss(0, self.NUDGE_NOISE_SIGMA)
            new_bi = max(0.1, bi - abs(eta) - 0.08)
            action = KernelAction.NUDGE
            desc = (
                f"{record.agent_a} × {record.agent_b} in threshold zone "
                f"(βi={bi:.3f}). Stochastic perturbation applied. "
                f"Projected βi → {new_bi:.3f}"
            )
        else:
            # Conflict: stronger intervention needed
            # Check if β provisioning can resolve it
            if bi < 1.6:
                new_bi = bi * 0.65
                action = KernelAction.NICHE_DIVERGENCE
                desc = (
                    f"{record.agent_a} × {record.agent_b} in conflict "
                    f"(βi={bi:.3f}). Niche divergence applied — "
                    f"resource partitioned. βi → {new_bi:.3f}"
                )
            else:
                new_bi = bi * 0.50
                action = KernelAction.PROVISION_BETA
                desc = (
                    f"{record.agent_a} × {record.agent_b} high conflict "
                    f"(βi={bi:.3f}). β provisioning triggered — "
                    f"additional resources allocated. βi → {new_bi:.3f}"
                )

        self._event_counter += 1
        event = BifurcationEvent(
            event_id=f"BIF-{self._event_counter:04d}",
            agent_a=record.agent_a,
            agent_b=record.agent_b,
            beta_i_pre=bi,
            beta_i_post=new_bi,
            action=action,
            description=desc,
            resolved=(new_bi < self.CONFLICT_THRESHOLD)
        )
        self.events.append(event)

    # ── OMEGA SERVICE NETWORK ──────────────────────────────────────────────

    def compute_omega(self) -> dict:
        """
        Compute service coupling matrix Ω metrics.
        β_service = λ_max(Ω) / n

        Simplified implementation: uses interaction frequency and
        quality as proxy for service coupling strength.
        """
        n = len(self.agents)
        if n == 0:
            return {"lambda_max": 0, "n": 0, "beta_service": 0, "edges": []}

        # Build adjacency weights from interaction history (last 100)
        recent = self.interactions[-100:]
        weights: dict[tuple, list] = {}
        for r in recent:
            key = tuple(sorted([r.agent_a, r.agent_b]))
            if key not in weights:
                weights[key] = []
            weights[key].append(1.0 - r.i_factor)  # higher = more cooperative

        edges = []
        total_weight = 0.0
        for (a, b), vals in weights.items():
            avg = sum(vals) / len(vals)
            edges.append({
                "agent_a": a,
                "agent_b": b,
                "weight": round(avg, 3),
                "interaction_type": (
                    "cooperative" if avg > 0.30 else
                    "threshold"   if avg > 0.0  else
                    "conflict"
                )
            })
            total_weight += avg

        # λ_max approximation: sum of weights / sqrt(n)
        lambda_max = total_weight / max(1, math.sqrt(n))
        beta_service = lambda_max / n if n > 0 else 0

        return {
            "lambda_max":    round(lambda_max, 4),
            "n":             n,
            "beta_service":  round(beta_service, 4),
            "edges":         edges,
        }

    # ── ECOSYSTEM HEALTH ──────────────────────────────────────────────────

    def cooperation_index(self) -> float:
        """
        CI = 1 - mean(βi)  across all recent interactions.
        Target: CI > 0.75 (ecosystem in cooperative basin).
        """
        recent = self.interactions[-50:]
        if not recent:
            return 1.0
        mean_bi = sum(r.beta_i for r in recent) / len(recent)
        return max(0.0, min(1.0, 1.0 - mean_bi))

    def ecosystem_health(self) -> dict:
        """
        Full ecosystem snapshot for the Harmony Dashboard.
        """
        recent = self.interactions[-50:]
        n_agents = len(self.agents)

        if recent:
            type_counts = {t.value: 0 for t in InteractionType}
            for r in recent:
                type_counts[r.interaction_type.value] += 1
            mean_i    = sum(r.i_factor for r in recent) / len(recent)
            mean_beta_i = sum(r.beta_i for r in recent) / len(recent)
        else:
            type_counts = {t.value: 0 for t in InteractionType}
            mean_i      = 0.0
            mean_beta_i = 0.0

        mean_phi     = (sum(a.phi for a in self.agents.values()) / n_agents
                        if n_agents else 0.0)
        mean_epsilon = (sum(a.epsilon for a in self.agents.values()) / n_agents
                        if n_agents else 0.0)

        status_counts = {}
        for a in self.agents.values():
            status_counts[a.status.value] = status_counts.get(a.status.value, 0) + 1

        recent_events = [e.to_dict() for e in self.events[-10:]]

        return {
            "cooperation_index":    round(self.cooperation_index(), 4),
            "mean_i_factor":        round(mean_i, 4),
            "mean_beta_i":          round(mean_beta_i, 4),
            "mean_phi":             round(mean_phi, 4),
            "mean_epsilon":         round(mean_epsilon, 4),
            "n_agents":             n_agents,
            "n_interactions_total": len(self.interactions),
            "interaction_breakdown":type_counts,
            "agent_status_counts":  status_counts,
            "beta_environment":     self.beta.to_dict(),
            "omega":                self.compute_omega(),
            "recent_events":        recent_events,
            "threshold_zone_count": type_counts.get("threshold", 0),
            "conflict_count":       type_counts.get("conflict", 0),
        }

    def get_all_agents(self) -> list[dict]:
        return [a.to_dict() for a in self.agents.values()]

    def get_recent_interactions(self, n: int = 20) -> list[dict]:
        return [r.to_dict() for r in self.interactions[-n:]]

    def get_recent_events(self, n: int = 20) -> list[dict]:
        return [e.to_dict() for e in self.events[-n:]]

    def provision_beta(self, resource: str, value: float):
        """Human portal: adjust environmental suitability."""
        setattr(self.beta, resource, max(0.1, min(3.0, value)))

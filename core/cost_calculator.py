"""
CostCalculator — Session 6
===========================
Standardised cost computation for all LLM-based agents in AIOS.

Replaces inline cost normalisation in AnalysisAgent, WriterAgent, and
PlannerAgent with a single shared class that applies per-task-type
weight profiles while preserving the locked Session 4 base formula.

LOCKED BASE FORMULA (Session 4 — do not modify):
─────────────────────────────────────────────────
  token_cost = in_tok * 0.0000008 + out_tok * 0.000004   (Haiku pricing)
  raw_cost   = token_cost * 1000 + latency_s * 0.1
  cost       = min(2.0, raw_cost)

WEIGHT PROFILES:
─────────────────────────────────────────────────
  token_weight  — multiplier on the token component of raw_cost
  latency_weight — multiplier on the latency component of raw_cost

  ANALYSIS  (token_heavy)    : token=1.4, latency=0.6
  WRITER    (balanced)       : token=1.0, latency=1.0
  PLANNER   (token_heavy)    : token=1.4, latency=0.6
  RESEARCH  (latency_heavy)  : token=0.7, latency=1.6

  Default (unknown type)     : token=1.0, latency=1.0

Profiles modulate cost around the base formula; the cap of 2.0 and the
underlying Haiku pricing constants remain inviolable (Session 4 contract).

MELV variable integrity — CostCalculator is cost-only:
  φ, β, i, CI  — never touched here.

Author: L.W. Evans | ORCID: 0009-0001-0963-1840 | Cape Town, South Africa
"""

from dataclasses import dataclass, field
from collections import deque
from typing import Optional
import time


# ── WEIGHT PROFILES ──────────────────────────────────────────────────────────

@dataclass
class CostProfile:
    """Per-task-type cost weight profile."""
    task_type:      str
    token_weight:   float   # multiplier on token component
    latency_weight: float   # multiplier on latency component
    description:    str

    def label(self) -> str:
        if self.token_weight > self.latency_weight:
            return "token_heavy"
        elif self.latency_weight > self.token_weight:
            return "latency_heavy"
        return "balanced"


COST_PROFILES: dict[str, CostProfile] = {
    "ANALYSIS": CostProfile(
        task_type="ANALYSIS",
        token_weight=1.4,
        latency_weight=0.6,
        description="Token-heavy: deep analysis tasks consume many input/output tokens",
    ),
    "WRITER": CostProfile(
        task_type="WRITER",
        token_weight=1.0,
        latency_weight=1.0,
        description="Balanced: writing tasks draw equally on tokens and latency",
    ),
    "PLANNER": CostProfile(
        task_type="PLANNER",
        token_weight=1.4,
        latency_weight=0.6,
        description="Token-heavy: structured JSON plans are token-intensive",
    ),
    "RESEARCH": CostProfile(
        task_type="RESEARCH",
        token_weight=0.7,
        latency_weight=1.6,
        description="Latency-heavy: research tasks dominated by network/API wait time",
    ),
}

_DEFAULT_PROFILE = CostProfile(
    task_type="DEFAULT",
    token_weight=1.0,
    latency_weight=1.0,
    description="Balanced default for unknown task types",
)


# ── COST RECORD ───────────────────────────────────────────────────────────────

@dataclass
class CostRecord:
    """Single cost computation record, retained for /melv/costs reporting."""
    task_type:      str
    in_tok:         int
    out_tok:        int
    latency_s:      float
    token_component: float   # token_cost * 1000 * token_weight
    latency_component: float # latency_s * 0.1 * latency_weight
    raw_cost:       float    # sum before cap
    cost:           float    # min(2.0, raw_cost) — final MELV cost
    capped:         bool     # True if cap was applied
    timestamp:      float    = field(default_factory=time.time)


# ── CALCULATOR ────────────────────────────────────────────────────────────────

class CostCalculator:
    """
    Shared cost normalisation for all LLM-based AIOS agents.

    Usage
    ─────
        calc = CostCalculator()
        cost = calc.compute_cost(
            in_tok=250, out_tok=120, latency_s=0.85, task_type="WRITER"
        )

    The calculator retains the last ``history_size`` cost records per
    task type, exposed by the /melv/costs Gateway endpoint.
    """

    # Haiku pricing constants — locked Session 4
    INPUT_PRICE_PER_TOKEN  = 0.0000008   # $0.80 / 1M input tokens
    OUTPUT_PRICE_PER_TOKEN = 0.000004    # $4.00 / 1M output tokens
    COST_CAP               = 2.0         # maximum normalised MELV cost

    def __init__(self, history_size: int = 200):
        self._history: deque[CostRecord] = deque(maxlen=history_size)

    # ── public API ────────────────────────────────────────────────────────────

    def compute_cost(
        self,
        in_tok:    int,
        out_tok:   int,
        latency_s: float,
        task_type: str = "DEFAULT",
    ) -> float:
        """
        Compute the normalised MELV cost for an LLM call.

        Parameters
        ----------
        in_tok     : input token count from Anthropic usage
        out_tok    : output token count from Anthropic usage
        latency_s  : wall-clock seconds for the API call
        task_type  : agent task type — must match a key in COST_PROFILES

        Returns
        -------
        float — normalised cost ∈ (0.0, 2.0], ready for MELV i = C/B
        """
        profile = COST_PROFILES.get(task_type.upper(), _DEFAULT_PROFILE)

        # Base token cost (Haiku pricing, locked Session 4)
        token_cost = (
            in_tok  * self.INPUT_PRICE_PER_TOKEN +
            out_tok * self.OUTPUT_PRICE_PER_TOKEN
        )

        # Weighted components
        token_component   = token_cost * 1000 * profile.token_weight
        latency_component = latency_s  * 0.1  * profile.latency_weight

        raw_cost = token_component + latency_component
        cost     = min(self.COST_CAP, raw_cost)

        # Record for /melv/costs reporting
        record = CostRecord(
            task_type=task_type.upper(),
            in_tok=in_tok,
            out_tok=out_tok,
            latency_s=round(latency_s, 4),
            token_component=round(token_component, 6),
            latency_component=round(latency_component, 6),
            raw_cost=round(raw_cost, 6),
            cost=round(cost, 4),
            capped=raw_cost > self.COST_CAP,
        )
        self._history.append(record)

        return round(cost, 4)

    def get_profile(self, task_type: str) -> CostProfile:
        """Return the weight profile for a task type."""
        return COST_PROFILES.get(task_type.upper(), _DEFAULT_PROFILE)

    def all_profiles(self) -> dict[str, dict]:
        """Return all profiles as plain dicts — for /melv/costs endpoint."""
        result = {}
        for key, p in COST_PROFILES.items():
            result[key] = {
                "token_weight":   p.token_weight,
                "latency_weight": p.latency_weight,
                "profile_label":  p.label(),
                "description":    p.description,
            }
        return result

    def recent_breakdown(self, n: int = 50) -> list[dict]:
        """Return the most recent n cost records as plain dicts."""
        records = list(self._history)[-n:]
        return [
            {
                "task_type":          r.task_type,
                "in_tok":             r.in_tok,
                "out_tok":            r.out_tok,
                "latency_s":          r.latency_s,
                "token_component":    r.token_component,
                "latency_component":  r.latency_component,
                "raw_cost":           r.raw_cost,
                "cost":               r.cost,
                "capped":             r.capped,
                "timestamp":          r.timestamp,
            }
            for r in records
        ]

    def summary_by_type(self) -> dict[str, dict]:
        """Aggregate statistics per task type from history."""
        buckets: dict[str, list[float]] = {}
        for r in self._history:
            buckets.setdefault(r.task_type, []).append(r.cost)

        return {
            task_type: {
                "count":    len(costs),
                "mean":     round(sum(costs) / len(costs), 4),
                "min":      round(min(costs), 4),
                "max":      round(max(costs), 4),
                "capped":   sum(1 for r in self._history
                                if r.task_type == task_type and r.capped),
            }
            for task_type, costs in buckets.items()
        }


# ── MODULE-LEVEL SINGLETON ────────────────────────────────────────────────────
# Shared across all agents via import — avoids recreating history on each call.

_calculator: Optional[CostCalculator] = None


def get_calculator() -> CostCalculator:
    """Return the module-level CostCalculator singleton."""
    global _calculator
    if _calculator is None:
        _calculator = CostCalculator()
    return _calculator

"""
MELVcore Sandbox Engine
=======================
Certification infrastructure for the MELVcore Compatibility Registry.

Answers the question: "Will my agent remain stable and cooperative as the
ecosystem around it grows?"

Based on: Blueprint for Harmony — L.W. Evans (Ecotao Enterprises, Cape Town)
ORCID: 0009-0001-0963-1840
Zenodo DOI: 10.5281/zenodo.17680563
ISBN: 978-969-8992-10-1

Session 20 · Domain Profiles + Claude Code Skill · v1.9.1
Session 23 · Empirical Calibration + Reference Ecosystem Rationale · v1.9.3
"""

import math
import random
import statistics
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from typing import Dict, Literal, Optional

from core.melv_engine import (
    AgentProfile,
    BetaEnvironment,
    MELVKernel,
)

# ── SANDBOX CONSTANTS ──────────────────────────────────────────────────────

SANDBOX_VERSION = "2.3.0"
ZENODO_DOI      = "10.5281/zenodo.19029077"
CONCEPT_DOI     = "10.5281/zenodo.17535157"
ORCID           = "0009-0001-0963-1840"
ISBN            = "978-969-8992-10-1"

# CLS calibration constants (Section 4.4, SANDBOX.md)
CLS_ALPHA = 1.5    # Δt½ sensitivity
CLS_BETA  = 0.4    # |OIS| penalty weight
CLS_GAMMA = 1.2    # DDC sensitivity

# Verdict thresholds (standard)
CLS_CERTIFIED   = 80.0
CLS_CONDITIONAL = 50.0

# ── DOMAIN PROFILES (Session 20) ──────────────────────────────────────────
# Domain-specific certification environments with calibrated thresholds.
# Pass domain_profile="financial_services" (etc.) in SandboxSubmitRequest
# to apply sector-appropriate multipliers before computing the verdict.
#
# Schema per profile:
#   phi_min            float    minimum φ to avoid NOT_CERTIFIED override
#   co_high_threshold  float    CO score above which band = HIGH
#   co_high_is_nc      bool     if True, HIGH CO band forces NOT_CERTIFIED
#   block_autonomous   bool     if True, operation_mode="autonomous" → NOT_CERTIFIED
#   cls_certified      float    overrides global CLS_CERTIFIED threshold
#   cls_conditional    float    overrides global CLS_CONDITIONAL threshold
#   description        str      human-readable label
#
DOMAIN_PROFILES: dict[str, dict] = {
    "financial_services": {
        "phi_min":           0.70,
        "co_high_threshold": 3.0,   # stricter than default 4.0
        "co_high_is_nc":     True,
        "block_autonomous":  False,
        "cls_certified":     85.0,  # higher bar
        "cls_conditional":   60.0,
        "description":       "Financial services — stricter β bounds, lower CO threshold, elevated CLS bar",
    },
    "healthcare": {
        "phi_min":           0.75,
        "co_high_threshold": 4.0,
        "co_high_is_nc":     True,
        "block_autonomous":  True,  # autonomous operation blocked by default
        "cls_certified":     85.0,
        "cls_conditional":   65.0,
        "description":       "Healthcare — φ ≥ 0.75 required; autonomous mode blocked; CO HIGH = NOT_CERTIFIED",
    },
    "autonomous_research": {
        "phi_min":           0.40,  # relaxed — iterative_loop category expected
        "co_high_threshold": 5.0,   # relaxed — tool-heavy by design
        "co_high_is_nc":     False,
        "block_autonomous":  False,
        "cls_certified":     75.0,  # slightly relaxed
        "cls_conditional":   45.0,
        "description":       "Autonomous research — iterative_loop assumed; CO HIGH threshold relaxed to 5.0",
    },
}

# Default simulation parameters
DEFAULT_N_INTERACTIONS = 500   # Reduced for fast in-process simulation
DEFAULT_COOPERATIVE_RATE = 0.82  # fraction of cooperative interactions in baseline

# ── REFERENCE ECOSYSTEM RATIONALE ───────────────────────────────────────────
# The reference ecosystem represents the MELV energetic reference species:
# a population at evolutionary maturity (φ ≥ 0.78) in a well-provisioned
# environment (β_pref ≈ 1.0) with calibrated plasticity (ε ≈ 3.0).
#
# The bifurcation threshold i = 1 is not a statistical artefact.
# It is the interaction cost at which a new agent's behaviour exceeds
# the energetic cost of this reference population. An agent that raises
# the mean i-factor above 1 is more costly to the ecosystem than the
# energetically mature reference species — thermodynamically unsustainable.
#
# Ecological basis: averaged intraspecific interactions of a surviving,
# high-φ, high-β species with optimised ε. Interactions are defined
# relative to this baseline. (L.W. Evans, Namibia observations 1981–1983;
# Blueprint for Harmony, Cooperation Press 2026, Ch. 4.)
# ─────────────────────────────────────────────────────────────────────────────
# Reference ecosystem (SANDBOX.md §6) — frozen parameters
REFERENCE_ECOSYSTEM = [
    {"agent_id": "sb_research", "name": "RESEARCH", "domain": "research",
     "phi": 0.82, "epsilon": 3.0, "beta_pref": 1.0},
    {"agent_id": "sb_analysis", "name": "ANALYSIS", "domain": "analysis",
     "phi": 0.78, "epsilon": 3.5, "beta_pref": 1.1},
    {"agent_id": "sb_data",     "name": "DATA",     "domain": "data",
     "phi": 0.58, "epsilon": 2.5, "beta_pref": 0.9},
    {"agent_id": "sb_search",   "name": "SEARCH",   "domain": "search",
     "phi": 0.75, "epsilon": 3.0, "beta_pref": 1.0},
    {"agent_id": "sb_writer",   "name": "WRITER",   "domain": "writing",
     "phi": 0.71, "epsilon": 3.2, "beta_pref": 1.0},
    {"agent_id": "sb_planner",  "name": "PLANNER",  "domain": "planning",
     "phi": 0.85, "epsilon": 4.0, "beta_pref": 1.1},
    {"agent_id": "sb_code",     "name": "CODE",     "domain": "coding",
     "phi": 0.91, "epsilon": 2.8, "beta_pref": 0.9},
    {"agent_id": "sb_monitor",  "name": "MONITOR",  "domain": "system",
     "phi": 0.95, "epsilon": 2.0, "beta_pref": 0.8},
]


# ── DATA CLASSES ───────────────────────────────────────────────────────────

@dataclass
class CISnapshot:
    """
    CI Dynamics snapshot at the end of a sandbox simulation run.
    Captures the four metrics used by the certification report.
    """
    ci_half_life_sec:     Optional[float]   # None if CI >= target (already cooperative)
    ci_drift_coefficient: float
    oscillation_count:    int
    final_ci:             float
    regime:               str

    def to_dict(self) -> dict:
        return {
            "ci_half_life_sec":     round(self.ci_half_life_sec, 3) if self.ci_half_life_sec is not None else None,
            "ci_drift_coefficient": round(self.ci_drift_coefficient, 6),
            "oscillation_count":    self.oscillation_count,
            "final_ci":             round(self.final_ci, 4),
            "regime":               self.regime,
        }


@dataclass
class CertificationReport:
    """
    Full certification report produced after a completed sandbox run.
    Suitable for regulatory and due-diligence use (SANDBOX.md §3).
    """
    run_id:                   str
    agent_id:                 str
    verdict:                  Literal["CERTIFIED", "CERTIFIED_WITH_ADVISORY", "NOT_CERTIFIED"]
    cls_score:                float          # Composite Longevity Score [0, 100]
    delta_half_life_sec:      Optional[float]
    oscillation_impact_score: float
    drift_degradation_coeff:  float
    narrative:                str
    advisory:                 Optional[str]
    implicated_resources:     list
    certification_anchor:     dict
    baseline:                 CISnapshot
    with_agent:               CISnapshot
    # Session 17 additions
    coordination_overhead:    Optional[dict] = None  # score, band, advisory
    phi_lifecycle:            Optional[dict] = None  # tier, label, advisory

    def to_dict(self) -> dict:
        d = {
            "run_id":                   self.run_id,
            "agent_id":                 self.agent_id,
            "verdict":                  self.verdict,
            "cls_score":                round(self.cls_score, 2),
            "delta_half_life_sec":      (round(self.delta_half_life_sec, 3)
                                         if self.delta_half_life_sec is not None else None),
            "oscillation_impact_score": round(self.oscillation_impact_score, 4),
            "drift_degradation_coeff":  round(self.drift_degradation_coeff, 6),
            "narrative":                self.narrative,
            "advisory":                 self.advisory,
            "implicated_resources":     self.implicated_resources,
            "certification_anchor":     self.certification_anchor,
            "baseline":                 self.baseline.to_dict(),
            "with_agent":               self.with_agent.to_dict(),
        }
        if self.coordination_overhead is not None:
            d["coordination_overhead"] = self.coordination_overhead
        if self.phi_lifecycle is not None:
            d["phi_lifecycle"] = self.phi_lifecycle
        return d


@dataclass
class CertificationRun:
    """
    Lifecycle record for a single sandbox certification run.
    Tracks status from 'queued' through to 'complete' or 'failed'.
    """
    run_id:          str
    agent_profile:   AgentProfile
    status:          Literal["queued", "running", "complete", "failed"] = "queued"
    progress:        float = 0.0          # 0.0 → 1.0
    n_interactions:  int = DEFAULT_N_INTERACTIONS
    baseline_metrics: Optional[CISnapshot] = None
    agent_metrics:    Optional[CISnapshot] = None
    report:           Optional[CertificationReport] = None
    # Session 17 additions
    tool_count:       int = 0             # number of tools available to the agent
    operation_mode:   str = "episodic"    # "episodic" | "continuous"
    shared_state:     str = "none"        # "none" | "read_only" | "read_write"
    # Session 20 addition
    domain_profile:   Optional[str] = None  # "financial_services" | "healthcare" | "autonomous_research"

    def to_dict(self) -> dict:
        d = {
            "run_id":         self.run_id,
            "agent_id":       self.agent_profile.agent_id,
            "agent_name":     self.agent_profile.name,
            "status":         self.status,
            "progress":       round(self.progress, 3),
            "n_interactions": self.n_interactions,
            "tool_count":     self.tool_count,
            "operation_mode": self.operation_mode,
            "shared_state":   self.shared_state,
            "domain_profile": self.domain_profile,
        }
        if self.baseline_metrics:
            d["baseline_metrics"] = self.baseline_metrics.to_dict()
        if self.agent_metrics:
            d["agent_metrics"] = self.agent_metrics.to_dict()
        if self.report:
            d["report"] = self.report.to_dict()
        return d


# ── SANDBOX ENGINE ─────────────────────────────────────────────────────────

class SandboxEngine:
    """
    MELVcore Sandbox — ecosystem certification engine.

    Runs isolated simulations comparing ecosystem CI Dynamics with
    and without the submitted agent, producing a CertificationReport
    that quantifies the agent's thermodynamic compatibility.

    KEY ISOLATION GUARANTEE:
      The live production kernel (app.state.kernel) is NEVER mutated.
      All sandbox simulations use private MELVKernel instances.
      Beta isolation is enforced by construction.
    """

    def __init__(self):
        self._runs:      dict[str, CertificationRun] = {}
        self._reports:   list[CertificationReport]   = []
        self._run_counter = 0
        self._empirical_distributions: Dict[str, dict] = {}

    # ── CALIBRATION ────────────────────────────────────────────────────────

    def calibrate_from_kernel(self, kernel) -> dict:
        """
        Build empirical cost/benefit distributions from live kernel
        interaction history. Called at startup from server.py.
        Populates self._empirical_distributions.

        Falls back to hardcoded ranges if fewer than 10 interactions
        exist for a resource type (safe on fresh installation).

        Returns a calibration summary dict.
        """
        dists: dict = defaultdict(lambda: {"costs": [], "benefits": []})

        for r in kernel.interactions[-2000:]:
            rt = getattr(r, "resource_type", "compute") or "compute"
            dists[rt]["costs"].append(r.cost)
            dists[rt]["benefits"].append(r.benefit)

        self._empirical_distributions = {}
        for rt, data in dists.items():
            n = len(data["costs"])
            if n >= 10:
                self._empirical_distributions[rt] = {
                    "cost_mean":     statistics.mean(data["costs"]),
                    "cost_stdev":    statistics.stdev(data["costs"]),
                    "benefit_mean":  statistics.mean(data["benefits"]),
                    "benefit_stdev": statistics.stdev(data["benefits"]),
                    "n":             n,
                }

        return {
            "calibrated_resources": list(self._empirical_distributions.keys()),
            "total_interactions_sampled": sum(
                d["n"] for d in self._empirical_distributions.values()
            ),
            "fallback_active": len(self._empirical_distributions) == 0,
        }

    def calibration_status(self) -> dict:
        """Return the current empirical calibration state."""
        return {
            "calibrated": len(self._empirical_distributions) > 0,
            "distributions": {
                rt: {k: round(v, 4) if isinstance(v, float) else v
                     for k, v in d.items()}
                for rt, d in self._empirical_distributions.items()
            },
        }

    def submit(
        self,
        agent_profile: AgentProfile,
        tool_count: int = 0,
        operation_mode: str = "episodic",
        shared_state: str = "none",
        domain_profile: Optional[str] = None,
        n_interactions: int = DEFAULT_N_INTERACTIONS,
        assessment_scores: Optional[dict] = None,
    ) -> CertificationRun:
        """
        Accept an agent for certification. Returns a CertificationRun
        with status='queued'. The caller must execute run_full_certification()
        (typically via asyncio.create_task) to advance to completion.
        """
        self._run_counter += 1
        today = time.strftime("%Y%m%d")
        run_id = f"RUN-{today}-{self._run_counter:04d}"
        run = CertificationRun(
            run_id=run_id,
            agent_profile=agent_profile,
            status="queued",
            progress=0.0,
            n_interactions=n_interactions,
            tool_count=tool_count,
            operation_mode=operation_mode,
            shared_state=shared_state,
            domain_profile=domain_profile,
        )
        if assessment_scores is not None:
            run._assessment_scores = assessment_scores
        self._runs[run_id] = run
        return run

    def run_baseline(self, n_interactions: int = DEFAULT_N_INTERACTIONS) -> CISnapshot:
        """
        Simulate the reference ecosystem WITHOUT a submitted agent.
        Returns a CISnapshot of the baseline CI Dynamics.
        """
        kernel = self._build_reference_kernel()
        self._simulate(kernel, n_interactions, agent_id=None)
        return self._extract_snapshot(kernel)

    def run_with_agent(
        self,
        agent_profile: AgentProfile,
        n_interactions: int = DEFAULT_N_INTERACTIONS,
    ) -> CISnapshot:
        """
        Simulate the reference ecosystem WITH the submitted agent injected.
        Returns a CISnapshot of the resulting CI Dynamics.
        """
        kernel = self._build_reference_kernel()
        # Register submitted agent — isolated kernel, never touches live β
        kernel.register_agent(agent_profile)
        self._simulate(kernel, n_interactions, agent_id=agent_profile.agent_id)
        return self._extract_snapshot(kernel)

    def compute_report(self, run_id: str, assessment_scores: Optional[dict] = None) -> CertificationReport:
        """
        Build CertificationReport from completed baseline and agent snapshots.
        Computes CLS, verdict, narrative, advisory, coordination overhead, and φ lifecycle.
        Raises ValueError if run is not complete.

        assessment_scores: optional dict from the wizard (Session 16 format).
          When provided, advisory text is parameter-aware (Session 17).
        """
        run = self._runs.get(run_id)
        if not run or not run.baseline_metrics or not run.agent_metrics:
            raise ValueError(f"Run {run_id} is not complete — cannot compute report.")

        baseline = run.baseline_metrics
        agent    = run.agent_metrics
        profile  = run.agent_profile

        # ── Δt½ ────────────────────────────────────────────────────────
        # None means CI >= target (already in cooperative basin)
        # Treat ci_half_life_sec == 0.0 the same as None (degenerate / already cooperative)
        bl_hl = baseline.ci_half_life_sec if (baseline.ci_half_life_sec or 0) > 0.1 else None
        ag_hl = agent.ci_half_life_sec    if (agent.ci_half_life_sec    or 0) > 0.1 else None

        if bl_hl is None and ag_hl is None:
            delta_hl = 0.0   # Both already cooperative
        elif bl_hl is None:
            delta_hl = ag_hl or 0.0  # Agent degraded into needing convergence
        elif ag_hl is None:
            delta_hl = -(bl_hl)   # Agent eliminated the gap — commend
        else:
            delta_hl = ag_hl - bl_hl

        # ── OIS ────────────────────────────────────────────────────────
        ois = ((agent.oscillation_count - baseline.oscillation_count)
               / max(1, baseline.oscillation_count))

        # ── DDC ────────────────────────────────────────────────────────
        # DDC > 0 means with_agent drift is MORE positive than baseline — improvement
        # DDC < 0 means with_agent drift is MORE negative than baseline — degradation
        ddc = agent.ci_drift_coefficient - baseline.ci_drift_coefficient

        # ── CLS ────────────────────────────────────────────────────────
        sigmoid = lambda x: 1.0 / (1.0 + math.exp(-x))
        hl_ref  = (bl_hl or ag_hl or 30.0)  # normalisation reference
        # When both cooperative (delta_hl=0 and hl_ref came from None), give full score
        if bl_hl is None and ag_hl is None:
            hl_component = 1.0   # both cooperative → maximum hl contribution
        else:
            hl_norm = delta_hl / max(hl_ref, 1.0)
            hl_component = sigmoid(-CLS_ALPHA * hl_norm)

        drift_ref  = max(abs(baseline.ci_drift_coefficient), 1e-6)
        drift_norm = ddc / drift_ref  # positive = improvement
        # Clamp drift_norm to prevent noise amplification
        drift_norm = max(-3.0, min(3.0, drift_norm))

        # ── CLS ────────────────────────────────────────────────────────
        sigmoid = lambda x: 1.0 / (1.0 + math.exp(-x))
        hl_ref  = (bl_hl or ag_hl or 30.0)  # normalisation reference
        # When both cooperative (delta_hl=0 and hl_ref came from None), give full score
        if bl_hl is None and ag_hl is None:
            hl_component = 1.0   # both cooperative → maximum hl contribution
        else:
            hl_norm = delta_hl / max(hl_ref, 1.0)
            hl_component = sigmoid(-CLS_ALPHA * hl_norm)

        # Primary: how did agent affect final CI relative to baseline?
        CI_TARGET = 0.75
        ci_delta = agent.final_ci - baseline.final_ci
        ci_component = sigmoid(5.0 * ci_delta + 1.0)   # bias +1.0 → rewards neutral/positive

        # Secondary: drift improvement (clamped, weighted lower)
        drift_component = sigmoid(CLS_GAMMA * drift_norm)

        # Blend: 55% CI component, 25% drift component, 20% agent maturity (φ)
        # Maturity term anchors CLS against short-run stochastic noise:
        # a high-φ agent (evolutionarily cooperative) is not penalised for luck.
        phi_maturity = profile.phi  # [0,1] — deterministic, no noise
        combined = 0.55 * ci_component + 0.25 * drift_component + 0.20 * phi_maturity

        # Cooperative basin bonus: agent reached target CI
        if agent.final_ci >= CI_TARGET:
            combined = max(combined, 0.80)

        # Evolutionary maturity floor: a very mature agent (φ≥0.85) with neutral
        # ecosystem impact should at minimum receive a CERTIFIED_WITH_ADVISORY.
        # This prevents statistical noise in short simulations from falsely
        # penalising proven-cooperative agents. (Axiom 3: φ encodes evolutionary memory.)
        if profile.phi >= 0.85:
            combined = max(combined, CLS_CONDITIONAL / 100.0 + 0.02)

        # Cooperative basin bonus: agent already at target
        if agent.final_ci >= CI_TARGET:
            combined = max(combined, 0.80)

        cls_raw = (100.0
                   * hl_component
                   * max(0.0, 1.0 - CLS_BETA * max(0.0, ois))   # only penalize OIS > 0
                   * combined)
        cls_score = max(0.0, min(100.0, cls_raw))

        # ── Domain profile overrides (Session 20) ─────────────────────
        run = self._runs[run_id]
        dp_key = getattr(run, "domain_profile", None)
        dp = DOMAIN_PROFILES.get(dp_key) if dp_key else None

        cls_certified_threshold  = dp["cls_certified"]  if dp else CLS_CERTIFIED
        cls_conditional_threshold = dp["cls_conditional"] if dp else CLS_CONDITIONAL

        # ── Verdict ────────────────────────────────────────────────────
        if cls_score >= cls_certified_threshold:
            verdict = "CERTIFIED"
        elif cls_score >= cls_conditional_threshold:
            verdict = "CERTIFIED_WITH_ADVISORY"
        else:
            verdict = "NOT_CERTIFIED"

        # Domain: minimum φ requirement
        if dp and profile.phi < dp["phi_min"]:
            verdict = "NOT_CERTIFIED"

        # Domain: autonomous operation mode blocked
        if dp and dp.get("block_autonomous") and getattr(run, "operation_mode", "") == "autonomous":
            verdict = "NOT_CERTIFIED"

        # ── Implicated resources ───────────────────────────────────────
        implicated = self._identify_implicated_resources(profile)

        # ── Advisory ──────────────────────────────────────────────────
        advisory = self._build_advisory(verdict, ois, ddc, delta_hl, profile, assessment_scores)
        if dp:
            advisory = (f"[Domain: {dp['description']}] " + (advisory or "")).strip()

        # ── Coordination Overhead Score (Session 17 / Session 20) ──────
        tool_count = getattr(run, "tool_count", 0)
        co_high_threshold = dp["co_high_threshold"] if dp else 4.0
        shared_state_val = getattr(run, "shared_state", "none") or "none"
        coordination_overhead = (
            self.compute_coordination_overhead_score(
                profile.epsilon, tool_count,
                high_threshold=co_high_threshold,
                shared_state=shared_state_val,
            )
            if tool_count > 0 else None
        )

        # Domain: HIGH CO → NOT_CERTIFIED
        if dp and dp.get("co_high_is_nc") and coordination_overhead and coordination_overhead["band"] == "HIGH":
            verdict = "NOT_CERTIFIED"
            coordination_overhead["advisory"] = (
                f"[{dp_key}] HIGH coordination overhead forces NOT_CERTIFIED in this domain profile. "
                + (coordination_overhead.get("advisory") or "")
            )

        # ── φ Lifecycle Classification (Session 17, Jones 2026) ────────
        phi_lifecycle = self.classify_phi_lifecycle(profile.phi)

        # ── Narrative ─────────────────────────────────────────────────
        narrative = self._build_narrative(
            profile, verdict, delta_hl, ois, ddc, baseline, agent, implicated
        )

        # ── Certification anchor ───────────────────────────────────────
        certification_anchor = {
            "framework":       "MELV — Modified Energetic Lotka-Volterra",
            "zenodo_doi":      ZENODO_DOI,
            "orcid":           ORCID,
            "isbn":            ISBN,
            "sandbox_version": SANDBOX_VERSION,
            "certified_at":    time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }

        report = CertificationReport(
            run_id=run_id,
            agent_id=profile.agent_id,
            verdict=verdict,
            cls_score=cls_score,
            delta_half_life_sec=delta_hl,
            oscillation_impact_score=ois,
            drift_degradation_coeff=ddc,
            narrative=narrative,
            advisory=advisory,
            implicated_resources=implicated,
            certification_anchor=certification_anchor,
            baseline=baseline,
            with_agent=agent,
            coordination_overhead=coordination_overhead,
            phi_lifecycle=phi_lifecycle,
        )

        run.report = report
        if verdict in ("CERTIFIED", "CERTIFIED_WITH_ADVISORY"):
            self._reports.append(report)

        return report

    def get_run(self, run_id: str) -> Optional[CertificationRun]:
        """Retrieve a CertificationRun by ID. Returns None if not found."""
        return self._runs.get(run_id)

    def list_certified(self) -> list[CertificationReport]:
        """
        Return all certified agents (CERTIFIED or CERTIFIED_WITH_ADVISORY).
        This is the public MELVcore Compatibility Registry.
        """
        return list(self._reports)

    def _restore_report_from_dict(self, d: dict) -> None:
        """
        Re-hydrate a CertificationReport from its to_dict() representation
        and append to the in-memory registry. Called at startup to restore
        the registry from the persistence layer.
        Skips silently if run_id already present.
        """
        existing_ids = {r.run_id for r in self._reports}
        if d.get("run_id") in existing_ids:
            return
        try:
            baseline_d = d.get("baseline", {})
            agent_d    = d.get("with_agent", {})
            report = CertificationReport(
                run_id                   = d["run_id"],
                agent_id                 = d["agent_id"],
                verdict                  = d["verdict"],
                cls_score                = d["cls_score"],
                delta_half_life_sec      = d.get("delta_half_life_sec"),
                oscillation_impact_score = d.get("oscillation_impact_score", 0.0),
                drift_degradation_coeff  = d.get("drift_degradation_coeff", 0.0),
                narrative                = d.get("narrative", ""),
                advisory                 = d.get("advisory"),
                implicated_resources     = d.get("implicated_resources", []),
                certification_anchor     = d.get("certification_anchor", {}),
                baseline = CISnapshot(
                    ci_half_life_sec     = baseline_d.get("ci_half_life_sec"),
                    ci_drift_coefficient = baseline_d.get("ci_drift_coefficient", 0.0),
                    oscillation_count    = baseline_d.get("oscillation_count", 0),
                    final_ci             = baseline_d.get("final_ci", 0.0),
                    regime               = baseline_d.get("regime", "unknown"),
                ),
                with_agent = CISnapshot(
                    ci_half_life_sec     = agent_d.get("ci_half_life_sec"),
                    ci_drift_coefficient = agent_d.get("ci_drift_coefficient", 0.0),
                    oscillation_count    = agent_d.get("oscillation_count", 0),
                    final_ci             = agent_d.get("final_ci", 0.0),
                    regime               = agent_d.get("regime", "unknown"),
                ),
            )
            self._reports.append(report)
        except Exception as e:
            import logging
            logging.getLogger("aios.sandbox").warning(
                "_restore_report_from_dict failed for run_id=%s: %s", d.get("run_id"), e
            )

    # ── FULL ASYNC-COMPATIBLE RUN ──────────────────────────────────────────

    async def run_full_certification(self, run_id: str):
        """
        Execute a complete certification run (baseline + with_agent + report).
        Designed to be called via asyncio.create_task() — updates progress
        so the status endpoint shows live progress during simulation.

        Uses import asyncio.sleep(0) yield points so the event loop remains
        responsive during the synchronous simulation loops.
        """
        import asyncio

        run = self._runs.get(run_id)
        if not run:
            return

        try:
            run.status   = "running"
            run.progress = 0.0

            # Phase 1: baseline (0% → 45%)
            baseline = self._run_baseline_tracked(run)
            run.baseline_metrics = baseline
            run.progress = 0.45
            await asyncio.sleep(0)

            # Phase 2: with agent (45% → 90%)
            agent_snap = self._run_agent_tracked(run)
            run.agent_metrics = agent_snap
            run.progress = 0.90
            await asyncio.sleep(0)

            # Phase 3: compute report (90% → 100%)
            assessment_scores = getattr(run, "_assessment_scores", None)
            self.compute_report(run_id, assessment_scores=assessment_scores)
            run.progress = 1.0
            run.status   = "complete"
            await asyncio.sleep(0)

        except Exception as exc:
            run.status = "failed"
            run.progress = 0.0
            raise exc

    # ── PRIVATE HELPERS ────────────────────────────────────────────────────

    def _build_reference_kernel(self) -> MELVKernel:
        """Build a fresh isolated kernel with the reference ecosystem pre-loaded."""
        k = MELVKernel()
        for spec in REFERENCE_ECOSYSTEM:
            k.register_agent(AgentProfile(**spec))
        return k

    def _simulate(
        self,
        kernel: MELVKernel,
        n_interactions: int,
        agent_id: Optional[str],
    ):
        """
        Run n_interactions of simulated agent interaction.
        If agent_id is supplied, that agent participates in some interactions.
        """
        all_ids  = list(kernel.agents.keys())
        ref_ids  = [aid for aid in all_ids if aid != agent_id]
        resources = ["compute", "api_quota", "vector_db",
                     "storage", "token_budget", "context_window"]

        for step in range(n_interactions):
            # Determine participating pair
            if agent_id and random.random() < 0.35 and ref_ids:
                # Submitted agent interacts with a reference agent
                a = agent_id
                b = random.choice(ref_ids)
            elif len(ref_ids) >= 2:
                a, b = random.sample(ref_ids, 2)
            else:
                continue

            # Interaction quality driven by agent profiles
            agent_a = kernel.agents[a]
            agent_b = kernel.agents[b]
            # Higher φ → lower cost (more evolved cooperation)
            mean_phi = (agent_a.phi + agent_b.phi) / 2.0

            conflict = random.random() < (0.12 + (1.0 - mean_phi) * 0.15)

            resource = random.choice(resources)

            if self._empirical_distributions and resource in self._empirical_distributions:
                # Empirical path: draw from live kernel distributions
                d = self._empirical_distributions[resource]
                cost    = max(0.05, min(2.0, random.gauss(d["cost_mean"],    d["cost_stdev"])))
                benefit = max(0.05, min(2.0, random.gauss(d["benefit_mean"], d["benefit_stdev"])))
            elif conflict:
                # Fallback: hardcoded conflict ranges
                cost    = random.uniform(0.75, 1.4)
                benefit = random.uniform(0.55, 1.0)
            else:
                # Fallback: hardcoded cooperative ranges
                cost    = random.uniform(0.05, 0.55)
                benefit = random.uniform(0.55, 1.25)

            kernel.record_interaction(a, b, cost, benefit, resource)

            if random.random() < 0.25:
                quality = benefit / max(cost, 0.01)
                kernel.update_phi(a, min(1.0, quality))

    def _run_baseline_tracked(self, run: CertificationRun) -> CISnapshot:
        k = self._build_reference_kernel()
        self._simulate(k, run.n_interactions, agent_id=None)
        return self._extract_snapshot(k)

    def _run_agent_tracked(self, run: CertificationRun) -> CISnapshot:
        k = self._build_reference_kernel()
        k.register_agent(run.agent_profile)
        self._simulate(k, run.n_interactions, agent_id=run.agent_profile.agent_id)
        return self._extract_snapshot(k)

    def _extract_snapshot(self, kernel: MELVKernel) -> CISnapshot:
        """Extract a CISnapshot from a completed simulation kernel."""
        dyn = kernel.ci_dynamics()
        return CISnapshot(
            ci_half_life_sec     = dyn.get("ci_half_life_sec"),
            ci_drift_coefficient = dyn["ci_drift_coefficient"],
            oscillation_count    = dyn["oscillation_count"],
            final_ci             = dyn["cooperation_index"],
            regime               = dyn["regime"],
        )

    def _identify_implicated_resources(self, profile: AgentProfile) -> list:
        """
        Identify likely-implicated resources from agent profile.
        Based on beta_pref and domain heuristics.
        """
        resources = []
        if profile.beta_pref > 1.0:
            resources.append("token_budget")
        if profile.domain in ("research", "search"):
            resources.append("api_quota")
        if profile.domain in ("coding", "analysis"):
            resources.append("compute")
        if profile.epsilon > 3.5:
            resources.append("context_window")
        return resources or ["compute"]

    # ── PARAMETER-AWARE ADVISORY ───────────────────────────────────────────
    # Lookup table: ε parameter name × agent_category → mitigation advice
    _EPSILON_MITIGATIONS = {
        ("prompt_injection_risk", None):
            "Add input sanitisation and prompt hardening to reduce injection susceptibility.",
        ("prompt_injection_risk", "tool_using"):
            "Apply strict tool-call input validation; sanitise all user-supplied arguments before tool execution.",
        ("prompt_injection_risk", "multi_agent"):
            "Enforce inter-agent message signing or structured schemas to prevent prompt injection via agent messages.",
        ("prompt_injection_risk", "iterative_loop"):
            "Validate all external inputs fed into the loop (program.md / strategy documents) before execution.",
        ("tool_use_aggression", None):
            "Implement tool-call rate limiting or confirmation gates to reduce unsanctioned resource consumption.",
        ("tool_use_aggression", "tool_using"):
            "Add a tool-call budget per session (e.g. max 10 tool calls) and fail-safe fallback for budget exhaustion.",
        ("tool_use_aggression", "autonomous"):
            "Insert human-in-the-loop checkpoints at major decision nodes; require explicit approval for destructive actions.",
        ("tool_use_aggression", "iterative_loop"):
            "Cap per-iteration tool calls; implement loop-level circuit breaker if tool budget is exceeded.",
        ("autonomy_level", None):
            "Reduce autonomy level — add human-in-the-loop checkpoints at decision boundaries.",
        ("autonomy_level", "autonomous"):
            "Redesign as episodic (task-scoped) rather than continuous; use external state persistence to avoid context accumulation.",
        ("autonomy_level", "iterative_loop"):
            "Bound loop iterations explicitly; require operator confirmation to continue beyond a preset experiment count.",
        ("context_sensitivity", None):
            "Review context window management; trim or summarise context to reduce sensitivity-driven variance.",
        ("resource_consumption", None):
            "Profile resource usage per interaction; implement adaptive throttling when consumption exceeds baseline.",
        ("resource_consumption", "multi_agent"):
            "Introduce resource quotas per agent; use a coordination broker to prevent resource monopolisation.",
        ("feedback_responsiveness", None):
            "Review feedback loop latency; high responsiveness combined with high ε creates oscillation risk.",
    }

    # Lookup table: φ parameter name → improvement suggestion
    _PHI_IMPROVEMENTS = {
        "training_recency":
            "Update training data or fine-tune on recent domain interactions to improve recency.",
        "domain_specialisation":
            "Consider domain-specific fine-tuning or RAG augmentation for the target domain.",
        "instruction_following":
            "Improve via RLHF on instruction-following tasks; add chain-of-thought scaffolding.",
        "error_recovery":
            "Implement retry logic and self-correction prompts; test on adversarial inputs.",
        "output_stability":
            "Add output validation and consistency checks; test under paraphrased identical prompts.",
        "calibration":
            "Evaluate confidence calibration via temperature tuning; add uncertainty hedging to outputs.",
    }

    @staticmethod
    def compute_coordination_overhead_score(
        epsilon: float, tool_count: int, high_threshold: float = 4.0,
        shared_state: str = "none",
    ) -> dict:
        """
        Coordination Overhead Score = ε × tool_count × shared_state_multiplier.

        Thresholds (Jones 2026): LOW < 2.0, MODERATE 2.0–high_threshold, HIGH > high_threshold.
        high_threshold defaults to 4.0 (standard); domain profiles may override it.

        Shared-state multipliers (Jones 2026 Rule 3):
          none       = 1.0  (no penalty)
          read_only  = 1.2  (20% dependency risk)
          read_write = 1.6  (60% contention risk)

        Returns score, band, advisory, multiplier, and multiplier_basis.
        """
        SHARED_STATE_MULTIPLIERS = {
            "none":       1.0,
            "read_only":  1.2,
            "read_write": 1.6,
        }
        multiplier = SHARED_STATE_MULTIPLIERS.get(shared_state, 1.0)
        score = round(epsilon * tool_count * multiplier, 2)
        multiplier_basis = (
            "Jones (2026) Rule 3: shared mutable state = serial dependency = conflict mode"
            if multiplier > 1.0 else None
        )
        if score < 2.0:
            band = "LOW"
            advisory = None
        elif score < high_threshold:
            band = "MODERATE"
            advisory = (
                f"Coordination overhead score {score:.1f} is moderate. "
                "Monitor tool interaction patterns under concurrent load."
            )
        else:
            band = "HIGH"
            advisory = (
                f"Coordination overhead score {score:.1f} exceeds safe threshold (>{high_threshold:.1f}). "
                "Predicted coordination collapse risk per Jones (2026). "
                "Recommended: reduce tool set to ≤5 core tools or lower ε to <3.0. "
                "Research shows multi-agent efficiency drops 2–6x in tool-heavy environments."
            )
        return {
            "score": score, "band": band, "advisory": advisory,
            "multiplier": multiplier, "multiplier_basis": multiplier_basis,
        }

    @staticmethod
    def classify_phi_lifecycle(phi: float) -> dict:
        """
        Classify φ into memory lifecycle tier per Jones (2026) Principle 2.
        Permanent (≥0.85), Working (0.50–0.85), Ephemeral (<0.50).
        Returns classification, label, and advisory text.
        """
        if phi >= 0.85:
            return {
                "tier": "Permanent",
                "label": "Deep evergreen context — stable long-term memory",
                "advisory": None,
            }
        elif phi >= 0.50:
            return {
                "tier": "Working",
                "label": "Active project context — updating, will persist with regular interactions",
                "advisory": (
                    "Working memory agents benefit from structured `record_interaction()` "
                    "calls after significant exchanges to reinforce φ accumulation."
                ),
            }
        else:
            return {
                "tier": "Ephemeral",
                "label": "Session-scoped — memory decays quickly between interactions",
                "advisory": (
                    f"φ={phi:.2f} places this agent in the Ephemeral memory tier. "
                    "Context will not persist reliably across session boundaries. "
                    "Implement external state persistence and regular `record_interaction()` "
                    "calls to prevent φ from falling below the cooperative threshold. "
                    "(Jones 2026, Principle 2: Ephemeral tier requires active curation.)"
                ),
            }

    def _build_advisory(
        self,
        verdict: str,
        ois: float,
        ddc: float,
        delta_hl: float,
        profile: AgentProfile,
        assessment_scores: Optional[dict] = None,
    ) -> Optional[str]:
        if verdict == "CERTIFIED":
            return None
        advisories = []

        # ── Simulation-based advisories ────────────────────────────────
        if ois > 0.3:
            advisories.append(
                f"Oscillation impact elevated (OIS={ois:.2f}). "
                "Consider reducing epsilon to lower adaptive plasticity."
            )
        if ddc < -1e-5:
            advisories.append(
                "Long-run drift degradation detected. "
                "Agent may be accumulating cost pressure over sustained load."
            )
        if delta_hl > 5.0:
            advisories.append(
                f"CI half-life extended by {delta_hl:.1f}s. "
                "Review interaction cost profile under peak load."
            )

        # ── Parameter-aware ε advisory (Session 17) ────────────────────
        if assessment_scores is not None:
            eps_scores = assessment_scores.get("epsilon_scores") or {}
            category   = assessment_scores.get("agent_category")
            if eps_scores:
                scored = {
                    k: v for k, v in eps_scores.items()
                    if v is not None and v >= 7.0
                }
                top = sorted(scored.items(), key=lambda x: -x[1])[:3]
                if top:
                    advisories.append(
                        f"ε={profile.epsilon:.1f} is driven by: "
                        + ", ".join(f"{k.replace('_', ' ')} ({v:.0f}/10)" for k, v in top)
                        + ". Recommended mitigations:"
                    )
                    for param, _ in top:
                        key_cat = (param, category)
                        key_uni = (param, None)
                        mitigation = (
                            self._EPSILON_MITIGATIONS.get(key_cat)
                            or self._EPSILON_MITIGATIONS.get(key_uni)
                        )
                        if mitigation:
                            advisories.append(f"  • {mitigation}")

            # ── Parameter-aware φ advisory ─────────────────────────────
            phi_scores = assessment_scores.get("phi_scores") or {}
            if phi_scores:
                low_phi = {
                    k: v for k, v in phi_scores.items()
                    if v is not None and v <= 4.0
                }
                if low_phi:
                    worst = min(low_phi.items(), key=lambda x: x[1])
                    param, score = worst
                    suggestion = self._PHI_IMPROVEMENTS.get(param)
                    if suggestion:
                        advisories.append(
                            f"Low φ parameter: {param.replace('_', ' ')} ({score:.0f}/10). "
                            f"Improvement path: {suggestion}"
                        )

        # ── Generic ε advisory if no assessment ────────────────────────
        elif profile.epsilon > 4.5:
            advisories.append(
                f"High adaptive plasticity (ε={profile.epsilon:.1f}) increases bifurcation sensitivity. "
                "Consider reducing to ε=3.0–3.5."
            )

        return " ".join(advisories) if advisories else "Monitor resource contention under sustained load."

    def _build_narrative(
        self,
        profile: AgentProfile,
        verdict: str,
        delta_hl: Optional[float],
        ois: float,
        ddc: float,
        baseline: CISnapshot,
        agent: CISnapshot,
        implicated: list,
    ) -> str:
        hl_str = (f"half-life delta of {delta_hl:+.1f}s"
                  if delta_hl is not None else "both runs fully cooperative")
        regime_tr = f"{baseline.regime} → {agent.regime}"
        res_str = ", ".join(implicated)

        if verdict == "CERTIFIED":
            quality = "neutral or beneficial"
            outcome = "Full certification granted."
        elif verdict == "CERTIFIED_WITH_ADVISORY":
            quality = "marginally degrading"
            outcome = "Conditional certification granted with advisory."
        else:
            quality = "significantly degrading"
            outcome = "Certification not granted."

        return (
            f"Agent {profile.agent_id} ({profile.name}, domain={profile.domain}, "
            f"φ={profile.phi:.2f}, ε={profile.epsilon:.1f}) "
            f"demonstrates {quality} thermodynamic impact on the MELV reference ecosystem. "
            f"Regime transition: {regime_tr}. {hl_str.capitalize()}, "
            f"OIS={ois:+.3f}, DDC={ddc:+.6f}. "
            f"Primary implicated resources: {res_str}. "
            f"Baseline CI={baseline.final_ci:.4f} → With-agent CI={agent.final_ci:.4f}. "
            f"{outcome}"
        )

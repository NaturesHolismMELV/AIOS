"""
MELVcore Engine
===============
Thermodynamic governance kernel for agent ecosystems.
Based on the Modified Energetic Lotka-Volterra (MELV) framework.
Blueprint for Harmony — L.W. Evans | ORCID: 0009-0001-0963-1840 | Cape Town, South Africa
ORCID: 0009-0001-0963-1840

Core principle: cooperation emerges thermodynamically when βi < 1.0

══════════════════════════════════════════════════════════════════
CANONICAL VARIABLE DEFINITIONS — DO NOT DEVIATE
══════════════════════════════════════════════════════════════════

  φ (phi)   EVOLUTIONARY MATURITY / FITNESS — INTERNAL to the agent.
            Increases as the agent matures, specialises, or adapts.
            The giraffe's long neck raises its φ.
            φ is NOT set by the environment. φ is NOT the same as β.
            Range: [0.0, 1.0]

  β (beta)  ENVIRONMENTAL SUITABILITY — EXTERNAL. Set by the
            environment configuration, NEVER by the agent.
            The acacia crown niche has high β because it is rich and
            uncontested. β is NOT something the agent increases.
            Range: [0.1, 3.0] (kernel-managed)

  i         INTERACTION COST RATIO — computed from C (cost) and B
            (benefit) of an interaction between two agents.
            i = C_AB / B_AB
            NOT a property of a single agent. NOT the same as φ or β.

  CI        COOPERATION INDEX — system-level measure of cooperative
            equilibrium. CI = 1 - mean(βi) across recent interactions.
            Target: CI > 0.75 (ecosystem in cooperative basin).
            NOT an agent property. NOT the same as φ.

CORRECT niche divergence:  agent.phi += niche_divergence_benefit
                           environment.beta['niche'] = 0.95
WRONG  niche divergence:   agent.beta += niche_divergence_bonus  ← NEVER

GATEWAY API ENFORCEMENT:
  The Gateway API at POST /melv/interact enforces these definitions.
  Any payload attempting to set β from the agent side is rejected HTTP 422.
  Agents report phi (φ) only. The kernel reads β from BetaEnvironment.

══════════════════════════════════════════════════════════════════
"""

import hashlib
import math
import time
import random
import json
import numpy as np
from dataclasses import dataclass, field, asdict
from typing import Optional
from enum import Enum

# ── SIGMOID QUORUM GATE CONSTANTS (Session 25 — ABM V2.1 verified ③) ──────
# Quorum sensing correspondence (MAIES Event 2 — Gemini, Nadell 2016):
#   MELV: f(φ·β) = 1/(1+exp(-k(φ·β − τ)))  ≡  P(coop) = 1/(1+exp(-k(N − N_thresh)))
# DO NOT CHANGE τ or k without a new ABM run (sensitivity=1.0, specificity=0.997)
QUORUM_TAU           = 0.5    # τ — sigmoid inflection point (ABM V2.1 φ×β boundary)
QUORUM_K             = 10.0   # k — sharpness (ABM V2.1 optimised)
PROVISION_STEP_FLOOR = 0.05   # minimum PROVISION_BETA step (healthy ecosystem)
PROVISION_STEP_CEIL  = 0.50   # maximum PROVISION_BETA step (stressed ecosystem)

# ── ε DECOMPOSITION CONSTANTS (Session 26 — v2.2.0) ───────────────────────
# ε_effective = ε_intrinsic + ε_ecosystem          [MASTER EQUATION — unchanged]
# ε_intrinsic  — agent-side plasticity (learned, heritable, domain-specific)
# ε_ecosystem  — infrastructure friction (tool latency, API limits, etc.)
#                  [formerly ε_environmental; renamed Session 29 — alias preserved]
# ε_architectural — boundary condition (NOT in master equation; diagnostic only)
#                   Session 29 addition. Computed once at registration.
#
# TOOL FRICTION WEIGHTS: relative cost of each resource type per interaction.
# Derived from the BetaEnvironment resource ordering and typical AI agent
# deployment profiles. ② theoretically grounded — not yet empirically calibrated.
TOOL_FRICTION_WEIGHTS: dict = {
    "compute":        1.0,   # baseline
    "api_quota":      1.4,   # external API calls: higher latency variance
    "vector_db":      1.2,   # retrieval overhead
    "storage":        0.8,   # lowest friction (async I/O)
    "token_budget":   1.3,   # LLM token pressure amplifies plasticity cost
    "context_window": 1.1,   # context management overhead
}

# ── ε_architectural CONSTANTS (Session 29 — v2.5.0) ──────────────────────
# MAIES Event 5: Grok derived three-scalar formulation from thermodynamic
# first principles. Gemini independently confirmed. Biological grounding:
# Oxpecker-Giraffe mutualism (L.W. Evans).
#
# Tool category weights for Approach A (architectural friction).
# These weight the FIXED structural characteristics of the agent's tool set,
# not the current runtime β values. High weight = more architectural friction.
#   agent-native:       0.2  — internal calls, minimal boundary crossing
#   fast_rest:          0.5  — standard REST, low latency
#   standard:           1.0  — baseline tool category
#   human_bottlenecked: 1.5  — human-in-loop, approval gates
#   legacy:             2.0  — legacy systems, slow boundaries, interrupted state
#
# From Session 26 Brief, confirmed in MAIES Event 5 (Grok second response).
ARCH_CATEGORY_WEIGHTS: dict = {
    "agent_native":        0.2,
    "fast_rest":           0.5,
    "standard":            1.0,
    "human_bottlenecked":  1.5,
    "legacy":              2.0,
}

# Policy threshold: when ε_architectural exceeds this, cap β provisioning
# and trigger architectural recommendation (Grok derivation, MAIES Event 5).
# ③ theoretical — confirmed by biological derivation and MAIES Event 5.
ARCH_RECOMMENDATION_THRESHOLD = 3.0   # ε_architectural > 3.0 → arch recommendation
ARCH_BETA_MULTIPLIER_CAP      = 0.6   # max β provisioning multiplier when arch is high

# OXPECKER agent archetype values (Grok second response, MAIES Event 5).
# High ε_architectural because recycling touches interrupted state, slow
# boundaries, and human-in-loop checkpoints.
OXPECKER_ARCH_EPSILON_LOW  = 2.8
OXPECKER_ARCH_EPSILON_HIGH = 3.5
OXPECKER_ECOSYSTEM_WEIGHT  = 0.5   # confirmed: fast Haiku call, biological derivation

# ── ε_intrinsic PER-AGENT VARIANCE (Session 30 — v2.6.0) ─────────────────
# Individual agents of the same type vary around their type mean just as
# conspecifics vary in adaptive plasticity. The type default is the species
# mean; the agent_id hash seeds a reproducible Gaussian perturbation.
#
# Design principles:
#   - Deterministic: same agent_id always → same ε_intrinsic across restarts
#   - ε is a structural property of the agent's architecture, not a fresh
#     random draw. The hash seed preserves this across Railway restarts.
#   - Sigma = 0.3: meaningful spread (±0.3) without swamping type signal
#
# Epistemic status: ② theoretical — sigma principled, not calibrated against
# empirical agent performance distributions.
EPSILON_VARIANCE_SIGMA   = 0.3    # Gaussian σ around type mean
EPSILON_VARIANCE_FLOOR   = 0.5    # minimum ε_intrinsic (no agent is fully rigid)
EPSILON_VARIANCE_CEILING = 7.5    # maximum ε_intrinsic (headroom below 8.0)

# Type-default ε_intrinsic (species means). Per-agent values perturb around these.
# Source: agent assessment calibration across Sessions 16–29.
EPSILON_TYPE_DEFAULTS: dict = {
    "RESEARCH":  3.2,
    "ANALYSIS":  5.5,
    "WRITER":    2.4,
    "CODE":      3.0,
    "MONITOR":   2.5,
    "PLANNER":   1.8,
    "DATA":      5.2,
    "SEARCH":    4.5,
    "OXPECKER":  1.5,
}


def _perturbed_epsilon(agent_id: str, base_epsilon: float) -> float:
    """
    Return a deterministic per-agent ε_intrinsic perturbed around base_epsilon.

    Seeds a Gaussian from the MD5 of agent_id — reproducible across restarts.
    If agent_id is empty, returns base_epsilon unperturbed (backward compat).

    Session 30 (v2.6.0). Epistemic status: ② theoretical.
    """
    if not agent_id:
        return round(base_epsilon, 4)
    seed = int(hashlib.md5(agent_id.encode("utf-8")).hexdigest()[:8], 16)
    rng  = random.Random(seed)
    perturbed = rng.gauss(base_epsilon, EPSILON_VARIANCE_SIGMA)
    clamped   = max(EPSILON_VARIANCE_FLOOR, min(EPSILON_VARIANCE_CEILING, perturbed))
    return round(clamped, 4)


# Speed-to-Cooperation (STC) normalisation: seconds at which STC = 1.0
# A freshly registered agent with mean ε in a calibrated sandbox reaches
# CI_TARGET in ≈ STC_REFERENCE_SECONDS in the reference environment.
# ② theoretical — will be updated when Session 23 empirical data warrants.
STC_REFERENCE_SECONDS = 120.0

# Diagnosis thresholds (② theoretical, principled)
VOLATILE_EPSILON_THRESHOLD   = 6.0   # ε_intrinsic ≥ 6.0 (necessary but not sufficient)
ENV_BOTTLENECK_THRESHOLD     = 1.5   # ε_ecosystem ≥ 1.5 → ENV_BOTTLENECKED
LEGACY_PHI_THRESHOLD         = 0.35  # retained for backward compat — see RANGE_MISMATCH
LEGACY_EPSILON_THRESHOLD     = 4.0   # retained for backward compat — see RANGE_MISMATCH

# Session 30c (v2.7.0): ε semantic realignment
# AGENT_VOLATILE fires only when ε_intrinsic exceeds what φ and β can support.
# High ε alone is adaptive range (an asset), not volatility (a liability).
# Mismatch = high adaptive range in an environment that cannot support it.
VOLATILE_PHI_CEILING         = 0.65  # φ below this = niche not mature enough to support high ε
VOLATILE_BETA_CEILING        = 1.0   # β below this = environment not rich enough to support high ε

# RANGE_MISMATCH: replaces LEGACY_CANDIDATE with correct framing.
# Low φ + high ε = high adaptive range in an immature niche. Not legacy — developing.
RANGE_MISMATCH_PHI_CEILING   = 0.35  # same trigger as legacy: φ ≤ 0.35
RANGE_MISMATCH_EPS_FLOOR     = 4.0   # ε_effective ≥ 4.0

# STC support factor: high φ × β relative to ε_effective reduces STC.
# A high-ε agent in a supportive environment converges to cooperation quickly.
STC_SUPPORT_REFERENCE        = 0.5   # φ × β / ε_effective at which full support applies (② theoretical)
STC_SUPPORT_REDUCTION        = 0.5   # maximum STC reduction when fully supported

# Mismatch fraction threshold for dominant_bottleneck = "mismatch"
MISMATCH_DOMINANT_THRESHOLD  = 0.25  # >25% agents mismatched → ecosystem bottleneck is mismatch

# ── β_norm CORRECTION (Session 35 — v3.1.0) ──────────────────────────────
# MAIES Form C correction C1. Canonical reference v1.2.
#
# The cooperation-evolution equation uses β_norm = β/(1+β) ∈ (0,1),
# not raw β ∈ [0.1, 3.0]. Raw β is retained for:
#   - Gateway condition R = C/B (unchanged)
#   - Quorum gate φ·β (unchanged — ABM V2.1 calibrated on raw β)
#   - BetaEnvironment provisioning (unchanged)
#
# Only the master equation term uses β_norm.
#
# Epistemic status: ③ canonical correction (MAIES Form C, nine-system
# convergence, canonical document v1.2, Session 35).

def _beta_norm(beta_raw: float) -> float:
    """
    β_norm = β/(1+β) ∈ (0,1).

    Maps raw β from [0.1, 3.0] to (0,1):
      β=0.1 → β_norm≈0.091
      β=0.5 → β_norm≈0.333
      β=1.0 → β_norm=0.500
      β=2.0 → β_norm≈0.667
      β=3.0 → β_norm=0.750

    Used ONLY in the master equation i(t) = i₀ × (1 − ε × φ(t) × β_norm(t)).
    NOT used in quorum gate, gateway condition, or β provisioning.

    MAIES Form C correction C1. Canonical reference v1.5.6 Part II.
    Session 35 (v3.1.0). Epistemic status: ③.
    """
    return beta_raw / (1.0 + beta_raw)


def _compute_i_inf(
    i0:       float,
    eta:      float,
    epsilon:  float,
    beta_raw: float,
) -> float:
    """
    Compute i_∞ — the cooperation-evolution parameter function at φ=1.

    i_∞ = i₀ × (1 − η × tanh(ε × β_norm / η))

    where β_norm = β/(1+β) ∈ (0,1)
          φ = 1  (parameter function evaluated at full maturity)

    This is a PARAMETER FUNCTION, not the current i(t).
    It characterises the long-run equilibrium the system converges toward.
    Used in the canonical gate condition β×i_∞ < 1 and the stagnation
    detector Δ_gate = β×i_∞ − 1.

    Parameters
    ----------
    i0       : baseline interaction cost ratio (AIOS: 1.0 normalised)
    eta      : saturation capacity η ∈ (0, 1]; use ETA_CANONICAL_DEFAULT
               if L3 posterior unavailable
    epsilon  : ε_effective — adaptive plasticity
    beta_raw : raw β ∈ [0.1, 3.0] (from BetaEnvironment or observe())

    Canonical Reference: v1.5.6 Equation 7, FB-ABM V1.0 confirmed June 2026.
    Epistemic status: ③ (Jacobian-derived + FB-ABM V1.0 confirmed).
    """
    eta_safe = max(0.01, min(1.0, eta))
    beta_n   = _beta_norm(beta_raw)
    arg      = epsilon * beta_n / eta_safe
    return i0 * (1.0 - eta_safe * math.tanh(arg))


# Computational floor for i(t) — guard against unbounded negative values
# for high-ε agents pending full tanh saturation form (Session 36 scope).
I_FLOOR = -5.0   # i(t) clamped to [I_FLOOR, 1.0] — ② theoretical

# ── EQUATION 7 φ DYNAMICS CONSTANTS (Session 35 — v3.1.0) ────────────────
# Canonical reference v1.5.6 Part IV Equation 7 (FB-ABM V1.0 confirmed June 2026).
#
# CANONICAL FORM (alignment pass — v3.3.0):
#   dφ/dt = α(1−φ) × H(1−β×i_∞) × max(0,1−i(t))   [BUILD]
#           − δ×D(t)×φ × H(β×i_∞−1)                 [DECAY]
#
# where i_∞ = i₀ × (1 − η × tanh(ε × β_norm / η))   [parameter function at φ=1]
#       β_norm = β/(1+β)
#       D(t) = max(0, ΔC/C_base + ΔTAX/TAX_base)
#       α=0.01, δ=0.10 (α ≪ δ — 10× asymmetry)
#
# Gate hierarchy:
#   Tier 1 — CANONICAL: H(1−β×i_∞) / H(β×i_∞−1)  — Jacobian-derived,
#             FB-ABM V1.0 confirmed. Governs all structural φ decisions.
#   Tier 2 — EMPIRICAL PROXY: φ·β > 0.3  — ABM V2.1 binary classifier
#             (405 runs, sensitivity=1.0, specificity=0.997).
#             Diagnostic only. NOT used in φ dynamics.
#   Tier 3 — ILLUSTRATIVE/LEGACY: τ=0.5 sigmoid (_quorum_gate())
#             Retained for soft agent routing and provisioning step scaling
#             ONLY. NOT used in structural gate decisions.
#
# PHI_GATEWAY_THRESHOLD = 0.50 — RETIRED from structural gate role.
#   Retained as exported constant for backward compatibility only.
#   The R < 0.50 scalar gate is superseded by the Jacobian-derived β×i_∞=1
#   boundary surface. See carryover §1 "What is settled" — C5 finding.
#   DO NOT use PHI_GATEWAY_THRESHOLD in any new structural gate logic.
#
# α ≪ δ: φ decays faster than it builds (asymmetry confirmed ABM V2.1 T1.5).
# Epistemic status: ③ (Jacobian-derived; FB-ABM V1.0 confirmed June 2026).

PHI_BUILD_RATE_ALPHA  = 0.01   # α — φ build rate; ABM T1.5 confirmed
PHI_DECAY_RATE_DELTA  = 0.10   # δ — φ decay rate; δ ≫ α (10× asymmetry)
PHI_GATEWAY_THRESHOLD = 0.50   # RETIRED — backward compat only; see note above

# ── CANONICAL GATE PARAMETERS (Alignment pass — v3.3.0) ──────────────────
# i₀ — baseline interaction cost ratio (AIOS: normalised to 1.0 per BI-NLS).
#   Canonical domain: i₀ > 1. The BI-NLS telemetry uses i₀=1.0 (normalised).
# η  — saturation capacity (bee-flower calibration: 0.93).
#   Overridden per-agent by BI-NLS L3 posterior once ≥100 L1 records exist.
I0_CANONICAL          = 1.0    # i₀ normalised baseline (see telemetry.py BI-NLS)
ETA_CANONICAL_DEFAULT = 0.93   # η default; bee-flower calibration (canonical Ref v1.5.6)

# ── SESSION 37 — DUNGBEETLE + IRREVERSIBILITY CONSTANTS (v3.2.0) ─────────
# Canonical reference v1.2 Part VI Items 7, 11, 12.
#
# Dungbeetle condition (MAIES Form C, May 2026):
#   Node v is a Dungbeetle iff:
#     beta_service(Omega) >= 0.5  AND  beta_service(Omega_{-v}) < 0.5
#   Sensitivity score: S_v = beta_service(Omega) - beta_service(Omega_{-v})
#   Epistemic status: theoretical (threshold 0.5 = PHI_GATEWAY_THRESHOLD,
#   consistent with Nadell et al. quorum analogy; empirical cal. pending).
#
# Irreversibility boundary:
#   phi_viable ~= 1 - 1/(epsilon x beta_norm x eta)
#   Three governance zones:
#     VIABLE            : phi > phi_viable
#     RECOVERABLE_URGENT: phi_irrev <= phi <= phi_viable
#     IRREVERSIBLE      : phi < phi_irrev
#   T_rec = (1/alpha) x ln((1-phi_current)/(1-phi_viable)) / f_eligible
#   Epistemic status: theoretical (ABM Test Suite 3 validation pending).

DUNGBEETLE_THRESHOLD = 0.50   # beta_service(Omega) threshold for Dungbeetle
PHI_IRREV_DEFAULT    = 0.10   # phi below which recovery is operationally irreversible
T_GOV_DEFAULT        = 100.0  # default governance horizon (time units)
                               # for phi_irrev = 1 - exp(-alpha x T_gov)


@dataclass
class EpsilonProfile:
    """
    Session 29 (v2.5.0): Three-scalar ε profile for a single agent assessment.

    MASTER EQUATION (unchanged):
      ε_effective = ε_intrinsic + ε_ecosystem

    Three scalars:
      ε_intrinsic     [0–8]   — agent learned plasticity (per-agent, persistent)
      ε_ecosystem     [0,∞)   — Approach B, live: infrastructure friction
                                 (formerly ε_environmental; alias preserved)
      ε_architectural [0,∞)   — Approach A, STATIC: boundary condition only.
                                 NOT in master equation. Computed once at
                                 registration from tool category weights.
                                 When high + CI low → architectural recommendation.

    MAIES Event 5 (Grok): ε_architectural never enters the master equation.
    It is a fixed thermal resistance. Provisioning β is futile against it.

    Diagnosis badges (mutually non-exclusive):
      AGENT_VOLATILE      — ε_intrinsic ≥ 6.0 AND φ < 0.65 AND β < 1.0:
                            adaptive range exceeds niche support (mismatch, not pathology)
      ENV_BOTTLENECKED    — ε_ecosystem ≥ 1.5: infrastructure friction amplifying cost
      RANGE_MISMATCH      — φ ≤ 0.35 AND ε_effective ≥ 4.0: high adaptive range in
                            immature niche — developing agent, not legacy (Session 30c)
      ARCH_BOUNDARY_HIGH  — ε_architectural > 3.0: β provisioning capped; arch
                            recommendation fired instead

    STC (Speed-to-Cooperation):
      Estimated time for this agent to reach CI_TARGET in current environment.
      STC = STC_REFERENCE_SECONDS × (ε_effective / EPSILON_REFERENCE) × (1 / β_mean)
    """
    agent_id:               str
    epsilon_intrinsic:      float   # agent-side component
    epsilon_ecosystem:      float   # infrastructure-side component (live)
    epsilon_architectural:  float   # boundary condition (static, diagnostic only)
    epsilon_effective:      float   # ε_intrinsic + ε_ecosystem (master equation)
    phi:                    float   # current φ at time of assessment
    beta_mean:              float   # mean β across all resources
    stc_seconds:            float   # Speed-to-Cooperation estimate
    badges:                 list    # diagnosis badges
    resource_friction:      dict    # per-resource friction breakdown
    interpretation:         str     # plain-language summary
    architectural_recommendation: Optional[str] = None  # fires when arch > threshold + CI low

    # Backward-compat alias: epsilon_environmental → epsilon_ecosystem
    @property
    def epsilon_environmental(self) -> float:
        """Backward-compatible alias for epsilon_ecosystem (Session 26 consumers)."""
        return self.epsilon_ecosystem

# ── CI DYNAMICS CONSTANTS ──────────────────────────────────────────────────
CI_TARGET            = 0.75      # Cooperative basin threshold
CI_HISTORY_MAX       = 1000      # Maximum timestamped CI readings to retain
CI_DCIDT_WINDOW      = 10        # Rolling window for dCI/dt computation (readings)
CI_DRIFT_WINDOW      = 500       # Long-run drift window (readings)
CI_OSCILLATION_WINDOW = 60.0     # Seconds: oscillation detection look-back
CI_OSCILLATION_MIN_AMPLITUDE = 0.05  # Minimum CI drop to register as oscillation

# ── tanh φ UPDATE CONSTANTS (Session 10 — DeepSeek independent derivation) ─
# Axiom 3: φ changes on a SLOW timescale relative to individual interactions.
# Axiom 8: heterogeneity maintained by Gaussian noise.
# dφ/dt = (1/τ_φ) · [tanh(γ · mean_surplus) − φ] + η_φ(t)
TAU_PHI_FACTOR = 0.01    # τ_φ factor — keeps φ slow relative to interactions (Axiom 3)
PHI_GAIN       = 2.0     # γ  — tanh sensitivity to surplus
WINDOW_SIZE    = 10      # δ  — surplus memory window (interactions)
NOISE_SIGMA    = 0.002   # η_φ — Axiom 8 heterogeneity noise


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
    preferred_resource: Optional[str] = None   # Session 22: niche routing tag
    surplus_window: list = field(default_factory=list)  # Session 22: φ persistence fix

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
    resource_type: str = "compute"  # Session 23: resource type for empirical calibration
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
            "resource_type":    self.resource_type,
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
class OscillationEvent:
    """
    Recorded when CI crosses the 0.75 cooperative target and then
    falls back below it within the oscillation detection window.

    Characterises whether the ecosystem exhibits damped, undamped,
    or divergent behaviour around the cooperative attractor.
    """
    event_id:      str
    ci_peak:       float          # Highest CI reading in the crossing
    ci_trough:     float          # CI value at fall-back detection
    amplitude:     float          # ci_peak - ci_trough
    period_sec:    float          # Time from peak to trough (seconds)
    timestamp:     float = field(default_factory=time.time)
    implicated_pairs: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "event_id":         self.event_id,
            "ci_peak":          round(self.ci_peak, 4),
            "ci_trough":        round(self.ci_trough, 4),
            "amplitude":        round(self.amplitude, 4),
            "period_sec":       round(self.period_sec, 2),
            "timestamp":        self.timestamp,
            "implicated_pairs": self.implicated_pairs,
        }


@dataclass
class BetaEnvironment:
    """
    Environmental suitability (β) for different resource types.
    β modulates how effectively agents can interact.
    High β = abundant niches = lower effective i-factor.
    """
    compute:        float = 1.0   # CPU/GPU availability
    api_quota:      float = 0.9   # External API bandwidth
    vector_db:      float = 1.2   # Vector DB I/O
    storage:        float = 0.8   # File/blob storage
    token_budget:   float = 1.1   # LLM token allocation
    context_window: float = 1.0   # Per-call context capacity (distinct from token budget)

    def get(self, resource: str) -> float:
        return getattr(self, resource, 1.0)

    def set(self, resource: str, value: float):
        """Set β for a resource type — called only by the kernel (oxpecker, provision_beta)."""
        if hasattr(self, resource):
            setattr(self, resource, max(0.1, min(3.0, value)))

    def mean(self) -> float:
        vals = [self.compute, self.api_quota, self.vector_db,
                self.storage, self.token_budget, self.context_window]
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

    def __init__(self, persistence=None):
        self.agents:       dict[str, AgentProfile]    = {}
        self.interactions: list[InteractionRecord]     = []
        self.events:       list[BifurcationEvent]      = []
        self.beta:         BetaEnvironment             = BetaEnvironment()
        self._event_counter = 0
        # Session 7: contention depth per agent pair key "agentA::agentB"
        # Increments on each threshold/conflict event, resets on cooperative
        self._contention_depth: dict[str, int] = {}
        # Session 9: CI Dynamics — timestamped history and oscillation tracking
        self._ci_history: list[tuple[float, float]] = []  # (timestamp, ci_value)
        self._osc_counter: int = 0
        self.oscillation_events: list[OscillationEvent] = []
        self._osc_above_target: bool = False   # True once CI has crossed 0.75
        self._osc_peak_ci:  float = 0.0
        self._osc_peak_ts:  float = 0.0
        # Session 12: optional persistence store (injected from server.py)
        self._persistence = persistence

    def _pair_key(self, agent_a: str, agent_b: str) -> str:
        """Canonical key for an agent pair (order-normalised)."""
        return "::".join(sorted([agent_a, agent_b]))

    def get_contention_depth(self, agent_a: str, agent_b: str) -> int:
        """Return current contention depth for an agent pair."""
        return self._contention_depth.get(self._pair_key(agent_a, agent_b), 0)

    def get_pair_pattern(
        self,
        agent_a: str,
        agent_b: str,
        short_window: int = 20,
    ) -> dict:
        """
        Session 22 Fix A1 — Query governance history for a specific agent pair.

        Two horizons:
          short — last `short_window` in-memory BifurcationEvents
          long  — full persistence history for the pair (if persistence attached)

        Returns a structured dict used by _kernel_respond() to decide
        intervention strength.
        """
        from collections import Counter

        key = self._pair_key(agent_a, agent_b)

        # Short horizon: scan in-memory events
        short = [
            e for e in self.events[-short_window:]
            if self._pair_key(e.agent_a, e.agent_b) == key
        ]

        # Long horizon: persistence query
        long = []
        if self._persistence:
            long = self._persistence.load_pair_events(agent_a, agent_b)

        def dominant(events):
            if not events:
                return None
            actions = [
                e.action.value if hasattr(e, 'action') else e.get('action')
                for e in events
            ]
            actions = [a for a in actions if a is not None]
            if not actions:
                return None
            return Counter(actions).most_common(1)[0][0]

        def conflict_rate(events):
            if not events:
                return 0.0
            conflicts = sum(
                1 for e in events
                if (e.beta_i_pre if hasattr(e, 'beta_i_pre')
                    else e.get('beta_i_pre', 0)) >= 1.0
            )
            return conflicts / len(events)

        # escalation_needed: this would be the 3rd+ consecutive same-action event
        # get_pair_pattern is called BEFORE the new event is appended, so we check
        # whether the last 2 in-memory events are the same action (making the current
        # the 3rd consecutive matching event).
        escalation_needed = False
        if len(short) >= 2:
            last_two_actions = [
                e.action.value if hasattr(e, 'action') else e.get('action')
                for e in short[-2:]
            ]
            escalation_needed = len(set(last_two_actions)) == 1

        return {
            'short_event_count':         len(short),
            'long_event_count':          len(long),
            'dominant_action_short':     dominant(short),
            'dominant_action_long':      dominant(long),
            'escalation_needed':         escalation_needed,
            'structurally_incompatible': (
                len(long) >= 10 and conflict_rate(long) > 0.70
            ),
            'situational': (
                len(long) >= 10 and conflict_rate(long) < 0.30
            ),
        }

    # ── AGENT MANAGEMENT ──────────────────────────────────────────────────

    def register_agent(self, profile: AgentProfile) -> AgentProfile:
        """Register a new agent in the ecosystem."""
        self.agents[profile.agent_id] = profile
        if self._persistence:
            self._persistence.save_agent(profile)
        return profile

    def unregister_agent(self, agent_id: str) -> bool:
        """
        Remove an agent from the ecosystem and delete from persistence.
        Returns True if found and removed, False if not found.
        Used by demo_router to clean up stress agents after demo sessions.
        Session 30b — demo agent lifecycle management.
        """
        if agent_id not in self.agents:
            return False
        del self.agents[agent_id]
        if self._persistence:
            self._persistence.delete_agent(agent_id)
        return True

    def get_agent(self, agent_id: str) -> Optional[AgentProfile]:
        return self.agents.get(agent_id)

    def update_phi(self, agent_id: str, outcome_quality: float):
        """
        Update agent's φ (evolutionary maturity) based on task outcome.

        Session 10 — tanh φ Enhancement (DeepSeek independent derivation):
          dφ/dt = (1/τ_φ) · [tanh(γ · mean_surplus) − φ] + η_φ(t)

        Three theoretical advantages over the previous linear update:
          1. Natural boundedness in [0,1] — saturation is intrinsic, not clamped
          2. Diminishing returns at high maturity (giraffe constraint, Axiom 3)
          3. Surplus memory window — φ responds to patterns, not single outcomes

        Backward compatibility: explicit max/min clamp retained so existing
        threshold assertions (MATURING → ACTIVE at φ ≥ 0.75) are unaffected.
        """
        agent = self.agents.get(agent_id)
        if not agent:
            return

        # Update surplus memory window
        surplus = outcome_quality - 0.5          # centred: positive = good outcome
        agent.surplus_window.append(surplus)
        if len(agent.surplus_window) > WINDOW_SIZE:
            agent.surplus_window = agent.surplus_window[-WINDOW_SIZE:]

        mean_surplus = sum(agent.surplus_window) / len(agent.surplus_window)

        # tanh target — shifted from (-1,1) to (0,1) by (tanh + 1) / 2
        phi_target = (math.tanh(PHI_GAIN * mean_surplus) + 1.0) / 2.0

        # Slow relaxation toward target (Axiom 3: τ_φ >> τ_interaction)
        delta = TAU_PHI_FACTOR * (phi_target - agent.phi)
        noise = random.gauss(0, NOISE_SIGMA)   # Axiom 8: heterogeneity

        agent.phi = max(0.0, min(1.0, agent.phi + delta + noise))
        agent.task_count += 1

        # Update success rate rolling average
        agent.success_rate = (
            (agent.success_rate * (agent.task_count - 1) + outcome_quality)
            / agent.task_count
        )

        # Promote status based on maturity (thresholds unchanged)
        if agent.phi >= 0.75 and agent.status == AgentStatus.MATURING:
            agent.status = AgentStatus.ACTIVE
        elif agent.phi >= 0.90:
            agent.status = AgentStatus.ACTIVE

        # Session 22 Fix C2: periodic flush — max drift between flushes ≈ 0.10
        if self._persistence and agent.task_count % 10 == 0:
            self._persistence.save_agent(agent)

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
            beta=beta,
            resource_type=resource_type,
        )
        self.interactions.append(record)
        if self._persistence:
            self._persistence.save_interaction(record)
        # Session 9: update CI history after every interaction
        self._record_ci_snapshot()

        # Session 7: track contention depth per agent pair
        pair_key = self._pair_key(agent_a, agent_b)
        if record.interaction_type in (InteractionType.THRESHOLD, InteractionType.CONFLICT):
            self._contention_depth[pair_key] = self._contention_depth.get(pair_key, 0) + 1
            self._kernel_respond(record, resource_type=resource_type)
        else:
            # Cooperative: reset contention depth for this pair
            self._contention_depth[pair_key] = 0

        # Session 37: L1 telemetry — wire cost/benefit into SQLite for BI-NLS η estimation
        try:
            import os as _os
            from core.telemetry import AIOSTelemetry as _Tel, L1Record as _L1
            _db = _os.environ.get(
                "AIOS_DB_PATH",
                _os.path.join(
                    _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))),
                    "aios_state.db",
                ),
            )
            _tel = _Tel(_db)
            _tel.log_l1(_L1(agent_id=agent_a, c_proxy=cost, b_proxy=benefit, tax_proxy=0.0))
            _tel.close()
        except Exception:
            pass  # telemetry non-critical

        return record

    def _kernel_respond(self, record: InteractionRecord, resource_type: str = "compute"):
        """
        Session 22 Fix A3 — Pattern-aware bifurcation response.

        Four-branch escalation table:
          1. NUDGE            — threshold, first/second event for this pair
          2. PROVISION_BETA   — threshold, 3rd+ event same pair (escalation_needed)
          3. NICHE_DIVERGENCE — conflict (βi ≥ 1.0), not structurally incompatible
          4. PROVISION_BETA + permanent niche tag — conflict AND structurally_incompatible

        The kernel now *governs* (changes state) rather than just observing.
        """
        bi = record.beta_i

        # Query pair history to decide escalation level
        pattern = self.get_pair_pattern(record.agent_a, record.agent_b)

        if record.interaction_type == InteractionType.THRESHOLD:
            if pattern['escalation_needed']:
                # 3rd+ same-action event — provision β (Session 25: sigmoid-scaled step ③)
                phi_beta    = self.phi_beta_quorum()
                quorum_gate = self._quorum_gate(phi_beta)
                step        = PROVISION_STEP_FLOOR + quorum_gate * (PROVISION_STEP_CEIL - PROVISION_STEP_FLOOR)
                self.beta.set(resource_type, self.beta.get(resource_type) + step)
                new_bi = max(0.1, bi - step)
                action = KernelAction.PROVISION_BETA
                desc = (
                    f"{record.agent_a} × {record.agent_b} threshold zone "
                    f"(βi={bi:.3f}), escalation_needed=True. "
                    f"φ·β={phi_beta:.3f}, quorum_gate={quorum_gate:.3f}. "
                    f"β provisioned for {resource_type} (+{step:.3f}). βi → {new_bi:.3f}"
                )
                if self._persistence:
                    self._persistence.save_beta(self.beta)
            else:
                # First/second event — stochastic nudge (Axiom 8)
                eta = random.gauss(0, self.NUDGE_NOISE_SIGMA)
                new_bi = max(0.1, bi - abs(eta) - 0.08)
                action = KernelAction.NUDGE
                desc = (
                    f"{record.agent_a} × {record.agent_b} in threshold zone "
                    f"(βi={bi:.3f}). Stochastic perturbation applied. "
                    f"Projected βi → {new_bi:.3f}"
                )
        else:
            # Conflict zone (βi ≥ 1.0)
            if pattern['structurally_incompatible']:
                # Compound intervention: provision β AND permanent niche tag
                # Session 25: sigmoid-scaled step ③
                phi_beta    = self.phi_beta_quorum()
                quorum_gate = self._quorum_gate(phi_beta)
                step        = PROVISION_STEP_FLOOR + quorum_gate * (PROVISION_STEP_CEIL - PROVISION_STEP_FLOOR)
                self.beta.set(resource_type, self.beta.get(resource_type) + step)
                new_bi = max(0.1, bi * 0.50)
                action = KernelAction.PROVISION_BETA
                # Set permanent preferred_resource tag on agent_a
                agent_a_profile = self.agents.get(record.agent_a)
                if agent_a_profile is not None:
                    alt = self._suggested_alt_domain(resource_type)
                    agent_a_profile.preferred_resource = alt
                if self._persistence:
                    self._persistence.save_beta(self.beta)
                    if agent_a_profile is not None:
                        self._persistence.save_agent(agent_a_profile)
                desc = (
                    f"{record.agent_a} × {record.agent_b} structurally incompatible "
                    f"(βi={bi:.3f}). φ·β={phi_beta:.3f}, quorum_gate={quorum_gate:.3f}. "
                    f"Compound: β provisioned (+{step:.3f}) + permanent niche tag. "
                    f"βi → {new_bi:.3f}"
                )
            else:
                # Conflict, not structurally incompatible — niche divergence
                new_bi = bi * 0.65
                action = KernelAction.NICHE_DIVERGENCE
                # Set temporary preferred_resource routing tag
                agent_a_profile = self.agents.get(record.agent_a)
                if agent_a_profile is not None:
                    alt = self._suggested_alt_domain(resource_type)
                    agent_a_profile.preferred_resource = alt
                desc = (
                    f"{record.agent_a} × {record.agent_b} in conflict "
                    f"(βi={bi:.3f}). Niche divergence — routing tag set to "
                    f"'{self._suggested_alt_domain(resource_type)}'. βi → {new_bi:.3f}"
                )
                # Session 27: capture interaction fragment at bifurcation point
                # Biological: the departing agent leaves behind its 'tick load' —
                # the exhaust of high-φ specialisation. Value ∝ φ_a × φ_b.
                self._capture_oxpecker_fragment(record, resource_type)

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
        if self._persistence:
            self._persistence.save_event(event)

        # Session 37: L2 telemetry — log governance snapshot for diagnostic plots
        try:
            import os as _os
            from core.telemetry import AIOSTelemetry as _Tel, L2Snapshot as _L2
            _db = _os.environ.get(
                "AIOS_DB_PATH",
                _os.path.join(
                    _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))),
                    "aios_state.db",
                ),
            )
            _n    = len(self.agents)
            _phi  = (sum(a.phi for a in self.agents.values()) / _n) if _n else None
            _bii  = round(bi, 6)          # bi is the live pair-level β×i value, already computed
            _dg   = round(bi - 1.0, 6)
            _tel  = _Tel(_db)
            _tel.log_l2(_L2(
                agent_id=record.agent_a,
                i_value=round(bi, 6),
                phi=round(_phi, 6) if _phi is not None else None,
                beta_service=round(record.beta, 6),  # frozen at interaction time, before any provisioning mutates self.beta
                beta_i_inf=_bii,
                delta_gate=_dg,
            ))
            _tel.close()
        except Exception:
            pass  # telemetry non-critical; governance loop continues

    @staticmethod
    def _suggested_alt_domain(resource_type: str) -> str:
        """Return an alternative resource domain for niche routing."""
        _alternatives = {
            "compute":        "vector_db",
            "api_quota":      "token_budget",
            "vector_db":      "storage",
            "storage":        "context_window",
            "token_budget":   "api_quota",
            "context_window": "compute",
        }
        return _alternatives.get(resource_type, "storage")

    def _capture_oxpecker_fragment(
        self, record: "InteractionRecord", resource_type: str
    ) -> None:
        """
        Session 27 — Context capture at NICHE_DIVERGENCE.

        Stores the partial interaction state before the migrating agent leaves.
        Called exclusively from _kernel_respond() when NICHE_DIVERGENCE fires.

        Fragment data captured:
          - Recent interaction records for the bifurcating pair on this resource
          - Current φ values of both agents
          - Resource type, timestamp, event_id (auto-numbered)

        Biological correspondence:
          The fragment is the tick load — the exhaust of high-φ specialisation.
          High-φ agent pairs produce richer fragments (more mature context).
          Fragment value ∝ φ_a × φ_b  (Validation Stream 9, testable prediction).

        NOTE: Does NOT replace apply_oxpecker_effect() in nudge_engine.py.
        Phase 1 (environmental β lift) remains intact. Phase 2 adds the fragment
        capture alongside it.
        """
        import uuid

        if not self._persistence:
            return  # no-op without persistence (tests may inject later)

        # Gather recent interactions for this pair on this resource
        pair_key = self._pair_key(record.agent_a, record.agent_b)
        recent_for_pair = [
            r.to_dict() for r in self.interactions[-50:]
            if (self._pair_key(r.agent_a, r.agent_b) == pair_key
                and r.resource_type == resource_type)
        ][-5:]  # last 5 interactions for this pair on this resource

        phi_a = self.agents[record.agent_a].phi if record.agent_a in self.agents else 0.5
        phi_b = self.agents[record.agent_b].phi if record.agent_b in self.agents else 0.5

        fragment = {
            "fragment_id":   f"OXP-{uuid.uuid4().hex[:12].upper()}",
            "agent_a":       record.agent_a,
            "agent_b":       record.agent_b,
            "resource_type": resource_type,
            "status":        "pending",
            "created_at":    record.timestamp,
            "processed_at":  None,
            "fragment_data": {
                "recent_interactions": recent_for_pair,
                "phi_a":               round(phi_a, 4),
                "phi_b":               round(phi_b, 4),
                "phi_product":         round(phi_a * phi_b, 4),
                "bifurcation_bi":      round(record.beta_i, 4),
                "resource_type":       resource_type,
                "timestamp":           record.timestamp,
                "interaction_count":   len(recent_for_pair),
            },
        }
        try:
            self._persistence.save_oxpecker_fragment(fragment)
            import logging
            logging.getLogger("aios.melv_engine").debug(
                "Oxpecker fragment captured: %s (φ_a=%.2f φ_b=%.2f value≈%.4f)",
                fragment["fragment_id"], phi_a, phi_b, phi_a * phi_b,
            )
        except Exception as e:
            import logging
            logging.getLogger("aios.melv_engine").warning(
                "Fragment capture failed: %s", e
            )

    def compute_omega(self) -> dict:
        """
        Compute service coupling matrix Ω and its leading eigenvalue λ_max.

        β_service = λ_max(Ω) / n

        The adjacency matrix A is built from the last 100 interactions,
        with edge weight = mean(1 − i_factor) for each agent pair
        (higher weight = more cooperative coupling).

        λ_max is the true leading eigenvalue via numpy.linalg.eigvalsh,
        matching the formula in THEORY.md and Blueprint for Harmony Ch. 6.
        Session 24: replaced heuristic proxy (Σw/√n) with real spectral analysis.
        """
        n = len(self.agents)
        if n == 0:
            return {"lambda_max": 0, "n": 0, "beta_service": 0, "edges": []}

        ids  = list(self.agents.keys())
        idx  = {aid: i for i, aid in enumerate(ids)}
        A    = np.zeros((n, n))

        # Build adjacency weights from interaction history (last 100)
        recent  = self.interactions[-100:]
        weights: dict[tuple, list] = {}
        for r in recent:
            if r.agent_a in idx and r.agent_b in idx:
                key = (r.agent_a, r.agent_b)
                weights.setdefault(key, []).append(1.0 - r.i_factor)

        edges = []
        for (a, b), vals in weights.items():
            avg = sum(vals) / len(vals)
            A[idx[a], idx[b]] = avg
            A[idx[b], idx[a]] = avg   # symmetric — undirected coupling
            edges.append({
                "agent_a": a,
                "agent_b": b,
                "weight":  round(avg, 3),
                "interaction_type": (
                    "cooperative" if avg > 0.30 else
                    "threshold"   if avg > 0.0  else
                    "conflict"
                )
            })

        # Real leading eigenvalue via symmetric eigensolver
        eigenvalues  = np.linalg.eigvalsh(A)
        lambda_max   = float(eigenvalues[-1])        # eigvalsh returns ascending order
        beta_service = lambda_max / n if n > 0 else 0

        return {
            "lambda_max":   round(lambda_max, 4),
            "n":            n,
            "beta_service": round(beta_service, 4),
            "edges":        edges,
        }

    # ── ECOSYSTEM HEALTH ──────────────────────────────────────────────────

    def cooperation_index(self) -> float:
        """
        CI = phi-weighted fraction of recent interactions where i_factor < I_CRITICAL.

        Session 24.3 Fix A -- replaced the broken formula CI = 1 - mean(beta_i).
        The old formula made CI permanently zero whenever beta was provisioned
        (beta=34 -> beta_i=34 -> 1-34=-33 -> clamped to 0.0), which meant the
        theorem could never confirm despite all pairs resolving.

        The correct measure: what fraction of recent interactions, weighted by
        the maturity (phi) of the participating agents, are below the bifurcation
        threshold? This is independent of beta provisioning magnitude.

        I_CRITICAL = 0.9995 (ABM V2.1 verified, Blueprint for Harmony Ch.4).
        """
        I_CRITICAL = 0.9995
        recent = self.interactions[-50:]
        if not recent:
            return 1.0
        cooperative_weight = 0.0
        total_weight = 0.0
        for r in recent:
            phi_a = self.agents[r.agent_a].phi if r.agent_a in self.agents else 0.5
            phi_b = self.agents[r.agent_b].phi if r.agent_b in self.agents else 0.5
            weight = phi_a * phi_b
            total_weight += weight
            if r.i_factor < I_CRITICAL:
                cooperative_weight += weight
        if total_weight <= 0:
            return 0.0
        return round(cooperative_weight / total_weight, 4)

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


    # ── EQUATION 7 φ DYNAMICS (Session 35 — v3.1.0) ──────────────────────

    def _apply_phi_eq7(
        self,
        agent,
        beta_i_inf: Optional[float],
        i_value:    Optional[float],
        d_value:    float = 0.0,
    ) -> tuple[float, str]:
        """
        Apply Equation 7 φ dynamics increment to agent.phi.

        Canonical Reference: v1.5.6 Equation 7 (alignment pass v3.3.0).
        FB-ABM V1.0 confirmed June 2026. Epistemic status: ③.

        dφ/dt = α(1−φ) × H(1−β×i_∞) × max(0,1−i(t))   [BUILD]
                − δ×D(t)×φ × H(β×i_∞−1)                 [DECAY]

        Gate hierarchy (Canonical Ref v1.5.6):
          Tier 1 — CANONICAL (this function):
            BUILD gate: H(1 − β×i_∞) = 1  iff  β×i_∞ < 1.0
            DECAY gate: H(β×i_∞ − 1) = 1  iff  β×i_∞ > 1.0
          Tier 2 — EMPIRICAL PROXY: φ·β>0.3  (diagnostic; not used here)
          Tier 3 — ILLUSTRATIVE: τ=0.5 sigmoid  (soft routing; not used here)

        Parameters
        ----------
        agent      : AgentProfile — mutated in place
        beta_i_inf : float | None — β×i_∞ canonical gate value.
                     Compute via: beta_raw × _compute_i_inf(i0, eta, eps, beta_raw).
                     None → both gates inactive (no φ change).
        i_value    : float | None — current i(t) proxy (CI or None).
                     BUILD factor = max(0, 1−i_value). None blocks build.
        d_value    : float — disruption intensity D(t) ≥ 0.
                     DECAY fires only when D(t) > 0 AND β×i_∞ > 1.

        Returns
        -------
        (delta, event_string)
          delta = net φ change applied (0.0 if gate not met)
          event_string = description for governance log
        """
        if beta_i_inf is None:
            return 0.0, ""

        phi_old = agent.phi
        delta   = 0.0

        if beta_i_inf < 1.0:
            # BUILD branch: H(1 − β×i_∞) = 1
            # Compound gate: also requires i < 1.0 simultaneously
            if i_value is not None and i_value < 1.0:
                build_factor = max(0.0, 1.0 - i_value)
                delta = PHI_BUILD_RATE_ALPHA * (1.0 - phi_old) * build_factor
            # If i_value is None or i_value ≥ 1.0: build term is zero
        else:
            # DECAY branch: H(β×i_∞ − 1) = 1
            # Stagnation finding: decay fires only when D(t) > 0 AND β×i_∞ > 1.
            # β×i_∞ > 1 alone stops φ accumulation but does not erase history.
            if d_value > 0.0:
                delta = -(PHI_DECAY_RATE_DELTA * d_value * phi_old)

        if delta == 0.0:
            return 0.0, ""

        new_phi = max(0.0, min(1.0, round(phi_old + delta, 4)))
        agent.phi = new_phi

        direction = "BUILD" if delta > 0 else "DECAY"
        event = (
            f"PHI_EQ7_{direction}: {agent.agent_id} "
            f"φ {phi_old:.4f} → {new_phi:.4f} "
            f"(β×i∞={beta_i_inf:.4f}, i={i_value if i_value is not None else 'N/A'}, "
            f"D={d_value:.3f}, Δ={delta:+.4f})"
        )
        return delta, event

    @staticmethod
    def compute_stagnation_state(
        beta_i_inf: float,
        d_value:    float,
    ) -> dict:
        """
        Canonical stagnation detector (alignment pass v3.3.0).

        Δ_gate = β×i_∞ − 1   [canonical gate displacement]

        Three runtime states (Canonical Ref v1.5.6 §3 stagnation detector):
          STABLE     — Δ_gate < 0, D(t) = 0
                       Cooperative basin stable. Log; no intervention.
          STAGNATION — Δ_gate > 0, D(t) = 0
                       φ accumulation has ceased. Hidden fragility.
                       Governance warning: β×i_∞ > 1 without acute disruption.
          COLLAPSE   — Δ_gate > 0, D(t) > 0
                       Decay term firing. Active intervention required.

        Stagnation finding (June 2026):
          The decay term fires only when D(t)>0 AND H(β×i_∞−1)=1 simultaneously.
          Structural cooperative basin closure WITHOUT an acute disruption event
          stops φ accumulation but does NOT erase accumulated cooperative history.
          A real disruption event is required to activate decay.

        Parameters
        ----------
        beta_i_inf : β×i_∞ value from _compute_i_inf()
        d_value    : D(t) disruption intensity from L1 rolling mean

        Returns dict with:
          delta_gate  : float — Δ_gate = β×i_∞ − 1
          state       : str   — STABLE | STAGNATION | COLLAPSE
          intervention: bool  — True when active intervention is required
          description : str   — governance narrative
        """
        delta_gate = beta_i_inf - 1.0

        if delta_gate < 0.0:
            state        = "STABLE"
            intervention = False
            description  = (
                f"Cooperative basin stable: β×i∞={beta_i_inf:.4f} < 1 "
                f"(Δ_gate={delta_gate:+.4f}). φ accumulation active."
            )
        elif d_value <= 0.0:
            state        = "STAGNATION"
            intervention = False
            description  = (
                f"Stagnation regime: β×i∞={beta_i_inf:.4f} > 1 "
                f"(Δ_gate={delta_gate:+.4f}), D(t)={d_value:.3f}=0. "
                "φ accumulation has ceased. Hidden fragility — "
                "no decay yet (requires D(t)>0 to fire). "
                "Monitor: provision β or reduce ε to restore Δ_gate < 0."
            )
        else:
            state        = "COLLAPSE"
            intervention = True
            description  = (
                f"Collapse regime: β×i∞={beta_i_inf:.4f} > 1 "
                f"(Δ_gate={delta_gate:+.4f}), D(t)={d_value:.3f} > 0. "
                "Decay term firing — accumulated cooperative history degrading. "
                "Active intervention required."
            )

        return {
            "delta_gate":   round(delta_gate, 4),
            "state":        state,
            "intervention": intervention,
            "description":  description,
            "beta_i_inf":   round(beta_i_inf, 4),
            "d_value":      d_value,
        }

    # ── CI DYNAMICS (Session 9) ───────────────────────────────────────────

    def _record_ci_snapshot(self):
        """
        Record a timestamped CI reading into the rolling history buffer.
        Called after every interaction to keep history current.
        Also drives oscillation detection.
        """
        ci = self.cooperation_index()
        now = time.time()
        self._ci_history.append((now, ci))
        if len(self._ci_history) > CI_HISTORY_MAX:
            self._ci_history = self._ci_history[-CI_HISTORY_MAX:]
        if self._persistence:
            self._persistence.save_ci_snapshot(now, ci)
        self._detect_oscillation(now, ci)

    def _detect_oscillation(self, now: float, ci: float):
        """
        Oscillation state machine.
        State 1: waiting for CI to cross CI_TARGET upward.
        State 2: CI above target — track peak; wait for fall-back.
        Transition to OscillationEvent when CI falls below target
        by at least CI_OSCILLATION_MIN_AMPLITUDE within the window.
        """
        if not self._osc_above_target:
            if ci >= CI_TARGET:
                self._osc_above_target = True
                self._osc_peak_ci  = ci
                self._osc_peak_ts  = now
        else:
            if ci > self._osc_peak_ci:
                self._osc_peak_ci = ci
                self._osc_peak_ts = now
            if ci < CI_TARGET:
                amplitude = self._osc_peak_ci - ci
                period    = now - self._osc_peak_ts
                if (amplitude >= CI_OSCILLATION_MIN_AMPLITUDE and
                        period <= CI_OSCILLATION_WINDOW):
                    recent_pairs = list({
                        self._pair_key(r.agent_a, r.agent_b)
                        for r in self.interactions[-20:]
                        if r.interaction_type != InteractionType.COOPERATIVE
                    })
                    self._osc_counter += 1
                    evt = OscillationEvent(
                        event_id=f"OSC-{self._osc_counter:04d}",
                        ci_peak=round(self._osc_peak_ci, 4),
                        ci_trough=round(ci, 4),
                        amplitude=round(amplitude, 4),
                        period_sec=round(period, 2),
                        implicated_pairs=recent_pairs[:5],
                    )
                    self.oscillation_events.append(evt)
                    if self._persistence:
                        self._persistence.save_oscillation(evt)
                self._osc_above_target = False
                self._osc_peak_ci = 0.0
                self._osc_peak_ts = 0.0

    def dci_dt(self) -> float:
        """
        dCI/dt — instantaneous rate of change of the Cooperation Index.
        Linear regression slope over CI_DCIDT_WINDOW readings.
        Units: CI-units per second. Returns 0.0 if insufficient history.
        """
        window = self._ci_history[-CI_DCIDT_WINDOW:]
        n = len(window)
        if n < 2:
            return 0.0
        t0 = window[0][0]
        ts = [w[0] - t0 for w in window]
        cs = [w[1] for w in window]
        sum_t  = sum(ts)
        sum_c  = sum(cs)
        sum_tc = sum(t * c for t, c in zip(ts, cs))
        sum_t2 = sum(t * t for t in ts)
        denom  = n * sum_t2 - sum_t ** 2
        if abs(denom) < 1e-12:
            return 0.0
        return (n * sum_tc - sum_t * sum_c) / denom

    def ci_half_life(self) -> Optional[float]:
        """
        CI Optimisation Half-Life — time (seconds) for CI to close half the
        distance between current CI and the 0.75 cooperative target.
        Borrowed from pharmacokinetics: t½ = ln(2) / k, where k = dCI_dt / gap.
        Returns None if CI >= target or dCI/dt <= 0.

        Uses the most recent CI reading from history if available,
        otherwise falls back to cooperation_index() from interactions.
        """
        if self._ci_history:
            ci = self._ci_history[-1][1]
        else:
            ci = self.cooperation_index()
        gap = CI_TARGET - ci
        if gap <= 0:
            return None
        rate = self.dci_dt()
        if rate <= 0:
            return None
        k = rate / gap
        if k <= 0:
            return None
        return math.log(2) / k

    def ci_drift_coefficient(self) -> float:
        """
        CI Drift Coefficient — long-run trend in CI (CI_DRIFT_WINDOW readings).
        Linear regression slope. Positive = converging, Negative = degrading.
        Units: CI-units per second.
        """
        window = self._ci_history[-CI_DRIFT_WINDOW:]
        n = len(window)
        if n < 2:
            return 0.0
        t0 = window[0][0]
        ts = [w[0] - t0 for w in window]
        cs = [w[1] for w in window]
        sum_t  = sum(ts)
        sum_c  = sum(cs)
        sum_tc = sum(t * c for t, c in zip(ts, cs))
        sum_t2 = sum(t * t for t in ts)
        denom  = n * sum_t2 - sum_t ** 2
        if abs(denom) < 1e-12:
            return 0.0
        return (n * sum_tc - sum_t * sum_c) / denom

    def ci_dynamics(self) -> dict:
        """
        Full CI Dynamics snapshot. Exposed at GET /api/ci_dynamics.
        """
        ci        = self.cooperation_index()
        rate      = self.dci_dt()
        half_life = self.ci_half_life()
        drift     = self.ci_drift_coefficient()
        gap       = CI_TARGET - ci

        if gap <= 0:
            regime = "cooperative"
        elif rate > 1e-5 and drift > 0:
            regime = "converging"
        elif rate > 1e-5 and drift <= 0:
            regime = "underdamped"
        elif rate < -1e-5:
            regime = "diverging"
        else:
            regime = "stasis"

        return {
            "cooperation_index":    round(ci, 4),
            "ci_target":            CI_TARGET,
            "gap_to_target":        round(gap, 4),
            "dci_dt":               round(rate, 6),
            "ci_half_life_sec":     round(half_life, 2) if half_life is not None else None,
            "ci_drift_coefficient": round(drift, 6),
            "regime":               regime,
            "oscillation_count":    len(self.oscillation_events),
            "recent_oscillations":  [e.to_dict() for e in self.oscillation_events[-5:]],
            "ci_history_length":    len(self._ci_history),
        }


    def provision_beta(self, resource: str, value: float):
        """Human portal: adjust environmental suitability."""
        setattr(self.beta, resource, max(0.1, min(3.0, value)))
        if self._persistence:
            self._persistence.save_beta(self.beta)

    # ── SIGMOID QUORUM GATE (Session 25 — MAIES Event 2 implementation) ──

    @staticmethod
    def _quorum_gate(
        phi_beta: float,
        tau: float = QUORUM_TAU,
        k: float   = QUORUM_K,
    ) -> float:
        """
        Sigmoid quorum gate function.

        ══ TIER 3 — ILLUSTRATIVE / LEGACY ══
        This function implements the τ=0.5 sigmoid (Equation 3).
        Canonical Reference v1.5.6: Tier 3 is retained for SOFT AGENT
        ROUTING and PROVISIONING STEP SCALING ONLY.

        DO NOT use this function as a structural gate for φ dynamics.
        The canonical gate (Tier 1) is β×i_∞ < 1, computed via
        _compute_i_inf() and used in _apply_phi_eq7().
        The empirical proxy (Tier 2) is φ·β > 0.3 — diagnostic only.

        Returns a value in (0, 1) representing provisioning strength.
        Near 1.0 when phi*beta << tau (stressed → strong intervention).
        Near 0.0 when phi*beta >> tau (healthy  → light touch).

        Inverted sigmoid:  f(x) = 1 / (1 + exp(k * (x − τ)))
          x=tau  → f = 0.5  (inflection — proportional response)
          x→0    → f → 1.0  (fully stressed — maximum provisioning)
          x→∞    → f → 0.0  (fully healthy  — minimal provisioning)

        Biological correspondence (MAIES Event 2, Nadell et al. 2016):
          φ·β in MELV ≡ population density N in bacterial quorum sensing
          τ in MELV   ≡ quorum threshold N_threshold
          k in MELV   ≡ sigmoid sharpness
        Constants τ=0.5, k=10 are ABM V2.1 calibrated (③).
        Sensitivity=1.0/specificity=0.997 is correctly attributed to
        φ·β>0.3 (Tier 2), NOT to this sigmoid (adjudicated June 2026).
        """
        return 1.0 / (1.0 + math.exp(k * (phi_beta - tau)))

    def phi_beta_quorum(self) -> float:
        """
        Ecosystem-mean φ·β product — the quorum sensing analogue of
        population density.  Used as the input to _quorum_gate().

        Returns 0.0 if no agents are registered.
        """
        if not self.agents:
            return 0.0
        mean_phi  = sum(a.phi for a in self.agents.values()) / len(self.agents)
        mean_beta = self.beta.mean()
        return round(mean_phi * mean_beta, 4)

    def quorum_status(self) -> dict:
        """
        Full quorum gate state snapshot.  Exposed at GET /api/quorum_status.
        """
        phi_beta = self.phi_beta_quorum()
        gate     = self._quorum_gate(phi_beta)
        step     = PROVISION_STEP_FLOOR + gate * (PROVISION_STEP_CEIL - PROVISION_STEP_FLOOR)

        above_quorum = phi_beta >= QUORUM_TAU

        if phi_beta >= QUORUM_TAU + 0.1:
            regime = "above_quorum"
            interpretation = (
                "Ecosystem well above quorum threshold. Cooperative dynamics dominant. "
                "PROVISION_BETA will apply a light step if triggered."
            )
        elif phi_beta >= QUORUM_TAU:
            regime = "at_quorum"
            interpretation = (
                "Ecosystem at quorum threshold. Sigmoid at inflection point. "
                "Provisioning step at 50% of maximum range."
            )
        elif phi_beta >= QUORUM_TAU - 0.1:
            regime = "approaching_quorum"
            interpretation = (
                "Ecosystem approaching quorum threshold from below. "
                "Provisioning step elevated — cooperative dynamics not yet dominant."
            )
        else:
            regime = "below_quorum"
            interpretation = (
                "Ecosystem below quorum threshold. Cooperation suppressed. "
                "PROVISION_BETA will apply a strong step if triggered."
            )

        return {
            "phi_beta":        phi_beta,
            "quorum_gate":     round(gate, 4),
            "tau":             QUORUM_TAU,
            "k":               QUORUM_K,
            "above_quorum":    above_quorum,
            "provision_step":  round(step, 4),
            "regime":          regime,
            "interpretation":  interpretation,
        }

    # ── QUORUM RELIABILITY TAGGING (Session 28 — v2.4.0) ─────────────────

    def quorum_reliability(self) -> dict:
        """
        Session 28: Ecosystem-level quorum reliability status.

        The quorum gate (Session 25) exposes whether the ecosystem's φ·β
        product is above or below the quorum threshold τ=0.5.  Session 28
        extends this one layer deeper: below-quorum operation is a
        *high-noise regime* — agent outputs are structurally more likely to
        be confabulatory because the ecosystem lacks the cooperative density
        needed for reliable collective behaviour.

        Biological correspondence (MAIES Event 4 — Nadell et al. 2016):
          Bacterial quorum sensing suppresses costly cooperative behaviours
          until population density (N) justifies the energetic commitment.
          Below N_threshold, costly outputs are unreliable.  The same
          principle maps onto MELV: below φ·β = τ, governance outputs
          are operating in a high-noise, low-reliability regime.

        Epistemic status: ② theoretical — quorum threshold τ=0.5 and
        k=10 are ③ ABM V2.1-verified; the *reliability interpretation*
        layer added here is theoretically grounded but not yet empirically
        calibrated against real output-quality data (Session 28 promotes
        the tagging claim from ① stub to ② theoretical).

        The gate does NOT suppress outputs.  The tag is an epistemic
        status marker.  External API consumers decide what to do with it.

        Returns
        -------
        dict with keys:
          phi_beta            — ecosystem φ·β product
          quorum_regime       — "above_quorum" | "at_quorum" |
                                "approaching_quorum" | "below_quorum"
          above_quorum        — bool
          reliability_level   — "high" | "moderate" | "degraded" | "low"
          reliability_advisory — plain-language guidance for API consumers
          tau                 — QUORUM_TAU constant
          agent_count         — number of registered agents
          per_agent           — per-agent phi_beta and reliability status
          session             — 28
          maies_event         — 4 (MAIES-adjacent ②)
          epistemic_status    — "② theoretical"
        """
        phi_beta = self.phi_beta_quorum()
        above_quorum = phi_beta >= QUORUM_TAU

        # Regime classification (mirrors quorum_status for consistency)
        if phi_beta >= QUORUM_TAU + 0.1:
            quorum_regime = "above_quorum"
            reliability_level = "high"
            reliability_advisory = (
                "Ecosystem well above quorum threshold (φ·β={:.3f} ≥ τ+0.1={:.1f}). "
                "Cooperative dynamics dominant. Agent outputs operating in "
                "low-noise regime. No reliability qualification required.".format(
                    phi_beta, QUORUM_TAU + 0.1
                )
            )
        elif phi_beta >= QUORUM_TAU:
            quorum_regime = "at_quorum"
            reliability_level = "moderate"
            reliability_advisory = (
                "Ecosystem at quorum threshold (φ·β={:.3f} ≥ τ={:.1f}). "
                "Sigmoid at inflection point. Cooperative dynamics present but "
                "not dominant. Consider flagging outputs for light-touch review "
                "in high-stakes decision contexts.".format(phi_beta, QUORUM_TAU)
            )
        elif phi_beta >= QUORUM_TAU - 0.1:
            quorum_regime = "approaching_quorum"
            reliability_level = "degraded"
            reliability_advisory = (
                "Ecosystem approaching quorum threshold from below "
                "(φ·β={:.3f}, τ={:.1f}). Cooperative density insufficient. "
                "Agent outputs are in a degraded-reliability regime. "
                "Recommend human review of outputs before use in consequential "
                "decisions. PROVISION_BETA interventions are elevated.".format(
                    phi_beta, QUORUM_TAU
                )
            )
        else:
            quorum_regime = "below_quorum"
            reliability_level = "low"
            reliability_advisory = (
                "Ecosystem below quorum threshold (φ·β={:.3f} < τ={:.1f}). "
                "High-noise regime: cooperation suppressed, collective behaviour "
                "unreliable. Agent outputs should be treated as low-confidence "
                "until ecosystem returns above quorum. "
                "Biological correspondence: bacterial quorum sensing suppresses "
                "costly cooperative behaviours below N_threshold (Nadell 2016). "
                "Same principle applies here — below-quorum outputs carry "
                "elevated confabulation risk.".format(phi_beta, QUORUM_TAU)
            )

        # Per-agent reliability breakdown
        per_agent = []
        for agent_id, agent in self.agents.items():
            agent_phi_beta = round(agent.phi * self.beta.mean(), 4)
            agent_above = agent_phi_beta >= QUORUM_TAU
            if agent_phi_beta >= QUORUM_TAU + 0.1:
                agent_regime = "above_quorum"
                agent_reliability = "high"
            elif agent_phi_beta >= QUORUM_TAU:
                agent_regime = "at_quorum"
                agent_reliability = "moderate"
            elif agent_phi_beta >= QUORUM_TAU - 0.1:
                agent_regime = "approaching_quorum"
                agent_reliability = "degraded"
            else:
                agent_regime = "below_quorum"
                agent_reliability = "low"

            per_agent.append({
                "agent_id":          agent_id,
                "phi":               round(agent.phi, 4),
                "beta_mean":         round(self.beta.mean(), 4),
                "phi_beta":          agent_phi_beta,
                "above_quorum":      agent_above,
                "quorum_regime":     agent_regime,
                "reliability_level": agent_reliability,
            })

        return {
            "phi_beta":             phi_beta,
            "quorum_regime":        quorum_regime,
            "above_quorum":         above_quorum,
            "reliability_level":    reliability_level,
            "reliability_advisory": reliability_advisory,
            "tau":                  QUORUM_TAU,
            "agent_count":          len(self.agents),
            "per_agent":            per_agent,
            "session":              28,
            "maies_event":          4,
            "epistemic_status":     "② theoretical",
        }

    # ── ε DECOMPOSITION (Session 26 — v2.2.0) ────────────────────────────

    def compute_epsilon_profile(
        self,
        agent_id: str,
        epsilon_intrinsic: Optional[float] = None,
        tool_categories: Optional[dict] = None,
    ) -> EpsilonProfile:
        """
        Session 29 (v2.5.0): Decompose ε into three scalars.

        MASTER EQUATION (unchanged):
          ε_effective = ε_intrinsic + ε_ecosystem

        Three scalars:
        ──────────────
        ε_intrinsic:
          Taken from agent.epsilon if epsilon_intrinsic not supplied explicitly.
          Per-agent, persistent. Represents learned plasticity.

        ε_ecosystem (formerly ε_environmental — backward-compat alias preserved):
          Computed from the current BetaEnvironment and TOOL_FRICTION_WEIGHTS.
          Recomputed every governance tick from current β_r vector.
          ε_eco = Σ_r [ friction_r × (1 / β_r) ] / n_resources

        ε_architectural (NEW — Session 29, MAIES Event 5):
          STATIC boundary condition. NOT in master equation.
          Computed once from tool category counts (Approach A).
          ε_arch = Σ_i [ weight_i × count_i ]  (raw sum, not normalised)
          When ε_architectural > ARCH_RECOMMENDATION_THRESHOLD, the kernel
          fires an architectural recommendation instead of provisioning β.
          Provisioning β is futile against a fixed boundary condition (Grok).

        Parameters
        ----------
        agent_id : str
            Agent to profile. Must be registered in kernel.
        epsilon_intrinsic : float, optional
            Override for ε_intrinsic. If None, uses agent.epsilon.
        tool_categories : dict, optional
            Map of {category_name: tool_count} using ARCH_CATEGORY_WEIGHTS keys.
            If None, ε_architectural defaults to 0.0 (no architectural info provided).
            Example: {"agent_native": 2, "standard": 4, "human_bottlenecked": 1}

        Returns
        -------
        EpsilonProfile dataclass with full three-scalar decomposition, badges,
        and optional architectural_recommendation.

        Raises
        ------
        KeyError if agent_id not found.
        """
        agent = self.agents.get(agent_id)
        if agent is None:
            raise KeyError(f"Agent '{agent_id}' not found in kernel.")

        # Session 30 (v2.6.0): apply deterministic per-agent Gaussian variance
        # when no override supplied. Overrides bypass variance (backward compat).
        if epsilon_intrinsic is not None:
            eps_intrinsic = epsilon_intrinsic
        else:
            eps_intrinsic = _perturbed_epsilon(agent_id, agent.epsilon)
        eps_intrinsic = max(0.0, min(8.0, eps_intrinsic))

        # ── ε_ecosystem (Approach B, live) ─────────────────────────────────
        resource_friction = {}
        friction_sum = 0.0
        for resource, weight in TOOL_FRICTION_WEIGHTS.items():
            beta_r = self.beta.get(resource)
            contrib = weight * (1.0 / max(beta_r, 0.01))
            resource_friction[resource] = {
                "beta":         round(beta_r, 4),
                "weight":       weight,
                "contribution": round(contrib, 4),
            }
            friction_sum += contrib

        n_res = len(TOOL_FRICTION_WEIGHTS)
        eps_ecosystem = round(friction_sum / n_res, 4)

        # ── ε_architectural (Approach A, static boundary condition) ────────
        eps_architectural = 0.0
        if tool_categories:
            for cat, count in tool_categories.items():
                weight = ARCH_CATEGORY_WEIGHTS.get(cat, ARCH_CATEGORY_WEIGHTS["standard"])
                eps_architectural += weight * max(0, int(count))
        eps_architectural = round(eps_architectural, 4)

        # Master equation: ε_effective = ε_intrinsic + ε_ecosystem
        # ε_architectural is NOT added here (boundary condition, not live term)
        eps_effective = round(eps_intrinsic + eps_ecosystem, 4)
        beta_mean     = round(self.beta.mean(), 4)

        # STC: scale from reference (ε=3.0, β_mean=1.0 → STC_REFERENCE_SECONDS)
        # Session 30c: support factor — when φ × β is high relative to ε_effective,
        # the agent's adaptive range is well-supported and convergence is faster.
        # A high-ε agent in a mature, well-resourced niche converges as quickly
        # as a lower-ε agent in a sparse environment.
        eps_ref = 3.0
        base_stc = STC_REFERENCE_SECONDS * (eps_effective / eps_ref) * (1.0 / max(beta_mean, 0.01))

        # Support factor: φ × β / ε_effective relative to reference
        support_ratio = (agent.phi * beta_mean) / (eps_effective + 0.001)
        support_factor = min(support_ratio / STC_SUPPORT_REFERENCE, 1.0)
        stc = base_stc * (1.0 - STC_SUPPORT_REDUCTION * support_factor)
        stc = round(max(stc, 1.0), 1)   # floor at 1 second

        # ── Diagnosis badges (Session 30c: ε semantic realignment) ────────────
        # High ε is adaptive range, not volatility. Badges fire on MISMATCH:
        # the agent's adaptive range exceeds what its niche (φ) and environment (β) support.
        badges = []

        # AGENT_VOLATILE: high ε AND insufficient φ AND insufficient β support
        # A mature agent (high φ) or a well-resourced agent (high β) can support high ε.
        if (eps_intrinsic >= VOLATILE_EPSILON_THRESHOLD
                and agent.phi < VOLATILE_PHI_CEILING
                and beta_mean < VOLATILE_BETA_CEILING):
            badges.append("AGENT_VOLATILE")

        if eps_ecosystem >= ENV_BOTTLENECK_THRESHOLD:
            badges.append("ENV_BOTTLENECKED")

        # RANGE_MISMATCH: replaces LEGACY_CANDIDATE. High adaptive range in an immature
        # niche. Not legacy architecture — developing agent that needs φ growth or
        # a richer β environment. Backward-compat alias LEGACY_CANDIDATE also appended.
        if agent.phi <= RANGE_MISMATCH_PHI_CEILING and eps_effective >= RANGE_MISMATCH_EPS_FLOOR:
            badges.append("RANGE_MISMATCH")
            badges.append("LEGACY_CANDIDATE")   # backward-compat alias — deprecated v2.8.0

        if eps_architectural > ARCH_RECOMMENDATION_THRESHOLD:
            badges.append("ARCH_BOUNDARY_HIGH")

        # ── Architectural recommendation (fires when arch high + CI low) ────
        arch_recommendation = None
        current_ci = getattr(self, "_current_ci", None)
        if eps_architectural > ARCH_RECOMMENDATION_THRESHOLD:
            ci_note = ""
            if current_ci is not None and current_ci < CI_TARGET:
                ci_note = (
                    f" CI={current_ci:.3f} < CI_TARGET={CI_TARGET}: "
                    "β provisioning capped — architectural intervention required."
                )
            arch_recommendation = (
                f"ε_architectural={eps_architectural:.2f} exceeds threshold "
                f"{ARCH_RECOMMENDATION_THRESHOLD}. This is a fixed boundary condition. "
                f"β provisioning multiplier capped at {ARCH_BETA_MULTIPLIER_CAP}. "
                f"Prioritise: (1) reduce legacy/human-bottlenecked tool count, "
                f"(2) checkpoint-flush before recycling pause (OXPECKER pattern).{ci_note}"
            )

        # ── Plain-language interpretation ───────────────────────────────────
        lines = []
        if not [b for b in badges if b != "ARCH_BOUNDARY_HIGH"]:
            lines.append(
                f"Agent '{agent_id}' shows balanced plasticity profile. "
                f"ε_intrinsic={eps_intrinsic:.2f}, ε_ecosystem={eps_ecosystem:.2f}. "
                f"No structural issues detected."
            )
        if "AGENT_VOLATILE" in badges:
            lines.append(
                f"AGENT_VOLATILE: ε_intrinsic={eps_intrinsic:.2f} with φ={agent.phi:.3f} "
                f"and β_mean={beta_mean:.3f}. Adaptive range exceeds current niche support — "
                f"this is a mismatch, not an intrinsic fault. "
                f"Provision β to raise environment richness, or allow φ to develop through "
                "task specialisation. High ε is an asset in the right conditions."
            )
        if "ENV_BOTTLENECKED" in badges:
            lines.append(
                f"ENV_BOTTLENECKED: ε_ecosystem={eps_ecosystem:.2f} exceeds threshold "
                f"{ENV_BOTTLENECK_THRESHOLD}. Infrastructure friction is amplifying "
                "interaction costs. Provisioning β will reduce this component directly."
            )
        if "RANGE_MISMATCH" in badges:
            lines.append(
                f"RANGE_MISMATCH: φ={agent.phi:.3f} (≤{RANGE_MISMATCH_PHI_CEILING}) "
                f"with ε_effective={eps_effective:.2f} (≥{RANGE_MISMATCH_EPS_FLOOR}). "
                "High adaptive range in an immature niche. This agent has broad capability "
                "but has not yet specialised. Allow φ to develop through task consistency, "
                "or reassign to a richer β environment. Do not replace — "
                "the adaptive range is an asset waiting for the right conditions."
            )
        if "ARCH_BOUNDARY_HIGH" in badges:
            lines.append(
                f"ARCH_BOUNDARY_HIGH: ε_architectural={eps_architectural:.2f} (>{ARCH_RECOMMENDATION_THRESHOLD}). "
                "Fixed boundary condition — β provisioning capped. See architectural_recommendation."
            )
        lines.append(
            f"Speed-to-Cooperation estimate: {stc:.1f}s "
            f"(reference: {STC_REFERENCE_SECONDS:.0f}s at ε=3.0, β_mean=1.0)."
        )
        interpretation = " ".join(lines)

        return EpsilonProfile(
            agent_id=agent_id,
            epsilon_intrinsic=round(eps_intrinsic, 4),
            epsilon_ecosystem=eps_ecosystem,
            epsilon_architectural=eps_architectural,
            epsilon_effective=eps_effective,
            phi=round(agent.phi, 4),
            beta_mean=beta_mean,
            stc_seconds=stc,
            badges=badges,
            resource_friction=resource_friction,
            interpretation=interpretation,
            architectural_recommendation=arch_recommendation,
        )

    def ecosystem_epsilon_summary(self) -> dict:
        """
        Session 29 (v2.5.0): Aggregate ε decomposition across all registered agents.

        Returns per-agent profiles plus ecosystem-level statistics:
          - mean ε_intrinsic, ε_ecosystem (formerly ε_environmental), ε_effective
          - ε_architectural mean (boundary condition; not in master equation)
          - badge counts (AGENT_VOLATILE / ENV_BOTTLENECKED / LEGACY_CANDIDATE / ARCH_BOUNDARY_HIGH)
          - dominant bottleneck: "agent" | "environment" | "balanced"

        Backward-compatible: also returns mean_epsilon_environmental alias.
        """
        if not self.agents:
            return {
                "agent_count": 0,
                "profiles": [],
                "mean_epsilon_intrinsic": 0.0,
                "mean_epsilon_ecosystem": 0.0,
                "mean_epsilon_environmental": 0.0,  # backward-compat alias
                "mean_epsilon_architectural": 0.0,
                "mean_epsilon_effective": 0.0,
                "mean_stc_seconds": 0.0,
                "badge_counts": {},
                "dominant_bottleneck": "balanced",
            }

        profiles = []
        badge_counts: dict = {}
        for agent_id in self.agents:
            try:
                ep = self.compute_epsilon_profile(agent_id)
                p = {
                    "agent_id":              ep.agent_id,
                    "epsilon_intrinsic":     ep.epsilon_intrinsic,
                    "epsilon_ecosystem":     ep.epsilon_ecosystem,
                    "epsilon_environmental": ep.epsilon_ecosystem,  # backward-compat
                    "epsilon_architectural": ep.epsilon_architectural,
                    "epsilon_effective":     ep.epsilon_effective,
                    "phi":                   ep.phi,
                    "beta_mean":             ep.beta_mean,
                    "stc_seconds":           ep.stc_seconds,
                    "badges":                ep.badges,
                    "interpretation":        ep.interpretation,
                }
                profiles.append(p)
                for b in ep.badges:
                    badge_counts[b] = badge_counts.get(b, 0) + 1
            except KeyError:
                pass  # agent removed mid-iteration

        n = len(profiles)
        mean_intr  = round(sum(p["epsilon_intrinsic"]     for p in profiles) / n, 4)
        mean_eco   = round(sum(p["epsilon_ecosystem"]     for p in profiles) / n, 4)
        mean_arch  = round(sum(p["epsilon_architectural"] for p in profiles) / n, 4)
        mean_eff   = round(sum(p["epsilon_effective"]     for p in profiles) / n, 4)
        mean_stc   = round(sum(p["stc_seconds"]           for p in profiles) / n, 1)

        # Session 30c: dominant_bottleneck uses mismatch fraction, not raw ε comparison.
        # An ecosystem of healthy high-ε agents (high φ, high β) is NOT agent-bottlenecked.
        # Mismatch = agents whose adaptive range exceeds their niche support.
        mismatched = sum(
            1 for p in profiles
            if (p.get("epsilon_intrinsic", 0) >= VOLATILE_EPSILON_THRESHOLD
                and p.get("phi", 1.0) < VOLATILE_PHI_CEILING
                and p.get("beta_mean", 1.0) < VOLATILE_BETA_CEILING)
        )
        mismatch_fraction = mismatched / n if n > 0 else 0.0

        if mismatch_fraction > MISMATCH_DOMINANT_THRESHOLD:
            dominant = "mismatch"
        elif mean_eco > mean_intr * 1.5:
            dominant = "environment"
        else:
            dominant = "balanced"

        return {
            "agent_count":                  n,
            "profiles":                     profiles,
            "mean_epsilon_intrinsic":       mean_intr,
            "mean_epsilon_ecosystem":       mean_eco,
            "mean_epsilon_environmental":   mean_eco,   # backward-compat alias
            "mean_epsilon_architectural":   mean_arch,
            "mean_epsilon_effective":       mean_eff,
            "mean_stc_seconds":             mean_stc,
            "badge_counts":                 badge_counts,
            "dominant_bottleneck":          dominant,
        }

    # ── SESSION 33 — observe() GOVERNANCE LOOP EXTENSION ──────────────────

    def apply_observation(self, observation_result) -> dict:
        """
        Apply an ObservationResult to the MELVKernel governance loop.

        Session 33 (v2.9.0): Integrates the observe() primitive output
        into the running kernel — updating agent φ, provisioning β from
        ResourcePolicy signals, recording the CI snapshot, and emitting
        governance events where warranted.

        This is the bridge between the observe() pipeline (external signals)
        and the MELVKernel governance loop (cooperative equilibrium tracking).

        Parameters
        ----------
        observation_result : ObservationResult
            Output of ObservationComputer.compute(). Must have agent_id
            matching a registered kernel agent (or a new agent is auto-
            registered at φ=0.5 as MATURING).

        Returns
        -------
        dict with keys:
          agent_updated    — True if kernel agent state was modified
          phi_applied      — new kernel φ value for the agent
          beta_provisioned — True if β was provisioned from observe result
          ci_snapshot      — CI value recorded (or None)
          governance_events — list of event descriptions emitted
          warnings          — list of non-fatal issues
        """
        from core.observe_schema import ObservationResult  # local import — avoid circular

        result = observation_result
        agent_id = result.agent_id
        events: list[str] = []
        warnings: list[str] = []

        # ── 1. Ensure agent is registered ─────────────────────────────────
        if agent_id not in self.agents:
            profile = AgentProfile(
                agent_id=agent_id,
                name=agent_id.upper(),
                domain="observed",
                phi=result.phi.value if result.phi.computable else 0.5,
                epsilon=result.epsilon.effective,
                status=AgentStatus.MATURING,
            )
            self.register_agent(profile)
            events.append(f"AUTO_REGISTERED: {agent_id} (from observe() signal)")
        else:
            agent = self.agents[agent_id]

        agent = self.agents[agent_id]

        # ── 2. Update φ if computed and status ②+ ──────────────────────────
        phi_applied = agent.phi
        if result.phi.computable and result.phi.status >= 2:
            # Drive kernel φ toward the observed value
            # Blend: kernel φ relaxes 20% per observe() call toward observed
            tau_observe = 0.2
            new_phi = agent.phi + tau_observe * (result.phi.value - agent.phi)
            new_phi = max(0.0, min(1.0, round(new_phi, 4)))
            agent.phi = new_phi
            phi_applied = new_phi
            events.append(
                f"PHI_UPDATED: {agent_id} φ {result.phi.value:.4f} → "
                f"kernel φ {new_phi:.4f}"
            )

            # Status promotion
            if agent.phi >= 0.75 and agent.status == AgentStatus.MATURING:
                agent.status = AgentStatus.ACTIVE
                events.append(f"STATUS_PROMOTED: {agent_id} MATURING → ACTIVE")

            # ── 2b. Equation 7 φ dynamics (alignment pass v3.3.0) ────────
            # Canonical gate: β×i_∞ < 1  (Jacobian-derived, FB-ABM V1.0 confirmed).
            # Compute β×i_∞ from observe() output: β, ε_effective, and η.
            # η is taken from result if available, else ETA_CANONICAL_DEFAULT.
            beta_i_inf = getattr(result, "beta_i_inf", None)
            if beta_i_inf is None and result.beta.computable:
                # Compute inline if observe_compute didn't supply it
                eta = ETA_CANONICAL_DEFAULT
                beta_i_inf = result.beta.value * _compute_i_inf(
                    I0_CANONICAL,
                    eta,
                    result.epsilon.effective,
                    result.beta.value,
                )

            eq7_delta, eq7_event = self._apply_phi_eq7(
                agent=agent,
                beta_i_inf=beta_i_inf,
                i_value=result.ci,   # CI proxy for max(0,1−i); None if gate unmet
                d_value=result.d_value,
            )
            if eq7_delta != 0.0:
                phi_applied = agent.phi
                events.append(eq7_event)

            # ── 2c. Stagnation detector ───────────────────────────────────
            if beta_i_inf is not None:
                stag = self.compute_stagnation_state(beta_i_inf, result.d_value)
                delta_gate = stag["delta_gate"]
                if stag["state"] != "STABLE":
                    events.append(
                        f"STAGNATION_DETECTOR [{stag['state']}]: {stag['description']}"
                    )
                # Always log telemetry for plotting against theoretical boundary
                events.append(
                    f"TELEMETRY: β×i∞={beta_i_inf:.4f} Δ_gate={delta_gate:+.4f} "
                    f"φ={phi_applied:.4f} D={result.d_value:.3f} "
                    f"state={stag['state']}"
                )
                # Attempt L2 telemetry write (non-blocking)
                try:
                    import os as _os
                    from core.telemetry import AIOSTelemetry as _Tel, L2Snapshot as _L2
                    _db = _os.environ.get(
                        "AIOS_DB_PATH",
                        _os.path.join(
                            _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))),
                            "aios_state.db",
                        ),
                    )
                    _tel = _Tel(_db)
                    _snap = _L2(
                        agent_id=agent_id,
                        phi=phi_applied,
                        beta_service=result.beta.value if result.beta.computable else None,
                        r_value=getattr(result, "r_value", None),
                        ci=result.ci,
                        beta_i_inf=round(beta_i_inf, 6),
                        delta_gate=round(delta_gate, 6),
                    )
                    _tel.log_l2(_snap)
                    _tel.close()
                except Exception:
                    pass  # telemetry non-critical; governance loop continues

        # ── 3. Provision β from observe() β result if status ③ ────────────
        beta_provisioned = False
        if result.beta.computable and result.beta.status >= 3:
            # Map ObservationResult β → kernel BetaEnvironment resource
            # Use "compute" as the primary resource (most general)
            beta_val = result.beta.value
            if 0.1 <= beta_val <= 3.0:
                self.beta.set("compute", round(beta_val, 4))
                beta_provisioned = True
                events.append(
                    f"BETA_PROVISIONED: compute β={beta_val:.4f} "
                    f"(from observe() β status ③)"
                )

        # ── 4. Update agent ε from observe() ε_effective ──────────────────
        eps_eff = result.epsilon.effective
        if eps_eff != agent.epsilon:
            old_eps = agent.epsilon
            agent.epsilon = round(eps_eff, 4)
            events.append(
                f"EPSILON_UPDATED: {agent_id} ε {old_eps:.4f} → {eps_eff:.4f}"
            )

        # ── 5. Record CI snapshot ──────────────────────────────────────────
        ci_snapshot = result.ci
        if ci_snapshot is not None:
            self._record_ci_snapshot()
            events.append(f"CI_SNAPSHOT: {ci_snapshot:.4f}")

        # ── 6. φ/σ divergence governance event ────────────────────────────
        if (result.phi_sigma_divergence is not None and
                result.phi_sigma_divergence > 0.2):
            events.append(
                f"DOMAIN_SHIFT_SIGNAL: φ/σ divergence = "
                f"{result.phi_sigma_divergence:.3f} for {agent_id}"
            )

        # ── 7. Persist agent if changed ────────────────────────────────────
        if self._persistence:
            self._persistence.save_agent(agent)

        return {
            "agent_updated":     True,
            "phi_applied":       phi_applied,
            "beta_provisioned":  beta_provisioned,
            "ci_snapshot":       ci_snapshot,
            "governance_events": events,
            "warnings":          warnings,
        }

    # ── SESSION 37 — DUNGBEETLE NODES ────────────────────────────────────────

    def compute_dungbeetle_nodes(self) -> dict:
        """
        Identify Dungbeetle nodes in the current service coupling graph Omega.

        Formal condition (canonical reference v1.2 Part VI Item 7):
          Node v is a Dungbeetle iff:
            beta_service(Omega) >= DUNGBEETLE_THRESHOLD
            AND
            beta_service(Omega_{-v}) < DUNGBEETLE_THRESHOLD

        That is: the full ecosystem is above the cooperation threshold, but
        removing node v alone drops the ecosystem below it.  v is therefore
        the critical enabler of cooperative viability.

        Sensitivity score:
          S_v = beta_service(Omega) - beta_service(Omega_{-v})

        Higher S_v = greater single-node dependency (governance risk).

        Epistemic status: DUNGBEETLE_THRESHOLD = 0.50 = PHI_GATEWAY_THRESHOLD,
        consistent with the Nadell et al. quorum analogy (MAIES Event 2).
        Empirical calibration pending (ABM Test Suite 3).

        Returns
        -------
        dict with keys:
          beta_service_full   : float  — beta_service of full Omega
          threshold           : float  — DUNGBEETLE_THRESHOLD (0.50)
          quorum_met          : bool   — beta_service_full >= threshold
          dungbeetle_nodes    : list[dict]  — one entry per Dungbeetle node:
                                  agent_id, beta_service_without, sensitivity,
                                  is_dungbeetle=True
          non_dungbeetle_nodes: list[dict]  — agents with is_dungbeetle=False
          warnings            : list[str]
        """
        warnings_out = []

        # Full ecosystem beta_service
        omega_full   = self.compute_omega()
        beta_full    = omega_full["beta_service"]
        n            = omega_full["n"]

        if n == 0:
            return {
                "beta_service_full":    0.0,
                "threshold":            DUNGBEETLE_THRESHOLD,
                "quorum_met":           False,
                "dungbeetle_nodes":     [],
                "non_dungbeetle_nodes": [],
                "warnings":             ["No agents registered — Dungbeetle analysis requires n >= 2"],
            }

        if n == 1:
            warnings_out.append("Only one agent — leave-one-out requires n >= 2")
            return {
                "beta_service_full":    round(beta_full, 4),
                "threshold":            DUNGBEETLE_THRESHOLD,
                "quorum_met":           beta_full >= DUNGBEETLE_THRESHOLD,
                "dungbeetle_nodes":     [],
                "non_dungbeetle_nodes": [],
                "warnings":             warnings_out,
            }

        quorum_met = beta_full >= DUNGBEETLE_THRESHOLD

        dungbeetle_nodes    = []
        non_dungbeetle_nodes = []

        ids = list(self.agents.keys())

        for agent_id in ids:
            # ── Leave-one-out: recompute Omega_{-v} ─────────────────────────
            # Temporarily capture the leave-one-out adjacency
            n_minus = n - 1
            if n_minus == 0:
                beta_without = 0.0
            else:
                remaining = [aid for aid in ids if aid != agent_id]
                idx_r     = {aid: i for i, aid in enumerate(remaining)}
                A_r       = np.zeros((n_minus, n_minus))

                recent  = self.interactions[-100:]
                weights_r: dict[tuple, list] = {}
                for rec in recent:
                    if rec.agent_a in idx_r and rec.agent_b in idx_r:
                        key = (rec.agent_a, rec.agent_b)
                        weights_r.setdefault(key, []).append(1.0 - rec.i_factor)

                for (a, b), vals in weights_r.items():
                    avg = sum(vals) / len(vals)
                    A_r[idx_r[a], idx_r[b]] = avg
                    A_r[idx_r[b], idx_r[a]] = avg

                if n_minus == 1:
                    # Scalar case: single agent has no coupling
                    beta_without = 0.0
                else:
                    eigs_r       = np.linalg.eigvalsh(A_r)
                    lambda_max_r = float(eigs_r[-1])
                    beta_without = lambda_max_r / n_minus

            sensitivity = round(beta_full - beta_without, 4)
            beta_without = round(beta_without, 4)

            # Dungbeetle condition: full >= threshold AND without < threshold
            is_dungbeetle = quorum_met and (beta_without < DUNGBEETLE_THRESHOLD)

            entry = {
                "agent_id":            agent_id,
                "beta_service_without": beta_without,
                "sensitivity":         sensitivity,

                "is_dungbeetle":       is_dungbeetle,
            }

            if is_dungbeetle:
                dungbeetle_nodes.append(entry)
            else:
                non_dungbeetle_nodes.append(entry)

        # Sort Dungbeetle nodes by sensitivity descending (highest risk first)
        dungbeetle_nodes.sort(key=lambda x: x["sensitivity"], reverse=True)

        if quorum_met and not dungbeetle_nodes:
            warnings_out.append(
                "Ecosystem above threshold but no Dungbeetle nodes found: "
                "cooperative viability is distributed (resilient topology)."
            )

        return {
            "beta_service_full":    round(beta_full, 4),
            "threshold":            DUNGBEETLE_THRESHOLD,
            "quorum_met":           quorum_met,
            "dungbeetle_nodes":     dungbeetle_nodes,
            "non_dungbeetle_nodes": non_dungbeetle_nodes,
            "warnings":             warnings_out,
        }

    # ── SESSION 37 — IRREVERSIBILITY BOUNDARY DIAGNOSTIC ─────────────────────

    def irreversibility_diagnostic(
        self,
        agent_id: str,
        eta:      float = 0.93,
        f_eligible: float = 1.0,
        t_gov:    float = T_GOV_DEFAULT,
    ) -> dict:
        """
        Compute the irreversibility boundary diagnostic for an agent.

        Canonical reference v1.2 Part VI Items 11, 12 (Discovery 1 & 2).
        Carryover document v3.0 Session 39 specification.

        Three-zone governance classification:
          VIABLE            : phi > phi_viable
          RECOVERABLE_URGENT: phi_irrev <= phi <= phi_viable
          IRREVERSIBLE      : phi < phi_irrev

        Parameters
        ----------
        agent_id    : str   — agent to evaluate
        eta         : float — saturation capacity in (0,1]; default bee-flower 0.93
        f_eligible  : float — fraction of post-disruption time with R < 0.50 AND
                              i < 1 simultaneously (measured externally from L1);
                              defaults to 1.0 (pessimistic: recovery clock never
                              freezes) until L1 telemetry provides empirical data.
        t_gov       : float — governance horizon in time units for phi_irrev;
                              default T_GOV_DEFAULT = 100 units.

        Derived quantities
        ------------------
        phi_viable ~= 1 - 1 / (epsilon x beta_norm x eta)
          The floor phi must exceed for cooperative viability.
          Lower eta => shallower attractor => smaller viable zone.
          Returns None (uncomputable) when epsilon x beta_norm x eta <= 0.

        phi_irrev = 1 - exp(-alpha x t_gov)
          The phi below which recovery within the governance horizon is
          operationally impossible given the build rate alpha.
          Uses PHI_BUILD_RATE_ALPHA (theoretical; ABM T1.5 pending).

        T_rec = (1/alpha) x ln((1-phi_current)/(1-phi_viable)) / f_eligible
          Expected recovery time from phi_current back to phi_viable.
          Returns None when phi_current >= phi_viable (already viable).
          Returns math.inf when f_eligible = 0 (recovery clock frozen).
          f_eligible < 1.0 extends T_rec proportionally (path-dependency).

        Returns
        -------
        dict with keys:
          agent_id        : str
          phi_current     : float
          epsilon         : float
          beta_norm       : float
          eta             : float
          phi_viable      : float | None
          phi_irrev       : float
          zone            : str   — VIABLE | RECOVERABLE_URGENT | IRREVERSIBLE
          zone_color      : str   — GREEN | AMBER | RED (dashboard convenience)
          t_rec           : float | None  — expected recovery time (None if already viable)
          f_eligible      : float
          alpha           : float — PHI_BUILD_RATE_ALPHA used
          warnings        : list[str]
          epistemic_status: str   — theoretical (ABM Test Suite 3 validation pending)
        """
        from core.melv_engine import _beta_norm  # already in scope but explicit

        warnings_out = []

        if agent_id not in self.agents:
            return {
                "agent_id":         agent_id,
                "zone":             "UNKNOWN",
                "zone_color":       "GREY",
                "warnings":         [f"Agent '{agent_id}' not registered in kernel"],
                "epistemic_status": "not_applicable",
            }

        agent   = self.agents[agent_id]
        phi_cur = agent.phi
        epsilon = agent.epsilon

        # beta_norm from BetaEnvironment (compute resource — most general)
        beta_raw  = self.beta.get("compute")
        beta_norm_val = _beta_norm(beta_raw)

        alpha = PHI_BUILD_RATE_ALPHA

        # ── phi_viable ───────────────────────────────────────────────────────
        denom = epsilon * beta_norm_val * eta
        if denom <= 1e-9:
            phi_viable = None
            warnings_out.append(
                f"phi_viable uncomputable: epsilon x beta_norm x eta = {denom:.6f} <= 0. "
                "Agent may have epsilon=0 or beta=0."
            )
        else:
            phi_viable_raw = 1.0 - 1.0 / denom
            # phi_viable is meaningful only in (0, 1); clamp for display
            phi_viable = max(0.0, min(1.0, round(phi_viable_raw, 5)))
            if phi_viable_raw <= 0.0:
                warnings_out.append(
                    f"phi_viable = {phi_viable_raw:.4f} <= 0: cooperative viability "
                    "does not require minimum phi — system is below saturation regime."
                )
                phi_viable = 0.0
            elif phi_viable_raw >= 1.0:
                warnings_out.append(
                    f"phi_viable = {phi_viable_raw:.4f} >= 1.0: epsilon x beta_norm x eta "
                    "is too low to sustain cooperation regardless of phi. "
                    "Increase epsilon, beta, or eta first."
                )
                phi_viable = 1.0

        # ── phi_irrev ────────────────────────────────────────────────────────
        phi_irrev = round(1.0 - math.exp(-alpha * t_gov), 5)

        # ── T_rec ────────────────────────────────────────────────────────────
        t_rec = None
        if phi_viable is not None and phi_cur < phi_viable:
            if f_eligible <= 0.0:
                t_rec = math.inf
                warnings_out.append(
                    "f_eligible = 0: recovery clock is frozen (R >= 0.50 or i >= 1.0 "
                    "in all post-disruption time). Recovery is path-blocked."
                )
            else:
                inner = (1.0 - phi_cur) / max(1.0 - phi_viable, 1e-9)
                if inner <= 0:
                    t_rec = math.inf
                else:
                    t_rec = round((1.0 / alpha) * math.log(inner) / f_eligible, 2)
                    if t_rec < 0:
                        # phi_cur > phi_viable path (shouldn't reach here but guard it)
                        t_rec = 0.0

        # ── Zone classification ───────────────────────────────────────────────
        # Priority order: IRREVERSIBLE > VIABLE > RECOVERABLE_URGENT
        # phi_irrev check must come first — an agent can have phi > phi_viable(clamped=0)
        # but still be below phi_irrev, which means recovery is impossible regardless.
        if phi_viable is None:
            zone       = "UNKNOWN"
            zone_color = "GREY"
            warnings_out.append("Zone classification unavailable: phi_viable not computable.")
        elif phi_cur < phi_irrev:
            zone       = "IRREVERSIBLE"
            zone_color = "RED"
            warnings_out.append(
                f"IRREVERSIBLE: phi={phi_cur:.4f} < phi_irrev={phi_irrev:.4f}. "
                f"Recovery within t_gov={t_gov} time units is operationally impossible "
                f"at current build rate alpha={alpha}. Governance intervention required."
            )
        elif phi_cur > phi_viable:
            zone       = "VIABLE"
            zone_color = "GREEN"
        else:
            zone       = "RECOVERABLE_URGENT"
            zone_color = "AMBER"
            if t_rec is not None and t_rec != math.inf:
                warnings_out.append(
                    f"RECOVERABLE_URGENT: phi={phi_cur:.4f}. "
                    f"Estimated recovery time T_rec={t_rec:.1f} units "
                    f"(f_eligible={f_eligible:.2f}). Monitor phi build rate."
                )

        return {
            "agent_id":         agent_id,
            "phi_current":      round(phi_cur, 5),
            "epsilon":          round(epsilon, 5),
            "beta_norm":        round(beta_norm_val, 5),
            "eta":              round(eta, 5),
            "phi_viable":       phi_viable,
            "phi_irrev":        phi_irrev,
            "zone":             zone,
            "zone_color":       zone_color,
            "t_rec":            t_rec,
            "f_eligible":       f_eligible,
            "alpha":            alpha,
            "warnings":         warnings_out,
            "epistemic_status": (
                "② theoretical — phi_viable formula canonical v1.2; "
                "ABM Test Suite 3 validation pending"
            ),
        }

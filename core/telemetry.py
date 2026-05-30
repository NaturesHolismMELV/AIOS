"""
telemetry.py — MELVcore Three-Layer Logging + η Estimation (v3.1.1)
====================================================================

Session 36 deliverable: three-layer logging architecture and BI-NLS
η (saturation capacity) estimation for the cooperation-evolution equation
in its saturation form (Equation 1a).

Architecture
------------
  L1  Per-exchange  — C_proxy, B_proxy, TAX_proxy, timestamps, agent IDs
                       Rotating 7-day buffer in SQLite (telemetry_l1)
  L2  Per-snapshot  — i(t), φ, β_service, R, σ, CI, η_estimate, RSE
                       Persistent in SQLite (telemetry_l2)
  L3  Per-cycle     — η_posterior, variance, last_updated, interaction_count
                       Persistent in SQLite (telemetry_l3)

η Estimation — BI-NLS Algorithm
---------------------------------
Sensitivity function S(u):
    S(u) = tanh(u) − u / cosh²(u)
    u    = ε × φ × β_norm / η

Well-identified when S(u) > 0.3, i.e. u > 0.3.
Gauss-Newton update (damped):
    η_{k+1} = η_k − [Σ rⱼ · S(uⱼ)] / [Σ S(uⱼ)² + λ]

RSE Thresholds (from canonical reference v1.2, Appendix B)
    RSE_EXCELLENT   < 0.02   (2%)   — η well-estimated, high confidence
    RSE_ACCEPTABLE  < 0.05   (5%)   — usable estimate
    RSE_POOR        ≥ 0.10  (10%)   — unreliable; flag for review

η Governance Thresholds
    ETA_STABLE_THRESHOLD    < 0.05  (5%)  quarterly variation — monitor only
    ETA_DECLINING_THRESHOLD > 0.15  (15%) quarterly decline   — investigate
    ETA_CRITICAL_RATIO        0.7         η < 0.7×η_arch      — critical alert

D(t) — Disruption Intensity (Equation 7 gate)
    D(t) = max(0, ΔC/C_base + ΔTAX/TAX_base)
    Populated by L1 rolling mean; initially 0.0 until sufficient L1 data.

Author: Laurence W. Evans | ORCID: 0009-0001-0963-1840 | Cape Town, South Africa
Session: 36 · Version: 3.1.1
"""

from __future__ import annotations

import math
import sqlite3
import time
import threading
import logging
from dataclasses import dataclass, field
from typing import Optional, List

logger = logging.getLogger(__name__)

# ── RSE Thresholds ────────────────────────────────────────────────────────────

RSE_EXCELLENT   = 0.02   # < 2%  : η well-estimated, high confidence
RSE_ACCEPTABLE  = 0.05   # 2–5%  : usable estimate
RSE_POOR        = 0.10   # ≥ 10% : unreliable; flag for review

# ── η Governance Thresholds ──────────────────────────────────────────────────

ETA_STABLE_THRESHOLD    = 0.05   # < 5% quarterly variation → monitor only
ETA_DECLINING_THRESHOLD = 0.15   # > 15% quarterly decline  → investigate
ETA_CRITICAL_RATIO      = 0.70   # η < 0.7 × η_arch         → critical alert

# ── η Initial / Architectural Values ─────────────────────────────────────────

ETA_ARCHITECTURAL_DEFAULT = 0.93   # bee-flower mutualism calibration
ETA_INITIAL               = 0.93   # starting estimate before BI-NLS converges
ETA_IDENTIFICATION_THRESHOLD = 0.3  # S(u) > 0.3 required; u ≈ 0.3 needed
ETA_MIN_INTERACTIONS      = 100    # minimum L1 records before BI-NLS runs

# ── Proxy Weights ─────────────────────────────────────────────────────────────
# These are dimensionless scale factors; adjust per deployment profile.

COMPUTE_COST_PER_TOKEN  = 1.0e-6   # cost per input token (normalised)
LATENCY_PENALTY_SCALE   = 0.001    # latency penalty per ms above baseline
ERROR_RATE_SCALE        = 1.0      # full C-unit penalty at error_rate = 1.0
TASK_COMPLETION_WEIGHT  = 1.0      # B_proxy contribution from completion score
DOWNSTREAM_UTILITY_WEIGHT = 0.5    # B_proxy contribution from downstream utility
INFORMATION_GAIN_WEIGHT = 0.3      # B_proxy contribution from information gain

L1_RETENTION_DAYS = 7              # rotate L1 records older than this

# ── SQLite Schema Fragment ────────────────────────────────────────────────────
# Added to the main aios_state.db via AIOSTelemetry.apply_schema().

TELEMETRY_SCHEMA = """
CREATE TABLE IF NOT EXISTS telemetry_l1 (
    rowid            INTEGER PRIMARY KEY AUTOINCREMENT,
    agent_id         TEXT    NOT NULL,
    session_id       TEXT,
    task_type        TEXT,
    c_proxy          REAL    NOT NULL,
    b_proxy          REAL    NOT NULL,
    tax_proxy        REAL    NOT NULL DEFAULT 0.0,
    token_count      INTEGER,
    latency_ms       REAL,
    error_rate       REAL    DEFAULT 0.0,
    task_completion  REAL    DEFAULT 0.0,
    downstream_utility REAL  DEFAULT 0.0,
    information_gain REAL    DEFAULT 0.0,
    timestamp        REAL    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_tl1_agent ON telemetry_l1(agent_id);
CREATE INDEX IF NOT EXISTS idx_tl1_ts    ON telemetry_l1(timestamp);

CREATE TABLE IF NOT EXISTS telemetry_l2 (
    rowid          INTEGER PRIMARY KEY AUTOINCREMENT,
    agent_id       TEXT    NOT NULL,
    session_id     TEXT,
    i_value        REAL,
    phi            REAL,
    beta_service   REAL,
    r_value        REAL,
    sigma          REAL,
    ci             REAL,
    eta_estimate   REAL,
    rse            REAL,
    rse_band       TEXT,
    timestamp      REAL    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_tl2_agent ON telemetry_l2(agent_id);
CREATE INDEX IF NOT EXISTS idx_tl2_ts    ON telemetry_l2(timestamp);

CREATE TABLE IF NOT EXISTS telemetry_l3 (
    agent_id          TEXT    PRIMARY KEY,
    eta_posterior     REAL    NOT NULL DEFAULT 0.93,
    eta_variance      REAL    NOT NULL DEFAULT 1.0,
    eta_architectural REAL    NOT NULL DEFAULT 0.93,
    interaction_count INTEGER NOT NULL DEFAULT 0,
    last_updated      REAL    NOT NULL,
    governance_flag   TEXT
);
"""

# ─────────────────────────────────────────────────────────────────────────────
# Dataclasses
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class L1Record:
    """
    Per-exchange telemetry record (Layer 1).

    C_proxy  = token_count × compute_cost + latency_penalty + error_rate_degradation
    B_proxy  = task_completion_score + downstream_utility + information_gain
    TAX_proxy= protocol_negotiation_time + context_window_management + alignment_checking
               expressed as dimensionless fraction of C_proxy (canonical C6).
    """
    agent_id:           str
    c_proxy:            float
    b_proxy:            float
    tax_proxy:          float           = 0.0
    session_id:         Optional[str]   = None
    task_type:          Optional[str]   = None
    token_count:        Optional[int]   = None
    latency_ms:         Optional[float] = None
    error_rate:         float           = 0.0
    task_completion:    float           = 0.0
    downstream_utility: float           = 0.0
    information_gain:   float           = 0.0
    timestamp:          float           = field(default_factory=time.time)


@dataclass
class L2Snapshot:
    """
    Per-batch/session telemetry snapshot (Layer 2).

    Captures all governance-relevant state variables at a point in time.
    rse_band is one of 'EXCELLENT', 'ACCEPTABLE', 'POOR', or None if not
    yet estimated.
    """
    agent_id:       str
    i_value:        Optional[float] = None   # i(t) cooperation-evolution value
    phi:            Optional[float] = None   # φ perpetuity
    beta_service:   Optional[float] = None   # β_service (raw)
    r_value:        Optional[float] = None   # R = C/B gateway ratio
    sigma:          Optional[float] = None   # σ niche matching coefficient
    ci:             Optional[float] = None   # CI cooperation index
    eta_estimate:   Optional[float] = None   # current η estimate
    rse:            Optional[float] = None   # residual standard error
    rse_band:       Optional[str]   = None   # 'EXCELLENT'|'ACCEPTABLE'|'POOR'
    session_id:     Optional[str]   = None
    timestamp:      float           = field(default_factory=time.time)


@dataclass
class L3EtaEstimate:
    """
    Per-estimation-cycle η posterior (Layer 3).

    Populated by BI-NLS algorithm after MIN_INTERACTIONS L1 records.
    governance_flag is one of None, 'STABLE', 'DECLINING', 'CRITICAL'.
    """
    agent_id:          str
    eta_posterior:     float = ETA_INITIAL
    eta_variance:      float = 1.0
    eta_architectural: float = ETA_ARCHITECTURAL_DEFAULT
    interaction_count: int   = 0
    last_updated:      float = field(default_factory=time.time)
    governance_flag:   Optional[str] = None   # None | STABLE | DECLINING | CRITICAL


# ─────────────────────────────────────────────────────────────────────────────
# BI-NLS η Estimation
# ─────────────────────────────────────────────────────────────────────────────

def _sensitivity(u: float) -> float:
    """
    S(u) — BI-NLS sensitivity function.

    S(u) = tanh(u) − u / cosh²(u)

    Properties:
      - S(0) = 0
      - S(u) > 0 for u > 0
      - Peaks at u ≈ 1.2 with S ≈ 0.43
      - Well-identified when S(u) > 0.3  (u > ~0.7)
      - Numerically stable: cosh clipped at 1e6 to prevent overflow
    """
    # Guard against very large u to prevent cosh overflow
    u_clamped = min(abs(u), 500.0) * (1.0 if u >= 0 else -1.0)
    try:
        ch = math.cosh(u_clamped)
        ch_sq = ch * ch
        # cosh grows exponentially; tanh saturates → S → 0 for large u
        return math.tanh(u_clamped) - u_clamped / ch_sq
    except (OverflowError, ZeroDivisionError):
        return 0.0


def _is_identified(u: float) -> bool:
    """Return True when S(u) > ETA_IDENTIFICATION_THRESHOLD (u ≈ 0.7+)."""
    return _sensitivity(abs(u)) > ETA_IDENTIFICATION_THRESHOLD


def estimate_eta_binls(
    observations: List[dict],
    eta_init: float = ETA_INITIAL,
    lambda_damp: float = 1e-4,
    max_iter: int = 50,
    tol: float = 1e-6,
) -> dict:
    """
    BI-NLS (Bounded Iterative Non-Linear Least Squares) η estimator.

    Each observation is a dict with keys:
        i_observed  : float  — observed i(t)
        epsilon     : float  — ε_effective
        phi         : float  — φ value
        beta_norm   : float  — β_norm = β/(1+β)

    Equation 1a prediction:
        i_pred = i₀ × (1 − η × tanh(ε × φ × β_norm / η))
    Here we assume i₀ = 1.0 (normalised). If i₀ ≠ 1.0, scale externally.

    Gauss-Newton update (damped):
        η_{k+1} = η_k − [Σ rⱼ · S(uⱼ)] / [Σ S(uⱼ)² + λ]

    Returns dict with:
        eta         : float — final η estimate
        rse         : float — residual standard error
        rse_band    : str   — 'EXCELLENT' | 'ACCEPTABLE' | 'POOR'
        converged   : bool
        iterations  : int
        n_obs       : int   — number of observations used
        n_identified: int   — observations where S(u) > threshold
        warnings    : list[str]
    """
    warnings_out: List[str] = []

    # Filter to valid observations with all required fields
    valid = []
    for obs in observations:
        try:
            i_obs   = float(obs["i_observed"])
            eps     = float(obs["epsilon"])
            phi     = float(obs["phi"])
            bn      = float(obs["beta_norm"])
            if any(math.isnan(v) or math.isinf(v) for v in (i_obs, eps, phi, bn)):
                continue
            if phi <= 0 or bn <= 0 or eps <= 0:
                continue
            valid.append((i_obs, eps, phi, bn))
        except (KeyError, ValueError, TypeError):
            continue

    n_obs = len(valid)
    if n_obs < 10:
        warnings_out.append(
            f"Insufficient observations for BI-NLS: {n_obs} < 10 required"
        )
        return {
            "eta": eta_init,
            "rse": None,
            "rse_band": None,
            "converged": False,
            "iterations": 0,
            "n_obs": n_obs,
            "n_identified": 0,
            "warnings": warnings_out,
        }

    eta = max(0.01, min(1.0, eta_init))
    converged = False
    n_iter = 0

    for n_iter in range(1, max_iter + 1):
        numerator   = 0.0
        denominator = lambda_damp   # damping
        n_identified = 0

        for i_obs, eps, phi, bn in valid:
            u = eps * phi * bn / eta
            s = _sensitivity(u)
            if _sensitivity(abs(u)) > ETA_IDENTIFICATION_THRESHOLD:
                n_identified += 1
            # Residual: r = i_observed − i_predicted
            # i_pred = 1 − η × tanh(u)   [i₀ = 1.0 normalised]
            i_pred = 1.0 - eta * math.tanh(u)
            r = i_obs - i_pred
            # Jacobian w.r.t. η: ∂i_pred/∂η = −S(u)  (from canonical Appendix B)
            numerator   += r * s
            denominator += s * s

        delta = -numerator / denominator   # canonical: η_{k+1} = η_k − Σr·S / Σ S²
        eta_new = eta + delta
        eta_new = max(0.01, min(1.0, eta_new))

        if abs(eta_new - eta) < tol:
            eta = eta_new
            converged = True
            break
        eta = eta_new

    # Compute RSE
    ss_res = 0.0
    for i_obs, eps, phi, bn in valid:
        u = eps * phi * bn / eta
        i_pred = 1.0 - eta * math.tanh(u)
        ss_res += (i_obs - i_pred) ** 2
    rse = math.sqrt(ss_res / max(n_obs - 1, 1))

    # Classify RSE band
    if rse < RSE_EXCELLENT:
        rse_band = "EXCELLENT"
    elif rse < RSE_ACCEPTABLE:
        rse_band = "ACCEPTABLE"
    else:
        rse_band = "POOR"
        warnings_out.append(
            f"RSE={rse:.4f} ≥ {RSE_POOR}: η estimate unreliable; flag for review"
        )

    if n_identified < n_obs * 0.5:
        warnings_out.append(
            f"Only {n_identified}/{n_obs} observations are well-identified "
            f"(S(u) > {ETA_IDENTIFICATION_THRESHOLD}). "
            f"Accumulate more cooperative interactions (target ≥ 100)."
        )

    return {
        "eta":          eta,
        "rse":          rse,
        "rse_band":     rse_band,
        "converged":    converged,
        "iterations":   n_iter,
        "n_obs":        n_obs,
        "n_identified": n_identified,
        "warnings":     warnings_out,
    }


def rse_band(rse: float) -> str:
    """Classify an RSE value into 'EXCELLENT', 'ACCEPTABLE', or 'POOR'."""
    if rse < RSE_EXCELLENT:
        return "EXCELLENT"
    elif rse < RSE_ACCEPTABLE:
        return "ACCEPTABLE"
    return "POOR"


# ─────────────────────────────────────────────────────────────────────────────
# D(t) computation from L1 rolling data
# ─────────────────────────────────────────────────────────────────────────────

def compute_d_value(
    recent_records: List[L1Record],
    c_base:   Optional[float] = None,
    tax_base: Optional[float] = None,
) -> float:
    """
    Compute D(t) disruption intensity from recent L1 records.

    D(t) = max(0, ΔC/C_base + ΔTAX/TAX_base)   [Equation 7]

    ΔC   = mean(recent c_proxy)   − c_base
    ΔTAX = mean(recent tax_proxy) − tax_base

    If c_base or tax_base is None, uses the mean of all provided records
    as the baseline (bootstrap mode — D(t) will be 0 until a stable base
    is established).

    Returns 0.0 if fewer than 2 records are provided.
    """
    if len(recent_records) < 2:
        return 0.0

    c_values   = [r.c_proxy   for r in recent_records]
    tax_values = [r.tax_proxy for r in recent_records]

    mean_c   = sum(c_values)   / len(c_values)
    mean_tax = sum(tax_values) / len(tax_values)

    base_c   = c_base   if c_base   is not None else mean_c
    base_tax = tax_base if tax_base is not None else mean_tax

    delta_c   = (mean_c   - base_c)   / max(base_c,   1e-9)
    delta_tax = (mean_tax - base_tax) / max(base_tax, 1e-9)

    return max(0.0, delta_c + delta_tax)


# ─────────────────────────────────────────────────────────────────────────────
# η Governance Flag
# ─────────────────────────────────────────────────────────────────────────────

def eta_governance_flag(
    eta_current:     float,
    eta_previous:    Optional[float],
    eta_architectural: float = ETA_ARCHITECTURAL_DEFAULT,
) -> Optional[str]:
    """
    Return a governance flag for η changes.

    None         — insufficient history
    'STABLE'     — < 5% quarterly variation
    'DECLINING'  — > 15% quarterly decline
    'CRITICAL'   — η < 0.7 × η_architectural
    """
    if eta_current < ETA_CRITICAL_RATIO * eta_architectural:
        return "CRITICAL"

    if eta_previous is None:
        return None

    if eta_previous <= 0:
        return None

    change = abs(eta_current - eta_previous) / eta_previous
    decline = (eta_previous - eta_current) / eta_previous

    if decline > ETA_DECLINING_THRESHOLD:
        return "DECLINING"
    if change < ETA_STABLE_THRESHOLD:
        return "STABLE"
    return None


# ─────────────────────────────────────────────────────────────────────────────
# AIOSTelemetry — Thread-safe telemetry layer
# ─────────────────────────────────────────────────────────────────────────────

class AIOSTelemetry:
    """
    Thread-safe telemetry layer for MELVcore three-layer logging.

    Connects to the existing aios_state.db (passed as db_path) and adds
    telemetry_l1, telemetry_l2, telemetry_l3 tables.  All methods are
    synchronous and safe to call from FastAPI background tasks.

    Usage
    -----
        telemetry = AIOSTelemetry(db_path)
        telemetry.log_l1(record)
        telemetry.log_l2(snapshot)
        eta_state = telemetry.get_l3(agent_id)
        telemetry.run_eta_cycle(agent_id, observations)
    """

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._lock   = threading.Lock()
        self._conn: Optional[sqlite3.Connection] = None
        self._connect()

    def _connect(self) -> None:
        self._conn = sqlite3.connect(
            self.db_path,
            check_same_thread=False,
            isolation_level=None,
        )
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self.apply_schema()
        logger.debug("AIOSTelemetry: schema applied to %s", self.db_path)

    def apply_schema(self) -> None:
        """Idempotent — CREATE TABLE IF NOT EXISTS."""
        with self._lock:
            self._conn.executescript(TELEMETRY_SCHEMA)

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None

    # ── L1 ─────────────────────────────────────────────────────────────────

    def log_l1(self, record: L1Record) -> None:
        """Append a per-exchange L1 record and rotate old rows."""
        with self._lock:
            try:
                self._conn.execute(
                    """INSERT INTO telemetry_l1
                       (agent_id, session_id, task_type,
                        c_proxy, b_proxy, tax_proxy,
                        token_count, latency_ms, error_rate,
                        task_completion, downstream_utility, information_gain,
                        timestamp)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (
                        record.agent_id,
                        record.session_id,
                        record.task_type,
                        record.c_proxy,
                        record.b_proxy,
                        record.tax_proxy,
                        record.token_count,
                        record.latency_ms,
                        record.error_rate,
                        record.task_completion,
                        record.downstream_utility,
                        record.information_gain,
                        record.timestamp,
                    ),
                )
                self._rotate_l1()
            except Exception as exc:
                logger.warning("log_l1 failed: %s", exc)

    def _rotate_l1(self) -> None:
        """Delete L1 records older than L1_RETENTION_DAYS."""
        cutoff = time.time() - L1_RETENTION_DAYS * 86400
        self._conn.execute(
            "DELETE FROM telemetry_l1 WHERE timestamp < ?", (cutoff,)
        )

    def get_l1_recent(
        self,
        agent_id: str,
        n: int = 200,
    ) -> List[L1Record]:
        """Return up to n most recent L1 records for agent_id."""
        with self._lock:
            try:
                rows = self._conn.execute(
                    """SELECT agent_id, session_id, task_type,
                              c_proxy, b_proxy, tax_proxy,
                              token_count, latency_ms, error_rate,
                              task_completion, downstream_utility, information_gain,
                              timestamp
                       FROM telemetry_l1
                       WHERE agent_id = ?
                       ORDER BY timestamp DESC LIMIT ?""",
                    (agent_id, n),
                ).fetchall()
                return [
                    L1Record(
                        agent_id=r[0],
                        session_id=r[1],
                        task_type=r[2],
                        c_proxy=r[3],
                        b_proxy=r[4],
                        tax_proxy=r[5],
                        token_count=r[6],
                        latency_ms=r[7],
                        error_rate=r[8] or 0.0,
                        task_completion=r[9] or 0.0,
                        downstream_utility=r[10] or 0.0,
                        information_gain=r[11] or 0.0,
                        timestamp=r[12],
                    )
                    for r in rows
                ]
            except Exception as exc:
                logger.warning("get_l1_recent failed: %s", exc)
                return []

    def get_l1_count(self, agent_id: str) -> int:
        """Return total L1 record count for agent_id (within retention window)."""
        with self._lock:
            try:
                row = self._conn.execute(
                    "SELECT COUNT(*) FROM telemetry_l1 WHERE agent_id = ?",
                    (agent_id,),
                ).fetchone()
                return row[0] if row else 0
            except Exception as exc:
                logger.warning("get_l1_count failed: %s", exc)
                return 0

    # ── L2 ─────────────────────────────────────────────────────────────────

    def log_l2(self, snapshot: L2Snapshot) -> None:
        """Append a per-snapshot L2 record."""
        with self._lock:
            try:
                self._conn.execute(
                    """INSERT INTO telemetry_l2
                       (agent_id, session_id,
                        i_value, phi, beta_service, r_value, sigma, ci,
                        eta_estimate, rse, rse_band, timestamp)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (
                        snapshot.agent_id,
                        snapshot.session_id,
                        snapshot.i_value,
                        snapshot.phi,
                        snapshot.beta_service,
                        snapshot.r_value,
                        snapshot.sigma,
                        snapshot.ci,
                        snapshot.eta_estimate,
                        snapshot.rse,
                        snapshot.rse_band,
                        snapshot.timestamp,
                    ),
                )
            except Exception as exc:
                logger.warning("log_l2 failed: %s", exc)

    def get_l2_recent(self, agent_id: str, n: int = 100) -> List[L2Snapshot]:
        """Return up to n most recent L2 snapshots for agent_id."""
        with self._lock:
            try:
                rows = self._conn.execute(
                    """SELECT agent_id, session_id,
                              i_value, phi, beta_service, r_value, sigma, ci,
                              eta_estimate, rse, rse_band, timestamp
                       FROM telemetry_l2
                       WHERE agent_id = ?
                       ORDER BY timestamp DESC LIMIT ?""",
                    (agent_id, n),
                ).fetchall()
                return [
                    L2Snapshot(
                        agent_id=r[0],
                        session_id=r[1],
                        i_value=r[2],
                        phi=r[3],
                        beta_service=r[4],
                        r_value=r[5],
                        sigma=r[6],
                        ci=r[7],
                        eta_estimate=r[8],
                        rse=r[9],
                        rse_band=r[10],
                        timestamp=r[11],
                    )
                    for r in rows
                ]
            except Exception as exc:
                logger.warning("get_l2_recent failed: %s", exc)
                return []

    # ── L3 ─────────────────────────────────────────────────────────────────

    def get_l3(self, agent_id: str) -> L3EtaEstimate:
        """
        Return the current L3 η estimate for agent_id.
        Creates a default record if none exists.
        """
        with self._lock:
            try:
                row = self._conn.execute(
                    """SELECT agent_id, eta_posterior, eta_variance,
                              eta_architectural, interaction_count,
                              last_updated, governance_flag
                       FROM telemetry_l3 WHERE agent_id = ?""",
                    (agent_id,),
                ).fetchone()
                if row:
                    return L3EtaEstimate(
                        agent_id=row[0],
                        eta_posterior=row[1],
                        eta_variance=row[2],
                        eta_architectural=row[3],
                        interaction_count=row[4],
                        last_updated=row[5],
                        governance_flag=row[6],
                    )
                # Initialise default
                default = L3EtaEstimate(agent_id=agent_id)
                self._save_l3(default)
                return default
            except Exception as exc:
                logger.warning("get_l3 failed: %s", exc)
                return L3EtaEstimate(agent_id=agent_id)

    def _save_l3(self, estimate: L3EtaEstimate) -> None:
        """Upsert L3 record (must be called with lock held)."""
        self._conn.execute(
            """INSERT INTO telemetry_l3
               (agent_id, eta_posterior, eta_variance,
                eta_architectural, interaction_count,
                last_updated, governance_flag)
               VALUES (?,?,?,?,?,?,?)
               ON CONFLICT(agent_id) DO UPDATE SET
                 eta_posterior     = excluded.eta_posterior,
                 eta_variance      = excluded.eta_variance,
                 eta_architectural = excluded.eta_architectural,
                 interaction_count = excluded.interaction_count,
                 last_updated      = excluded.last_updated,
                 governance_flag   = excluded.governance_flag""",
            (
                estimate.agent_id,
                estimate.eta_posterior,
                estimate.eta_variance,
                estimate.eta_architectural,
                estimate.interaction_count,
                estimate.last_updated,
                estimate.governance_flag,
            ),
        )

    # ── η Estimation Cycle ─────────────────────────────────────────────────

    def run_eta_cycle(
        self,
        agent_id:     str,
        observations: List[dict],
        eta_init:     Optional[float] = None,
    ) -> dict:
        """
        Run one full BI-NLS estimation cycle for agent_id.

        observations — list of dicts, each with:
            i_observed, epsilon, phi, beta_norm

        Updates the L3 record and returns the estimation result dict
        from estimate_eta_binls() augmented with governance_flag.

        Governance flag is set based on comparison with the previous L3
        η_posterior. Requires ≥ ETA_MIN_INTERACTIONS L1 records (checked
        externally by the caller; this method does not gate on L1 count).
        """
        prev_l3 = self.get_l3(agent_id)
        init_eta = eta_init if eta_init is not None else prev_l3.eta_posterior

        result = estimate_eta_binls(observations, eta_init=init_eta)

        flag = eta_governance_flag(
            eta_current=result["eta"],
            eta_previous=prev_l3.eta_posterior if prev_l3.interaction_count > 0 else None,
            eta_architectural=prev_l3.eta_architectural,
        )

        new_l3 = L3EtaEstimate(
            agent_id=agent_id,
            eta_posterior=result["eta"],
            eta_variance=result.get("rse", 1.0) or 1.0,
            eta_architectural=prev_l3.eta_architectural,
            interaction_count=prev_l3.interaction_count + result["n_obs"],
            last_updated=time.time(),
            governance_flag=flag,
        )

        with self._lock:
            self._save_l3(new_l3)

        result["governance_flag"] = flag
        return result

    def compute_d_value_for_agent(
        self,
        agent_id: str,
        c_base:   Optional[float] = None,
        tax_base: Optional[float] = None,
        n_recent: int = 50,
    ) -> float:
        """
        Compute D(t) for agent_id from its recent L1 records.

        Returns 0.0 if fewer than 2 L1 records exist (bootstrap mode).
        """
        records = self.get_l1_recent(agent_id, n=n_recent)
        return compute_d_value(records, c_base=c_base, tax_base=tax_base)


# ─────────────────────────────────────────────────────────────────────────────
# C / B / TAX Proxy Builders
# ─────────────────────────────────────────────────────────────────────────────

def build_c_proxy(
    token_count:    int   = 0,
    latency_ms:     float = 0.0,
    error_rate:     float = 0.0,
    latency_baseline_ms: float = 500.0,
) -> float:
    """
    C_proxy = token_count × COMPUTE_COST_PER_TOKEN
              + max(0, latency_ms − latency_baseline_ms) × LATENCY_PENALTY_SCALE
              + error_rate × ERROR_RATE_SCALE

    All terms are non-negative and dimensionless.
    """
    token_cost    = token_count * COMPUTE_COST_PER_TOKEN
    latency_extra = max(0.0, latency_ms - latency_baseline_ms)
    latency_cost  = latency_extra * LATENCY_PENALTY_SCALE
    error_cost    = error_rate * ERROR_RATE_SCALE
    return token_cost + latency_cost + error_cost


def build_b_proxy(
    task_completion:    float = 0.0,
    downstream_utility: float = 0.0,
    information_gain:   float = 0.0,
) -> float:
    """
    B_proxy = task_completion × TASK_COMPLETION_WEIGHT
              + downstream_utility × DOWNSTREAM_UTILITY_WEIGHT
              + information_gain × INFORMATION_GAIN_WEIGHT

    All inputs are normalised to [0, 1] before weighting.
    """
    tc  = max(0.0, min(1.0, task_completion))    * TASK_COMPLETION_WEIGHT
    du  = max(0.0, min(1.0, downstream_utility)) * DOWNSTREAM_UTILITY_WEIGHT
    ig  = max(0.0, min(1.0, information_gain))   * INFORMATION_GAIN_WEIGHT
    return tc + du + ig


def build_tax_proxy(
    c_proxy:                    float,
    protocol_negotiation_frac:  float = 0.05,
    context_management_frac:    float = 0.03,
    alignment_checking_frac:    float = 0.02,
) -> float:
    """
    TAX_proxy = C_proxy × (protocol_negotiation + context_management + alignment_checking)

    Expressed as a dimensionless fraction of C_proxy (canonical C6 form).
    Default fractions from Kimi K2.6 MAIES Form C analysis:
      protocol negotiation ≈ 5%
      context management   ≈ 3%
      alignment checking   ≈ 2%
    """
    frac = protocol_negotiation_frac + context_management_frac + alignment_checking_frac
    return c_proxy * max(0.0, min(1.0, frac))

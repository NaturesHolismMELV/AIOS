"""
persistence.py — MELVcore SQLite Persistence Layer
===================================================
Session 12 · v1.4.0

Provides durable storage for all kernel state so the ecosystem
survives server restarts. Uses a single SQLite file (aios_state.db)
with one table per entity type, all JSON-serialised for simplicity
and forward compatibility.

Design principles
-----------------
- Zero mandatory schema migrations: all rows are JSON blobs; schema
  changes require no ALTER TABLE.
- Append-only writes for interactions, CI history, events, and
  oscillation events — never mutate history.
- Agents and beta are upserted (current state only).
- Sandbox runs/reports are stored for the registry.
- Startup restore is idempotent: re-registering an agent that already
  exists in the kernel is safe (kernel skips duplicates).

Usage (from server.py)
----------------------
    from core.persistence import AIOSPersistence
    store = AIOSPersistence()            # opens / creates aios_state.db
    store.restore_kernel(kernel)        # load persisted state at startup
    # … pass store to kernel so it can call save_ methods on write …
"""

import json
import logging
import os
import sqlite3
import time
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from core.melv_engine import MELVKernel

logger = logging.getLogger("aios.persistence")

# Default DB path — next to the project root (overridable via env var)
DEFAULT_DB_PATH = os.environ.get(
    "AIOS_DB_PATH",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "aios_state.db"),
)

# ── SCHEMA ─────────────────────────────────────────────────────────────────

_SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS agents (
    agent_id   TEXT PRIMARY KEY,
    data       TEXT NOT NULL,          -- JSON blob (AgentProfile.to_dict())
    updated_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS beta_state (
    id         INTEGER PRIMARY KEY CHECK (id = 1),  -- singleton row
    data       TEXT NOT NULL,          -- JSON blob (BetaEnvironment.to_dict())
    updated_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS interactions (
    rowid      INTEGER PRIMARY KEY AUTOINCREMENT,
    agent_a    TEXT NOT NULL,
    agent_b    TEXT NOT NULL,
    data       TEXT NOT NULL,          -- JSON blob (InteractionRecord.to_dict())
    timestamp  REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_interactions_ts ON interactions(timestamp);

CREATE TABLE IF NOT EXISTS bifurcation_events (
    rowid      INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id   TEXT NOT NULL,
    data       TEXT NOT NULL,          -- JSON blob (BifurcationEvent.to_dict())
    timestamp  REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS ci_history (
    rowid      INTEGER PRIMARY KEY AUTOINCREMENT,
    t          REAL NOT NULL,
    ci         REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_ci_history_t ON ci_history(t);

CREATE TABLE IF NOT EXISTS oscillation_events (
    rowid      INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id   TEXT NOT NULL,
    data       TEXT NOT NULL,          -- JSON blob (OscillationEvent.to_dict())
    timestamp  REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS sandbox_reports (
    rowid             INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id            TEXT NOT NULL UNIQUE,
    agent_id          TEXT NOT NULL,
    agent_domain      TEXT,
    verdict           TEXT NOT NULL,
    cls_score         REAL NOT NULL,
    data              TEXT NOT NULL,        -- JSON blob (CertificationReport.to_dict())
    certified_at      TEXT,
    assessment_scores TEXT                  -- JSON blob (AssessmentScores) or NULL
);
CREATE INDEX IF NOT EXISTS idx_sandbox_verdict ON sandbox_reports(verdict);

CREATE TABLE IF NOT EXISTS theorem_state (
    id         INTEGER PRIMARY KEY CHECK (id = 1),  -- singleton row
    data       TEXT NOT NULL,          -- JSON blob (full _theorem_state dict)
    updated_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS oxpecker_fragments (
    fragment_id   TEXT PRIMARY KEY,
    agent_a       TEXT NOT NULL,
    agent_b       TEXT NOT NULL,
    resource_type TEXT NOT NULL,
    fragment_data TEXT NOT NULL,   -- JSON blob
    status        TEXT DEFAULT 'pending',
    created_at    REAL NOT NULL,
    processed_at  REAL
);
CREATE INDEX IF NOT EXISTS idx_oxpecker_status ON oxpecker_fragments(status);
"""

# How many rows to keep per append-only table (rolling window)
MAX_INTERACTIONS  = 10_000
MAX_CI_HISTORY    = 5_000
MAX_BIF_EVENTS    = 2_000
MAX_OSC_EVENTS    = 1_000
MAX_OXP_FRAGMENTS = 500


class AIOSPersistence:
    """
    Thread-safe SQLite persistence for MELVcore kernel state.

    All public methods are synchronous and safe to call from FastAPI
    background tasks or the main thread; SQLite's WAL mode allows
    concurrent readers with a single writer.
    """

    def __init__(self, db_path: str = DEFAULT_DB_PATH):
        self.db_path = db_path
        self._conn: Optional[sqlite3.Connection] = None
        self._connect()
        logger.info("AIOSPersistence: opened %s", db_path)

    # ── LIFECYCLE ──────────────────────────────────────────────────────────

    def _connect(self):
        self._conn = sqlite3.connect(
            self.db_path,
            check_same_thread=False,   # FastAPI runs in multiple threads
            isolation_level=None,       # autocommit; we manage transactions
        )
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.executescript(_SCHEMA)
        logger.debug("AIOSPersistence: schema applied")

    def close(self):
        if self._conn:
            self._conn.close()
            self._conn = None

    # ── AGENTS ─────────────────────────────────────────────────────────────

    def save_agent(self, profile) -> None:
        """Upsert an AgentProfile (called on register_agent)."""
        try:
            self._conn.execute(
                """INSERT INTO agents (agent_id, data, updated_at)
                   VALUES (?, ?, ?)
                   ON CONFLICT(agent_id) DO UPDATE SET
                     data=excluded.data,
                     updated_at=excluded.updated_at""",
                (profile.agent_id, json.dumps(profile.to_dict()), time.time()),
            )
        except Exception as e:
            logger.warning("save_agent failed: %s", e)

    def load_agents(self) -> list[dict]:
        """Return all persisted agent dicts."""
        try:
            rows = self._conn.execute(
                "SELECT data FROM agents ORDER BY updated_at"
            ).fetchall()
            return [json.loads(r[0]) for r in rows]
        except Exception as e:
            logger.warning("load_agents failed: %s", e)
            return []

    # ── BETA STATE ─────────────────────────────────────────────────────────

    def save_beta(self, beta) -> None:
        """Upsert the singleton BetaEnvironment row."""
        try:
            self._conn.execute(
                """INSERT INTO beta_state (id, data, updated_at)
                   VALUES (1, ?, ?)
                   ON CONFLICT(id) DO UPDATE SET
                     data=excluded.data,
                     updated_at=excluded.updated_at""",
                (json.dumps(beta.to_dict()), time.time()),
            )
        except Exception as e:
            logger.warning("save_beta failed: %s", e)

    def load_beta(self) -> Optional[dict]:
        """Return persisted beta dict, or None if not yet stored."""
        try:
            row = self._conn.execute(
                "SELECT data FROM beta_state WHERE id=1"
            ).fetchone()
            return json.loads(row[0]) if row else None
        except Exception as e:
            logger.warning("load_beta failed: %s", e)
            return None

    # ── INTERACTIONS ───────────────────────────────────────────────────────

    def save_interaction(self, record) -> None:
        """Append an InteractionRecord (called after every record_interaction)."""
        try:
            self._conn.execute(
                "INSERT INTO interactions (agent_a, agent_b, data, timestamp) VALUES (?,?,?,?)",
                (record.agent_a, record.agent_b,
                 json.dumps(record.to_dict()), record.timestamp),
            )
            self._trim("interactions", MAX_INTERACTIONS)
        except Exception as e:
            logger.warning("save_interaction failed: %s", e)

    def load_interactions(self, n: int = MAX_INTERACTIONS) -> list[dict]:
        """Return the last n interaction dicts."""
        try:
            rows = self._conn.execute(
                "SELECT data FROM interactions ORDER BY timestamp DESC LIMIT ?", (n,)
            ).fetchall()
            return [json.loads(r[0]) for r in reversed(rows)]
        except Exception as e:
            logger.warning("load_interactions failed: %s", e)
            return []

    # ── BIFURCATION EVENTS ─────────────────────────────────────────────────

    def save_event(self, event) -> None:
        """Append a BifurcationEvent."""
        try:
            self._conn.execute(
                "INSERT INTO bifurcation_events (event_id, data, timestamp) VALUES (?,?,?)",
                (event.event_id, json.dumps(event.to_dict()), event.timestamp),
            )
            self._trim("bifurcation_events", MAX_BIF_EVENTS)
        except Exception as e:
            logger.warning("save_event failed: %s", e)

    def load_events(self, n: int = MAX_BIF_EVENTS) -> list[dict]:
        """Return the last n bifurcation event dicts."""
        try:
            rows = self._conn.execute(
                "SELECT data FROM bifurcation_events ORDER BY timestamp DESC LIMIT ?", (n,)
            ).fetchall()
            return [json.loads(r[0]) for r in reversed(rows)]
        except Exception as e:
            logger.warning("load_events failed: %s", e)
            return []

    def load_pair_events(
        self, agent_a: str, agent_b: str, n: int = 200
    ) -> list[dict]:
        """
        Session 22 Fix A2 — All persisted bifurcation events for a specific pair.
        Used by MELVKernel.get_pair_pattern() for long-horizon pattern queries.
        """
        try:
            rows = self._conn.execute(
                """SELECT data FROM bifurcation_events
                   WHERE (data LIKE ? AND data LIKE ?)
                   ORDER BY timestamp DESC LIMIT ?""",
                (f'%{agent_a}%', f'%{agent_b}%', n)
            ).fetchall()
            return [json.loads(r[0]) for r in rows]
        except Exception as e:
            logger.warning("load_pair_events failed: %s", e)
            return []

    # ── CI HISTORY ─────────────────────────────────────────────────────────

    def save_ci_snapshot(self, t: float, ci: float) -> None:
        """Append a single (t, ci) reading."""
        try:
            self._conn.execute(
                "INSERT INTO ci_history (t, ci) VALUES (?,?)", (t, ci)
            )
            self._trim("ci_history", MAX_CI_HISTORY)
        except Exception as e:
            logger.warning("save_ci_snapshot failed: %s", e)

    def load_ci_history(self, n: int = MAX_CI_HISTORY) -> list[tuple[float, float]]:
        """Return the last n (t, ci) tuples in chronological order."""
        try:
            rows = self._conn.execute(
                "SELECT t, ci FROM ci_history ORDER BY t DESC LIMIT ?", (n,)
            ).fetchall()
            return list(reversed(rows))
        except Exception as e:
            logger.warning("load_ci_history failed: %s", e)
            return []

    # ── OSCILLATION EVENTS ─────────────────────────────────────────────────

    def save_oscillation(self, event) -> None:
        """Append an OscillationEvent."""
        try:
            self._conn.execute(
                "INSERT INTO oscillation_events (event_id, data, timestamp) VALUES (?,?,?)",
                (event.event_id, json.dumps(event.to_dict()), event.timestamp),
            )
            self._trim("oscillation_events", MAX_OSC_EVENTS)
        except Exception as e:
            logger.warning("save_oscillation failed: %s", e)

    def load_oscillation_events(self, n: int = MAX_OSC_EVENTS) -> list[dict]:
        """Return the last n oscillation event dicts."""
        try:
            rows = self._conn.execute(
                "SELECT data FROM oscillation_events ORDER BY timestamp DESC LIMIT ?", (n,)
            ).fetchall()
            return [json.loads(r[0]) for r in reversed(rows)]
        except Exception as e:
            logger.warning("load_oscillation_events failed: %s", e)
            return []

    # ── SANDBOX REPORTS ────────────────────────────────────────────────────

    def save_sandbox_report(self, report, assessment_scores=None) -> None:
        """Upsert a CertificationReport (verdict != NOT_CERTIFIED only).

        assessment_scores — optional dict produced by the phi/epsilon assessment wizard.
        Stored as a JSON blob in the assessment_scores column for audit trail.
        """
        try:
            d = report.to_dict()
            anchor = d.get("certification_anchor", {})
            scores_json = json.dumps(assessment_scores) if assessment_scores else None
            self._conn.execute(
                """INSERT INTO sandbox_reports
                       (run_id, agent_id, agent_domain, verdict, cls_score, data,
                        certified_at, assessment_scores)
                   VALUES (?,?,?,?,?,?,?,?)
                   ON CONFLICT(run_id) DO UPDATE SET
                     verdict=excluded.verdict,
                     cls_score=excluded.cls_score,
                     data=excluded.data,
                     certified_at=excluded.certified_at,
                     assessment_scores=excluded.assessment_scores""",
                (
                    report.run_id,
                    report.agent_id,
                    report.agent_domain,
                    report.verdict,
                    report.cls_score,
                    json.dumps(d),
                    anchor.get("certified_at", ""),
                    scores_json,
                ),
            )
        except Exception as e:
            logger.warning("save_sandbox_report failed: %s", e)

    def load_sandbox_reports(self, certified_only: bool = False) -> list[dict]:
        """Return all (or only certified) sandbox reports."""
        try:
            if certified_only:
                rows = self._conn.execute(
                    "SELECT data FROM sandbox_reports WHERE verdict != 'NOT_CERTIFIED' ORDER BY rowid"
                ).fetchall()
            else:
                rows = self._conn.execute(
                    "SELECT data FROM sandbox_reports ORDER BY rowid"
                ).fetchall()
            return [json.loads(r[0]) for r in rows]
        except Exception as e:
            logger.warning("load_sandbox_reports failed: %s", e)
            return []

    # ── THEOREM STATE ──────────────────────────────────────────────────────

    def save_theorem_state(self, state: dict) -> None:
        """
        Session 24.2 Fix B — Persist the active theorem experiment state so
        Railway restarts do not wipe the prediction baseline.
        ci_at_prediction: null was the symptom — this is the cure.
        """
        try:
            self._conn.execute(
                """INSERT INTO theorem_state (id, data, updated_at)
                   VALUES (1, ?, ?)
                   ON CONFLICT(id) DO UPDATE SET
                     data=excluded.data,
                     updated_at=excluded.updated_at""",
                (json.dumps(state, default=str), time.time()),
            )
        except Exception as e:
            logger.warning("save_theorem_state failed: %s", e)

    def load_theorem_state(self) -> Optional[dict]:
        """Return persisted theorem state, or None if not yet stored."""
        try:
            row = self._conn.execute(
                "SELECT data FROM theorem_state WHERE id=1"
            ).fetchone()
            return json.loads(row[0]) if row else None
        except Exception as e:
            logger.warning("load_theorem_state failed: %s", e)
            return None

    # ── RESTORE ────────────────────────────────────────────────────────────

    def restore_kernel(self, kernel: "MELVKernel") -> dict:
        """
        Hydrate kernel state from the database at startup.

        Returns a summary dict of what was restored.
        """
        from core.melv_engine import (
            AgentProfile, AgentStatus, BetaEnvironment,
            InteractionRecord, BifurcationEvent, OscillationEvent, KernelAction,
        )

        summary = {
            "agents": 0, "interactions": 0, "events": 0,
            "ci_history": 0, "oscillations": 0, "beta_restored": False,
        }

        # ── Beta ───────────────────────────────────────────────────────────
        beta_data = self.load_beta()
        if beta_data:
            for resource, value in beta_data.items():
                if hasattr(kernel.beta, resource):
                    setattr(kernel.beta, resource, value)
            summary["beta_restored"] = True

        # ── Agents ─────────────────────────────────────────────────────────
        for ad in self.load_agents():
            if ad["agent_id"] not in kernel.agents:
                try:
                    status = AgentStatus(ad.get("status", "maturing"))
                    profile = AgentProfile(
                        agent_id    = ad["agent_id"],
                        name        = ad["name"],
                        domain      = ad["domain"],
                        phi         = ad.get("phi", 0.5),
                        epsilon     = ad.get("epsilon", 3.0),
                        beta_pref   = ad.get("beta_pref", 1.0),
                        status      = status,
                        capabilities= ad.get("capabilities", []),
                        created_at  = ad.get("created_at", time.time()),
                        task_count  = ad.get("task_count", 0),
                        success_rate= ad.get("success_rate", 0.0),
                        preferred_resource = ad.get("preferred_resource", None),
                        surplus_window     = ad.get("surplus_window", []),
                    )
                    kernel.agents[profile.agent_id] = profile
                    summary["agents"] += 1
                except Exception as e:
                    logger.warning("restore_kernel: skip agent %s: %s", ad.get("agent_id"), e)

        # ── Interactions ───────────────────────────────────────────────────
        for rd in self.load_interactions():
            try:
                record = InteractionRecord(
                    agent_a       = rd["agent_a"],
                    agent_b       = rd["agent_b"],
                    cost          = rd["cost"],
                    benefit       = rd["benefit"],
                    beta          = rd["beta"],
                    resource_type = rd.get("resource_type", "compute"),
                    timestamp     = rd["timestamp"],
                )
                kernel.interactions.append(record)
                summary["interactions"] += 1
            except Exception as e:
                logger.debug("restore_kernel: skip interaction: %s", e)
        # Trim to kernel max
        if len(kernel.interactions) > 5000:
            kernel.interactions = kernel.interactions[-5000:]

        # ── Bifurcation events ─────────────────────────────────────────────
        for ed in self.load_events():
            try:
                action = KernelAction(ed.get("action", "none"))
                event = BifurcationEvent(
                    event_id   = ed["event_id"],
                    agent_a    = ed["agent_a"],
                    agent_b    = ed["agent_b"],
                    beta_i_pre = ed["beta_i_pre"],
                    beta_i_post= ed["beta_i_post"],
                    action     = action,
                    description= ed["description"],
                    timestamp  = ed["timestamp"],
                    resolved   = ed.get("resolved", False),
                )
                kernel.events.append(event)
                summary["events"] += 1
            except Exception as e:
                logger.debug("restore_kernel: skip event: %s", e)
        # Restore event counter
        if kernel.events:
            last = kernel.events[-1].event_id  # "BIF-0042"
            try:
                kernel._event_counter = int(last.split("-")[-1])
            except Exception:
                pass

        # ── CI history ────────────────────────────────────────────────────
        kernel._ci_history = list(self.load_ci_history())
        summary["ci_history"] = len(kernel._ci_history)

        # ── Oscillation events ────────────────────────────────────────────
        for od in self.load_oscillation_events():
            try:
                osc = OscillationEvent(
                    event_id        = od["event_id"],
                    ci_peak         = od["ci_peak"],
                    ci_trough       = od["ci_trough"],
                    amplitude       = od["amplitude"],
                    period_sec      = od["period_sec"],
                    timestamp       = od["timestamp"],
                    implicated_pairs= od.get("implicated_pairs", []),
                )
                kernel.oscillation_events.append(osc)
                kernel._osc_counter += 1
                summary["oscillations"] += 1
            except Exception as e:
                logger.debug("restore_kernel: skip oscillation: %s", e)

        logger.info(
            "AIOSPersistence: restored — agents=%d interactions=%d "
            "events=%d ci_history=%d oscillations=%d beta=%s",
            summary["agents"], summary["interactions"], summary["events"],
            summary["ci_history"], summary["oscillations"], summary["beta_restored"],
        )
        return summary

    # ── OXPECKER FRAGMENTS (Session 27) ───────────────────────────────────

    def save_oxpecker_fragment(self, fragment: dict) -> None:
        """Insert a new oxpecker fragment. Rolling cap at MAX_OXP_FRAGMENTS."""
        try:
            self._conn.execute(
                """INSERT OR REPLACE INTO oxpecker_fragments
                   (fragment_id, agent_a, agent_b, resource_type,
                    fragment_data, status, created_at, processed_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    fragment["fragment_id"],
                    fragment["agent_a"],
                    fragment["agent_b"],
                    fragment["resource_type"],
                    json.dumps(fragment["fragment_data"]),
                    fragment.get("status", "pending"),
                    fragment["created_at"],
                    fragment.get("processed_at"),
                ),
            )
            self._trim_oxpecker_fragments()
        except Exception as e:
            logger.warning("save_oxpecker_fragment failed: %s", e)

    def update_oxpecker_fragment_status(
        self, fragment_id: str, status: str, processed_at: float
    ) -> None:
        """Mark a fragment as 'summarised' or 'recycled'."""
        try:
            self._conn.execute(
                """UPDATE oxpecker_fragments
                   SET status=?, processed_at=?
                   WHERE fragment_id=?""",
                (status, processed_at, fragment_id),
            )
        except Exception as e:
            logger.warning("update_oxpecker_fragment_status failed: %s", e)

    def load_pending_fragments(self, limit: int = 10) -> list:
        """Return up to `limit` pending fragments (oldest first)."""
        try:
            rows = self._conn.execute(
                """SELECT fragment_id, agent_a, agent_b, resource_type,
                          fragment_data, status, created_at, processed_at
                   FROM oxpecker_fragments
                   WHERE status='pending'
                   ORDER BY created_at ASC
                   LIMIT ?""",
                (limit,),
            ).fetchall()
            result = []
            for r in rows:
                result.append({
                    "fragment_id":   r[0],
                    "agent_a":       r[1],
                    "agent_b":       r[2],
                    "resource_type": r[3],
                    "fragment_data": json.loads(r[4]),
                    "status":        r[5],
                    "created_at":    r[6],
                    "processed_at":  r[7],
                })
            return result
        except Exception as e:
            logger.warning("load_pending_fragments failed: %s", e)
            return []

    def oxpecker_fragment_counts(self) -> dict:
        """Return counts by status for the oxpecker_status endpoint."""
        try:
            rows = self._conn.execute(
                "SELECT status, COUNT(*) FROM oxpecker_fragments GROUP BY status"
            ).fetchall()
            counts = {"pending": 0, "summarised": 0, "recycled": 0, "total": 0}
            for status, cnt in rows:
                counts[status] = cnt
                counts["total"] += cnt
            return counts
        except Exception as e:
            logger.warning("oxpecker_fragment_counts failed: %s", e)
            return {"pending": 0, "summarised": 0, "recycled": 0, "total": 0}

    def _trim_oxpecker_fragments(self) -> None:
        """Keep oxpecker_fragments within rolling cap (500 rows), oldest first."""
        try:
            count = self._conn.execute(
                "SELECT COUNT(*) FROM oxpecker_fragments"
            ).fetchone()[0]
            if count > MAX_OXP_FRAGMENTS:
                excess = count - MAX_OXP_FRAGMENTS
                self._conn.execute(
                    """DELETE FROM oxpecker_fragments WHERE fragment_id IN
                       (SELECT fragment_id FROM oxpecker_fragments
                        ORDER BY created_at ASC LIMIT ?)""",
                    (excess,),
                )
        except Exception as e:
            logger.warning("_trim_oxpecker_fragments failed: %s", e)

    # ── UTILITIES ──────────────────────────────────────────────────────────

    def _trim(self, table: str, max_rows: int) -> None:
        """
        Delete oldest rows if table exceeds max_rows.
        Called after each append to keep the DB bounded.
        Uses rowid ordering (append order = insertion order).
        """
        try:
            count = self._conn.execute(
                f"SELECT COUNT(*) FROM {table}"
            ).fetchone()[0]
            if count > max_rows:
                excess = count - max_rows
                self._conn.execute(
                    f"DELETE FROM {table} WHERE rowid IN "
                    f"(SELECT rowid FROM {table} ORDER BY rowid ASC LIMIT ?)",
                    (excess,),
                )
        except Exception as e:
            logger.warning("_trim(%s) failed: %s", table, e)

    def stats(self) -> dict:
        """Return row counts for all tables — useful for health checks."""
        tables = [
            "agents", "beta_state", "interactions",
            "bifurcation_events", "ci_history",
            "oscillation_events", "sandbox_reports", "theorem_state",
            "oxpecker_fragments",
        ]
        result = {}
        for t in tables:
            try:
                result[t] = self._conn.execute(
                    f"SELECT COUNT(*) FROM {t}"
                ).fetchone()[0]
            except Exception:
                result[t] = -1
        result["db_path"] = self.db_path
        return result

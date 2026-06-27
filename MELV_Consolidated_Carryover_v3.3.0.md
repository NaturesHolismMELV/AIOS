# MELV Consolidated Carryover — v3.3.0 Completion State
**Date**: 2026-06-27  
**Git commit**: `de9cbb1` — "alignment pass v3.3.0: canonical eq7, beta*i_inf gate, stagnation detector, E01-E50 tests"  
**Status**: Deployed to Railway via GitHub (origin/main, HEAD)

---

## 1. What Is Complete

The alignment pass v3.3.0 is fully implemented, tested (50 tests, all green), committed, and live on Railway. No further work is needed on the six tasks below.

| Task | File | Status |
|------|------|--------|
| T1 | `core/melv_engine.py` | ✅ canonical eq7, `_compute_i_inf()`, stagnation detector, gate constants |
| T2 | `core/observe_schema.py` | ✅ `beta_i_inf`, `delta_gate` fields on `ObservationResult` |
| T3 | `core/observe_compute.py` | ✅ computes `beta_i_inf` with L3 posterior η override |
| T4 | `core/telemetry.py` | ✅ L2 schema migration, `L2Snapshot` fields, `log_l2` / `get_l2_recent` updated |
| T5 | `tests/test_session35.py` | ✅ 50 tests E01–E50, all passing |
| T6 | Railway deployment | ✅ pushed and live |
| T7 | `api/server.py` | ✅ `GET /api/telemetry/l2/{agent_id}` added; `beta_i_inf`/`delta_gate` added to `TelemetryL2Request` |

---

## 2. Permanent Constraints (Never Violate)

- **ABM files are read-only historical**: `abm/melv_abm_v22.py` and `abm/abm_v22_runner.py` must NOT be modified under any circumstances.
- **Anti-circularity protocol** applies to derivation and adjudication work only — NOT to implementation tasks.
- **`PHI_GATEWAY_THRESHOLD = 0.50`** is RETIRED from structural gate role but must remain exported from `melv_engine.py` as a constant for backward compatibility. Do not remove it.
- **`I0_CANONICAL = 1.0`** — normalized baseline, do not change.
- **`ETA_CANONICAL_DEFAULT = 0.93`** — bee-flower calibration. Overridden by BI-NLS L3 posterior only when ≥ 100 L1 records are available.

---

## 3. Canonical Equation 7 (v3.3.0 — DO NOT MODIFY)

```
dφ/dt = α(1−φ) × H(1−β×i_∞) × max(0,1−i(t))   [BUILD branch]
        − δ×D(t)×φ × H(β×i_∞−1)                 [DECAY branch]

where i_∞ = i₀ × (1 − η × tanh(ε × β_norm / η))  [parameter function at φ=1]
      β_norm = β/(1+β) ∈ (0,1)
```

**Tier 1 canonical gate**: `β×i_∞ < 1` (Jacobian-derived stability condition)  
**Stagnation detector**: `Δ_gate = β×i_∞ − 1`  
- `Δ < 0` → STABLE  
- `Δ ≥ 0, D=0` → STAGNATION (no intervention)  
- `Δ ≥ 0, D>0` → COLLAPSE (intervention=True)

---

## 4. CRITICAL: Where β×i∞ Data Lives (Do Not Ask for L2 SQLite)

**L2 SQLite tables (`telemetry_l2`) are EMPTY in production.** This is by design:

- The pair-level governance loop (bifurcation decisions) runs entirely **in-memory** in the AIOS kernel.
- `apply_observation()` writes to L2 SQLite only when explicitly called with a real `ObservationResult` — this path is not used by the live pair-interaction engine.
- Railway's filesystem is ephemeral: any SQLite data written during a session is lost on the next deploy.

**The correct data source for β×i∞ diagnostics is `/api/events`:**

```
GET https://web-production-e14d1.up.railway.app/api/events?n=200
```

Each event has: `event_id`, `agent_a`, `agent_b`, `beta_i_pre`, `beta_i_post`, `action`, `resolved`, `timestamp`, `description`.

**Do NOT attempt to use:**
- `GET /api/telemetry/l2/{agent_id}` → always returns count=0
- `GET /api/telemetry/l1/{agent_id}` → always returns count=0 (L1 requires explicit POST calls)
- Local `aios_state.db` → has no tables (schema never applied locally)

---

## 5. Production System State (as of 2026-06-27)

From `/api/health`:
- 436 agents, 5120 interactions, CI = 1.0, mean_beta_i = 0.0624
- Events numbered BIF-109978 → BIF-110027 (50 events returned by n=50 query)
- Production URL: `https://web-production-e14d1.up.railway.app/`
- Three action types observed: `nudge`, `niche_divergence`, `provision_beta`
- `demo_stress_9b72ef9e` is a test agent with β=57 (structural outlier, exclude from analysis)

**Diagnostic plots already generated** (saved to repo root):
- `melv_canonical_gate_diagnostics_v3.3.0.png` — 4-panel figure:
  1. β×i∞ distribution at trigger (production agents)
  2. Pre vs post scatter by action type
  3. Resolution rate by action (nudge ~88%, niche_divergence ~35%, provision_beta 0%)
  4. Reduction ratio vs initial β×i∞ magnitude

**Key empirical findings from events data:**
- niche_divergence reduces β×i∞ by ~35% per intervention (×0.65 ratio holds across 1–8 range)
- nudge reduces β×i∞ by ~20% (stochastic, operates only in threshold zone 0.75–1.0)
- provision_beta is structural — does not resolve in single step

---

## 6. Key Implementation Details

### `core/melv_engine.py`
Constants added (after existing constants block):
```python
PHI_BUILD_RATE_ALPHA  = 0.01
PHI_DECAY_RATE_DELTA  = 0.10
PHI_GATEWAY_THRESHOLD = 0.50   # RETIRED — backward compat only
I0_CANONICAL          = 1.0
ETA_CANONICAL_DEFAULT = 0.93
```

`_compute_i_inf(i0, eta, epsilon, beta_raw)`:
```python
eta_safe = max(0.01, min(1.0, eta))
beta_n   = _beta_norm(beta_raw)
arg      = epsilon * beta_n / eta_safe
return i0 * (1.0 - eta_safe * math.tanh(arg))
```

`_apply_phi_eq7(self, agent, beta_i_inf, i_value, d_value)`:
- If `beta_i_inf is None` → return `(0.0, "")` (no-op)
- If `beta_i_inf < 1.0` → BUILD branch
- Else → DECAY branch

`compute_stagnation_state(beta_i_inf, d_value)` → staticmethod returning dict with `delta_gate`, `state`, `intervention`, `description`.

L2 write block (line ~2168) is inside `except Exception: pass` — silent on failure.

### `core/observe_schema.py`
`ObservationResult` new fields:
```python
beta_i_inf: Optional[float] = None
delta_gate: Optional[float] = None
```
`ScoredValue.computable`: `status=0` → False. Critical for E34 test.

### `core/observe_compute.py`
Imports `_compute_i_inf`, `I0_CANONICAL`, `ETA_CANONICAL_DEFAULT` from `core.melv_engine`.  
Computes `beta_i_inf` with L3 η override at end of `compute()`.

### `core/telemetry.py`
L2 schema: `beta_i_inf REAL`, `delta_gate REAL` columns added via migration.  
`get_l2_recent(agent_id, n)` — requires `agent_id` argument (not optional).

### `api/server.py`
- `TelemetryL2Request`: added `beta_i_inf: Optional[float] = None`, `delta_gate: Optional[float] = None`
- `POST /api/telemetry/l2`: passes new fields through to `L2Snapshot`
- `GET /api/telemetry/l2/{agent_id}`: new endpoint (returns count=0 in production — see §4)

### `tests/test_session35.py`
50 tests E01–E50, all pass. E34 requires `beta.status=0` to force `computable=False`.  
File must be written via bash Python if editing — Edit tool Windows path does not reliably sync to bash mount.

---

## 7. Filesystem Notes

- Edit tool: `C:\Users\web\AIOS\...` — may not sync to bash mount
- Bash mount: `/sessions/charming-lucid-knuth/mnt/AIOS/...`
- Write critical files via `python3 -c "open(...,'w').write(...)"` in bash when running pytest
- Stale `.pyc` in `tests/__pycache__/` can cause pytest to use old bytecode — changing file size forces recompile
- `conftest.py` at repo root is empty, cannot be deleted from bash mount — leave as-is
- `.git/index.lock` stale lock: `Remove-Item C:\Users\web\AIOS\.git\index.lock` from PowerShell

---

## 8. Frontend / Deployment State

- `frontend/dashboard13.html`: v3.2.0, stub panels for Telemetry/Dungbeetle/Irreversibility — no update needed for v3.3.0 deployment
- `frontend/landing.html`: v1.9.0, marketing only
- Railway: auto-deploys from GitHub main

---

## 9. What Is NOT Done (Next Session Scope)

### Deferred diagnostic plots (original carryover §3)
The three originally deferred plots (irreversibility boundary, inefficiency plateau, η recovery) require L1/L2 SQLite data which is not accumulating in production. **These are blocked pending a telemetry pipeline change** — either Railway Volume persistence for `aios_state.db` or an external DB.

Instead, the events-based diagnostic plot was generated (§5 above). If more events-based analysis is needed, use `/api/events?n=200`.

### Still explicitly deferred (not alignment pass scope)
- Clean preprint draft
- Governance recipe document
- φ decomposition adjudication

---

## 10. Untracked Files (Do Not Commit)

```
abm/__init__.py          — ABM module init, read-only historical scope
abm/abm_v22_output/      — ABM run outputs
abm/abm_v22_runner.py    — ABM runner, read-only historical scope
aios_state.db-journal    — SQLite WAL journal, never commit
conftest.py              — empty, leave as-is
```

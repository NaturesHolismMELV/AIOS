"""
plot1_irreversibility_boundary.py
MELV Diagnostic Plot 1 — Irreversibility Boundary
Laurence W. Evans | ORCID 0009-0001-0963-1840 | Cape Town, South Africa

Shows φ vs Δ_gate (= β×i∞ − 1) for all post-fix L2 rows, colour-coded by zone:
  STABLE       : Δ_gate < 0          (green)
  STAGNATION   : Δ_gate ≥ 0, no D    (amber)  — no intervention flag in L2; shown as Δ∈[0,1)
  COLLAPSE     : Δ_gate ≥ 1          (red)

Data source: Railway API — agents known to have post-fix rows (beta_service = null).
Post-fix filter: beta_service IS NULL (null in JSON).
"""

import urllib.request
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASE_URL = "https://web-production-e14d1.up.railway.app"

# Agents confirmed to have post-fix rows. Extend if more are discovered.
AGENTS = [
    "RESEARCH-9914",
    "WRITER-7b3b",
]

OUTPUT_FILE = "melv_plot1_irreversibility_boundary.png"

# Zone boundaries on delta_gate axis
STAGNATION_THRESHOLD = 0.0   # delta_gate >= 0 → not STABLE
COLLAPSE_THRESHOLD   = 1.0   # delta_gate >= 1 → COLLAPSE (deep stagnation / intervention)

# Canonical reference line
CANONICAL_GATE = 0.0  # β×i∞ = 1, i.e. delta_gate = 0

# ---------------------------------------------------------------------------
# Fetch data
# ---------------------------------------------------------------------------
def fetch_l2(agent_id):
    url = f"{BASE_URL}/api/telemetry/l2/{agent_id}"
    with urllib.request.urlopen(url, timeout=15) as resp:
        data = json.loads(resp.read())
    return data.get("records", [])

rows = []
for agent in AGENTS:
    try:
        records = fetch_l2(agent)
        for r in records:
            # Post-fix filter: beta_service must be null
            if r.get("beta_service") is not None:
                continue
            phi = r.get("phi")
            dg  = r.get("delta_gate")
            bii = r.get("beta_i_inf")
            ts  = r.get("timestamp")
            if phi is None or dg is None:
                continue
            rows.append({
                "agent_id":   agent,
                "phi":        float(phi),
                "delta_gate": float(dg),
                "beta_i_inf": float(bii) if bii is not None else None,
                "timestamp":  float(ts)  if ts  is not None else None,
            })
        print(f"  {agent}: {len([r for r in records if r.get('beta_service') is None])} post-fix rows")
    except Exception as e:
        print(f"  WARNING: could not fetch {agent}: {e}")

print(f"Total post-fix rows loaded: {len(rows)}")

if not rows:
    print("ERROR: No post-fix rows found. Aborting.")
    raise SystemExit(1)

# ---------------------------------------------------------------------------
# Classify zones
# ---------------------------------------------------------------------------
def classify(dg):
    if dg < STAGNATION_THRESHOLD:
        return "STABLE"
    elif dg < COLLAPSE_THRESHOLD:
        return "STAGNATION"
    else:
        return "COLLAPSE"

for r in rows:
    r["zone"] = classify(r["delta_gate"])

zone_colours = {
    "STABLE":     "#2ca02c",   # green
    "STAGNATION": "#ff7f0e",   # amber
    "COLLAPSE":   "#d62728",   # red
}
zone_markers = {
    "STABLE":     "o",
    "STAGNATION": "s",
    "COLLAPSE":   "^",
}

counts = {z: sum(1 for r in rows if r["zone"] == z) for z in ["STABLE", "STAGNATION", "COLLAPSE"]}
print(f"Zone counts — STABLE: {counts['STABLE']}, STAGNATION: {counts['STAGNATION']}, COLLAPSE: {counts['COLLAPSE']}")

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 6))

for zone in ["STABLE", "STAGNATION", "COLLAPSE"]:
    subset = [r for r in rows if r["zone"] == zone]
    if not subset:
        continue
    xs = [r["delta_gate"] for r in subset]
    ys = [r["phi"]        for r in subset]
    ax.scatter(xs, ys,
               c=zone_colours[zone],
               marker=zone_markers[zone],
               s=90, zorder=5,
               label=f"{zone} (n={len(subset)})")

# Canonical gate line at delta_gate = 0
ax.axvline(x=CANONICAL_GATE, color="black", linewidth=1.5, linestyle="--",
           label="β×i∞ = 1 (canonical gate, T1)", zorder=4)

# Collapse threshold at delta_gate = 1
ax.axvline(x=COLLAPSE_THRESHOLD, color="#d62728", linewidth=1.0, linestyle=":",
           label="δ_gate = 1 (collapse boundary)", zorder=4)

# Zone background shading
xlim_min = min(r["delta_gate"] for r in rows) - 0.5
xlim_max = max(r["delta_gate"] for r in rows) + 0.5
ax.axvspan(xlim_min, 0,            alpha=0.06, color="#2ca02c")   # STABLE
ax.axvspan(0,        1,            alpha=0.06, color="#ff7f0e")   # STAGNATION
ax.axvspan(1,        xlim_max,     alpha=0.06, color="#d62728")   # COLLAPSE

# Zone labels at top
y_top = ax.get_ylim()[1] if ax.get_ylim()[1] > 0.85 else 0.87
ax.text((xlim_min + 0) / 2,  0.995, "STABLE",     ha="center", va="top",
        fontsize=9, color="#2ca02c", transform=ax.get_xaxis_transform())
ax.text(0.5,                  0.995, "STAGNATION", ha="center", va="top",
        fontsize=9, color="#ff7f0e", transform=ax.get_xaxis_transform())
ax.text((1 + xlim_max) / 2,  0.995, "COLLAPSE",   ha="center", va="top",
        fontsize=9, color="#d62728", transform=ax.get_xaxis_transform())

ax.set_xlabel("Δ_gate = β×i∞ − 1", fontsize=11)
ax.set_ylabel("φ (mean across agents at governance event)", fontsize=11)
ax.set_title("MELV Diagnostic Plot 1 — Irreversibility Boundary\n"
             "φ vs Δ_gate across STABLE / STAGNATION / COLLAPSE zones\n"
             "MELVcore v3.3.1 | Post-fix L2 telemetry | Laurence W. Evans",
             fontsize=10)

ax.set_xlim(xlim_min, xlim_max)
ax.set_ylim(0.79, 0.82)   # φ varies narrowly around ~0.805 in current system
ax.legend(loc="lower right", fontsize=9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_FILE, dpi=150, bbox_inches="tight")
print(f"Saved: {OUTPUT_FILE}")

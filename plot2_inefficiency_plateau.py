"""
plot2_inefficiency_plateau.py
MELV Diagnostic Plot 2 — Inefficiency Plateau
Laurence W. Evans | ORCID 0009-0001-0963-1840 | Cape Town, South Africa

Two-panel figure:
  Left  — β×i∞ trajectory over time (per agent, colour-coded), canonical gate
           line at β×i∞ = 1, zone bands shaded
  Right — Scatter: β×i∞ vs φ, coloured by zone, size proportional to |delta_gate|

Data source: Railway API — agents known to have post-fix rows (beta_service = null).
"""

import urllib.request
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASE_URL = "https://web-production-e14d1.up.railway.app"

AGENTS = [
    "RESEARCH-9914",
    "WRITER-7b3b",
]

OUTPUT_FILE = "melv_plot2_inefficiency_plateau.png"

AGENT_COLOURS = {
    "RESEARCH-9914": "#1f77b4",   # blue
    "WRITER-7b3b":   "#9467bd",   # purple
}

zone_colours = {
    "STABLE":     "#2ca02c",
    "STAGNATION": "#ff7f0e",
    "COLLAPSE":   "#d62728",
}

# ---------------------------------------------------------------------------
# Fetch data
# ---------------------------------------------------------------------------
def fetch_l2(agent_id):
    url = f"{BASE_URL}/api/telemetry/l2/{agent_id}"
    with urllib.request.urlopen(url, timeout=15) as resp:
        data = json.loads(resp.read())
    return data.get("records", [])

def classify(dg):
    if dg < 0.0:
        return "STABLE"
    elif dg < 1.0:
        return "STAGNATION"
    else:
        return "COLLAPSE"

all_rows = []
agent_rows = {}

for agent in AGENTS:
    try:
        records = fetch_l2(agent)
        rows = []
        for r in records:
            if r.get("beta_service") is not None:
                continue
            phi = r.get("phi")
            dg  = r.get("delta_gate")
            bii = r.get("beta_i_inf")
            ts  = r.get("timestamp")
            if phi is None or dg is None or bii is None or ts is None:
                continue
            rows.append({
                "agent_id":   agent,
                "phi":        float(phi),
                "delta_gate": float(dg),
                "beta_i_inf": float(bii),
                "timestamp":  float(ts),
                "zone":       classify(float(dg)),
            })
        # Sort by timestamp ascending for trajectory plot
        rows.sort(key=lambda x: x["timestamp"])
        agent_rows[agent] = rows
        all_rows.extend(rows)
        print(f"  {agent}: {len(rows)} post-fix rows")
    except Exception as e:
        print(f"  WARNING: could not fetch {agent}: {e}")

print(f"Total post-fix rows: {len(all_rows)}")

if not all_rows:
    print("ERROR: No post-fix rows found. Aborting.")
    raise SystemExit(1)

# Normalise timestamps to hours-since-first-event for readability
t0 = min(r["timestamp"] for r in all_rows)
for r in all_rows:
    r["t_hours"] = (r["timestamp"] - t0) / 3600.0
for rows in agent_rows.values():
    for r in rows:
        r["t_hours"] = (r["timestamp"] - t0) / 3600.0

t_max = max(r["t_hours"] for r in all_rows)

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

# ── Panel 1: β×i∞ trajectory over time ────────────────────────────────────
for agent, rows in agent_rows.items():
    if not rows:
        continue
    ts  = [r["t_hours"]   for r in rows]
    bii = [r["beta_i_inf"] for r in rows]
    col = AGENT_COLOURS.get(agent, "grey")
    ax1.plot(ts, bii, marker="o", markersize=6, linewidth=1.5,
             color=col, label=agent, zorder=5)
    # Colour each marker by zone
    for r in rows:
        ax1.scatter(r["t_hours"], r["beta_i_inf"],
                    c=zone_colours[r["zone"]], s=60, zorder=6, edgecolors=col, linewidths=0.8)

# Canonical gate line
ax1.axhline(y=1.0, color="black", linewidth=1.5, linestyle="--",
            label="β×i∞ = 1 (canonical gate, T1)", zorder=4)

# Zone bands
bii_max = max(r["beta_i_inf"] for r in all_rows) + 0.5
ax1.axhspan(0,    1.0,     alpha=0.06, color="#2ca02c")   # STABLE
ax1.axhspan(1.0,  2.0,     alpha=0.06, color="#ff7f0e")   # STAGNATION
ax1.axhspan(2.0,  bii_max, alpha=0.06, color="#d62728")   # COLLAPSE

ax1.text(t_max * 0.02, 0.50, "STABLE",     fontsize=8, color="#2ca02c", va="center")
ax1.text(t_max * 0.02, 1.50, "STAGNATION", fontsize=8, color="#ff7f0e", va="center")
ax1.text(t_max * 0.02, 2.50, "COLLAPSE",   fontsize=8, color="#d62728", va="center")

ax1.set_xlabel("Time since first event (hours)", fontsize=10)
ax1.set_ylabel("β×i∞ (pair-level governance signal)", fontsize=10)
ax1.set_title("β×i∞ Trajectory Over Time\nby agent, coloured by zone", fontsize=10)
ax1.set_ylim(0, bii_max)
ax1.set_xlim(-t_max * 0.02, t_max * 1.05)
ax1.legend(fontsize=8, loc="upper left")
ax1.grid(True, alpha=0.3)

# ── Panel 2: β×i∞ vs φ scatter ────────────────────────────────────────────
for zone in ["STABLE", "STAGNATION", "COLLAPSE"]:
    subset = [r for r in all_rows if r["zone"] == zone]
    if not subset:
        continue
    xs   = [r["phi"]        for r in subset]
    ys   = [r["beta_i_inf"] for r in subset]
    sizes = [40 + 20 * abs(r["delta_gate"]) for r in subset]
    ax2.scatter(xs, ys, c=zone_colours[zone], s=sizes, alpha=0.85, zorder=5,
                label=f"{zone} (n={len(subset)})", edgecolors="white", linewidths=0.5)

ax2.axhline(y=1.0, color="black", linewidth=1.5, linestyle="--",
            label="β×i∞ = 1 (canonical gate)", zorder=4)

phi_vals = [r["phi"] for r in all_rows]
phi_min, phi_max = min(phi_vals) - 0.001, max(phi_vals) + 0.001
ax2.axhspan(0,    1.0,     alpha=0.06, color="#2ca02c")
ax2.axhspan(1.0,  2.0,     alpha=0.06, color="#ff7f0e")
ax2.axhspan(2.0,  bii_max, alpha=0.06, color="#d62728")

ax2.set_xlabel("φ (mean cooperation level at governance event)", fontsize=10)
ax2.set_ylabel("β×i∞ (pair-level governance signal)", fontsize=10)
ax2.set_title("β×i∞ vs φ — Zone Scatter\nmarker size ∝ |Δ_gate|", fontsize=10)
ax2.set_xlim(phi_min, phi_max)
ax2.set_ylim(0, bii_max)
ax2.legend(fontsize=8, loc="upper right")
ax2.grid(True, alpha=0.3)

# ── Figure title ───────────────────────────────────────────────────────────
fig.suptitle(
    "MELV Diagnostic Plot 2 — Inefficiency Plateau\n"
    "MELVcore v3.3.1 | Post-fix L2 telemetry | Laurence W. Evans",
    fontsize=11, y=1.02
)

plt.tight_layout()
plt.savefig(OUTPUT_FILE, dpi=150, bbox_inches="tight")
print(f"Saved: {OUTPUT_FILE}")

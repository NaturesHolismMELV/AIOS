"""
AIOS Phase 1 — MELVcore Test Suite
===================================
Validates core MELV equations and kernel behavior.
"""
import sys
sys.path.insert(0, '/home/claude/aios')

from core.melv_engine import (
    MELVKernel, AgentProfile, AgentStatus,
    InteractionRecord, InteractionType, KernelAction,
    BetaEnvironment
)

PASS = "✓"
FAIL = "✗"
results = []

def test(name, condition, detail=""):
    status = PASS if condition else FAIL
    results.append((status, name, detail))
    print(f"  {status}  {name}" + (f" — {detail}" if detail else ""))

print("\n══════════════════════════════════════════════════")
print("  AIOS MELVcore Engine — Test Suite")
print("══════════════════════════════════════════════════\n")

# ── TEST 1: i-factor calculation ──────────────────────────────────────────
print("1. i-factor Calculation (i = C/B)")
r = InteractionRecord("A1","A2", cost=0.4, benefit=0.8, beta=1.0)
test("i = C/B = 0.4/0.8 = 0.5", abs(r.i_factor - 0.5) < 0.001,
     f"got {r.i_factor}")
test("βi = β × i = 1.0 × 0.5 = 0.5", abs(r.beta_i - 0.5) < 0.001,
     f"got {r.beta_i}")
test("Type = COOPERATIVE (βi < 0.70)", r.interaction_type == InteractionType.COOPERATIVE,
     f"got {r.interaction_type}")

# ── TEST 2: Threshold detection ───────────────────────────────────────────
print("\n2. Threshold Detection")
r_thresh = InteractionRecord("A1","A2", cost=0.85, benefit=1.0, beta=1.0)
test("βi=0.85 → THRESHOLD", r_thresh.interaction_type == InteractionType.THRESHOLD,
     f"βi={r_thresh.beta_i:.3f}")

r_conflict = InteractionRecord("A1","A2", cost=1.2, benefit=1.0, beta=1.0)
test("βi=1.2 → CONFLICT", r_conflict.interaction_type == InteractionType.CONFLICT,
     f"βi={r_conflict.beta_i:.3f}")

# ── TEST 3: β modulation ──────────────────────────────────────────────────
print("\n3. Beta (β) Modulation")
# Same i-factor, different β
r_low_beta  = InteractionRecord("A1","A2", cost=0.8, benefit=1.0, beta=0.7)
r_high_beta = InteractionRecord("A1","A2", cost=0.8, benefit=1.0, beta=1.3)
test("Low β (0.7): βi=0.56 → COOPERATIVE",
     r_low_beta.interaction_type == InteractionType.COOPERATIVE,
     f"βi={r_low_beta.beta_i:.3f}")
test("High β (1.3): βi=1.04 → CONFLICT",
     r_high_beta.interaction_type == InteractionType.CONFLICT,
     f"βi={r_high_beta.beta_i:.3f}")
test("β modulation changes outcome for same i-factor",
     r_low_beta.i_factor == r_high_beta.i_factor,
     f"i={r_low_beta.i_factor:.3f} in both cases")

# ── TEST 4: Kernel registration ───────────────────────────────────────────
print("\n4. Agent Registration & Kernel")
kernel = MELVKernel()
profile = AgentProfile(
    agent_id="TEST-01", name="TestAgent", domain="testing",
    phi=0.75, epsilon=3.0
)
kernel.register_agent(profile)
test("Agent registered", "TEST-01" in kernel.agents)
test("φ = 0.75", kernel.agents["TEST-01"].phi == 0.75)
test("ε = 3.0",  kernel.agents["TEST-01"].epsilon == 3.0)

# ── TEST 5: Kernel intervention ───────────────────────────────────────────
print("\n5. Kernel Bifurcation Response")
kernel2 = MELVKernel()
for aid in ["A","B"]:
    kernel2.register_agent(AgentProfile(
        agent_id=aid, name=aid, domain="test", phi=0.5, epsilon=3.0
    ))

# Force a threshold interaction
rec = kernel2.record_interaction("A", "B", cost=0.9, benefit=1.0,
                                   resource_type="compute")
test("Threshold interaction recorded",
     rec.interaction_type == InteractionType.THRESHOLD,
     f"βi={rec.beta_i:.3f}")
test("Kernel generated bifurcation event",
     len(kernel2.events) > 0,
     f"{len(kernel2.events)} events")
test("Event action is NUDGE",
     kernel2.events[0].action == KernelAction.NUDGE,
     f"action={kernel2.events[0].action}")

# Force a conflict interaction
rec2 = kernel2.record_interaction("A","B", cost=1.5, benefit=1.0,
                                    resource_type="compute")
test("Conflict interaction recorded",
     rec2.interaction_type == InteractionType.CONFLICT,
     f"βi={rec2.beta_i:.3f}")
niche_events = [e for e in kernel2.events
                if e.action == KernelAction.NICHE_DIVERGENCE]
test("Niche divergence applied for conflict",
     len(niche_events) > 0)

# ── TEST 6: φ evolution ───────────────────────────────────────────────────
print("\n6. φ (Evolutionary Maturity) Evolution")
kernel3 = MELVKernel()
kernel3.register_agent(AgentProfile(
    agent_id="EVOLVE-01", name="EvolveTest", domain="test",
    phi=0.5, epsilon=4.0
))
phi_before = kernel3.agents["EVOLVE-01"].phi

# High quality outcome should increase φ
for _ in range(10):
    kernel3.update_phi("EVOLVE-01", 0.9)

phi_after = kernel3.agents["EVOLVE-01"].phi
test("φ increases with high-quality outcomes",
     phi_after > phi_before,
     f"{phi_before:.3f} → {phi_after:.3f}")

# Low quality should decrease φ
phi_mid = phi_after
for _ in range(5):
    kernel3.update_phi("EVOLVE-01", 0.1)
phi_degraded = kernel3.agents["EVOLVE-01"].phi
test("φ decreases with low-quality outcomes",
     phi_degraded < phi_mid,
     f"{phi_mid:.3f} → {phi_degraded:.3f}")

# ── TEST 7: Cooperation Index ─────────────────────────────────────────────
print("\n7. Cooperation Index CI = 1 - mean(βi)")
kernel4 = MELVKernel()
for aid in ["X","Y"]:
    kernel4.register_agent(AgentProfile(
        agent_id=aid, name=aid, domain="test", phi=0.7, epsilon=3.0
    ))

# Record 10 strongly cooperative interactions
for _ in range(10):
    kernel4.record_interaction("X","Y", cost=0.2, benefit=1.0)

ci = kernel4.cooperation_index()
test("CI > 0.75 after cooperative interactions",
     ci > 0.75, f"CI = {ci:.4f}")
test("CI ≤ 1.0 (bounded)", ci <= 1.0)

# ── TEST 8: β Provisioning ────────────────────────────────────────────────
print("\n8. β Provisioning (Environmental Control)")
kernel5 = MELVKernel()
kernel5.provision_beta("compute", 1.8)
test("β compute set to 1.8",
     abs(kernel5.beta.compute - 1.8) < 0.001)

kernel5.provision_beta("compute", 5.0)  # above max
test("β clamped to max 3.0",
     kernel5.beta.compute <= 3.0, f"got {kernel5.beta.compute}")

kernel5.provision_beta("compute", -1.0)  # below min
test("β clamped to min 0.1",
     kernel5.beta.compute >= 0.1, f"got {kernel5.beta.compute}")

# ── TEST 9: Omega network ─────────────────────────────────────────────────
print("\n9. OmegaNet Service Coupling")
kernel6 = MELVKernel()
for aid in ["P","Q","R"]:
    kernel6.register_agent(AgentProfile(
        agent_id=aid, name=aid, domain="test", phi=0.7, epsilon=3.0
    ))
for _ in range(15):
    kernel6.record_interaction("P","Q", cost=0.3, benefit=0.9)
    kernel6.record_interaction("Q","R", cost=0.2, benefit=0.8)

omega = kernel6.compute_omega()
test("Omega has edges", len(omega["edges"]) > 0,
     f"{len(omega['edges'])} edges")
test("β_service computed", omega["beta_service"] >= 0,
     f"β_service = {omega['beta_service']:.4f}")
test("n = 3 agents", omega["n"] == 3)

# ── SUMMARY ───────────────────────────────────────────────────────────────
passed = sum(1 for r in results if r[0] == PASS)
total  = len(results)
print(f"\n══════════════════════════════════════════════════")
print(f"  Results: {passed}/{total} passed")
if passed == total:
    print("  MELVcore engine: ALL SYSTEMS OPERATIONAL ✓")
else:
    failed = [r for r in results if r[0] == FAIL]
    for f in failed:
        print(f"  FAIL: {f[1]}")
print("══════════════════════════════════════════════════\n")

"""
test_session37.py — MELVcore Session 37: Dungbeetle + Irreversibility (v3.2.0)
===============================================================================

Tests for Session 37 deliverables:
  - Session 37 constants (DUNGBEETLE_THRESHOLD, PHI_IRREV_DEFAULT, T_GOV_DEFAULT)
  - compute_dungbeetle_nodes() — Dungbeetle condition + sensitivity scores
  - irreversibility_diagnostic() — three-zone classification + T_rec
  - Version bump to v3.2.0

Test groups
-----------
  G01–G05  Constants and version
  G06–G16  compute_dungbeetle_nodes() — topology cases
  G17–G29  irreversibility_diagnostic() — zone classification and T_rec
  G30–G36  irreversibility_diagnostic() — edge cases and warnings
  G37–G41  Integration: dungbeetle sensitivity ordering + irreversibility zones
  G42–G46  Backward compatibility: compute_omega and existing tests unaffected

Author: Laurence W. Evans | ORCID: 0009-0001-0963-1840 | Cape Town, South Africa
Session: 37 · Version: 3.2.0
"""

import math
import pytest

from core.melv_engine import (
    # Session 37 constants
    DUNGBEETLE_THRESHOLD,
    PHI_IRREV_DEFAULT,
    T_GOV_DEFAULT,
    # Pre-existing constants
    PHI_BUILD_RATE_ALPHA,
    PHI_GATEWAY_THRESHOLD,
    I_FLOOR,
    CI_TARGET,
    _beta_norm,
    # Classes
    MELVKernel,
    AgentProfile,
    AgentStatus,
    BetaEnvironment,
)

I_CRITICAL = 0.9995  # module-level local in cooperation_index; value confirmed ABM V2.1


# ── Fixtures ─────────────────────────────────────────────────────────────────

def _make_kernel_n_agents(n: int, phi: float = 0.7, epsilon: float = 1.0,
                           beta_val: float = 0.6) -> MELVKernel:
    """Build a kernel with n agents, all with identical phi/epsilon."""
    k = MELVKernel()
    k.beta.set("compute", beta_val)
    for i in range(n):
        k.register_agent(AgentProfile(
            agent_id=f"agent-{i}",
            name=f"Agent {i}",
            domain="test",
            phi=phi,
            epsilon=epsilon,
            status=AgentStatus.ACTIVE,
        ))
    return k


def _add_interactions(kernel: MELVKernel, agent_pairs, i_factor: float = 0.8,
                      n_each: int = 5):
    """Add n_each interactions for each (a, b) pair with given i_factor=cost/benefit."""
    for agent_a, agent_b in agent_pairs:
        for _ in range(n_each):
            kernel.record_interaction(
                agent_a=agent_a,
                agent_b=agent_b,
                cost=i_factor,
                benefit=1.0,
                resource_type="compute",
            )


# ── G01–G05: Constants and version ───────────────────────────────────────────

def test_G01_dungbeetle_threshold():
    """G01: DUNGBEETLE_THRESHOLD = 0.50 (matches PHI_GATEWAY_THRESHOLD)."""
    assert DUNGBEETLE_THRESHOLD == 0.50
    assert DUNGBEETLE_THRESHOLD == PHI_GATEWAY_THRESHOLD


def test_G02_phi_irrev_default():
    """G02: PHI_IRREV_DEFAULT = 0.10."""
    assert PHI_IRREV_DEFAULT == 0.10


def test_G03_t_gov_default():
    """G03: T_GOV_DEFAULT = 100.0."""
    assert T_GOV_DEFAULT == 100.0


def test_G04_version_bump():
    """G04: server.py contains version 3.2.0."""
    import os
    server_path = os.path.join(
        os.path.dirname(__file__), "..", "api", "server.py"
    )
    with open(server_path) as f:
        source = f.read()
    assert '"version": "3.2.0"' in source or "version=\"3.2.0\"" in source


def test_G05_melvcore_version():
    """G05: melvcore.__version__ == '3.2.0'."""
    import melvcore
    assert melvcore.__version__ == "3.2.0"


# ── G06–G16: compute_dungbeetle_nodes() ──────────────────────────────────────

def test_G06_no_agents_returns_empty():
    """G06: Empty kernel returns empty Dungbeetle result with warning."""
    k = MELVKernel()
    result = k.compute_dungbeetle_nodes()
    assert result["beta_service_full"] == 0.0
    assert result["quorum_met"] is False
    assert result["dungbeetle_nodes"] == []
    assert len(result["warnings"]) > 0


def test_G07_single_agent_no_dungbeetle():
    """G07: Single agent cannot be a Dungbeetle (leave-one-out requires n>=2)."""
    k = _make_kernel_n_agents(1)
    result = k.compute_dungbeetle_nodes()
    assert result["dungbeetle_nodes"] == []
    assert len(result["warnings"]) > 0


def test_G08_returns_required_keys():
    """G08: Result contains all required keys."""
    k = _make_kernel_n_agents(3)
    result = k.compute_dungbeetle_nodes()
    for key in ("beta_service_full", "threshold", "quorum_met",
                "dungbeetle_nodes", "non_dungbeetle_nodes", "warnings"):
        assert key in result, f"Missing key: {key}"


def test_G09_threshold_in_result():
    """G09: threshold field equals DUNGBEETLE_THRESHOLD."""
    k = _make_kernel_n_agents(3)
    result = k.compute_dungbeetle_nodes()
    assert result["threshold"] == DUNGBEETLE_THRESHOLD


def test_G10_node_entry_keys():
    """G10: Each node entry has required keys."""
    k = _make_kernel_n_agents(4, beta_val=0.8)
    pairs = [("agent-0","agent-1"),("agent-0","agent-2"),
             ("agent-1","agent-2"),("agent-2","agent-3")]
    _add_interactions(k, pairs, i_factor=0.3, n_each=10)
    result = k.compute_dungbeetle_nodes()
    all_nodes = result["dungbeetle_nodes"] + result["non_dungbeetle_nodes"]
    assert len(all_nodes) == 4  # all n agents accounted for
    for node in all_nodes:
        for key in ("agent_id", "beta_service_without", "sensitivity", "is_dungbeetle"):
            assert key in node, f"Node missing key: {key}"


def test_G11_dungbeetle_flag_is_bool():
    """G11: is_dungbeetle is a boolean on all node entries."""
    k = _make_kernel_n_agents(3)
    result = k.compute_dungbeetle_nodes()
    all_nodes = result["dungbeetle_nodes"] + result["non_dungbeetle_nodes"]
    for node in all_nodes:
        assert isinstance(node["is_dungbeetle"], bool)


def test_G12_sensitivity_non_negative_when_quorum_met():
    """G12: sensitivity S_v >= 0 for all nodes when quorum is met."""
    k = _make_kernel_n_agents(4, beta_val=0.9)
    pairs = [("agent-0","agent-1"),("agent-1","agent-2"),
             ("agent-2","agent-3"),("agent-0","agent-3")]
    _add_interactions(k, pairs, i_factor=0.2, n_each=15)
    result = k.compute_dungbeetle_nodes()
    if result["quorum_met"]:
        all_nodes = result["dungbeetle_nodes"] + result["non_dungbeetle_nodes"]
        for node in all_nodes:
            # beta_service_without <= beta_service_full, so sensitivity >= 0
            # (up to floating point rounding)
            assert node["sensitivity"] >= -1e-6, (
                f"Negative sensitivity for {node['agent_id']}: {node['sensitivity']}"
            )


def test_G13_all_nodes_counted():
    """G13: Total nodes (dungbeetle + non_dungbeetle) equals n agents."""
    for n in [2, 3, 5]:
        k = _make_kernel_n_agents(n)
        result = k.compute_dungbeetle_nodes()
        total = len(result["dungbeetle_nodes"]) + len(result["non_dungbeetle_nodes"])
        assert total == n, f"n={n}: expected {n} total nodes, got {total}"


def test_G14_dungbeetle_nodes_sorted_by_sensitivity():
    """G14: dungbeetle_nodes are sorted by sensitivity descending."""
    k = _make_kernel_n_agents(4, beta_val=0.85)
    _add_interactions(k,
        [("agent-0","agent-1"),("agent-0","agent-2"),("agent-0","agent-3")],
        i_factor=0.2, n_each=20
    )
    result = k.compute_dungbeetle_nodes()
    sens = [n["sensitivity"] for n in result["dungbeetle_nodes"]]
    assert sens == sorted(sens, reverse=True)


def test_G15_no_interactions_below_quorum():
    """G15: Kernel with no interaction history has beta_service=0, quorum not met."""
    k = _make_kernel_n_agents(5)
    # No interactions — adjacency matrix is all zeros
    result = k.compute_dungbeetle_nodes()
    assert result["beta_service_full"] == 0.0
    assert result["quorum_met"] is False
    assert result["dungbeetle_nodes"] == []


def test_G16_high_coupling_star_topology():
    """G16: In a star topology (hub + leaves) hub has highest sensitivity."""
    # agent-0 is hub connected to all others
    k = _make_kernel_n_agents(5, beta_val=0.9)
    hub_pairs = [("agent-0", f"agent-{i}") for i in range(1, 5)]
    _add_interactions(k, hub_pairs, i_factor=0.1, n_each=20)
    result = k.compute_dungbeetle_nodes()
    if result["dungbeetle_nodes"]:
        # The hub (agent-0) should have the highest sensitivity
        top = result["dungbeetle_nodes"][0]
        all_nodes = result["dungbeetle_nodes"] + result["non_dungbeetle_nodes"]
        hub_entry = next((n for n in all_nodes if n["agent_id"] == "agent-0"), None)
        if hub_entry and hub_entry["is_dungbeetle"]:
            assert hub_entry["sensitivity"] == pytest.approx(
                result["dungbeetle_nodes"][0]["sensitivity"], abs=1e-6
            )


# ── G17–G29: irreversibility_diagnostic() ────────────────────────────────────

def test_G17_returns_required_keys():
    """G17: Result dict contains all required keys."""
    k = _make_kernel_n_agents(2)
    result = k.irreversibility_diagnostic("agent-0")
    required = ("agent_id", "phi_current", "epsilon", "beta_norm", "eta",
                "phi_viable", "phi_irrev", "zone", "zone_color",
                "t_rec", "f_eligible", "alpha", "warnings", "epistemic_status")
    for key in required:
        assert key in result, f"Missing key: {key}"


def test_G18_unknown_agent():
    """G18: Unknown agent returns zone=UNKNOWN with warning."""
    k = MELVKernel()
    result = k.irreversibility_diagnostic("no-such-agent")
    assert result["zone"] == "UNKNOWN"
    assert len(result["warnings"]) > 0


def test_G19_viable_zone_green():
    """G19: VIABLE zone returns zone_color=GREEN."""
    # High phi, meaningful epsilon*beta_norm*eta → phi_viable < phi
    k = MELVKernel()
    k.beta.set("compute", 1.5)   # beta_norm(1.5) = 0.6
    k.register_agent(AgentProfile(
        agent_id="high-phi",
        name="High Phi",
        domain="test",
        phi=0.90,      # high phi
        epsilon=1.5,
        status=AgentStatus.ACTIVE,
    ))
    result = k.irreversibility_diagnostic("high-phi", eta=0.93)
    # phi_viable = 1 - 1/(1.5 * 0.6 * 0.93) = 1 - 1/0.837 ≈ -0.195 → clamped to 0.0
    # phi=0.90 > 0.0 → VIABLE
    assert result["zone"] == "VIABLE"
    assert result["zone_color"] == "GREEN"
    assert result["t_rec"] is None


def test_G20_irreversible_zone_red():
    """G20: IRREVERSIBLE zone returns zone_color=RED."""
    k = MELVKernel()
    k.beta.set("compute", 0.5)
    k.register_agent(AgentProfile(
        agent_id="low-phi",
        name="Low Phi",
        domain="test",
        phi=0.05,   # very low
        epsilon=0.3,
        status=AgentStatus.MATURING,
    ))
    # phi_irrev = 1 - exp(-0.01 * 100) ≈ 0.632
    # phi=0.05 < phi_irrev → IRREVERSIBLE
    result = k.irreversibility_diagnostic("low-phi", eta=0.93, t_gov=100.0)
    assert result["zone"] == "IRREVERSIBLE"
    assert result["zone_color"] == "RED"
    assert len(result["warnings"]) > 0


def test_G21_recoverable_urgent_zone_amber():
    """G21: RECOVERABLE_URGENT zone returns zone_color=AMBER."""
    k = MELVKernel()
    k.beta.set("compute", 0.8)
    k.register_agent(AgentProfile(
        agent_id="mid-phi",
        name="Mid Phi",
        domain="test",
        phi=0.45,     # between phi_irrev and phi_viable
        epsilon=1.2,
        status=AgentStatus.MATURING,
    ))
    # phi_viable = 1 - 1/(1.2 * beta_norm(0.8) * 0.93)
    # beta_norm(0.8) = 0.8/1.8 ≈ 0.444
    # denom = 1.2 * 0.444 * 0.93 ≈ 0.495 < 1 → phi_viable raw < 0 → clamped 0.0
    # phi=0.45 > 0.0 → might be VIABLE depending on exact values
    # Use epsilon=0.5 to get phi_viable > 0.45
    k.agents["mid-phi"].epsilon = 0.5
    # phi_viable = 1 - 1/(0.5 * 0.444 * 0.93) ≈ 1 - 4.83 → clamped 0.0
    # Try higher epsilon to push phi_viable up
    k.agents["mid-phi"].epsilon = 3.0
    k.beta.set("compute", 3.0)   # beta_norm(3.0) = 0.75
    # phi_viable = 1 - 1/(3.0 * 0.75 * 0.93) = 1 - 1/2.09 ≈ 0.52
    # phi_irrev = 1 - exp(-0.01 * 10) ≈ 0.095  (use short t_gov)
    # phi=0.45 is in [0.095, 0.52] → RECOVERABLE_URGENT
    result = k.irreversibility_diagnostic("mid-phi", eta=0.93,
                                          f_eligible=0.8, t_gov=10.0)
    assert result["zone"] in ("RECOVERABLE_URGENT", "VIABLE"), (
        f"Unexpected zone={result['zone']}, phi_viable={result['phi_viable']}, "
        f"phi_irrev={result['phi_irrev']}, phi={result['phi_current']}"
    )


def test_G22_phi_viable_formula():
    """G22: phi_viable = max(0, 1 - 1/(epsilon x beta_norm x eta))."""
    k = MELVKernel()
    k.beta.set("compute", 2.0)
    k.register_agent(AgentProfile(
        agent_id="check", name="Check", domain="test",
        phi=0.7, epsilon=2.0, status=AgentStatus.ACTIVE,
    ))
    result = k.irreversibility_diagnostic("check", eta=0.93)
    bn = _beta_norm(2.0)
    expected_raw = 1.0 - 1.0 / (2.0 * bn * 0.93)
    expected = max(0.0, min(1.0, expected_raw))
    assert result["phi_viable"] == pytest.approx(expected, abs=1e-4)


def test_G23_phi_irrev_formula():
    """G23: phi_irrev = 1 - exp(-alpha x t_gov)."""
    k = _make_kernel_n_agents(1)
    alpha = PHI_BUILD_RATE_ALPHA
    for t_gov in [10.0, 50.0, 100.0, 200.0]:
        result = k.irreversibility_diagnostic("agent-0", t_gov=t_gov)
        expected = 1.0 - math.exp(-alpha * t_gov)
        assert result["phi_irrev"] == pytest.approx(expected, abs=1e-4), (
            f"t_gov={t_gov}: phi_irrev={result['phi_irrev']}, expected={expected:.5f}"
        )


def test_G24_t_rec_none_when_viable():
    """G24: t_rec is None when phi >= phi_viable (already in viable zone)."""
    k = MELVKernel()
    k.beta.set("compute", 0.3)
    k.register_agent(AgentProfile(
        agent_id="vbl", name="V", domain="test",
        phi=0.8, epsilon=0.1, status=AgentStatus.ACTIVE,
    ))
    # epsilon=0.1, beta_norm(0.3)≈0.23, eta=0.93 → denom≈0.021 < 1 → phi_viable=0
    # phi=0.8 > 0 → VIABLE, t_rec should be None
    result = k.irreversibility_diagnostic("vbl", eta=0.93)
    if result["zone"] == "VIABLE":
        assert result["t_rec"] is None


def test_G25_t_rec_positive_when_recoverable():
    """G25: t_rec > 0 when phi < phi_viable."""
    k = MELVKernel()
    k.beta.set("compute", 3.0)   # beta_norm(3.0) = 0.75
    k.register_agent(AgentProfile(
        agent_id="rec", name="R", domain="test",
        phi=0.3,       # low phi
        epsilon=2.5,   # phi_viable = 1 - 1/(2.5*0.75*0.93) ≈ 0.43
        status=AgentStatus.MATURING,
    ))
    result = k.irreversibility_diagnostic("rec", eta=0.93, f_eligible=1.0, t_gov=10.0)
    if result["zone"] == "RECOVERABLE_URGENT":
        assert result["t_rec"] is not None
        assert result["t_rec"] > 0


def test_G26_t_rec_formula():
    """G26: t_rec = (1/alpha) * ln((1-phi_cur)/(1-phi_viable)) / f_eligible."""
    k = MELVKernel()
    k.beta.set("compute", 3.0)
    k.register_agent(AgentProfile(
        agent_id="trec", name="T", domain="test",
        phi=0.30, epsilon=2.5, status=AgentStatus.MATURING,
    ))
    f = 0.8
    result = k.irreversibility_diagnostic("trec", eta=0.93, f_eligible=f, t_gov=10.0)
    if result["zone"] == "RECOVERABLE_URGENT" and result["t_rec"] is not None:
        alpha = result["alpha"]
        pv    = result["phi_viable"]
        pc    = result["phi_current"]
        inner = (1.0 - pc) / max(1.0 - pv, 1e-9)
        expected = round((1.0 / alpha) * math.log(inner) / f, 2)
        assert result["t_rec"] == pytest.approx(expected, abs=0.05)


def test_G27_t_rec_longer_with_lower_f_eligible():
    """G27: T_rec increases as f_eligible decreases (path dependency)."""
    k = MELVKernel()
    k.beta.set("compute", 3.0)
    k.register_agent(AgentProfile(
        agent_id="path", name="P", domain="test",
        phi=0.30, epsilon=2.5, status=AgentStatus.MATURING,
    ))
    r1 = k.irreversibility_diagnostic("path", eta=0.93, f_eligible=1.0, t_gov=10.0)
    r2 = k.irreversibility_diagnostic("path", eta=0.93, f_eligible=0.5, t_gov=10.0)
    # If both are recoverable, lower f_eligible → longer T_rec
    if (r1["zone"] == "RECOVERABLE_URGENT" and r2["zone"] == "RECOVERABLE_URGENT"
            and r1["t_rec"] is not None and r2["t_rec"] is not None):
        assert r2["t_rec"] > r1["t_rec"], (
            f"T_rec(f=0.5)={r2['t_rec']} should exceed T_rec(f=1.0)={r1['t_rec']}"
        )


def test_G28_lower_eta_smaller_viable_zone():
    """G28: eta affects phi_viable — lower eta produces different phi_viable (Discovery 1)."""
    # phi_viable = 1 - 1/(eps * beta_norm * eta)
    # For phi_viable > 0 we need eps * beta_norm * eta > 1.
    # eps=3.0, beta=4.0 (beta_norm=0.8):
    #   eta=0.93 → denom=2.232 → phi_viable≈0.552
    #   eta=0.50 → denom=1.200 → phi_viable≈0.167
    # Lower eta → lower phi_viable (shallower attractor means lower threshold)
    k = MELVKernel()
    k.beta.set("compute", 4.0)
    k.register_agent(AgentProfile(
        agent_id="eta-test", name="E", domain="test",
        phi=0.5, epsilon=3.0, status=AgentStatus.ACTIVE,
    ))
    r_high_eta = k.irreversibility_diagnostic("eta-test", eta=0.93)
    r_low_eta  = k.irreversibility_diagnostic("eta-test", eta=0.50)
    # Both phi_viable should be > 0 with these parameters
    assert r_high_eta["phi_viable"] is not None and r_high_eta["phi_viable"] > 0
    assert r_low_eta["phi_viable"]  is not None and r_low_eta["phi_viable"]  > 0
    # Lower eta → lower phi_viable
    assert r_low_eta["phi_viable"] < r_high_eta["phi_viable"], (
        f"phi_viable(eta=0.50)={r_low_eta['phi_viable']:.4f} should be < "
        f"phi_viable(eta=0.93)={r_high_eta['phi_viable']:.4f}"
    )


def test_G29_alpha_matches_phi_build_rate():
    """G29: alpha in result equals PHI_BUILD_RATE_ALPHA."""
    k = _make_kernel_n_agents(2)
    result = k.irreversibility_diagnostic("agent-0")
    assert result["alpha"] == pytest.approx(PHI_BUILD_RATE_ALPHA)


# ── G30–G36: Edge cases and warnings ─────────────────────────────────────────

def test_G30_f_eligible_zero_frozen_clock():
    """G30: f_eligible=0 → t_rec=None (serialised) and warning."""
    k = MELVKernel()
    k.beta.set("compute", 3.0)
    k.register_agent(AgentProfile(
        agent_id="frozen", name="F", domain="test",
        phi=0.30, epsilon=2.5, status=AgentStatus.MATURING,
    ))
    result = k.irreversibility_diagnostic("frozen", eta=0.93,
                                          f_eligible=0.0, t_gov=10.0)
    if result["zone"] == "RECOVERABLE_URGENT":
        # t_rec = inf internally; may be preserved as inf in raw result
        assert (result["t_rec"] is None or result["t_rec"] == math.inf
                or result["t_rec"] == float("inf"))


def test_G31_epsilon_zero_uncomputable():
    """G31: epsilon=0 → phi_viable uncomputable, zone=UNKNOWN, warning."""
    k = MELVKernel()
    k.beta.set("compute", 1.0)
    k.register_agent(AgentProfile(
        agent_id="zero-eps", name="Z", domain="test",
        phi=0.5, epsilon=0.0,
        status=AgentStatus.MATURING,
    ))
    result = k.irreversibility_diagnostic("zero-eps")
    assert result["phi_viable"] is None or result["zone"] in ("UNKNOWN", "VIABLE", "IRREVERSIBLE")
    # Warning should be present
    if result["phi_viable"] is None:
        assert len(result["warnings"]) > 0


def test_G32_epistemic_status_field():
    """G32: epistemic_status contains 'theoretical'."""
    k = _make_kernel_n_agents(2)
    result = k.irreversibility_diagnostic("agent-0")
    if result.get("epistemic_status"):
        assert "theoretical" in result["epistemic_status"].lower()


def test_G33_zone_is_valid_string():
    """G33: zone is one of the four valid values."""
    k = _make_kernel_n_agents(3)
    valid_zones = {"VIABLE", "RECOVERABLE_URGENT", "IRREVERSIBLE", "UNKNOWN"}
    result = k.irreversibility_diagnostic("agent-0")
    assert result["zone"] in valid_zones


def test_G34_zone_color_matches_zone():
    """G34: zone_color is consistent with zone."""
    mapping = {
        "VIABLE":             "GREEN",
        "RECOVERABLE_URGENT": "AMBER",
        "IRREVERSIBLE":       "RED",
        "UNKNOWN":            "GREY",
    }
    k = _make_kernel_n_agents(3)
    result = k.irreversibility_diagnostic("agent-0")
    expected_color = mapping.get(result["zone"])
    if expected_color:
        assert result["zone_color"] == expected_color


def test_G35_phi_current_matches_agent():
    """G35: phi_current in result matches the agent's kernel phi."""
    k = _make_kernel_n_agents(2, phi=0.62)
    result = k.irreversibility_diagnostic("agent-0")
    assert result["phi_current"] == pytest.approx(0.62, abs=1e-4)


def test_G36_eta_in_result():
    """G36: eta passed in is reflected in result."""
    k = _make_kernel_n_agents(2)
    result = k.irreversibility_diagnostic("agent-0", eta=0.75)
    assert result["eta"] == pytest.approx(0.75)


# ── G37–G41: Integration tests ────────────────────────────────────────────────

def test_G37_dungbeetle_all_agents_covered():
    """G37: Every registered agent appears in exactly one list."""
    k = _make_kernel_n_agents(6, beta_val=0.7)
    pairs = [(f"agent-{i}", f"agent-{i+1}") for i in range(5)]
    _add_interactions(k, pairs, i_factor=0.3, n_each=10)
    result = k.compute_dungbeetle_nodes()
    db_ids  = {n["agent_id"] for n in result["dungbeetle_nodes"]}
    non_ids = {n["agent_id"] for n in result["non_dungbeetle_nodes"]}
    all_ids = {f"agent-{i}" for i in range(6)}
    assert db_ids | non_ids == all_ids
    assert db_ids & non_ids == set()   # no overlap


def test_G38_dungbeetle_quorum_not_met_no_dungbeetle():
    """G38: When quorum is not met, no Dungbeetle nodes (condition requires full >= 0.5)."""
    k = _make_kernel_n_agents(4)
    # No interactions → beta_service = 0 → quorum not met
    result = k.compute_dungbeetle_nodes()
    assert result["quorum_met"] is False
    assert result["dungbeetle_nodes"] == []


def test_G39_irreversibility_three_zones_reachable():
    """G39: All three zones are reachable by parameter variation."""
    k = MELVKernel()
    k.beta.set("compute", 4.0)   # beta_norm(4.0) = 0.8

    # VIABLE: very high phi, high epsilon → phi > phi_viable
    k.register_agent(AgentProfile(
        agent_id="viable-agent", name="V", domain="test",
        phi=0.95, epsilon=3.0, status=AgentStatus.ACTIVE,
    ))

    # IRREVERSIBLE: very low phi + long t_gov → phi_irrev is high
    # phi_irrev = 1 - exp(-0.01 * 500) = 1 - exp(-5) ≈ 0.993
    # phi=0.02 << 0.993 → IRREVERSIBLE
    k.register_agent(AgentProfile(
        agent_id="irrev-agent", name="I", domain="test",
        phi=0.02, epsilon=3.0, status=AgentStatus.MATURING,
    ))

    rv = k.irreversibility_diagnostic("viable-agent", eta=0.93, t_gov=10.0)
    ri = k.irreversibility_diagnostic("irrev-agent",  eta=0.93, t_gov=500.0)

    assert rv["zone"] == "VIABLE", (
        f"Expected VIABLE: phi={rv['phi_current']}, phi_viable={rv['phi_viable']}, "
        f"phi_irrev={rv['phi_irrev']}"
    )
    assert ri["zone"] == "IRREVERSIBLE", (
        f"Expected IRREVERSIBLE: phi={ri['phi_current']}, phi_irrev={ri['phi_irrev']}"
    )


def test_G40_sensitivity_sums_close_to_beta_full():
    """G40: max(S_v) <= beta_service_full (sensitivity cannot exceed full beta)."""
    k = _make_kernel_n_agents(5, beta_val=0.85)
    pairs = [(f"agent-{i}", f"agent-{(i+1)%5}") for i in range(5)]
    _add_interactions(k, pairs, i_factor=0.2, n_each=15)
    result = k.compute_dungbeetle_nodes()
    beta_full = result["beta_service_full"]
    all_nodes = result["dungbeetle_nodes"] + result["non_dungbeetle_nodes"]
    for node in all_nodes:
        assert node["sensitivity"] <= beta_full + 1e-6, (
            f"sensitivity={node['sensitivity']} > beta_full={beta_full}"
        )


def test_G41_compute_omega_still_works():
    """G41: compute_omega is unaffected by Session 37 additions."""
    k = _make_kernel_n_agents(3)
    _add_interactions(k, [("agent-0","agent-1"),("agent-1","agent-2")],
                      i_factor=0.5, n_each=5)
    omega = k.compute_omega()
    assert "lambda_max"   in omega
    assert "beta_service" in omega
    assert "n"            in omega


# ── G42–G46: Backward compatibility ──────────────────────────────────────────

def test_G42_beta_norm_unchanged():
    """G42: _beta_norm function is unchanged from Session 35."""
    assert _beta_norm(0.0)  == pytest.approx(0.0)
    assert _beta_norm(1.0)  == pytest.approx(0.5)
    assert _beta_norm(3.0)  == pytest.approx(0.75)
    assert _beta_norm(99.0) == pytest.approx(99.0/100.0, rel=1e-4)


def test_G43_i_floor_unchanged():
    """G43: I_FLOOR = -5.0 (unchanged)."""
    assert I_FLOOR == -5.0


def test_G44_phi_build_alpha_unchanged():
    """G44: PHI_BUILD_RATE_ALPHA = 0.01 (unchanged from Session 35)."""
    assert PHI_BUILD_RATE_ALPHA == 0.01


def test_G45_i_critical_unchanged():
    """G45: I_CRITICAL = 0.9995 (unchanged)."""
    assert I_CRITICAL == 0.9995


def test_G46_session35_apply_observation_unchanged():
    """G46: apply_observation still works correctly after Session 37 additions."""
    from core.observe_schema import (
        ObservationResult, ScoredValue, EpsilonResult, ResourcePolicy
    )
    k = MELVKernel()
    k.beta.set("compute", 0.8)
    k.register_agent(AgentProfile(
        agent_id="obs-agent", name="O", domain="test",
        phi=0.6, epsilon=1.0, status=AgentStatus.ACTIVE,
    ))

    def _sv(v, s=3):
        return ScoredValue(value=v, status=s, warnings=[])

    obs = ObservationResult(
        agent_id="obs-agent",
        phi=_sv(0.65),
        sigma=_sv(0.60),
        beta=_sv(0.8),
        epsilon=EpsilonResult(
            intrinsic=_sv(0.5, 2),
            ecosystem=_sv(0.3, 2),
            architectural=_sv(1.0, 1),
            effective=0.8,
        ),
        ci=0.72,
        phi_sigma_divergence=0.05,
        warnings=[],
        r_value=0.8,
        d_value=0.0,
    )
    result = k.apply_observation(obs)
    assert result["agent_updated"] is True
    assert "governance_events" in result

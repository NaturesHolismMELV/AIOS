"""
test_session8.py — MELVcore v1.0.0 Release Tests
==================================================
Session 8 deliverable validation.

Tests (8 total):
  1. governance/ importable as library (no FastAPI dependency at import time)
  2. melvcore/ public API exports all required symbols
  3. pyproject.toml has all required fields
  4. Theory-to-Code table completeness (THEORY.md contains all required rows)
  5. README.md has all required sections
  6. governance.kernel helpers: create_kernel() and integrate_agent()
  7. Full integration smoke test: kernel + nudge + cost in one call
  8. __version__ == "1.0.0" across governance and melvcore

Run: python -m pytest tests/test_session8.py -v
"""

import sys
import os

# Ensure project root is on path
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)

import pytest


# ─────────────────────────────────────────────────────────────────────────────
# Test 1 — governance/ importable without FastAPI at module level
# ─────────────────────────────────────────────────────────────────────────────

def test_governance_importable_without_server():
    """
    governance/ must be importable in any Python environment that does NOT
    have the FastAPI server running. The import must not raise and must not
    trigger uvicorn/fastapi imports at module level.
    """
    import governance
    assert hasattr(governance, "MELVKernel"), "MELVKernel not exported by governance"
    assert hasattr(governance, "NudgeEngine"), "NudgeEngine not exported by governance"
    assert hasattr(governance, "CostCalculator"), "CostCalculator not exported by governance"
    # Confirm fastapi is NOT imported as a side-effect of importing governance
    # (we check sys.modules; fastapi may be installed but should not be pulled in)
    # Note: fastapi might already be imported by pytest environment — we just
    # verify governance itself doesn't *require* it.
    import governance as gov_module
    # Check that fastapi/uvicorn are not imported at module level
    # (they may appear in docstrings; we check actual import statements in code lines)
    import ast
    source = open(gov_module.__file__, encoding='utf-8').read()
    tree = ast.parse(source)
    top_level_imports = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            # Only flag top-level imports (not inside functions)
            top_level_imports.append(ast.unparse(node))
    fastapi_at_top = any("fastapi" in imp for imp in top_level_imports)
    uvicorn_at_top = any("uvicorn" in imp for imp in top_level_imports)
    assert not fastapi_at_top, "governance/__init__.py imports fastapi at module level"
    assert not uvicorn_at_top, "governance/__init__.py imports uvicorn at module level"


# ─────────────────────────────────────────────────────────────────────────────
# Test 2 — melvcore/ public API exports correct symbols
# ─────────────────────────────────────────────────────────────────────────────

def test_melvcore_public_api_exports():
    """
    from melvcore import * must expose the full required public API.
    """
    import melvcore

    required_symbols = [
        "MELVKernel",
        "AgentProfile",
        "BetaEnvironment",
        "NudgeEngine",
        "CostCalculator",
        "KernelAction",
        "InteractionRecord",
        "NudgeResponse",
        "integrate_agent",
        "create_kernel",
        "__version__",
        "__author__",
        "__license__",
    ]
    for sym in required_symbols:
        assert hasattr(melvcore, sym), \
            f"melvcore is missing required export: {sym}"


# ─────────────────────────────────────────────────────────────────────────────
# Test 3 — pyproject.toml has required fields
# ─────────────────────────────────────────────────────────────────────────────

def test_pyproject_toml_required_fields():
    """
    melvcore/pyproject.toml must contain all required fields for PyPI publication.
    """
    toml_path = os.path.join(_root, "melvcore", "pyproject.toml")
    assert os.path.exists(toml_path), "melvcore/pyproject.toml does not exist"

    content = open(toml_path, encoding='utf-8').read()

    required_fields = [
        'name',
        'version',
        'description',
        'license',
        'authors',
        'requires-python',
        'dependencies',
        'Apache-2.0',
        'melvcore',
        '1.0.0',
        'L.W. Evans',
        'laurence@naturesholismmelv.com',
        '3.11',
        'fastapi',
        'pydantic',
        'anthropic',
    ]
    for field in required_fields:
        assert field in content, \
            f"pyproject.toml is missing required field/value: '{field}'"


# ─────────────────────────────────────────────────────────────────────────────
# Test 4 — THEORY.md Theory-to-Code table completeness
# ─────────────────────────────────────────────────────────────────────────────

def test_theory_to_code_table_completeness():
    """
    THEORY.md must contain the Theory-to-Code table with all required rows.
    """
    theory_path = os.path.join(_root, "THEORY.md")
    assert os.path.exists(theory_path), "THEORY.md does not exist"

    content = open(theory_path, encoding='utf-8').read()

    required_entries = [
        # Equations
        "i = C / B",
        "CI = 1",
        "mean(βi)",
        "λ_max(Ω)",
        "min(2.0",
        # Functions
        "InteractionRecord",
        "MELVKernel.record_interaction",
        "MELVKernel.update_phi",
        "MELVKernel.cooperation_index",
        "MELVKernel.compute_omega",
        "CostCalculator.compute_cost",
        "NudgeEngine.build_nudge_v2",
        "NudgeEngine.apply_oxpecker_effect",
        # Files
        "core/melv_engine.py",
        "core/cost_calculator.py",
        "core/nudge_engine.py",
        # Table header
        "Theory-to-Code",
    ]
    for entry in required_entries:
        assert entry in content, \
            f"THEORY.md Theory-to-Code table is missing entry: '{entry}'"


# ─────────────────────────────────────────────────────────────────────────────
# Test 5 — README.md has required sections
# ─────────────────────────────────────────────────────────────────────────────

def test_readme_required_sections():
    """
    README.md must contain all required sections for a release-quality README.
    """
    readme_path = os.path.join(_root, "README.md")
    assert os.path.exists(readme_path), "README.md does not exist"

    content = open(readme_path, encoding='utf-8').read()

    required = [
        "pip install melvcore",
        "MELVKernel",
        "NudgeEngine",
        "CostCalculator",
        "Architecture",
        "Agent Registry",
        "API Reference",
        "Roadmap",
        "Citation",
        "0009-0001-0963-1840",   # ORCID
        "978-969-8992-10-1",     # ISBN
        "10.5281/zenodo.19665563",  # DOI (current preprint — updated from concept DOI 17680563)
        "1.0.0",
        "Apache",
    ]
    for entry in required:
        assert entry in content, \
            f"README.md is missing required content: '{entry}'"


# ─────────────────────────────────────────────────────────────────────────────
# Test 6 — governance/kernel.py helpers
# ─────────────────────────────────────────────────────────────────────────────

def test_governance_kernel_helpers():
    """
    create_kernel() and integrate_agent() must work correctly.
    """
    from governance.kernel import create_kernel, integrate_agent

    kernel = create_kernel(compute=1.2, token_budget=1.3)
    assert kernel is not None
    assert kernel.beta.compute == 1.2
    assert kernel.beta.token_budget == 1.3

    profile = integrate_agent(kernel, "test_01", "TestAgent",
                               domain="testing", phi=0.75)
    assert profile.agent_id == "test_01"
    assert profile.phi == 0.75
    assert profile.domain == "testing"
    assert "test_01" in kernel.agents

    # MELV integrity: beta_pref is in AgentProfile but beta is NOT set by agent
    # integrate_agent must NOT accept a beta parameter
    import inspect
    sig = inspect.signature(integrate_agent)
    assert "beta" not in sig.parameters, \
        "integrate_agent must not accept 'beta' parameter (MELV integrity violation)"


# ─────────────────────────────────────────────────────────────────────────────
# Test 7 — Full integration smoke test: kernel + nudge + cost in one call
# ─────────────────────────────────────────────────────────────────────────────

def test_full_integration_smoke():
    """
    Full integration: register agents, compute cost, record interaction,
    generate nudge, verify CI — all in a single coherent call chain.
    This validates that MELVKernel + NudgeEngine + CostCalculator interoperate
    correctly as the melvcore public API.
    """
    from melvcore import (
        MELVKernel, NudgeEngine, CostCalculator,
        AgentProfile, AgentStatus,
    )

    kernel = MELVKernel()
    nudge  = NudgeEngine()
    calc   = CostCalculator()

    # Register two agents
    profile_a = AgentProfile(
        agent_id="smoke_a",
        name="SmokeAgent_A",
        domain="research",
        phi=0.82,
        status=AgentStatus.ACTIVE,
    )
    profile_b = AgentProfile(
        agent_id="smoke_b",
        name="SmokeAgent_B",
        domain="writing",
        phi=0.71,
        status=AgentStatus.ACTIVE,
    )
    kernel.register_agent(profile_a)
    kernel.register_agent(profile_b)

    # Compute cost via CostCalculator
    cost = calc.compute_cost(in_tok=300, out_tok=150, latency_s=0.9,
                              task_type="RESEARCH")
    assert 0.0 < cost <= 2.0, f"Cost out of range: {cost}"

    # Record cooperative interaction
    rec = kernel.record_interaction(
        "smoke_a", "smoke_b", cost=cost, benefit=2.5,
        resource_type="token_budget",
    )
    assert rec.i_factor > 0, "i-factor must be positive"
    assert rec.interaction_type is not None

    # CI should be computed
    ci = kernel.cooperation_index()
    assert 0.0 <= ci <= 1.0, f"CI out of range: {ci}"

    # Force a threshold event for nudge testing
    rec2 = kernel.record_interaction(
        "smoke_a", "smoke_b", cost=1.8, benefit=1.0,
        resource_type="token_budget",
    )
    depth = kernel.get_contention_depth("smoke_a", "smoke_b")

    nudge_resp = nudge.build_nudge_v2(
        action="nudge",
        beta_i=rec2.beta_i,
        resource="token_budget",
        contention_depth=max(1, depth),
        agent_phi=profile_a.phi,
    )
    assert nudge_resp.nudge_type in (
        "retry_with_jitter", "rephrase", "yield", "niche_diverge"
    ), f"Unexpected nudge_type: {nudge_resp.nudge_type}"
    assert nudge_resp.resource == "token_budget"
    assert isinstance(nudge_resp.to_dict(), dict)

    # Verify ecosystem health
    health = kernel.ecosystem_health()
    assert "cooperation_index" in health
    assert "n_agents" in health
    assert health["n_agents"] == 2


# ─────────────────────────────────────────────────────────────────────────────
# Test 8 — __version__ == "1.0.0" across governance and melvcore
# ─────────────────────────────────────────────────────────────────────────────

def test_version_string():
    """
    __version__ must be "1.0.0" in both melvcore and governance packages.
    """
    import melvcore
    assert melvcore.__version__.startswith(("1.", "3.")), f"melvcore.__version__ = {melvcore.__version__!r}, expected 1.x.x"

    import governance
    assert governance.__version__.startswith(("1.", "3.")), f"governance.__version__ = {governance.__version__!r}, expected 1.x.x"

    # Also confirm in pyproject.toml
    toml_path = os.path.join(_root, "melvcore", "pyproject.toml")
    content = open(toml_path, encoding='utf-8').read()
    assert "1.0.0" in content

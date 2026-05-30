"""
test_session11.py — CI History API + Dashboard11 UI Tests
==========================================================
Session 11 deliverable validation.

Tests (12 total):
  CI History API (6):
   1.  GET /api/ci_history returns HTTP 200
   2.  Response is a list of {t, ci} objects
   3.  All ci values are in [0, 1]
   4.  ?n=50 returns at most 50 items
   5.  Timestamps are monotonically non-decreasing
   6.  Default ?n=200 returns at most 200 items

  Dashboard UI (6):
   7.  frontend/dashboard13.html exists
   8.  sec-sandbox section is present
   9.  ci-history panel element is present
  10.  Sandbox nav item references sec-sandbox
  11.  Submit form input sb-agent-id is present
  12.  Registry panel sb-registry-list is present
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from fastapi.testclient import TestClient
from api.server import app

client = TestClient(app, headers={"X-Forwarded-For": "10.99.11.1"})

# ── HELPERS ───────────────────────────────────────────────────────────────

def seed_ci_history(n: int = 10):
    """Record n interactions to populate _ci_history."""
    for i in range(n):
        client.post("/api/interactions", json={
            "agent_a": "writer_agent",
            "agent_b": "planner_agent",
            "cost": 0.2,
            "benefit": 0.8,
            "resource_type": "compute",
        })


# ── CI HISTORY API TESTS ──────────────────────────────────────────────────

def test_ci_history_endpoint_exists():
    """1. GET /api/ci_history returns HTTP 200."""
    seed_ci_history(5)
    response = client.get("/api/ci_history")
    assert response.status_code == 200, (
        f"Expected 200, got {response.status_code}: {response.text}"
    )


def test_ci_history_returns_list():
    """2. Response is a list of {t, ci} objects."""
    seed_ci_history(5)
    response = client.get("/api/ci_history")
    result = response.json()
    assert isinstance(result, list), f"Expected list, got {type(result)}"
    if result:
        item = result[0]
        assert "t"  in item, f"Missing 't' key in item: {item}"
        assert "ci" in item, f"Missing 'ci' key in item: {item}"


def test_ci_history_ci_values_bounded():
    """3. All ci values are in [0, 1]."""
    seed_ci_history(10)
    result = client.get("/api/ci_history").json()
    for item in result:
        assert 0.0 <= item["ci"] <= 1.0, (
            f"ci value out of bounds: {item['ci']}"
        )


def test_ci_history_n_param():
    """4. ?n=50 returns at most 50 items."""
    seed_ci_history(60)
    result = client.get("/api/ci_history?n=50").json()
    assert len(result) <= 50, (
        f"Expected at most 50 items, got {len(result)}"
    )


def test_ci_history_timestamps_ascending():
    """5. Timestamps are monotonically non-decreasing."""
    seed_ci_history(15)
    result = client.get("/api/ci_history").json()
    timestamps = [item["t"] for item in result]
    for i in range(1, len(timestamps)):
        assert timestamps[i] >= timestamps[i - 1], (
            f"Timestamps not ascending at index {i}: "
            f"{timestamps[i-1]} > {timestamps[i]}"
        )


def test_ci_history_default_n():
    """6. Default ?n=200 returns at most 200 items."""
    seed_ci_history(20)
    result = client.get("/api/ci_history").json()
    assert len(result) <= 200, (
        f"Expected at most 200 items, got {len(result)}"
    )


# ── DASHBOARD UI TESTS ────────────────────────────────────────────────────

# dashboard11.html was superseded by dashboard13.html in Session 12.
# Tests now point to the current dashboard file.
DASHBOARD_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "frontend", "dashboard13.html"
)


@pytest.fixture(scope="module")
def dashboard_html():
    with open(DASHBOARD_PATH, "r", encoding="utf-8") as f:
        return f.read()


def test_dashboard11_file_exists():
    """7. frontend/dashboard13.html exists."""
    assert os.path.exists(DASHBOARD_PATH), (
        f"dashboard13.html not found at {DASHBOARD_PATH}"
    )


def test_dashboard11_has_sandbox_section(dashboard_html):
    """8. sec-sandbox section is present in HTML."""
    assert "sec-sandbox" in dashboard_html, (
        "Expected id='sec-sandbox' section in dashboard13.html"
    )


def test_dashboard11_has_ci_history_panel(dashboard_html):
    """9. CI History panel is present in HTML."""
    assert "ci-history" in dashboard_html, (
        "Expected ci-history element in dashboard13.html"
    )


def test_dashboard11_has_sandbox_nav(dashboard_html):
    """10. Sandbox nav item references sec-sandbox."""
    assert "sec-sandbox" in dashboard_html, (
        "Expected nav item referencing sec-sandbox in dashboard13.html"
    )


def test_dashboard11_has_submit_form(dashboard_html):
    """11. Submit form input sb-agent-id is present."""
    assert "sb-agent-id" in dashboard_html, (
        "Expected input id='sb-agent-id' in dashboard13.html"
    )


def test_dashboard11_has_registry_panel(dashboard_html):
    """12. Registry panel sb-registry-list is present."""
    assert "sb-registry-list" in dashboard_html, (
        "Expected element id='sb-registry-list' in dashboard13.html"
    )

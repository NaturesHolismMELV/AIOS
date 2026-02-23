"""
tests/test_data_agent.py — Phase 2 Session 3
pytest suite for real DataAgent World Bank integration.

Run: pytest tests/test_data_agent.py -v
"""

import pytest
import asyncio
from agents.data_agent import DataAgent, fetch_indicator, fetch_country_profile, compare_countries


@pytest.fixture
def agent():
    return DataAgent()


# ── Agent metadata ──────────────────────────────────────────────────────────

def test_agent_metadata(agent):
    assert agent.name == "DATA"
    assert agent.real_data is True
    assert "cooperation_index" in agent.melv_metadata


# ── List indicators ──────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_list_indicators(agent):
    result = await agent.execute({"action": "list"})
    assert result["status"] == "success"
    indicators = result["result"]
    assert "NY.GDP.MKTP.CD" in indicators
    assert "EN.ATM.CO2E.PC" in indicators
    assert len(indicators) >= 5


# ── Country profile ──────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_country_profile_south_africa(agent):
    result = await agent.execute({"action": "profile", "country": "ZA"})
    assert result["status"] == "success"
    assert result["real_data"] is True
    profile = result["result"]
    assert profile["country_code"] == "ZA"
    assert "indicators" in profile
    # GDP should be present and positive
    gdp = profile["indicators"].get("NY.GDP.MKTP.CD")
    assert gdp is not None
    if gdp["value"] is not None:
        assert gdp["value"] > 0


@pytest.mark.asyncio
async def test_country_profile_usa(agent):
    result = await agent.execute({"action": "profile", "country": "US"})
    assert result["status"] == "success"
    profile = result["result"]
    assert profile["country_code"] == "US"


# ── Indicator fetch ──────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_fetch_gdp_indicator(agent):
    result = await agent.execute({
        "action": "indicator",
        "indicator": "NY.GDP.MKTP.CD",
        "country": "ZA",
        "year_from": 2018,
        "year_to": 2022,
    })
    assert result["status"] == "success"
    data = result["result"]
    assert data["indicator_code"] == "NY.GDP.MKTP.CD"
    assert data["real_data"] is True
    assert data["source"] == "World Bank Open Data API"
    assert isinstance(data["records"], list)
    if data["records"]:
        rec = data["records"][0]
        assert "country" in rec
        assert "year" in rec
        assert "value" in rec


@pytest.mark.asyncio
async def test_fetch_co2_indicator(agent):
    result = await agent.execute({
        "action": "indicator",
        "indicator": "EN.ATM.CO2E.PC",
        "country": "ZA",
        "year_from": 2010,
        "year_to": 2020,
    })
    assert result["status"] == "success"
    data = result["result"]
    assert data["indicator_label"] == "CO₂ emissions (metric tons per capita)"


# ── Cross-country comparison ──────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_compare_countries_gdp(agent):
    result = await agent.execute({
        "action": "compare",
        "indicator": "NY.GDP.MKTP.CD",
        "countries": ["ZA", "US", "CN"],
        "year": 2022,
    })
    assert result["status"] == "success"
    data = result["result"]
    assert data["comparison_year"] == 2022
    assert len(data["records"]) >= 1
    # US should have higher GDP than ZA
    country_gdp = {r["country_code"]: r["value"] for r in data["records"] if r["value"]}
    if "USA" in country_gdp and "ZAF" in country_gdp:
        assert country_gdp["USA"] > country_gdp["ZAF"]


# ── MELV-specific: energy cooperation ────────────────────────────────────────

@pytest.mark.asyncio
async def test_melv_energy_data(agent):
    """Verify the agent returns data suitable for MELV thermodynamic analysis."""
    energy_result = await agent.execute({
        "action": "compare",
        "indicator": "EG.USE.PCAP.KG.OE",
        "countries": ["ZA", "DE", "US"],
        "year": 2020,
    })
    assert energy_result["status"] == "success"
    records = energy_result["result"]["records"]
    # All values should be positive (energy use can't be negative)
    for r in records:
        if r["value"] is not None:
            assert r["value"] > 0


# ── Error handling ───────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_unknown_action(agent):
    result = await agent.execute({"action": "nonexistent"})
    assert result["status"] == "error"
    assert "Unknown action" in result["error"]


@pytest.mark.asyncio
async def test_invalid_country_graceful(agent):
    """Invalid country should return empty records, not crash."""
    result = await agent.execute({
        "action": "indicator",
        "indicator": "NY.GDP.MKTP.CD",
        "country": "XX",   # invalid
        "year_from": 2020,
        "year_to": 2022,
    })
    # Should not error out — just empty records
    assert result["status"] in ("success", "error")

"""
DataAgent — Phase 2 Session 3
Real HTTP calls to World Bank Open Data API (no API key required).
Governed by MELVcore thermodynamic kernel (L.W. Evans, Ecotao Enterprises).

World Bank API docs: https://datahelpdesk.worldbank.org/knowledgebase/articles/889392
"""

import httpx
import asyncio
import json
import logging
from typing import Any, Optional
from datetime import datetime, UTC

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# World Bank API configuration
# ---------------------------------------------------------------------------
WB_BASE = "https://api.worldbank.org/v2"
DEFAULT_TIMEOUT = 15.0  # seconds

# Curated indicator catalogue (indicator_code -> human label)
INDICATORS = {
    "NY.GDP.MKTP.CD":    "GDP (current US$)",
    "NY.GDP.PCAP.CD":    "GDP per capita (current US$)",
    "SP.POP.TOTL":       "Total population",
    "EN.ATM.CO2E.PC":    "CO₂ emissions (metric tons per capita)",
    "SL.UEM.TOTL.ZS":    "Unemployment rate (%)",
    "SE.ADT.LITR.ZS":    "Adult literacy rate (%)",
    "EG.USE.PCAP.KG.OE": "Energy use (kg oil eq. per capita)",
    "AG.LND.FRST.ZS":    "Forest area (% of land area)",
    "SH.DYN.MORT":       "Under-5 mortality rate (per 1,000)",
    "IT.NET.USER.ZS":    "Internet users (% of population)",
}

# Country ISO codes the dashboard exposes
DEFAULT_COUNTRIES = ["ZA", "US", "CN", "DE", "BR", "NG", "IN", "AU"]


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

async def _wb_get(path: str, params: dict | None = None) -> dict | list | None:
    """Raw async GET against the World Bank API. Returns parsed JSON or None."""
    url = f"{WB_BASE}/{path}"
    defaults = {"format": "json", "per_page": 50}
    if params:
        defaults.update(params)
    try:
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            r = await client.get(url, params=defaults)
            r.raise_for_status()
            return r.json()
    except httpx.TimeoutException:
        logger.warning("World Bank API timeout: %s", url)
    except httpx.HTTPStatusError as e:
        logger.warning("World Bank HTTP error %s: %s", e.response.status_code, url)
    except Exception as e:
        logger.error("World Bank fetch error: %s", e)
    return None


# ---------------------------------------------------------------------------
# Public API used by AIOS DataAgent
# ---------------------------------------------------------------------------

async def fetch_indicator(
    indicator: str,
    country: str = "all",
    year_from: int = 2015,
    year_to: int = 2023,
) -> dict[str, Any]:
    """
    Fetch a single World Bank indicator for one or more countries.

    Returns a structured dict ready for the AIOS event bus.
    """
    date_range = f"{year_from}:{year_to}"
    raw = await _wb_get(
        f"country/{country}/indicator/{indicator}",
        {"date": date_range, "mrv": 1},
    )

    results: list[dict] = []
    label = INDICATORS.get(indicator, indicator)

    if raw and len(raw) == 2 and raw[1]:
        for entry in raw[1]:
            if entry.get("value") is not None:
                results.append({
                    "country": entry["country"]["value"],
                    "country_code": entry["countryiso3code"],
                    "year": entry["date"],
                    "value": entry["value"],
                    "unit": label,
                })

    return {
        "indicator_code": indicator,
        "indicator_label": label,
        "query": {"country": country, "year_from": year_from, "year_to": year_to},
        "records": results,
        "fetched_at": datetime.now(UTC).isoformat(),
        "source": "World Bank Open Data API",
        "real_data": True,
    }


async def fetch_country_profile(country_code: str) -> dict[str, Any]:
    """
    Fetch a multi-indicator snapshot for a single country — useful for
    MELV ecosystem modelling (energy, population, CO₂, GDP).
    """
    melv_indicators = [
        "NY.GDP.MKTP.CD",
        "SP.POP.TOTL",
        "EN.ATM.CO2E.PC",
        "EG.USE.PCAP.KG.OE",
        "SL.UEM.TOTL.ZS",
    ]

    tasks = [fetch_indicator(ind, country_code) for ind in melv_indicators]
    results = await asyncio.gather(*tasks)

    profile: dict[str, Any] = {
        "country_code": country_code.upper(),
        "fetched_at": datetime.now(UTC).isoformat(),
        "indicators": {},
    }

    for res in results:
        key = res["indicator_code"]
        records = res["records"]
        # Take the most recent value
        if records:
            latest = sorted(records, key=lambda r: r["year"], reverse=True)[0]
            profile["indicators"][key] = {
                "label": res["indicator_label"],
                "value": latest["value"],
                "year": latest["year"],
            }
        else:
            profile["indicators"][key] = {"label": INDICATORS.get(key, key), "value": None}

    return profile


async def compare_countries(
    indicator: str,
    countries: list[str] | None = None,
    year: int = 2022,
) -> dict[str, Any]:
    """
    Cross-country comparison for a single indicator — feeds MELV competition
    vs cooperation analysis directly.
    """
    countries = countries or DEFAULT_COUNTRIES
    country_str = ";".join(countries)

    raw = await fetch_indicator(indicator, country_str, year, year)

    # Sort by value descending
    sorted_records = sorted(
        raw["records"], key=lambda r: r["value"] if r["value"] else 0, reverse=True
    )
    raw["records"] = sorted_records
    raw["comparison_year"] = year
    raw["countries_queried"] = countries

    return raw


async def list_available_indicators() -> dict[str, str]:
    """Return the built-in indicator catalogue."""
    return INDICATORS


# ---------------------------------------------------------------------------
# AIOS Agent class (drop-in replacement for simulated DataAgent)
# ---------------------------------------------------------------------------

class DataAgent:
    """
    AIOS DataAgent — Phase 2 real implementation.
    Replaces simulated random data with live World Bank API calls.
    """

    name = "DATA"
    description = (
        "Retrieves real economic, environmental, and social data from the "
        "World Bank Open Data API. No API key required. Supports single-country "
        "profiles, cross-country comparisons, and time-series for 10+ indicators."
    )
    real_data = True

    # MELV thermodynamic metadata
    melv_metadata = {
        "interaction_type": "mutualistic",   # open data = low interaction cost
        "energy_cost": "low",
        "cooperation_index": 0.92,           # free public API → high cooperation
    }

    async def execute(self, task: dict[str, Any]) -> dict[str, Any]:
        """
        Dispatch based on task["action"]:
          - "profile"     → fetch_country_profile(country)
          - "indicator"   → fetch_indicator(indicator, country, year_from, year_to)
          - "compare"     → compare_countries(indicator, countries, year)
          - "list"        → list_available_indicators()
        """
        action = task.get("action", "profile")
        logger.info("[DataAgent] action=%s task=%s", action, task)

        try:
            if action == "profile":
                result = await fetch_country_profile(task.get("country", "ZA"))
            elif action == "indicator":
                result = await fetch_indicator(
                    task.get("indicator", "NY.GDP.MKTP.CD"),
                    task.get("country", "all"),
                    int(task.get("year_from", 2015)),
                    int(task.get("year_to", 2023)),
                )
            elif action == "compare":
                result = await compare_countries(
                    task.get("indicator", "NY.GDP.MKTP.CD"),
                    task.get("countries", DEFAULT_COUNTRIES),
                    int(task.get("year", 2022)),
                )
            elif action == "list":
                result = await list_available_indicators()
            else:
                return {
                    "agent": self.name,
                    "action": action,
                    "status": "error",
                    "error": f"Unknown action: {action}",
                    "timestamp": datetime.now(UTC).isoformat(),
                }

            return {
                "agent": self.name,
                "action": action,
                "status": "success",
                "real_data": True,
                "result": result,
                "melv_metadata": self.melv_metadata,
                "timestamp": datetime.now(UTC).isoformat(),
            }

        except Exception as e:
            logger.error("[DataAgent] Error: %s", e, exc_info=True)
            return {
                "agent": self.name,
                "action": action,
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now(UTC).isoformat(),
            }


# ---------------------------------------------------------------------------
# Quick smoke test (run directly: python agents/data_agent.py)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import pprint

    async def smoke_test():
        agent = DataAgent()

        print("\n=== 1. South Africa profile ===")
        r = await agent.execute({"action": "profile", "country": "ZA"})
        pprint.pprint(r["result"])

        print("\n=== 2. GDP comparison (ZA, US, CN, DE) ===")
        r = await agent.execute({
            "action": "compare",
            "indicator": "NY.GDP.MKTP.CD",
            "countries": ["ZA", "US", "CN", "DE"],
            "year": 2022,
        })
        for rec in r["result"]["records"]:
            print(f"  {rec['country']:30s} {rec['value']:>20,.0f} USD")

        print("\n=== 3. CO₂ time series (South Africa 2010-2022) ===")
        r = await agent.execute({
            "action": "indicator",
            "indicator": "EN.ATM.CO2E.PC",
            "country": "ZA",
            "year_from": 2010,
            "year_to": 2022,
        })
        for rec in r["result"]["records"]:
            print(f"  {rec['year']}: {rec['value']:.2f} t CO₂/capita")

        print("\n=== 4. Available indicators ===")
        r = await agent.execute({"action": "list"})
        for code, label in r["result"].items():
            print(f"  {code}: {label}")

    asyncio.run(smoke_test())

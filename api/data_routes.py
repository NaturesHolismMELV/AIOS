"""
AIOS API routes for DataAgent — Phase 2 Session 3
Add these routes to api/server.py (or include as a router).

Usage in server.py:
    from api.data_routes import router as data_router
    app.include_router(data_router, prefix="/data", tags=["data"])
"""

from fastapi import APIRouter, Query
from typing import Optional
from agents.data_agent import DataAgent

router = APIRouter()
_agent = DataAgent()


@router.get("/profile/{country_code}")
async def country_profile(country_code: str):
    """
    Full MELV-relevant indicator snapshot for a country.
    Example: GET /data/profile/ZA
    """
    return await _agent.execute({"action": "profile", "country": country_code.upper()})


@router.get("/indicator/{indicator_code}")
async def indicator(
    indicator_code: str,
    country: str = Query("all", description="ISO2 code or 'all'"),
    year_from: int = Query(2015),
    year_to: int = Query(2023),
):
    """
    Time series for a specific indicator.
    Example: GET /data/indicator/NY.GDP.MKTP.CD?country=ZA&year_from=2010&year_to=2022
    """
    return await _agent.execute({
        "action": "indicator",
        "indicator": indicator_code,
        "country": country,
        "year_from": year_from,
        "year_to": year_to,
    })


@router.get("/compare/{indicator_code}")
async def compare(
    indicator_code: str,
    countries: str = Query("ZA;US;CN;DE;BR", description="Semicolon-separated ISO2 codes"),
    year: int = Query(2022),
):
    """
    Cross-country comparison for one indicator.
    Example: GET /data/compare/NY.GDP.MKTP.CD?countries=ZA;US;CN;DE&year=2022
    """
    country_list = [c.strip().upper() for c in countries.split(";") if c.strip()]
    return await _agent.execute({
        "action": "compare",
        "indicator": indicator_code,
        "countries": country_list,
        "year": year,
    })


@router.get("/indicators")
async def list_indicators():
    """List all available World Bank indicators."""
    return await _agent.execute({"action": "list"})


@router.get("/melv/energy-cooperation")
async def melv_energy_cooperation(
    countries: str = Query("ZA;US;CN;DE;BR;NG;IN;AU"),
    year: int = Query(2022),
):
    """
    MELV-specific endpoint: fetches energy use + CO₂ for cross-country
    thermodynamic cooperation analysis.
    """
    country_list = [c.strip().upper() for c in countries.split(";") if c.strip()]
    energy = await _agent.execute({
        "action": "compare",
        "indicator": "EG.USE.PCAP.KG.OE",
        "countries": country_list,
        "year": year,
    })
    co2 = await _agent.execute({
        "action": "compare",
        "indicator": "EN.ATM.CO2E.PC",
        "countries": country_list,
        "year": year,
    })
    return {
        "melv_analysis": "energy_cooperation",
        "year": year,
        "energy_use": energy["result"],
        "co2_emissions": co2["result"],
        "note": (
            "High energy efficiency + low CO₂ → thermodynamic cooperation signal. "
            "Low interaction cost (i) → MELV predicts cooperative equilibria."
        ),
    }

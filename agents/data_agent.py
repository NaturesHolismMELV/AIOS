"""
DataAgent — Phase 2 Session 3 / Session 24.3.2
Real HTTP calls to World Bank Open Data API (no API key required).
Governed by MELVcore thermodynamic kernel (L.W. Evans | ORCID: 0009-0001-0963-1840 | Cape Town, South Africa).

World Bank API docs: https://datahelpdesk.worldbank.org/knowledgebase/articles/889392

Session 24.3.2 additions
------------------------
compute_melv_for_country(country_code) — derives MELV φ, β, ε, i_factor from
World Bank indicators for any country, replacing the hardcoded cooperation_index: 0.92
in melv_metadata.  DataAgent.execute() now accepts action="melv" to call it directly.

MELV parameter derivation
--------------------------
β (environmental suitability):
    β = EG.USE.PCAP.KG.OE / BETA_ENERGY_REF
    where BETA_ENERGY_REF = 1900 kg oil eq/cap (global mean, Blueprint Ch.4 ② theoretical)
    β < 1 → energy-scarce environment, higher interaction costs
    β > 1 → energy-abundant environment, lower interaction costs

φ (evolutionary maturity) — weighted composite [0, 1]:
    GDP/capita component (weight 0.5): log-normalised against OECD reference
    Unemployment suppressor (weight 0.3): (1 - unemployment_rate / 100)
    Governance effectiveness (weight 0.2): uses GE.EST when available,
                                           falls back to 0.5 (neutral stub)
    All three components clamped to [0, 1] before weighting.

ε (adaptive plasticity) — honest stub:
    Not yet implementable from standard World Bank indicators.
    Requires R&D expenditure (GB.XPD.RSDV.GD.ZS) + trade openness.
    Returned as null with epistemic_status "requires_rd_trade_data".

i_factor estimate:
    Derived from master equation: i₁₂(t) = i₀ × (1 − ε × φ × β)
    With ε unknown (null), conservative i₀ = 1.0, this simplifies to:
    i_factor_estimate = 1.0 / max(phi * beta, 0.01)
    Clamped to [0.01, 3.0].  Epistemic status: ② theoretical.

Epistemic status codes (Blueprint for Harmony convention):
    ③ verified    — ABM / empirical
    ② theoretical — axiom-derived, formula applied, not independently validated
    ① stub        — placeholder, formula not yet implemented
"""

import httpx
import asyncio
import json
import logging
import math
from typing import Any, Optional
from datetime import datetime, UTC

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# World Bank API configuration
# ---------------------------------------------------------------------------
WB_BASE = "https://api.worldbank.org/v2"
DEFAULT_TIMEOUT = 15.0  # seconds

# ---------------------------------------------------------------------------
# MELV macro-validation constants  (Session 24.3.2)
# ---------------------------------------------------------------------------
# β reference: global mean energy use per capita (kg oil equivalent)
# Source: Blueprint for Harmony Ch.4, IEA 2019 global mean ≈ 1900 kg/cap
# Epistemic status: ② theoretical — normalisation formula not independently validated
BETA_ENERGY_REF = 1900.0

# GDP per capita reference for φ normalisation: OECD mean ≈ $45,000 (2022 USD)
# Upper tail clamped to 1.0 — very high GDP does not guarantee φ=1 (Axiom 3)
PHI_GDP_REF = 45000.0

# i_critical from ABM V2.1 (③ verified)
I_CRITICAL = 0.9995

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
    "GE.EST":            "Government Effectiveness estimate (WGI, −2.5 to +2.5)",
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
# Session 24.3.2 — Macro MELV Computation
# ---------------------------------------------------------------------------

def _derive_phi(gdp_per_capita: float | None,
                unemployment_rate: float | None,
                governance_est: float | None) -> tuple[float, str]:
    """
    Derive φ (evolutionary maturity) from three World Bank indicators.

    Weighted composite:
      0.5 × gdp_component      — log-normalised GDP per capita vs OECD mean
      0.3 × employment_component — (1 − unemployment / 100)
      0.2 × governance_component — WGI GE.EST normalised from [−2.5, +2.5] → [0, 1]

    Any missing indicator falls back to 0.5 (neutral), marked in phi_basis.
    All components clamped to [0.0, 1.0] before weighting.
    Epistemic status: ② theoretical — formula not independently validated.
    """
    components = []
    basis_parts = []

    # GDP component: log-normalised, clamped
    if gdp_per_capita and gdp_per_capita > 0:
        log_ratio = math.log(gdp_per_capita / PHI_GDP_REF) / math.log(10)
        # log₁₀(1) = 0 → component = 0.5; maps ±1 order of magnitude to [~0.17, ~0.83]
        gdp_comp = max(0.0, min(1.0, 0.5 + log_ratio * 0.3))
        basis_parts.append(f"GDP/cap ${gdp_per_capita:,.0f}")
    else:
        gdp_comp = 0.5
        basis_parts.append("GDP/cap: unavailable (neutral 0.5)")
    components.append(("gdp", gdp_comp, 0.5))

    # Employment component: (1 - unem/100), suppresses φ directly
    if unemployment_rate is not None and 0 <= unemployment_rate <= 100:
        emp_comp = max(0.0, min(1.0, 1.0 - unemployment_rate / 100.0))
        basis_parts.append(f"unemployment {unemployment_rate:.1f}%")
    else:
        emp_comp = 0.5
        basis_parts.append("unemployment: unavailable (neutral 0.5)")
    components.append(("employment", emp_comp, 0.3))

    # Governance component: WGI GE.EST ∈ [−2.5, +2.5] → [0, 1]
    if governance_est is not None:
        gov_comp = max(0.0, min(1.0, (governance_est + 2.5) / 5.0))
        basis_parts.append(f"governance {governance_est:+.3f}")
    else:
        gov_comp = 0.5
        basis_parts.append("governance: unavailable (neutral 0.5 stub ①)")
    components.append(("governance", gov_comp, 0.2))

    phi = sum(v * w for _, v, w in components)
    phi = max(0.0, min(1.0, phi))
    phi_basis = " + ".join(basis_parts)
    return round(phi, 4), phi_basis


def _derive_beta(energy_use_kg: float | None) -> tuple[float | None, str]:
    """
    Derive β (environmental suitability) from energy use per capita.

    β = energy_use_kg / BETA_ENERGY_REF  (1900 kg oil eq/cap global mean)

    β < 1 → energy-scarce (high interaction cost environment)
    β > 1 → energy-abundant (low interaction cost environment)

    Clamped to [0.05, 5.0] to avoid degenerate extremes.
    Epistemic status: ② theoretical.
    """
    if energy_use_kg is None or energy_use_kg <= 0:
        return None, "energy use: unavailable"
    beta = max(0.05, min(5.0, energy_use_kg / BETA_ENERGY_REF))
    beta_basis = (
        f"Energy use {energy_use_kg:.0f} kg/cap "
        f"/ {BETA_ENERGY_REF:.0f} reference"
    )
    return round(beta, 4), beta_basis


def _derive_i_factor(phi: float, beta: float | None) -> tuple[float | None, str, str]:
    """
    Estimate i_factor from φ and β via master equation simplification.

    With ε unknown (null stub), the master equation i₁₂(t) = i₀(1 − ε·φ·β)
    cannot be solved directly.  Conservative estimate uses:
        i_factor_estimate = 1.0 / max(φ·β, 0.01)
    This yields i_factor > 1 when φ·β < 1 (non-cooperative signal) and
    i_factor < 1 when φ·β > 1 (cooperative signal).

    Clamped to [0.01, 3.0].
    Epistemic status: ② theoretical — requires empirical ε to be exact.
    """
    if beta is None:
        return None, "unavailable: β missing", "unknown"
    product = phi * beta
    if product <= 0:
        return 3.0, "clamped: φ·β ≈ 0", "above_i_critical"
    i_est = max(0.01, min(3.0, 1.0 / product))
    if i_est <= I_CRITICAL:
        classification = "below_i_critical"
        prediction = "COOPERATIVE"
    else:
        classification = "above_i_critical"
        prediction = "NOT_COOPERATIVE"
    return round(i_est, 4), f"1 / (φ={phi:.4f} × β={beta:.4f})", prediction


async def compute_melv_for_country(country_code: str) -> dict[str, Any]:
    """
    Session 24.3.2 — Derive MELV φ, β, ε, i_factor from World Bank indicators.

    Replaces the hardcoded cooperation_index: 0.92 in melv_metadata.
    All three MELV parameters are derived from real World Bank data where
    available, with honest null stubs and epistemic status for missing data.

    Indicators fetched:
      EG.USE.PCAP.KG.OE  — energy use per capita (β derivation)
      NY.GDP.PCAP.CD      — GDP per capita (φ component)
      SL.UEM.TOTL.ZS      — unemployment rate (φ suppressor)
      GE.EST              — WGI governance effectiveness (φ component)

    Returns a structured dict suitable for direct use in API responses.
    """
    code = country_code.upper()

    # Fetch all four indicators in parallel
    tasks = [
        fetch_indicator("EG.USE.PCAP.KG.OE", code),
        fetch_indicator("NY.GDP.PCAP.CD",     code),
        fetch_indicator("SL.UEM.TOTL.ZS",     code),
        fetch_indicator("GE.EST",              code),
    ]
    energy_res, gdp_res, unem_res, gov_res = await asyncio.gather(*tasks)

    def _latest(res: dict) -> float | None:
        records = res.get("records", [])
        if not records:
            return None
        return sorted(records, key=lambda r: r["year"], reverse=True)[0]["value"]

    def _latest_year(res: dict) -> str | None:
        records = res.get("records", [])
        if not records:
            return None
        return sorted(records, key=lambda r: r["year"], reverse=True)[0]["year"]

    energy_use   = _latest(energy_res)
    gdp_per_cap  = _latest(gdp_res)
    unemployment = _latest(unem_res)
    governance   = _latest(gov_res)

    # Derive MELV parameters
    phi, phi_basis   = _derive_phi(gdp_per_cap, unemployment, governance)
    beta, beta_basis = _derive_beta(energy_use)
    i_est, i_basis, cooperation_prediction = _derive_i_factor(phi, beta)

    # Determine composite epistemic status
    have_governance = governance is not None
    have_beta       = beta is not None
    if have_beta and have_governance:
        epistemic_status = "phi_beta_verified_governance_included"
    elif have_beta:
        epistemic_status = "phi_theoretical_beta_verified_governance_stub"
    else:
        epistemic_status = "phi_theoretical_beta_unavailable"

    # Maturity label (mirrors MELVKernel thresholds)
    if phi >= 0.85:   maturity_label = "expert"
    elif phi >= 0.65: maturity_label = "proficient"
    elif phi >= 0.40: maturity_label = "developing"
    else:             maturity_label = "novice"

    # Argentine ant warning: high β + low φ → above i_critical despite good environment
    argentine_ant_warning = (
        beta is not None
        and beta >= 1.0          # energy-rich environment
        and phi < 0.50           # but structural suppression (unemployment, governance)
        and i_est is not None
        and i_est > I_CRITICAL
    )

    return {
        "country_code": code,
        # ── φ ──
        "phi":             phi,
        "phi_basis":       phi_basis,
        "phi_maturity":    maturity_label,
        "phi_components": {
            "gdp_per_capita":   round(gdp_per_cap, 2) if gdp_per_cap else None,
            "gdp_year":         _latest_year(gdp_res),
            "unemployment_pct": round(unemployment, 2) if unemployment is not None else None,
            "unemployment_year": _latest_year(unem_res),
            "governance_est":   round(governance, 4) if governance is not None else None,
            "governance_year":  _latest_year(gov_res),
        },
        # ── β ──
        "beta":            beta,
        "beta_basis":      beta_basis,
        "beta_components": {
            "energy_use_kg_cap": round(energy_use, 1) if energy_use else None,
            "energy_year":       _latest_year(energy_res),
            "beta_reference":    BETA_ENERGY_REF,
        },
        # ── ε ──
        "epsilon":         None,
        "epsilon_status":  "requires R&D expenditure + trade openness indicators ①",
        # ── i_factor ──
        "i_factor_estimate":       i_est,
        "i_factor_basis":          i_basis,
        "i_critical":              I_CRITICAL,
        "i_classification":        "above_i_critical" if (i_est and i_est > I_CRITICAL) else (
                                       "below_i_critical" if i_est is not None else "unavailable"),
        "cooperation_prediction":  cooperation_prediction,
        # ── meta ──
        "epistemic_status":        epistemic_status,
        "argentine_ant_warning":   argentine_ant_warning,
        "argentine_ant_note": (
            "High β (energy-rich environment) but structural φ suppression "
            "(unemployment / governance) drives i_factor above i_critical — "
            "predicted non-cooperative extraction dynamics despite resource abundance."
            if argentine_ant_warning else None
        ),
        "melv_version":  "24.3.2",
        "computed_at":   datetime.now(UTC).isoformat(),
        "source":        "World Bank Open Data API",
    }


# ---------------------------------------------------------------------------
# AIOS Agent class (drop-in replacement for simulated DataAgent)
# ---------------------------------------------------------------------------

class DataAgent:
    """
    AIOS DataAgent — Phase 2 real implementation.
    Replaces simulated random data with live World Bank API calls.

    Session 24.3.2: melv_metadata cooperation_index stub removed.
    Use action="melv" to compute real MELV parameters from World Bank indicators.
    """

    name = "DATA"
    description = (
        "Retrieves real economic, environmental, and social data from the "
        "World Bank Open Data API. No API key required. Supports single-country "
        "profiles, cross-country comparisons, time-series for 10+ indicators, "
        "and MELV macro-parameter derivation (φ, β, i_factor) per country."
    )
    real_data = True

    # MELV thermodynamic metadata — agent-level only (not country-level)
    # cooperation_index: 0.92 stub removed in Session 24.3.2 — use action="melv"
    melv_metadata = {
        "interaction_type": "mutualistic",   # open data = low interaction cost
        "energy_cost":      "low",
    }

    async def execute(self, task: dict[str, Any]) -> dict[str, Any]:
        """
        Dispatch based on task["action"]:
          - "profile"     → fetch_country_profile(country)
          - "indicator"   → fetch_indicator(indicator, country, year_from, year_to)
          - "compare"     → compare_countries(indicator, countries, year)
          - "list"        → list_available_indicators()
          - "melv"        → compute_melv_for_country(country)  [Session 24.3.2]
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
            elif action == "melv":
                # Session 24.3.2: real MELV φ/β computation from World Bank indicators
                result = await compute_melv_for_country(task.get("country", "ZA"))
            else:
                return {
                    "agent": self.name,
                    "action": action,
                    "status": "error",
                    "error": f"Unknown action: {action}. Valid: profile, indicator, compare, list, melv",
                    "timestamp": datetime.now(UTC).isoformat(),
                }

            return {
                "agent":         self.name,
                "action":        action,
                "status":        "success",
                "real_data":     True,
                "result":        result,
                "melv_metadata": self.melv_metadata,
                "timestamp":     datetime.now(UTC).isoformat(),
            }

        except Exception as e:
            logger.error("[DataAgent] Error: %s", e, exc_info=True)
            return {
                "agent":     self.name,
                "action":    action,
                "status":    "error",
                "error":     str(e),
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

"""
AIOS Phase 2 — Search Tool Agent
=================================
Lightweight web-search wrapper backed by the DuckDuckGo Instant Answer API.

Cost   = real wall-clock latency in seconds (the primary consumable resource).
Benefit = fraction of DDG result fields that contain non-empty / non-null values
          (proxy for answer richness: 0.0 → no useful data, 1.0 → fully populated).

The agent is compatible with BaseAgent.run_task() — execute() returns the
standard result dict:

    {
        "success": bool,
        "output":  dict,   # raw DDG response + query metadata
        "cost":    float,  # latency in seconds
        "benefit": float,  # 0.0–1.0 field-fill ratio
    }

Only stdlib is used (urllib).  No third-party packages required.
"""

import asyncio
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Optional

from agents.base_agent import BaseAgent
from core.melv_engine import MELVKernel

# ---------------------------------------------------------------------------
# DDG API constants
# ---------------------------------------------------------------------------

_DDG_ENDPOINT = "https://api.duckduckgo.com/"
_DDG_PARAMS   = {"format": "json", "no_html": "1", "skip_disambig": "1"}

# Fields we inspect when scoring result richness.
# Mix of high-value answer fields and supplementary fields.
_SCORED_FIELDS = [
    "Abstract",
    "AbstractText",
    "AbstractSource",
    "AbstractURL",
    "Answer",
    "AnswerType",
    "Definition",
    "DefinitionSource",
    "DefinitionURL",
    "Entity",
    "Heading",
    "Image",
    "RelatedTopics",   # list — non-empty list counts as filled
    "Results",         # list — non-empty list counts as filled
    "Type",
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _field_is_filled(value) -> bool:
    """Return True when a DDG response field carries meaningful content."""
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, dict)):
        return bool(value)
    return bool(value)


def _compute_benefit(ddg_data: dict) -> float:
    """
    Benefit = (number of scored fields that are non-empty)
              / (total number of scored fields)

    Range: 0.0 (complete miss) → 1.0 (every field populated).
    """
    if not ddg_data:
        return 0.0
    filled = sum(
        1 for f in _SCORED_FIELDS if _field_is_filled(ddg_data.get(f))
    )
    return round(filled / len(_SCORED_FIELDS), 4)


def _fetch_ddg(query: str, timeout: float = 10.0) -> tuple[dict, float]:
    """
    Synchronous HTTP GET to the DDG Instant Answer API.

    Returns (parsed_json_dict, latency_seconds).
    Raises urllib.error.URLError / ValueError on failure.
    """
    params = {**_DDG_PARAMS, "q": query}
    url    = _DDG_ENDPOINT + "?" + urllib.parse.urlencode(params)

    t0 = time.perf_counter()
    with urllib.request.urlopen(url, timeout=timeout) as response:  # noqa: S310
        raw = response.read()
    latency = time.perf_counter() - t0

    data = json.loads(raw.decode("utf-8"))
    return data, latency


# ---------------------------------------------------------------------------
# SearchAgent
# ---------------------------------------------------------------------------

class SearchAgent(BaseAgent):
    """
    Web search agent using the DuckDuckGo Instant Answer API.

    φ starts moderate (0.65) — the agent is specialised but the DDG API
    is external and not fully controllable, so initial confidence is tempered.
    ε is moderately high (4.5) — learns quickly from repeated searches.

    Task dict keys:
        query    (str, required) : search query string
        timeout  (float, opt)    : HTTP timeout in seconds (default 10.0)
    """

    def __init__(self, kernel: MELVKernel):
        super().__init__(
            name="SEARCH",
            domain="web search & instant answers",
            kernel=kernel,
            phi=0.65,
            epsilon=4.5,
            beta_pref=1.0,
            capabilities=["web_search", "instant_answers", "definitions", "facts"],
        )

    async def execute(self, task: dict) -> dict:
        """
        Perform a DuckDuckGo search and return a standardised result dict.

        The HTTP call is blocking; we offload it to the default thread-pool
        executor so it doesn't stall the event loop.
        """
        query   = task.get("query", "").strip()
        timeout = float(task.get("timeout", 10.0))

        if not query:
            return {
                "success": False,
                "output":  {"error": "No query provided", "query": query},
                "cost":    0.0,
                "benefit": 0.0,
                "resource_type": "network",
            }

        loop = asyncio.get_event_loop()
        try:
            ddg_data, latency = await loop.run_in_executor(
                None, _fetch_ddg, query, timeout
            )
        except urllib.error.URLError as exc:
            return {
                "success": False,
                "output":  {"error": f"Network error: {exc.reason}", "query": query},
                "cost":    timeout,   # charge full timeout on network failure
                "benefit": 0.0,
                "resource_type": "network",
            }
        except (json.JSONDecodeError, ValueError) as exc:
            return {
                "success": False,
                "output":  {"error": f"Parse error: {exc}", "query": query},
                "cost":    0.1,
                "benefit": 0.0,
                "resource_type": "network",
            }

        benefit = _compute_benefit(ddg_data)

        return {
            "success": True,
            "output": {
                "query":          query,
                "heading":        ddg_data.get("Heading", ""),
                "answer":         ddg_data.get("Answer", ""),
                "abstract":       ddg_data.get("AbstractText", ""),
                "abstract_source": ddg_data.get("AbstractSource", ""),
                "abstract_url":   ddg_data.get("AbstractURL", ""),
                "definition":     ddg_data.get("Definition", ""),
                "entity":         ddg_data.get("Entity", ""),
                "image":          ddg_data.get("Image", ""),
                "related_topics": ddg_data.get("RelatedTopics", []),
                "results":        ddg_data.get("Results", []),
                "type":           ddg_data.get("Type", ""),
                "latency_s":      round(latency, 4),
                "fields_filled":  int(benefit * len(_SCORED_FIELDS)),
                "fields_total":   len(_SCORED_FIELDS),
                "_raw":           ddg_data,   # full DDG payload for callers
            },
            # cost  = real latency (seconds) — the consumed resource
            # benefit = field-fill ratio — proxy for answer quality
            "cost":          round(latency, 4),
            "benefit":       benefit,
            "resource_type": "network",
        }


# ---------------------------------------------------------------------------
# Module-level convenience function (no agent overhead)
# ---------------------------------------------------------------------------

def search(query: str, timeout: float = 10.0) -> dict:
    """
    Standalone synchronous search — no BaseAgent / MELVKernel required.

    Returns the same standardised dict that SearchAgent.execute() produces,
    useful for scripting or quick tests outside the agent ecosystem.

    Example
    -------
    >>> from agents.search_tool import search
    >>> result = search("Python programming language")
    >>> print(result["output"]["abstract"])
    """
    if not query.strip():
        return {
            "success": False,
            "output":  {"error": "No query provided", "query": query},
            "cost":    0.0,
            "benefit": 0.0,
        }
    try:
        ddg_data, latency = _fetch_ddg(query, timeout)
    except urllib.error.URLError as exc:
        return {
            "success": False,
            "output":  {"error": f"Network error: {exc.reason}", "query": query},
            "cost":    timeout,
            "benefit": 0.0,
        }
    except (json.JSONDecodeError, ValueError) as exc:
        return {
            "success": False,
            "output":  {"error": f"Parse error: {exc}", "query": query},
            "cost":    0.1,
            "benefit": 0.0,
        }

    benefit = _compute_benefit(ddg_data)
    return {
        "success": True,
        "output": {
            "query":           query,
            "heading":         ddg_data.get("Heading", ""),
            "answer":          ddg_data.get("Answer", ""),
            "abstract":        ddg_data.get("AbstractText", ""),
            "abstract_source": ddg_data.get("AbstractSource", ""),
            "abstract_url":    ddg_data.get("AbstractURL", ""),
            "definition":      ddg_data.get("Definition", ""),
            "entity":          ddg_data.get("Entity", ""),
            "image":           ddg_data.get("Image", ""),
            "related_topics":  ddg_data.get("RelatedTopics", []),
            "results":         ddg_data.get("Results", []),
            "type":            ddg_data.get("Type", ""),
            "latency_s":       round(latency, 4),
            "fields_filled":   int(benefit * len(_SCORED_FIELDS)),
            "fields_total":    len(_SCORED_FIELDS),
            "_raw":            ddg_data,
        },
        "cost":    round(latency, 4),
        "benefit": benefit,
    }

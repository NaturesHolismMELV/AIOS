"""
AIOS Phase 1 — Concrete Agent Implementations
==============================================
Demo agents showing how domain-specific agents plug into the ecosystem.
Each agent has distinct φ, ε, domain characteristics.
"""

import asyncio
import random
import time
from agents.base_agent import BaseAgent
from core.melv_engine import MELVKernel
from agents.search_tool import SearchAgent


class ResearchAgent(BaseAgent):
    """
    Researches topics using available data sources.
    High φ (domain-optimized), moderate ε.
    """
    def __init__(self, kernel: MELVKernel):
        super().__init__(
            name="RESEARCH",
            domain="knowledge retrieval & synthesis",
            kernel=kernel,
            phi=0.82,
            epsilon=3.2,
            beta_pref=1.1,
            capabilities=["web_search", "summarization", "citation"]
        )

    async def execute(self, task: dict) -> dict:
        from agents.search_tool import search
        query = task.get("query", "cooperation thermodynamics")
        depth = task.get("depth", "standard")

        # Real search — cost is actual latency, benefit is result quality
        try:
            result = await asyncio.get_event_loop().run_in_executor(
                None, search, query
            )
            cost    = result.get("latency_s", 0.5)
            benefit = result.get("benefit", 0.3)
            summary = result.get("abstract") or result.get("answer") or \
                      f"Search completed for '{query}' — {result.get('fields_filled',0)} fields"
            sources = len(result.get("related_topics", [])) or 1
            success = True
        except Exception as e:
            cost, benefit, summary, sources, success = 1.0, 0.1, str(e), 0, False

        # φ maturity still modulates benefit — experienced agent extracts more value
        benefit = min(1.0, benefit * (0.7 + self.phi * 0.3))
        if depth == "deep":
            cost *= 1.4  # deep search costs more

        return {
            "success": success,
            "output": {
                "query":   query,
                "summary": summary,
                "sources": sources,
            },
            "cost":          round(cost, 4),
            "benefit":       round(benefit, 4),
            "resource_type": "network",
        }

class AnalysisAgent(BaseAgent):
    """
    Performs data analysis and pattern recognition.
    High φ, high ε (fast learner).
    """
    def __init__(self, kernel: MELVKernel):
        super().__init__(
            name="ANALYSIS",
            domain="data analysis & pattern recognition",
            kernel=kernel,
            phi=0.78,
            epsilon=5.5,
            beta_pref=1.2,
            capabilities=["statistical_analysis", "pattern_detection", "reporting"]
        )

    async def execute(self, task: dict) -> dict:
        import anthropic
        import os

        data_type  = task.get("data_type", "agent cooperation dynamics")
        complexity = task.get("complexity", 0.5)

        prompt = (
            f"You are a data analysis agent. Analyse the following topic and identify "
            f"key patterns, relationships, and insights.\n\n"
            f"Topic: {data_type}\n"
            f"Analysis depth: {'deep' if complexity > 0.7 else 'standard'}\n\n"
            f"Respond in exactly this format:\n"
            f"PATTERNS: [2-4 key patterns, comma separated]\n"
            f"INSIGHT: [one sentence core insight]\n"
            f"CONFIDENCE: [0.0-1.0]\n"
            f"COMPLEXITY: [low/medium/high]"
        )

        start = time.perf_counter()
        try:
            client = anthropic.Anthropic(
                api_key=os.environ.get("ANTHROPIC_API_KEY")
            )
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: client.messages.create(
                    model="claude-haiku-4-5-20251001",
                    max_tokens=200,
                    messages=[{"role": "user", "content": prompt}]
                )
            )
            latency = time.perf_counter() - start
            text    = response.content[0].text
            in_tok  = response.usage.input_tokens
            out_tok = response.usage.output_tokens

            # Cost: real token cost normalised to MELV range
            # Haiku: $0.80/M input, $4.00/M output
            token_cost = (in_tok * 0.0000008 + out_tok * 0.000004)
            cost = min(2.0, token_cost * 1000 + latency * 0.1)

            # Benefit: structural quality + extracted confidence
            has_patterns   = "PATTERNS:"   in text
            has_insight    = "INSIGHT:"    in text
            has_confidence = "CONFIDENCE:" in text
            structure_score = sum([has_patterns, has_insight, has_confidence]) / 3.0

            conf = 0.7
            if has_confidence:
                try:
                    line = [l for l in text.split('\n') if 'CONFIDENCE:' in l][0]
                    conf = float(line.split(':')[1].strip())
                except Exception:
                    pass

            benefit = min(1.0, structure_score * 0.5 + conf * 0.5)

            # φ maturity modulates benefit — expert analyst extracts more value
            benefit = min(1.0, benefit * (0.7 + self.phi * 0.3))

            return {
                "success": True,
                "output": {
                    "data_type":  data_type,
                    "analysis":   text,
                    "tokens":     in_tok + out_tok,
                    "latency_s":  round(latency, 3),
                },
                "cost":          round(cost, 4),
                "benefit":       round(benefit, 4),
                "resource_type": "token_budget",
            }

        except Exception as e:
            latency = time.perf_counter() - start
            return {
                "success": False,
                "output": {"error": str(e), "data_type": data_type},
                "cost":          round(min(2.0, latency), 4),
                "benefit":       0.05,
                "resource_type": "token_budget",
            }

class WritingAgent(BaseAgent):
    """
    Content generation and documentation.
    Moderate φ, low ε (deliberate writer).
    """
    def __init__(self, kernel: MELVKernel):
        super().__init__(
            name="WRITER",
            domain="content generation & documentation",
            kernel=kernel,
            phi=0.71,
            epsilon=2.4,
            beta_pref=0.9,
            capabilities=["drafting", "editing", "summarization", "documentation"]
        )

    async def execute(self, task: dict) -> dict:
        await asyncio.sleep(random.uniform(0.2, 0.5))
        content_type = task.get("content_type", "report")
        word_count   = task.get("word_count", 500)

        quality = 0.65 + self.phi * 0.35 + random.gauss(0, 0.06)
        cost    = 0.2 + (word_count / 1000) * 0.3

        return {
            "success": True,
            "output": {
                "content_type": content_type,
                "word_count":   word_count,
                "quality":      round(min(1.0, quality), 3),
            },
            "cost":    round(cost, 3),
            "benefit": round(min(1.0, quality), 3),
        }

class CodeAgent(BaseAgent):
    """
    Software development and debugging.
    Very high φ, very high ε.
    """
    def __init__(self, kernel: MELVKernel):
        super().__init__(
            name="CODE",
            domain="software development & debugging",
            kernel=kernel,
            phi=0.91,
            epsilon=6.8,
            beta_pref=1.3,
            capabilities=["python", "javascript", "testing", "debugging", "refactoring"]
        )

    async def execute(self, task: dict) -> dict:
        await asyncio.sleep(random.uniform(0.1, 0.4))
        task_type = task.get("task_type", "implement")
        language  = task.get("language", "python")

        quality = 0.75 + self.phi * 0.25 + random.gauss(0, 0.04)
        cost    = {"debug": 0.3, "implement": 0.5, "refactor": 0.4, "test": 0.3}.get(task_type, 0.4)

        return {
            "success": True,
            "output": {
                "task_type": task_type,
                "language":  language,
                "lines":     random.randint(20, 200),
                "tests_pass": random.random() > 0.1,
                "quality":   round(min(1.0, quality), 3),
            },
            "cost":    round(cost, 3),
            "benefit": round(min(1.0, quality), 3),
            "resource_type": "compute",
        }


class MonitorAgent(BaseAgent):
    """
    System observability and health monitoring.
    Very high φ, very high ε — oldest, most mature agent.
    Low interaction cost (lightweight).
    """
    def __init__(self, kernel: MELVKernel):
        super().__init__(
            name="MONITOR",
            domain="observability & health monitoring",
            kernel=kernel,
            phi=0.95,
            epsilon=7.5,
            beta_pref=0.8,
            capabilities=["metrics", "alerting", "logging", "tracing"]
        )

    async def execute(self, task: dict) -> dict:
        await asyncio.sleep(random.uniform(0.01, 0.05))
        metric_type = task.get("metric_type", "system")

        # Monitor is very efficient — low cost, reliable output
        quality = 0.88 + self.phi * 0.12 + random.gauss(0, 0.02)
        cost    = 0.08 + random.uniform(0, 0.05)

        return {
            "success": True,
            "output": {
                "metric_type": metric_type,
                "readings":    random.randint(10, 100),
                "anomalies":   random.randint(0, 2),
                "quality":     round(min(1.0, quality), 3),
            },
            "cost":    round(cost, 3),
            "benefit": round(min(1.0, quality), 3),
        }


class PlannerAgent(BaseAgent):
    """
    Strategic planning and task decomposition.
    High φ, low ε (deliberate planner).
    """
    def __init__(self, kernel: MELVKernel):
        super().__init__(
            name="PLANNER",
            domain="strategic planning & task orchestration",
            kernel=kernel,
            phi=0.85,
            epsilon=1.8,
            beta_pref=1.0,
            capabilities=["decomposition", "scheduling", "prioritization", "coordination"]
        )

    async def execute(self, task: dict) -> dict:
        await asyncio.sleep(random.uniform(0.15, 0.35))
        goal       = task.get("goal", "")
        n_subtasks = task.get("n_subtasks", 3)

        quality = 0.72 + self.phi * 0.28 + random.gauss(0, 0.05)
        cost    = 0.25 + n_subtasks * 0.05

        return {
            "success": True,
            "output": {
                "goal":      goal,
                "subtasks":  n_subtasks,
                "plan_quality": round(min(1.0, quality), 3),
            },
            "cost":    round(cost, 3),
            "benefit": round(min(1.0, quality), 3),
        }


class DataAgent(BaseAgent):
    """
    Data retrieval, transformation, and storage.
    Moderate φ (newer agent), high ε.
    """
    def __init__(self, kernel: MELVKernel):
        super().__init__(
            name="DATA",
            domain="data retrieval & transformation",
            kernel=kernel,
            phi=0.58,
            epsilon=5.2,
            beta_pref=1.1,
            capabilities=["sql", "api_calls", "etl", "caching"]
        )

    async def execute(self, task: dict) -> dict:
        await asyncio.sleep(random.uniform(0.1, 0.3))
        source      = task.get("source", "database")
        volume      = task.get("volume", "small")

        # Lower φ → slightly higher interaction cost
        quality = 0.55 + self.phi * 0.40 + random.gauss(0, 0.07)
        cost    = {"small": 0.2, "medium": 0.45, "large": 0.8}.get(volume, 0.35)

        return {
            "success": True,
            "output": {
                "source":  source,
                "volume":  volume,
                "records": random.randint(10, 10000),
                "quality": round(min(1.0, quality), 3),
            },
            "cost":    round(cost, 3),
            "benefit": round(min(1.0, quality), 3),
            "resource_type": "api_quota",
        }


# ── AGENT FACTORY ──────────────────────────────────────────────────────────

def create_default_ecosystem(kernel: MELVKernel) -> dict:
    """
    Instantiate the default Phase 1 agent ecosystem.
    Returns dict of agent_id → agent instance.
    """
    agents = [
        ResearchAgent(kernel),
        AnalysisAgent(kernel),
        WritingAgent(kernel),
        CodeAgent(kernel),
        MonitorAgent(kernel),
        PlannerAgent(kernel),
        DataAgent(kernel),
        SearchAgent(kernel),
    ]
    return {a.agent_id: a for a in agents}

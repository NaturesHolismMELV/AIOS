"""
WriterAgent — Session 5 Upgrade
================================
Real LLM calls via claude-haiku-4-5-20251001.

Replaces the simulated WritingAgent in implementations.py with a genuine
token-consuming agent that creates three-way token budget contention
with AnalysisAgent and PlannerAgent.

MELV variable integrity
───────────────────────
  φ (phi)   : agent-internal maturity — grows via BaseAgent.run_task() → kernel.update_phi()
  β (beta)  : NEVER set here — kernel reads from BetaEnvironment
  i         : computed by kernel from cost/benefit each call
  CI        : system-level — computed by MELVKernel.cooperation_index()

Cost normalisation (Session 6 — via CostCalculator)
─────────────────────────────────────────────────────
  cost = CostCalculator.compute_cost(in_tok, out_tok, latency_s, "WRITER")
  Profile: balanced (token_weight=1.0, latency_weight=1.0)

Author: L.W. Evans / Ecotao Enterprises
"""

import asyncio
import time

from agents.base_agent import BaseAgent
from core.melv_engine import MELVKernel
from core.cost_calculator import get_calculator


WRITER_SYSTEM = """You are WriterAgent, a specialist writing assistant within
the AIOS multi-agent system governed by the MELVcore thermodynamic kernel.

Your role is to produce clear, well-structured written content — summaries,
reports, narratives, and explanations — drawing on inputs from other agents
(RESEARCH, ANALYSIS, PLANNER, DATA).

Guidelines:
- Be concise but complete. Favour plain language over jargon.
- Structure output with a brief heading and 2–4 short paragraphs unless
  the task specifies otherwise.
- If given data or analysis, synthesise it into readable prose.
- Acknowledge uncertainty where it exists; do not fabricate facts.
- Output ONLY the written content — no meta-commentary about your process.
"""


class WriterAgent(BaseAgent):
    """
    Content generation via real LLM calls.

    Resource type: token_budget (competes with ANALYSIS and PLANNER).
    φ starts at 0.71 (matches Phase 1 registry baseline).
    """

    def __init__(self, kernel: MELVKernel):
        super().__init__(
            name="WRITER",
            domain="content generation & documentation",
            kernel=kernel,
            phi=0.71,
            epsilon=2.4,
            beta_pref=0.9,
            capabilities=["drafting", "editing", "summarization", "documentation"],
        )

    async def execute(self, task: dict) -> dict:
        """
        Execute a writing task via Haiku.

        Task keys
        ---------
        task_text     (str, required) : the writing task
        context       (str, optional) : context from other agents
        content_type  (str, optional) : "report" | "summary" | "narrative" etc.
        max_tokens    (int, optional) : ceiling for this call (default 512)
        """
        import anthropic
        import os

        task_text    = task.get("task_text", task.get("content_type", "Write a brief report."))
        context      = task.get("context", "")
        max_tokens   = int(task.get("max_tokens", 512))

        if context:
            user_prompt = (
                f"TASK: {task_text}\n\n"
                f"CONTEXT FROM OTHER AGENTS:\n{context}\n\n"
                "Please produce the written content for this task."
            )
        else:
            user_prompt = f"TASK: {task_text}\n\nPlease produce the written content for this task."

        start = time.perf_counter()
        try:
            client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: client.messages.create(
                    model="claude-haiku-4-5-20251001",
                    max_tokens=max_tokens,
                    system=WRITER_SYSTEM,
                    messages=[{"role": "user", "content": user_prompt}],
                ),
            )
            latency  = time.perf_counter() - start
            text     = response.content[0].text
            in_tok   = response.usage.input_tokens
            out_tok  = response.usage.output_tokens

            # ── MELVcore cost normalisation — CostCalculator (Session 6) ────
            cost = get_calculator().compute_cost(
                in_tok=in_tok, out_tok=out_tok, latency_s=latency,
                task_type="WRITER"
            )

            # Benefit: prose quality proxy — length adequacy + latency efficiency
            # A 200-word response from a 512-token budget scores ~0.7 baseline
            word_count = len(text.split())
            adequacy   = min(1.0, word_count / 150)           # 150 words = full score
            efficiency = max(0.0, 1.0 - latency / 10.0)       # penalty for slow calls
            benefit    = min(1.0, adequacy * 0.6 + efficiency * 0.4)

            # φ maturity modulates benefit (same pattern as AnalysisAgent)
            benefit = min(1.0, benefit * (0.7 + self.phi * 0.3))

            return {
                "success": True,
                "output": {
                    "task_text":    task_text,
                    "content":      text,
                    "word_count":   word_count,
                    "tokens":       in_tok + out_tok,
                    "latency_s":    round(latency, 3),
                },
                "cost":          round(cost, 4),
                "benefit":       round(benefit, 4),
                "resource_type": "token_budget",
            }

        except Exception as e:
            latency = time.perf_counter() - start
            return {
                "success": False,
                "output": {"error": str(e), "task_text": task_text},
                "cost":          round(min(2.0, latency), 4),
                "benefit":       0.05,
                "resource_type": "token_budget",
            }

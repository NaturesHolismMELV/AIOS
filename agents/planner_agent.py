"""
PlannerAgent — Session 5 Upgrade
==================================
Real LLM calls via claude-haiku-4-5-20251001.

Replaces the simulated PlannerAgent in implementations.py with a genuine
token-consuming agent. Completes the three-way LLM token budget competition:

    ANALYSIS  (φ=0.78, token_budget, real LLM)   ← Session 2
    WRITER    (φ=0.71, token_budget, real LLM)   ← Session 5
    PLANNER   (φ=0.85, token_budget, real LLM)   ← Session 5 ← you are here

This triangle gives the MELVcore kernel its first genuine 3-way resource
contention test — the bifurcation threshold can now be hit by real usage.

MELV variable integrity
───────────────────────
  φ (phi)   : agent-internal — grows via kernel.update_phi() in BaseAgent.run_task()
  β (beta)  : NEVER set here — owned by BetaEnvironment
  i         : kernel computes from cost/benefit
  CI        : kernel.cooperation_index()

Cost normalisation (Session 6 — via CostCalculator)
─────────────────────────────────────────────────────
  cost = CostCalculator.compute_cost(in_tok, out_tok, latency_s, "PLANNER")
  Profile: token_heavy (token_weight=1.4, latency_weight=0.6)

Author: L.W. Evans / Ecotao Enterprises
"""

import asyncio
import json
import re
import time

from agents.base_agent import BaseAgent
from core.melv_engine import MELVKernel
from core.cost_calculator import get_calculator


PLANNER_SYSTEM = """You are PlannerAgent, the task-decomposition and orchestration
specialist within the AIOS multi-agent system governed by the MELVcore kernel.

Your role is to receive a high-level goal and decompose it into a concrete, ordered
list of sub-tasks that other agents can execute.

Available agents: RESEARCH, ANALYSIS, DATA, WRITER, CODE, SEARCH, MONITOR

Output format — always return a valid JSON object with NO markdown fences:
{
  "goal": "<restated goal>",
  "steps": [
    {"step": 1, "agent": "<AGENT>", "task": "<what to do>", "depends_on": []},
    {"step": 2, "agent": "<AGENT>", "task": "<what to do>", "depends_on": [1]}
  ],
  "estimated_complexity": "low | medium | high",
  "notes": "<any caveats or assumptions>"
}

Rules:
- Assign each step to exactly one agent from the list above.
- List dependencies accurately — a step can only depend on earlier steps.
- Keep tasks atomic — one clear action per step.
- Output ONLY the JSON object — no prose, no markdown fences, no explanation.
"""


class PlannerAgent(BaseAgent):
    """
    Strategic planning and task decomposition via real LLM calls.

    Resource type: token_budget (competes with ANALYSIS and WRITER).
    φ starts at 0.85 (matches Phase 1 registry baseline).
    """

    def __init__(self, kernel: MELVKernel):
        super().__init__(
            name="PLANNER",
            domain="strategic planning & task orchestration",
            kernel=kernel,
            phi=0.85,
            epsilon=1.8,
            beta_pref=1.0,
            capabilities=["decomposition", "scheduling", "prioritization", "coordination"],
        )

    async def execute(self, task: dict) -> dict:
        """
        Decompose a goal into a structured execution plan via Haiku.

        Task keys
        ---------
        goal          (str, required) : high-level objective to decompose
        constraints   (str, optional) : budget, timing, preferred agents
        max_tokens    (int, optional) : ceiling for this call (default 400)
        """
        import anthropic
        import os

        goal        = task.get("goal", task.get("task_text", "Plan a research project."))
        constraints = task.get("constraints", "")
        max_tokens  = int(task.get("max_tokens", 400))

        if constraints:
            user_prompt = (
                f"GOAL: {goal}\n\n"
                f"CONSTRAINTS / CONTEXT:\n{constraints}\n\n"
                "Decompose this goal into an ordered execution plan."
            )
        else:
            user_prompt = f"GOAL: {goal}\n\nDecompose this goal into an ordered execution plan."

        start = time.perf_counter()
        try:
            client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: client.messages.create(
                    model="claude-haiku-4-5-20251001",
                    max_tokens=max_tokens,
                    system=PLANNER_SYSTEM,
                    messages=[{"role": "user", "content": user_prompt}],
                ),
            )
            latency = time.perf_counter() - start
            raw     = response.content[0].text
            in_tok  = response.usage.input_tokens
            out_tok = response.usage.output_tokens

            # ── MELVcore cost normalisation — CostCalculator (Session 6) ────
            cost = get_calculator().compute_cost(
                in_tok=in_tok, out_tok=out_tok, latency_s=latency,
                task_type="PLANNER"
            )

            # Parse the JSON plan
            plan = self._parse_plan(raw)
            n_steps = len(plan.get("steps", [])) if plan else 0

            # Benefit: plan quality proxy
            # Valid JSON with ≥2 steps = good plan; each step adds value
            if plan and n_steps >= 2:
                structure_score = min(1.0, n_steps / 5.0)   # 5 steps = full score
                benefit = min(1.0, 0.6 + structure_score * 0.4)
            elif plan:
                benefit = 0.5
            else:
                benefit = 0.1   # JSON parse failed — low quality

            # φ maturity modulates benefit
            benefit = min(1.0, benefit * (0.7 + self.phi * 0.3))

            return {
                "success": True,
                "output": {
                    "goal":        goal,
                    "plan":        plan,
                    "raw_output":  raw,
                    "n_steps":     n_steps,
                    "tokens":      in_tok + out_tok,
                    "latency_s":   round(latency, 3),
                },
                "cost":          round(cost, 4),
                "benefit":       round(benefit, 4),
                "resource_type": "token_budget",
            }

        except Exception as e:
            latency = time.perf_counter() - start
            return {
                "success": False,
                "output": {"error": str(e), "goal": goal},
                "cost":          round(min(2.0, latency), 4),
                "benefit":       0.05,
                "resource_type": "token_budget",
            }

    # ── helpers ──────────────────────────────────────────────────────────────

    def _parse_plan(self, raw: str) -> dict | None:
        """Strip any accidental markdown fences and parse JSON."""
        clean = re.sub(r"```(?:json)?|```", "", raw).strip()
        try:
            return json.loads(clean)
        except json.JSONDecodeError:
            # Try to extract first {...} block
            match = re.search(r"\{.*\}", clean, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group())
                except json.JSONDecodeError:
                    pass
        return None

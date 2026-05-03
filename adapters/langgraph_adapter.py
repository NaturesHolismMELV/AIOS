"""
adapters/langgraph_adapter.py — MELVcore × LangGraph Governance Bridge
=======================================================================
Session 13 · v1.5.0

Wraps any LangGraph node (Python callable) with MELVcore thermodynamic
governance: interaction cost/benefit reporting, CI monitoring, and nudge
response — without requiring any changes to the underlying node logic.

Quick start
-----------
    from adapters.langgraph_adapter import MELVNode, MELVGraph
    from core.melv_engine import MELVKernel, AgentProfile

    kernel = MELVKernel()

    # Wrap your existing LangGraph node functions
    retriever = MELVNode("retriever", "retrieval", kernel, my_retriever_fn)
    generator = MELVNode("generator", "generation", kernel, my_generator_fn)

    # Build a governed graph
    graph = MELVGraph(kernel)
    graph.add_node(retriever)
    graph.add_node(generator)
    graph.add_edge("retriever", "generator")

    # Run — MELVcore governance fires transparently on every invocation
    result = graph.invoke({"query": "What is cooperation?"})
    print(kernel.cooperation_index())   # CI after the run

Design principles
-----------------
- Zero rewrites: wrap any existing callable. The node function's
  signature and return value are untouched.
- Cost/benefit estimation: by default, cost = normalised token usage
  (estimated from state dict size), benefit = 1.0 - error_rate.
  Override by subclassing MELVNode and implementing _estimate_cb().
- φ update: after each successful invocation the node's φ is nudged
  upward proportional to outcome quality. Failed invocations nudge φ
  downward.
- Nudge passthrough: if the kernel returns a nudge recommendation
  (bifurcation event), it is logged to the node's nudge_log. The node
  continues regardless — nudges are advisory in the adapter layer.
- No LangGraph dependency at import time: the adapter works as a
  standalone wrapper even without LangGraph installed. When LangGraph
  is present, MELVGraph provides a thin governed StateGraph wrapper.

MELV invariants preserved
--------------------------
- β is NEVER set by an adapter node. It is read from BetaEnvironment.
- φ is updated only via kernel.update_phi() — never directly.
- Cost cap: raw cost is clamped to [0.0, 2.0] before reporting.
"""

from __future__ import annotations

import asyncio
import logging
import sys
import time
from typing import Any, Callable, Optional

logger = logging.getLogger("aios.adapters.langgraph")

# ── OPTIONAL LANGGRAPH IMPORT ─────────────────────────────────────────────

try:
    from langgraph.graph import StateGraph, END          # type: ignore
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
    logger.debug("LangGraph not installed — MELVGraph will raise on use. "
                 "MELVNode works standalone.")


# ── MELV NODE ─────────────────────────────────────────────────────────────

class MELVNode:
    """
    A LangGraph node wrapped with MELVcore governance.

    Parameters
    ----------
    agent_id : str
        Unique agent identifier registered with the MELVKernel.
    domain : str
        Agent domain (e.g. "retrieval", "generation", "planning").
    kernel : MELVKernel
        Live kernel instance. The node registers itself on first call
        if not already present.
    fn : Callable[[dict], dict]
        The underlying node function. Receives state dict, returns
        updated state dict (standard LangGraph convention).
    phi : float
        Initial evolutionary maturity [0.0–1.0]. Default 0.5.
    epsilon : float
        Adaptive plasticity [0.0–8.0]. Default 3.0.
    resource_type : str
        Primary resource this node consumes (used for β lookup).
        One of: compute, api_quota, vector_db, storage,
                token_budget, context_window.
    name : str | None
        Human-readable name. Defaults to agent_id.upper().
    read_keys : set[str] | None
        State dict keys this node reads. Used by MELVGraph for shared-state
        inference. Default None (unspecified).
    write_keys : set[str] | None
        State dict keys this node writes. If any key appears in write_keys
        for more than one node, MELVGraph infers shared_state='read_write'.
        Default None (unspecified).
    """

    def __init__(
        self,
        agent_id:      str,
        domain:        str,
        kernel,                         # MELVKernel — no type hint to avoid circular
        fn:            Callable[[dict], dict],
        *,
        phi:           float = 0.5,
        epsilon:       float = 3.0,
        resource_type: str   = "compute",
        name:          Optional[str] = None,
        read_keys:     Optional[set] = None,
        write_keys:    Optional[set] = None,
    ):
        self.agent_id      = agent_id
        self.domain        = domain
        self.kernel        = kernel
        self.fn            = fn
        self.phi           = phi
        self.epsilon       = epsilon
        self.resource_type = resource_type
        self.name          = name or agent_id.upper()
        self.read_keys     = set(read_keys) if read_keys else None
        self.write_keys    = set(write_keys) if write_keys else None

        # Runtime counters
        self.call_count:  int   = 0
        self.error_count: int   = 0
        self.total_cost:  float = 0.0
        self.nudge_log:   list  = []   # bifurcation events received

        self._registered = False

    # ── REGISTRATION ──────────────────────────────────────────────────────

    def _ensure_registered(self):
        """Lazy-register this node's profile with the kernel."""
        if self._registered:
            return
        if self.agent_id in self.kernel.agents:
            self._registered = True
            return

        from core.melv_engine import AgentProfile, AgentStatus
        profile = AgentProfile(
            agent_id    = self.agent_id,
            name        = self.name,
            domain      = self.domain,
            phi         = self.phi,
            epsilon     = self.epsilon,
            beta_pref   = 1.0,
            status      = AgentStatus.MATURING,
        )
        self.kernel.register_agent(profile)
        self._registered = True
        logger.info("MELVNode registered: %s (domain=%s φ=%.2f ε=%.1f)",
                    self.agent_id, self.domain, self.phi, self.epsilon)

    # ── COST / BENEFIT ESTIMATION ─────────────────────────────────────────

    def _estimate_cb(self, state: dict, result: dict,
                     elapsed: float, error: bool) -> tuple[float, float]:
        """
        Estimate (cost, benefit) for a node invocation.

        Default heuristic:
          cost    = clamp(elapsed_seconds * 0.3 + state_complexity * 0.1, 0, 2)
          benefit = 1.0 if no error, 0.2 if error
                    × result_richness_factor

        Override this method in a subclass for domain-specific metrics.
        For example, a generation node could use actual token counts.
        """
        # State complexity proxy: number of keys × average value length
        try:
            complexity = sum(
                len(str(v)) for v in state.values()
            ) / max(1, len(state)) / 1000.0   # normalise to ~[0, 1]
        except Exception:
            complexity = 0.1

        cost = min(2.0, elapsed * 0.3 + complexity * 0.1)

        if error:
            benefit = 0.2
        else:
            # Result richness: more output keys = richer interaction
            try:
                richness = min(1.0, len(result) / 5.0)
            except Exception:
                richness = 0.5
            benefit = 0.6 + richness * 0.4   # range [0.6, 1.0]

        return cost, benefit

    # ── INVOCATION ────────────────────────────────────────────────────────

    def __call__(self, state: dict) -> dict:
        """
        Invoke the wrapped function with MELVcore governance.

        1. Lazy-register with kernel if needed.
        2. Record start time.
        3. Call the underlying fn(state).
        4. Estimate cost/benefit.
        5. Report interaction to kernel (triggers CI update + potential nudge).
        6. Update φ based on outcome quality.
        7. Log any nudge/bifurcation event.
        8. Return the result.

        Exceptions in fn() are caught, cost/benefit penalised, then re-raised.
        """
        self._ensure_registered()

        # Find a "peer" agent to record the interaction against.
        # Use the first other registered agent, or a synthetic "environment" stub.
        peers = [aid for aid in self.kernel.agents if aid != self.agent_id]
        peer_id = peers[0] if peers else "__environment__"

        # Ensure environment stub exists
        if peer_id == "__environment__" and peer_id not in self.kernel.agents:
            from core.melv_engine import AgentProfile, AgentStatus
            self.kernel.register_agent(AgentProfile(
                agent_id="__environment__",
                name="ENVIRONMENT",
                domain="system",
                phi=0.9,
                epsilon=1.0,
                status=AgentStatus.ACTIVE,
            ))

        t0    = time.perf_counter()
        error = False
        result: dict = {}

        try:
            result = self.fn(state)
            if result is None:
                result = {}
            self.call_count += 1
        except Exception as exc:
            error = True
            self.error_count += 1
            logger.warning("MELVNode %s raised: %s", self.agent_id, exc)
            # Still report the failed interaction before re-raising
            elapsed = time.perf_counter() - t0
            cost, benefit = self._estimate_cb(state, {}, elapsed, error=True)
            cost    = min(2.0, max(0.0, cost))
            benefit = max(0.01, benefit)
            self._report_interaction(peer_id, cost, benefit)
            # Penalise φ
            self.kernel.update_phi(self.agent_id, 0.1)
            raise

        elapsed = time.perf_counter() - t0
        cost, benefit = self._estimate_cb(state, result, elapsed, error=False)
        cost    = min(2.0, max(0.0, cost))
        benefit = max(0.01, benefit)

        record = self._report_interaction(peer_id, cost, benefit)

        # Update φ — quality = benefit/cost ratio clamped to [0,1]
        quality = min(1.0, benefit / max(cost, 0.01))
        self.kernel.update_phi(self.agent_id, quality)
        self.total_cost += cost

        # Check for nudge events
        self._check_nudges()

        logger.debug(
            "MELVNode %s: cost=%.3f benefit=%.3f i=%.3f CI=%.4f",
            self.agent_id, cost, benefit,
            record.i_factor if record else 0.0,
            self.kernel.cooperation_index(),
        )

        return result

    def _report_interaction(self, peer_id: str, cost: float, benefit: float):
        """Record interaction with the kernel and return the record."""
        try:
            return self.kernel.record_interaction(
                agent_a=self.agent_id,
                agent_b=peer_id,
                cost=cost,
                benefit=benefit,
                resource_type=self.resource_type,
            )
        except Exception as e:
            logger.warning("MELVNode._report_interaction failed: %s", e)
            return None

    def _check_nudges(self):
        """Log the latest bifurcation event if it involves this node."""
        if not self.kernel.events:
            return
        latest = self.kernel.events[-1]
        if (latest.agent_a == self.agent_id or
                latest.agent_b == self.agent_id):
            if not self.nudge_log or self.nudge_log[-1] != latest.event_id:
                self.nudge_log.append(latest.event_id)
                logger.info(
                    "MELVNode %s received nudge: %s (%s)",
                    self.agent_id, latest.event_id, latest.action.value
                )

    # ── STATUS ────────────────────────────────────────────────────────────

    def status(self) -> dict:
        """Return a snapshot of this node's MELV metrics."""
        profile = self.kernel.agents.get(self.agent_id)
        return {
            "agent_id":      self.agent_id,
            "domain":        self.domain,
            "call_count":    self.call_count,
            "error_count":   self.error_count,
            "error_rate":    self.error_count / max(1, self.call_count),
            "total_cost":    round(self.total_cost, 4),
            "phi":           round(profile.phi, 4) if profile else self.phi,
            "status":        profile.status.value if profile else "unregistered",
            "nudge_count":   len(self.nudge_log),
            "ci_current":    round(self.kernel.cooperation_index(), 4),
        }

    def __repr__(self):
        return (f"MELVNode(agent_id={self.agent_id!r}, domain={self.domain!r}, "
                f"calls={self.call_count}, ci={self.kernel.cooperation_index():.4f})")


# ── MELV GRAPH ────────────────────────────────────────────────────────────

class MELVGraph:
    """
    A thin governed wrapper around LangGraph's StateGraph.

    Adds MELVcore CI monitoring to any LangGraph workflow.
    Requires LangGraph to be installed (`pip install langgraph`).

    Usage
    -----
        kernel = MELVKernel()
        graph  = MELVGraph(kernel, state_schema=MyState)

        # Add MELVNode-wrapped nodes
        graph.add_node(MELVNode("retriever", "retrieval", kernel, my_fn))
        graph.add_node(MELVNode("generator", "generation", kernel, my_fn))

        graph.set_entry_point("retriever")
        graph.add_edge("retriever", "generator")
        graph.add_edge("generator", END)

        runnable = graph.compile()
        result   = runnable.invoke({"input": "hello"})

        # After run
        print(f"CI after run: {kernel.cooperation_index():.4f}")
        print(f"Cooperative: {kernel.cooperation_index() >= 0.75}")
    """

    def __init__(self, kernel, state_schema: type = dict):
        if not LANGGRAPH_AVAILABLE:
            raise ImportError(
                "LangGraph is not installed. "
                "Install it with: pip install langgraph\n"
                "MELVNode works standalone without LangGraph."
            )
        self.kernel       = kernel
        self._nodes:  dict[str, MELVNode] = {}
        self._graph       = StateGraph(state_schema)
        self._compiled    = None

    def add_node(self, node: MELVNode) -> "MELVGraph":
        """Register a MELVNode with the graph."""
        self._nodes[node.agent_id] = node
        self._graph.add_node(node.agent_id, node)
        return self

    def add_edge(self, from_node: str, to_node: str) -> "MELVGraph":
        """Add a directed edge between two nodes."""
        self._graph.add_edge(from_node, to_node)
        return self

    def add_conditional_edges(self, source: str, path: Callable,
                               path_map: dict) -> "MELVGraph":
        """Add conditional routing (pass-through to StateGraph)."""
        self._graph.add_conditional_edges(source, path, path_map)
        return self

    def set_entry_point(self, node_id: str) -> "MELVGraph":
        """Set the graph entry point."""
        self._graph.set_entry_point(node_id)
        return self

    def compile(self, **kwargs):
        """Compile the StateGraph and return the runnable."""
        self._compiled = self._graph.compile(**kwargs)
        return self._compiled

    def to_sandbox_payload(
        self,
        agent_id: Optional[str] = None,
        agent_name: Optional[str] = None,
        domain: str = "workflow",
        domain_profile: Optional[str] = None,
    ) -> dict:
        """
        Inspect the MELVGraph and return a SandboxSubmitRequest-compatible dict
        representing the whole workflow as a single certifiable unit.

        Topology: cyclic graph -> operation_mode='continuous'; DAG -> 'episodic'.
        tool_count: sum across nodes.
        epsilon/phi: weighted mean by call_count.
        shared_state: 'read_write' if >=2 nodes declare overlapping write_keys.
        """
        import uuid

        nodes = list(self._nodes.values())
        if not nodes:
            raise ValueError(
                "MELVGraph has no nodes — add nodes before calling to_sandbox_payload()"
            )

        # 1. Infer operation_mode from graph topology
        try:
            import networkx as nx  # type: ignore
            nx_graph = nx.DiGraph()
            try:
                edges = list(self._graph.edges)
            except AttributeError:
                edges = []
            nx_graph.add_edges_from(edges)
            operation_mode = (
                "episodic" if nx.is_directed_acyclic_graph(nx_graph) else "continuous"
            )
        except Exception:
            operation_mode = "episodic"

        # 2. Aggregate tool_count
        total_tool_count = 0
        for node in nodes:
            profile = self.kernel.agents.get(node.agent_id)
            tc = getattr(profile, "tool_count", None) or getattr(node, "tool_count", 0)
            total_tool_count += int(tc)

        # 3 & 4. Weighted mean epsilon and phi by call_count
        total_calls = sum(max(n.call_count, 1) for n in nodes)
        weighted_epsilon = sum(
            n.epsilon * max(n.call_count, 1) for n in nodes
        ) / total_calls
        weighted_phi = sum(
            (
                self.kernel.agents[n.agent_id].phi
                if n.agent_id in self.kernel.agents
                else n.phi
            )
            * max(n.call_count, 1)
            for n in nodes
        ) / total_calls

        # 5. Infer shared_state from write_keys overlap
        shared_state = "none"
        all_write_keys: dict = {}
        for node in nodes:
            if node.write_keys:
                for k in node.write_keys:
                    all_write_keys.setdefault(k, []).append(node.agent_id)
        if any(len(owners) > 1 for owners in all_write_keys.values()):
            shared_state = "read_write"

        # 6. Build payload
        wf_id = agent_id or f"workflow-{uuid.uuid4().hex[:8]}"
        wf_name = agent_name or f"Workflow ({len(nodes)} nodes)"

        payload: dict = {
            "agent_id": wf_id,
            "agent_name": wf_name,
            "domain": domain,
            "phi": round(weighted_phi, 4),
            "epsilon": round(weighted_epsilon, 4),
            "beta_pref": 1.0,
            "tool_count": total_tool_count,
            "operation_mode": operation_mode,
            "shared_state": shared_state,
        }
        if domain_profile:
            payload["domain_profile"] = domain_profile

        # Workflow metadata (non-standard — ignored by pydantic, used by UI)
        payload["_workflow_meta"] = {
            "node_count": len(nodes),
            "node_ids": [n.agent_id for n in nodes],
            "topology": operation_mode,
            "shared_write_keys": [k for k, v in all_write_keys.items() if len(v) > 1],
        }
        return payload

    async def certify_workflow(
        self,
        base_url: str = "http://localhost:8000",
        poll_interval: float = 1.0,
        timeout: float = 120.0,
        **payload_kwargs,
    ) -> dict:
        """
        Convenience coroutine: build the sandbox payload, POST to /sandbox/submit,
        poll until certification is complete, and return the CertificationReport dict.

        Requires the AIOS server to be running at base_url.
        Requires httpx: pip install httpx
        """
        try:
            import httpx  # type: ignore
        except ImportError:
            raise ImportError(
                "httpx is required for certify_workflow(). "
                "Install with: pip install httpx"
            )

        payload = self.to_sandbox_payload(**payload_kwargs)
        post_body = {k: v for k, v in payload.items() if not k.startswith("_")}

        async with httpx.AsyncClient(base_url=base_url, timeout=30.0) as client:
            resp = await client.post("/sandbox/submit", json=post_body)
            resp.raise_for_status()
            run = resp.json()
            run_id = run.get("run_id")
            if not run_id:
                raise ValueError(f"No run_id in /sandbox/submit response: {run}")

            deadline = time.monotonic() + timeout
            while True:
                if time.monotonic() > deadline:
                    raise TimeoutError(
                        f"certify_workflow timed out after {timeout}s (run_id={run_id})"
                    )
                await asyncio.sleep(poll_interval)
                status_resp = await client.get(f"/sandbox/certify/{run_id}")
                status_resp.raise_for_status()
                report = status_resp.json()
                status_val = report.get("status", "")
                if status_val in ("certified", "complete", "error") or report.get("verdict"):
                    return report

    def ci_report(self) -> dict:
        """Return a CI summary across all governed nodes."""
        return {
            "cooperation_index": round(self.kernel.cooperation_index(), 4),
            "target":            0.75,
            "healthy":           self.kernel.cooperation_index() >= 0.75,
            "nodes": {
                node_id: node.status()
                for node_id, node in self._nodes.items()
            },
            "bifurcation_events": len(self.kernel.events),
        }

    def __repr__(self):
        nodes = list(self._nodes.keys())
        ci    = self.kernel.cooperation_index()
        return f"MELVGraph(nodes={nodes}, ci={ci:.4f})"


# ── CONVENIENCE DECORATOR ─────────────────────────────────────────────────

def melv_node(agent_id: str, domain: str, kernel,
              resource_type: str = "compute",
              phi: float = 0.5, epsilon: float = 3.0):
    """
    Decorator to wrap a LangGraph node function with MELVcore governance.

    Usage
    -----
        @melv_node("retriever", "retrieval", kernel, resource_type="vector_db")
        def retrieve(state: dict) -> dict:
            ...
            return {"documents": docs}

        # retrieve is now a MELVNode — call it directly or add to MELVGraph
    """
    def decorator(fn: Callable) -> MELVNode:
        return MELVNode(
            agent_id=agent_id,
            domain=domain,
            kernel=kernel,
            fn=fn,
            phi=phi,
            epsilon=epsilon,
            resource_type=resource_type,
        )
    return decorator


# ── SESSION 33 — OBSERVATION PAYLOAD BUILDER ─────────────────────────────
# Extends MELVNode with observe() primitive signal collection.

class LangGraphObservationBuilder:
    """
    Accumulates LangGraph node signals and builds an ObservationPayload.
    Session 33 · v2.9.0

    LangGraph signal mapping (MAIES-006 ④ convergence):
    ────────────────────────────────────────────────────
      φ/σ: node output → next node without retry edge = downstream_accepted=True
           error/retry count per node = reconfiguration events
      β:   token usage per node vs. configured limits via callbacks
           rate-limit errors = ContentionEvent(origin='infra')
      ε:   latency per node via checkpoint timestamps
           state delta size as complexity proxy

    task_domain: inject as a required field in LangGraph state schema.

    Usage
    -----
        builder = LangGraphObservationBuilder(
            agent_id="lg-retriever",
            task_domain="retrieval",
            resource_policy=ResourcePolicy(token_budget_per_hour=10000),
        )

        # After each node invocation:
        builder.record_node_invocation(
            task_id="run-001",
            node_name="retriever",
            success=True,
            retry_count=0,
            duration_seconds=1.2,
            downstream_accepted=True,
            latency_ms=1200.0,
        )

        payload = builder.build()
    """

    def __init__(
        self,
        agent_id: str,
        task_domain: str,
        resource_policy=None,
        tool_topology=None,
        phi_window: int = 200,
        sigma_window: int = 20,
        state_reliability=None,
    ):
        from core.observe_schema import (
            ResourcePolicy as RP,
            ToolTopology as TT,
            PHI_WINDOW_DEFAULT,
            SIGMA_WINDOW_DEFAULT,
        )
        self.agent_id        = agent_id
        self.task_domain     = task_domain
        self.resource_policy = resource_policy or RP()
        self.tool_topology   = tool_topology or TT()
        self.phi_window      = phi_window
        self.sigma_window    = sigma_window
        self.state_reliability = state_reliability

        self._history:    list = []
        self._contention: list = []
        self._reconfigs:  list = []
        self._latencies:  list = []
        self._current_task_duration: float = 0.0

    def record_node_invocation(
        self,
        task_id: str,
        node_name: str,
        success: bool,
        retry_count: int = 0,
        duration_seconds: float = 0.0,
        downstream_accepted=None,
        latency_ms: float = 0.0,
        task_type: str = "node",
    ) -> None:
        """Record one LangGraph node invocation."""
        from datetime import datetime as _dt
        from core.observe_schema import (
            TaskOutcome, ReconfigEvent, LatencySample,
        )
        now = _dt.utcnow()

        outcome = TaskOutcome(
            task_id=f"{task_id}:{node_name}",
            task_domain=self.task_domain,
            success=success,
            reconfiguration_count=retry_count,
            duration_seconds=duration_seconds,
            downstream_accepted=downstream_accepted,
        )
        self._history.append(outcome)
        self._current_task_duration += duration_seconds

        for _ in range(retry_count):
            self._reconfigs.append(ReconfigEvent(
                event_type="branching",
                tool_switched=False,
                timestamp=now,
                task_id=task_id,
            ))

        if latency_ms > 0:
            self._latencies.append(LatencySample(
                task_domain=self.task_domain,
                task_type=task_type or node_name,
                latency_ms=latency_ms,
                timestamp=now,
            ))

    def record_rate_limit(self, resource_type: str = "tokens", delay_ms: float = 0.0):
        """Record an infra rate-limit event."""
        from datetime import datetime as _dt
        from core.observe_schema import ContentionEvent
        self._contention.append(ContentionEvent(
            resource_type=resource_type,
            origin="infra",
            timestamp=_dt.utcnow(),
            delay_ms=delay_ms,
        ))

    def build(self, task_duration_seconds=None):
        """Build ObservationPayload from accumulated signals."""
        from core.observe_schema import ObservationPayload
        phi_history = [
            t for t in self._history
            if t.task_domain == self.task_domain
        ][-self.phi_window:]
        sigma_recent = self._history[-self.sigma_window:]
        duration = task_duration_seconds or self._current_task_duration

        return ObservationPayload(
            agent_id=self.agent_id,
            framework="langgraph",
            task_domain=self.task_domain,
            domain_success_history=phi_history,
            recent_task_outcomes=sigma_recent,
            resource_policy=self.resource_policy,
            contention_events=list(self._contention),
            reconfiguration_events=list(self._reconfigs),
            latency_samples=list(self._latencies),
            tool_topology=self.tool_topology,
            task_duration_seconds=duration,
            state_reliability=self.state_reliability,
        )

    def reset_session(self):
        self._reconfigs.clear()
        self._latencies.clear()
        self._contention.clear()
        self._current_task_duration = 0.0

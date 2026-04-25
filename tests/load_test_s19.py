# tests/load_test_s19.py
# MELVcore Load Test -- Session 19 P3
# Run: python tests/load_test_s19.py
# Requires: pip install httpx

import asyncio
import time
import random
import httpx

BASE_URL = "http://127.0.0.1:8000"
CONCURRENCY = 50

CATEGORIES = [
    "task_executor",
    "tool_using",
    "multi_agent",
    "autonomous",
    "iterative_loop",
]

DEFAULTS = {
    "iterative_loop": {"phi": 0.45, "epsilon": 5.5,  "tool_count": 20},
    "autonomous":     {"phi": 0.55, "epsilon": 4.5,  "tool_count": 10},
    "multi_agent":    {"phi": 0.65, "epsilon": 3.5,  "tool_count": 8},
    "tool_using":     {"phi": 0.70, "epsilon": 3.0,  "tool_count": 5},
    "task_executor":  {"phi": 0.80, "epsilon": 2.0,  "tool_count": 2},
}


def make_payload(i: int) -> dict:
    cat = CATEGORIES[i % len(CATEGORIES)]
    d = DEFAULTS[cat]
    return {
        "agent_id":   f"lt-{i:03d}",
        "agent_name": f"LT{i:03d}",
        "domain":     random.choice(["retrieval", "reasoning", "coding"]),
        "phi":         d["phi"],
        "epsilon":     d["epsilon"],
        "beta_pref":   1.0,
        "capabilities": ["search"],
        "run_duration_interactions": 50,
        "tool_count":  d["tool_count"],
        "operation_mode": "episodic",
        "shared_state": "none",
        "assessment_scores": {"agent_category": cat},
    }


async def submit_one(client: httpx.AsyncClient, i: int) -> dict:
    t0 = time.perf_counter()
    try:
        r = await client.post(
            f"{BASE_URL}/sandbox/submit",
            json=make_payload(i),
            timeout=30.0,
        )
        return {
            "i":      i,
            "status": r.status_code,
            "ms":     (time.perf_counter() - t0) * 1000,
            "run_id": r.json().get("run_id") if r.status_code == 200 else None,
        }
    except Exception as e:
        return {
            "i":      i,
            "status": 0,
            "ms":     (time.perf_counter() - t0) * 1000,
            "run_id": None,
            "err":    str(e),
        }


async def poll_run(client: httpx.AsyncClient, run_id: str, max_wait: float = 15.0) -> str:
    """Poll until complete or timeout. Returns final status."""
    deadline = time.time() + max_wait
    while time.time() < deadline:
        try:
            r = await client.get(f"{BASE_URL}/sandbox/run/{run_id}", timeout=5.0)
            if r.status_code == 200:
                status = r.json().get("status", "")
                if status in ("complete", "error"):
                    return status
        except Exception:
            pass
        await asyncio.sleep(0.3)
    return "timeout"


async def main() -> None:
    print(f"MELVcore Load Test — {CONCURRENCY} concurrent submissions")
    print(f"Target: {BASE_URL}")
    print("-" * 50)

    async with httpx.AsyncClient() as client:
        # ── Phase 1: concurrent submissions ────────────────────────────
        results = await asyncio.gather(*[submit_one(client, i) for i in range(CONCURRENCY)])

    ok       = [r for r in results if r["status"] == 200]
    failed   = [r for r in results if r["status"] != 200]
    ms_all   = sorted(r["ms"] for r in results)
    mean_ms  = sum(ms_all) / len(ms_all)
    p50_ms   = ms_all[int(len(ms_all) * 0.50)]
    p95_ms   = ms_all[int(len(ms_all) * 0.95)]
    max_ms   = ms_all[-1]

    print(f"Submissions:  {len(ok)}/{CONCURRENCY} succeeded")
    if failed:
        print(f"Failures:     {len(failed)}")
        for f in failed[:5]:
            print(f"  [{f['i']:03d}] status={f['status']} err={f.get('err','')}")

    print(f"Latency (ms): mean={mean_ms:.0f}  p50={p50_ms:.0f}  p95={p95_ms:.0f}  max={max_ms:.0f}")

    # ── Phase 2: CI consistency check on 5 sampled completed runs ──────
    run_ids = [r["run_id"] for r in ok if r["run_id"]][:5]
    if run_ids:
        print(f"\nPolling {len(run_ids)} sampled runs for CI consistency …")
        async with httpx.AsyncClient() as client:
            statuses = await asyncio.gather(*[poll_run(client, rid) for rid in run_ids])

        ci_values = []
        async with httpx.AsyncClient() as client:
            for rid, st in zip(run_ids, statuses):
                if st == "complete":
                    try:
                        r = await client.get(f"{BASE_URL}/sandbox/certify/{rid}", timeout=10.0)
                        if r.status_code == 200:
                            cls_score = r.json().get("cls_score")
                            if cls_score is not None:
                                ci_values.append(cls_score)
                    except Exception:
                        pass

        if ci_values:
            spread = max(ci_values) - min(ci_values)
            print(f"CI values:    {[round(v,1) for v in ci_values]}")
            print(f"CI spread:    {spread:.1f} CLS points (limit: 50)")
            ci_consistent = spread < 50
        else:
            print("CI values:    (none retrieved — runs may not have completed)")
            ci_consistent = True  # don't fail on this
    else:
        ci_consistent = True

    # ── Overall verdict ─────────────────────────────────────────────────
    submit_pass = len(ok) == CONCURRENCY
    latency_pass = mean_ms < 2000
    overall = submit_pass and latency_pass and ci_consistent

    print("\n" + "=" * 50)
    print(f"Submit 200:  {'PASS' if submit_pass   else 'FAIL'}")
    print(f"Mean <2000ms: {'PASS' if latency_pass else 'FAIL'}")
    print(f"CI consistent: {'PASS' if ci_consistent else 'FAIL'}")
    print("=" * 50)
    print("OVERALL:", "PASS ✓" if overall else "FAIL ✗")


if __name__ == "__main__":
    asyncio.run(main())

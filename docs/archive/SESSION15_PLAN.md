# Session 15 Plan — v1.7.0

## Hardening priorities (from manual Section 7.4)

### 1. Rate limit startup grace period (dashboard fix)
**Problem:** Dashboard fires all polling requests simultaneously on load,
causing a burst of 429s before settling. Cosmetically ugly, harmless functionally.
**Fix:** Add a `STARTUP_GRACE_SECONDS` (default 15) to `RateLimitMiddleware`.
During the grace window after server start, rate limiting is bypassed entirely.
After grace period, normal limits apply.
**Env var:** `AIOS_STARTUP_GRACE=15`
**Files:** `api/middleware.py`, `tests/test_session15.py`

### 2. Adversarial sandbox inputs
- Negative phi values (phi < 0)
- phi > 1.0
- epsilon = 0 (no plasticity)
- n_interactions = 0
- Very large n_interactions (> 10,000)
- Unicode/special characters in agent_id and domain fields
- Cost = 0, benefit = 0 edge case

### 3. MCP Inspector compliance
- Run MCP Inspector against all 4 tools
- Verify JSON schema validation on all inputs
- Confirm error responses follow MCP spec

### 4. Load test
- 50 concurrent certify_agent requests
- Confirm sandbox kernel isolation holds under concurrency
- Confirm live kernel is never mutated

### 5. arXiv preprint prep
- Title: "MELVcore: A Thermodynamic Governance Kernel for Multi-Agent AI Systems"
- Sections: Abstract, Introduction, MELV Framework, Implementation, Validation
- Cite: Zenodo DOI 10.5281/zenodo.17680563

### 6. Srinivasan (UCLA) outreach email
- Use DeepSeek independent MELV derivation as centrepiece
- Reference arXiv preprint once submitted

## Target test count: 77 → currently 245, target ~270 (25 new tests)
## Version: v1.7.0

### 7. landing.html end-to-end testing
- Load http://localhost:8000/demo and confirm page renders
- Submit sandbox certification via the landing page form
- Confirm result card shows verdict, CLS score, certification anchor
- Confirm rate limit (6th submission returns friendly error message)
- Confirm Zenodo DOI link is present and correct
- Confirm GitHub link points to NaturesHolismMELV/AIOS
- Add test_landing_sandbox_submit_returns_result to test_session15.py

# AIOS / MELVcore — Installation Guide

**Version:** v1.6.0 · Session 14  
**Repository:** https://github.com/NaturesHolismMELV/AIOS  
**Author:** L.W. Evans · Ecotao Enterprises, Cape Town  
**Zenodo DOI:** 10.5281/zenodo.17680563

---

## Requirements

- Python 3.10 or later
- pip
- A terminal (PowerShell on Windows, Terminal on macOS/Linux)
- An Anthropic API key (for real LLM agent calls — ANALYSIS, WRITER, PLANNER agents)

---

## 1 — Clone the repository

```bash
git clone https://github.com/NaturesHolismMELV/AIOS.git
cd AIOS
```

---

## 2 — Install Python dependencies

```bash
pip install -r requirements.txt
```

For development and running tests, also install:

```bash
pip install -r requirements-dev.txt
```

---

## 3 — Configure environment

Copy the example environment file and add your Anthropic API key:

```bash
cp .env.example .env
```

Edit `.env`:

```
ANTHROPIC_API_KEY=sk-ant-your-key-here

# Optional — leave unset for open access (dev mode)
# AIOS_API_KEY=your-api-key

# Optional — SQLite database path (default: ./aios_state.db)
# AIOS_DB_PATH=/path/to/aios_state.db

# Optional — rate limiting (defaults shown)
# AIOS_RATE_LIMIT_REQUESTS=60
# AIOS_RATE_LIMIT_WINDOW=60
# AIOS_SANDBOX_LIMIT=5
# AIOS_SANDBOX_WINDOW=3600
```

> **Note:** Without `ANTHROPIC_API_KEY`, the ANALYSIS, WRITER, and PLANNER agents
> will fail on real LLM calls. All other functionality (kernel, sandbox,
> CI Dynamics, MCP server) works without an API key.

---

## 4 — Start the server

```bash
python -m uvicorn api.server:app --reload
```

The server starts on `http://localhost:8000`.  
You should see the MELVcore startup banner in the terminal.

---

## 5 — Open the dashboard

Open `frontend/dashboard12.html` directly in your browser (no web server needed — it is a static file).

The status indicator in the top-right corner will turn green when connected.

---

## 6 — Run the test suite

```bash
python -m pytest tests/ -v
```

Expected: **245 tests passing**.

To run a specific session's tests:

```bash
python -m pytest tests/test_session14.py -v
```

---

## 7 — MCP server (Claude Desktop / Cursor integration)

The MCP server is mounted automatically when the AIOS server starts.

**Streamable HTTP (recommended):** `http://localhost:8000/mcp`  
**SSE (legacy):** `http://localhost:8000/mcp/sse`

### Claude Desktop configuration

Add to `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "melvcore": {
      "type": "sse",
      "url": "http://localhost:8000/mcp/sse"
    }
  }
}
```

For the hosted demo replace `localhost:8000` with your Railway/Render URL.

---

## 8 — Deploying to Railway (hosted demo)

See `DEPLOY.md` for full Railway and Render deployment instructions.

Quick start:

1. Push this repo to GitHub (private or public)
2. Create a new project at https://railway.app
3. Connect the GitHub repository
4. Set environment variables: `ANTHROPIC_API_KEY`, `AIOS_DB_PATH=/tmp/aios_state.db`
5. Railway auto-detects the `Procfile` and deploys

The public demo URL will be: `https://YOUR-APP.railway.app/demo`

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Dashboard shows OFFLINE | Server not running. Run `python -m uvicorn api.server:app --reload` |
| LLM agents fail | `ANTHROPIC_API_KEY` not set or invalid |
| `mcp` import error | Run `pip install mcp>=1.0` |
| Tests fail with rate limit errors | Tests set `AIOS_RATE_LIMIT_REQUESTS=500` via `pytest.ini` — ensure `pytest-env` is installed |
| Database locked | Stop all server instances before running tests |

---

*Ecotao Enterprises · L.W. Evans · Cape Town, South Africa*  
*Blueprint for Harmony · Cooperation Press, 2026 · ISBN 978-969-8992-10-1*

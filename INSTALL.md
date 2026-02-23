# INSTALL.md — Installation & Setup

## Requirements

- Python 3.11 or higher
- An Anthropic API key (free tier works — only the ANALYSIS agent uses it)
- Internet connection (RESEARCH, DATA, SEARCH agents call external APIs)

---

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/AIOS.git
cd AIOS
pip install -r requirements.txt
```

### Requirements file contents
```
fastapi
uvicorn
pydantic
anthropic
httpx
duckduckgo-search
pytest
pytest-asyncio
```

---

## API Key Setup

The ANALYSIS agent calls Claude Haiku. You need an Anthropic API key.

**Windows (permanent):**
```
setx ANTHROPIC_API_KEY "sk-ant-..."
```
Close and reopen PowerShell after running this.

**Linux / macOS:**
```bash
export ANTHROPIC_API_KEY="sk-ant-..."
# Add to ~/.bashrc or ~/.zshrc to make permanent
```

The RESEARCH, DATA, and SEARCH agents use free public APIs (DuckDuckGo, World Bank) — no keys required.

---

## Starting the Server

```bash
cd AIOS
python -m uvicorn api.server:app --reload
```

Server starts at `http://localhost:8000`

**If you see import errors:**
```bash
python -m pip install fastapi uvicorn pydantic anthropic httpx duckduckgo-search
```

---

## Opening the Dashboard

The live dashboard is a standalone HTML file — open it directly in your browser:

```
frontend/dashboard.html
```

Or navigate to it from the filesystem. No separate server needed for the dashboard.

The dashboard polls `http://localhost:8000/api/` every 3 seconds and shows:
- Cooperation Index gauge (target ≥ 0.75)
- Per-agent i-factor, φ maturity, ε energy
- Agent registry with ACTIVE / MATURING / THRESHOLD status
- OmegaNet interaction graph
- Bifurcation event log
- β Provisioning panel (resource scarcity by type)

---

## Running Tests

```bash
python -m pytest tests/ -v
```

**Phase 1 suite** (26 tests — MELVcore kernel, simulation):
```bash
python -m pytest tests/test_melv.py -v
```

**DataAgent suite** (10 tests — real World Bank API):
```bash
python -m pytest tests/test_data_agent.py -v
```

---

## Verifying the Setup

With the server running, test the key endpoints:

```bash
# System health
curl http://localhost:8000/api/health

# Agent registry
curl http://localhost:8000/api/agents

# Real World Bank data (South Africa GDP)
curl http://localhost:8000/data/profile/ZA

# Submit a task
curl -X POST http://localhost:8000/api/task \
  -H "Content-Type: application/json" \
  -d '{"task": "What is the current state of multi-agent AI systems?"}'
```

A healthy system returns a `cooperation_index` above 0.75 in the `/api/health` response.

---

## Windows Notes

- Always use `python -m pip` rather than bare `pip`
- Python location: typically `C:\Users\<user>\AppData\Local\Programs\Python\Python3XX\`
- Working directory: `C:\Users\<user>\AIOS`
- PowerShell is preferred over Command Prompt

---

## Troubleshooting

**`pytest` not recognised:**
```
python -m pytest tests/ -v
```

**`uvicorn` not found:**
```
python -m uvicorn api.server:app --reload
```

**Dashboard shows ERR on all panels:**  
Check that the server is running on port 8000. The dashboard connects to `http://localhost:8000/api/`.

**ANALYSIS agent errors:**  
Check your `ANTHROPIC_API_KEY` is set. On Windows, open a fresh PowerShell after `setx`.

**DATA agent timeout:**  
The World Bank API occasionally responds slowly. The agent has a 15-second timeout and will return an empty records list rather than crash. Retry after a moment.

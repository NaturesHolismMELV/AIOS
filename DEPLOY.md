# Deploying MELVcore Demo — v1.9.2 · Session 21.2

Two paths to a live public URL. **Railway is recommended** for a first
deployment — faster setup, usage-based billing (near-zero cost for a
demo with light traffic).

> **WeasyPrint note:** PDF certification reports require system graphics
> libraries (Pango, Cairo, GLib). These are declared in `nixpacks.toml`
> and installed automatically by Railway's Nixpacks builder. No manual
> action needed — just ensure `nixpacks.toml` is present in the repo root.

---

## Option A — Railway (recommended, ~15 minutes)

### Prerequisites
- GitHub account with the AIOS repo pushed
- Railway account at https://railway.app (free tier available)

### Steps

1. **Push the repo to GitHub**
   ```bash
   cd AIOS
   git init
   git add .
   git commit -m "MELVcore v1.5.0 — Session 13"
   git remote add origin https://github.com/YOUR_USERNAME/AIOS.git
   git push -u origin main
   ```

2. **Create a Railway project**
   - Go to https://railway.app/new
   - Choose **Deploy from GitHub repo**
   - Select your AIOS repository
   - Railway auto-detects the `Procfile` and builds with Nixpacks

3. **Set environment variables** (Railway Dashboard → Variables)

   | Variable | Value | Notes |
   |---|---|---|
   | `PORT` | set by Railway | auto-injected |
   | `AIOS_DB_PATH` | `/data/aios_state.db` | persistent volume path (see step 4) |
   | `AIOS_API_KEY` | your-secret-key | **required** before sharing public URL |
   | `AIOS_RATE_LIMIT_REQUESTS` | `60` | default |
   | `AIOS_SANDBOX_LIMIT` | `5` | per-IP per hour |

4. **Mount a persistent volume** (Railway Dashboard → Volumes)
   - Click **New Volume** → mount path: `/data`
   - This ensures the SQLite database survives redeploys
   - Without this, `aios_state.db` resets on every deploy (all certification history lost)

5. **Deploy** — Railway triggers on every push to `main`

6. **Your URLs**
   - Landing page: `https://YOUR-APP.railway.app/demo`
   - Dashboard:    `https://YOUR-APP.railway.app/frontend/dashboard12.html`
   - API docs:     `https://YOUR-APP.railway.app/docs`
   - Health check: `https://YOUR-APP.railway.app/health`

### Cost estimate
- Free tier: $5 credit/month, enough for ~500 hours of light traffic
- Starter ($5/month): always-on, no sleep
- Usage: roughly $0.000463/hour for a 512MB service = ~$0.33/day

---

## Option B — Render

### Steps

1. **Push to GitHub** (same as Railway step 1)

2. **Create a Web Service** at https://render.com/new
   - Connect your GitHub repo
   - Render detects `render.yaml` automatically
   - Build command: `pip install -r requirements.txt`
   - Start command: `uvicorn api.server:app --host 0.0.0.0 --port $PORT`

3. **Set environment variables** (Render Dashboard → Environment)
   - Same variables as Railway above
   - For `AIOS_API_KEY`: use **Secret Files** or the environment panel

4. **Free tier caveat**: Render free tier spins down after 15 minutes of
   inactivity. Upgrade to Starter ($7/month) for always-on.

---

## Static file serving (dashboard + landing page)

The server mounts `frontend/` as a static directory and serves:
- `/demo` → `frontend/landing.html` (public landing page)
- `/frontend/dashboard12.html` → full internal dashboard

To verify static serving works locally:
```bash
cd AIOS
uvicorn api.server:app --reload
# Open: http://localhost:8000/demo
```

---

## Environment variable reference

| Variable | Default | Description |
|---|---|---|
| `PORT` | `8000` | Server port (injected by Railway/Render) |
| `AIOS_DB_PATH` | `./aios_state.db` | SQLite persistence file path |
| `AIOS_API_KEY` | *(unset)* | Header key for protected endpoints. Unset = open access |
| `AIOS_RATE_LIMIT_REQUESTS` | `60` | Requests per IP per window |
| `AIOS_RATE_LIMIT_WINDOW` | `60` | Window size in seconds |
| `AIOS_SANDBOX_LIMIT` | `5` | Sandbox submissions per IP per hour |
| `AIOS_SANDBOX_WINDOW` | `3600` | Sandbox window in seconds |

---

## Sharing the demo

Once deployed, the URL to share is:

```
https://YOUR-APP.railway.app/demo
```

This page:
- Shows the live Cooperation Index of the running ecosystem
- Explains MELVcore in plain language (no prior knowledge required)
- Lets anyone submit an agent profile and get a CLS certification report
- Links to the full dashboard and GitHub

---

## Security notes

- The demo is rate-limited per IP: 60 req/min general, 5 sandbox submissions/hour
- Set `AIOS_API_KEY` to restrict `/sandbox/`, `/melv/`, and `/api/beta` endpoints
- The SQLite DB at `/tmp/aios_state.db` on Railway resets on deploy — acceptable for demo
- For production persistence, mount a Railway Volume and set `AIOS_DB_PATH` to the mount path

---

*L.W. Evans | ORCID: 0009-0001-0963-1840 | Cape Town, South Africa · MELVcore v1.9.2 · Session 21.2*

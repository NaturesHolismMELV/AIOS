# Security Policy — AIOS / MELVcore

## Scope

This document covers the security posture of the AIOS / MELVcore
Thermodynamic Agent Certification Platform
(`github.com/NaturesHolismMELV/AIOS`).

## Data Storage

- **No sensitive user data is stored.** The system records agent
  profiles (IDs, ε, φ, β values), certification run results, and
  interaction histories — all synthetic or user-supplied configuration
  values.
- **No PII, credentials, or private keys** are written to the SQLite
  database or any log files.
- The SQLite database (`aios_state.db`) holds certification run metadata
  only. It is safe to delete and recreate at any time.

## API Key Requirements

- Set the `AIOS_API_KEY` environment variable before exposing the server
  publicly.
- Protected paths (`/sandbox/`, `/melv/`, `/api/beta`) require the
  `X-API-Key` header when `AIOS_API_KEY` is set.
- The server logs a WARNING at startup if `AIOS_API_KEY` is unset.

## Rate Limits

The following rate limits are applied per client IP:

| Scope                              | Default limit        |
|------------------------------------|----------------------|
| General API                        | 60 requests / 60 s   |
| `/sandbox/submit` (POST)           | 5 submissions / hour |
| `/sandbox/assess/*` (POST)         | 5 requests / hour    |
| `/sandbox/run/*` (POST)            | 5 requests / hour    |

Limits are configurable via environment variables:
`AIOS_RATE_LIMIT_REQUESTS`, `AIOS_RATE_LIMIT_WINDOW`,
`AIOS_SANDBOX_LIMIT`, `AIOS_SANDBOX_WINDOW`.

## Deployment Checklist

Before sharing the public demo URL widely:

- [ ] `AIOS_API_KEY` set in Railway/Render deployment secrets
- [ ] Persistent SQLite volume mounted (`/data/aios_state.db`)
- [ ] MCP Inspector verified against `/mcp`
- [ ] Rate limits reviewed for expected traffic volume
- [ ] Dependencies scanned (Dependabot enabled)

## Responsible Disclosure

If you discover a security vulnerability in this project, please report
it privately by emailing:

**laurence@ecotao.co.za**

Please include:
- Description of the vulnerability
- Steps to reproduce
- Potential impact

We aim to respond within 5 business days and to publish a fix within 30
days of confirmed receipt.

Do not open a public GitHub issue for security vulnerabilities.

## Known Limitations

- Rate limiting is in-memory only (resets on restart). For
  multi-instance production deployments, replace with a shared store
  such as Redis.
- MCP server is unauthenticated by default; protect with API key
  middleware or restrict to trusted networks.

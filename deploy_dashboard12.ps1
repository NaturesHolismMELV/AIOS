# ─────────────────────────────────────────────────────────────────────────────
# deploy_dashboard12.ps1
# Deploy updated dashboard12.html to AIOS Railway project (incredible-friendship)
# Run from: C:\Users\web\AIOS
# Usage:    .\deploy_dashboard12.ps1 [-SourceFile "path\to\dashboard12.html"]
# ─────────────────────────────────────────────────────────────────────────────

param(
    [string]$SourceFile = "frontend\dashboard12.html"
)

$ErrorActionPreference = "Stop"

# ── 0. Resolve paths ──────────────────────────────────────────────────────────
$RepoRoot   = $PSScriptRoot          # directory containing this script
$Target     = Join-Path $RepoRoot "frontend\dashboard12.html"

if (-not (Test-Path $SourceFile)) {
    Write-Error "Source file not found: $SourceFile"
    exit 1
}

# ── 1. Copy updated file into the repo ───────────────────────────────────────
Write-Host "► Copying $SourceFile  →  $Target" -ForegroundColor Cyan
Copy-Item -Path $SourceFile -Destination $Target -Force

# ── 2. Git commit ─────────────────────────────────────────────────────────────
Write-Host "► Staging and committing..." -ForegroundColor Cyan
Set-Location $RepoRoot

git add frontend/dashboard12.html
git commit -m "cosmetic: OmegaNet canvas +25% height, node radius 4.5→6, label 7px→11px bright-blue"

if ($LASTEXITCODE -ne 0) {
    Write-Error "git commit failed. Check git status."
    exit 1
}

# ── 3. Push to main → triggers Railway auto-deploy ────────────────────────────
Write-Host "► Pushing to origin main..." -ForegroundColor Cyan
git push origin main

if ($LASTEXITCODE -ne 0) {
    Write-Error "git push failed."
    exit 1
}

Write-Host ""
Write-Host "✔ Pushed. Railway will redeploy automatically." -ForegroundColor Green
Write-Host "  Monitor: https://railway.app/project/incredible-friendship" -ForegroundColor DarkGray
Write-Host "  Live URL: https://web-production-e14d1.up.railway.app/frontend/dashboard12.html" -ForegroundColor DarkGray
Write-Host ""
Write-Host "Changes deployed:" -ForegroundColor Yellow
Write-Host "  • #omega-canvas height  200px → 250px  (+25%)" -ForegroundColor White
Write-Host "  • Node radius           4.5   → 6" -ForegroundColor White
Write-Host "  • Label font            7px   → 11px  Space Mono" -ForegroundColor White
Write-Host "  • Label colour          #4a6a8a (dim) → #00aaff (bright blue)" -ForegroundColor White
Write-Host "  • Label offset          13px  → 18px  (clears larger nodes)" -ForegroundColor White
Write-Host "  • Node glow shadowBlur  7     → 10" -ForegroundColor White

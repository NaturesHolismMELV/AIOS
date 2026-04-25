"""
MELVcore Certification Report â€” PDF Generator
==============================================
Session 19 Â· Wave 2 SaaS Â· v1.9.2

Renders a CertificationReport (to_dict()) into a professional PDF using
weasyprint (HTML â†’ PDF). Falls back to reportlab if weasyprint is absent.

Public entry-point:
    pdf_bytes = render_cert_pdf(report_dict, assessment_scores=None)
"""

from __future__ import annotations
import io
from datetime import datetime, timezone
from typing import Optional

# â”€â”€ CONSTANTS â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

ZENODO_DOI   = "10.5281/zenodo.19029077"
CONCEPT_DOI  = "10.5281/zenodo.17535157"
ORCID        = "0009-0001-0963-1840"
ISBN         = "978-969-8992-10-1"
GITHUB       = "github.com/NaturesHolismMELV/AIOS"

# Verdict colours
VERDICT_COLOUR = {
    "CERTIFIED":              "#1a7f37",
    "CERTIFIED_WITH_ADVISORY": "#b45309",
    "NOT_CERTIFIED":          "#b91c1c",
}
VERDICT_BG = {
    "CERTIFIED":              "#d1fae5",
    "CERTIFIED_WITH_ADVISORY": "#fef3c7",
    "NOT_CERTIFIED":          "#fee2e2",
}

# Ï† lifecycle tier colours
PHI_TIER_COLOUR = {
    "Permanent":  "#1d4ed8",
    "Working":    "#b45309",
    "Ephemeral":  "#6b7280",
}
PHI_TIER_BG = {
    "Permanent":  "#dbeafe",
    "Working":    "#fef3c7",
    "Ephemeral":  "#f3f4f6",
}

# CO band colours
CO_BAND_COLOUR = {
    "LOW":      "#1a7f37",
    "MODERATE": "#b45309",
    "HIGH":     "#b91c1c",
}
CO_BAND_BG = {
    "LOW":      "#d1fae5",
    "MODERATE": "#fef3c7",
    "HIGH":     "#fee2e2",
}


# â”€â”€ HTML TEMPLATE â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def _render_html(report: dict, assessment_scores: Optional[dict]) -> str:
    """Build the full HTML string for the PDF report."""

    verdict       = report["verdict"]
    cls_score     = report["cls_score"]
    agent_id      = report["agent_id"]
    run_id        = report["run_id"]
    narrative     = report.get("narrative", "")
    advisory      = report.get("advisory", "")
    anchor        = report.get("certification_anchor", {})
    certified_at  = anchor.get("certified_at", datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))
    phi_lc        = report.get("phi_lifecycle") or {}
    co            = report.get("coordination_overhead") or {}
    baseline      = report.get("baseline") or {}
    with_agent    = report.get("with_agent") or {}

    ois  = report.get("oscillation_impact_score", 0.0)
    ddc  = report.get("drift_degradation_coeff", 0.0)
    dhl  = report.get("delta_half_life_sec")

    # Ï† lifecycle badge
    phi_tier       = phi_lc.get("tier", "â€”")
    phi_label      = phi_lc.get("label", "")
    phi_advisory   = phi_lc.get("advisory", "")
    phi_tc         = PHI_TIER_COLOUR.get(phi_tier, "#374151")
    phi_tb         = PHI_TIER_BG.get(phi_tier, "#f9fafb")

    # CO badge
    co_score  = co.get("score", 0.0)
    co_band   = co.get("band", "â€”")
    co_adv    = co.get("advisory", "")
    co_c      = CO_BAND_COLOUR.get(co_band, "#374151")
    co_b      = CO_BAND_BG.get(co_band, "#f9fafb")

    # Top 3 high-risk Îµ parameters
    eps_rows = ""
    if assessment_scores:
        eps_scores = assessment_scores.get("epsilon_scores") or {}
        if eps_scores:
            scored = [(k, v) for k, v in eps_scores.items() if v is not None]
            scored.sort(key=lambda x: -x[1])
            top3 = scored[:3]
            for param, val in top3:
                bar_w = int((val / 10.0) * 100)
                bar_c = "#b91c1c" if val >= 7 else "#b45309" if val >= 5 else "#1a7f37"
                eps_rows += f"""
                <tr>
                  <td style="padding:4px 8px;font-size:11px;">{param.replace('_',' ').title()}</td>
                  <td style="padding:4px 8px;font-size:11px;">{val:.1f}/10</td>
                  <td style="padding:4px 8px;">
                    <div style="background:#e5e7eb;border-radius:3px;height:8px;width:120px;">
                      <div style="background:{bar_c};border-radius:3px;height:8px;width:{bar_w}%;"></div>
                    </div>
                  </td>
                </tr>"""

    eps_section = ""
    if eps_rows:
        eps_section = f"""
        <div class="section">
          <div class="section-title">Top Îµ Risk Parameters</div>
          <table style="width:100%;border-collapse:collapse;">
            <thead>
              <tr style="background:#f3f4f6;">
                <th style="padding:4px 8px;text-align:left;font-size:11px;">Parameter</th>
                <th style="padding:4px 8px;text-align:left;font-size:11px;">Score</th>
                <th style="padding:4px 8px;text-align:left;font-size:11px;">Risk Level</th>
              </tr>
            </thead>
            <tbody>{eps_rows}</tbody>
          </table>
        </div>"""

    # CI snapshot table
    def fmt_snap(snap: dict) -> str:
        if not snap:
            return "â€”"
        return (f"mean_ci={snap.get('mean_ci', 0.0):.4f} "
                f"regime={snap.get('regime', '?')} "
                f"interactions={snap.get('total_interactions', 0)}")

    # Advisory section
    advisory_section = ""
    if advisory:
        advisory_section = f"""
        <div class="section advisory-box">
          <div class="section-title" style="color:#92400e;">Advisory Notes</div>
          <p style="font-size:11px;margin:0;white-space:pre-wrap;">{advisory}</p>
        </div>"""

    # co advisory
    co_advisory_html = ""
    if co_adv:
        co_advisory_html = f'<p style="font-size:10px;color:#b91c1c;margin:4px 0 0;">{co_adv}</p>'

    # phi advisory
    phi_advisory_html = ""
    if phi_advisory:
        phi_advisory_html = f'<p style="font-size:10px;color:#374151;margin:4px 0 0;">{phi_advisory}</p>'

    # dhl display
    dhl_str = f"{dhl:+.3f}s" if dhl is not None else "N/A (both runs fully cooperative)"

    verdict_c = VERDICT_COLOUR.get(verdict, "#374151")
    verdict_b = VERDICT_BG.get(verdict, "#f9fafb")
    verdict_label = verdict.replace("_", " ")

    # CLS progress bar
    cls_bar_w = int(cls_score)
    cls_bar_c = "#1a7f37" if cls_score >= 80 else "#b45309" if cls_score >= 60 else "#b91c1c"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<style>
  @page {{ size: A4; margin: 18mm 16mm 18mm 16mm; }}
  * {{ box-sizing: border-box; }}
  body {{ font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
         font-size: 12px; color: #111827; margin: 0; padding: 0; }}
  .page {{ max-width: 700px; margin: 0 auto; }}
  .header {{ background: #0f172a; color: #f1f5f9; padding: 20px 24px 16px;
             border-radius: 4px 4px 0 0; }}
  .header-title {{ font-size: 18px; font-weight: 700; letter-spacing: 0.5px; margin: 0 0 2px; }}
  .header-sub {{ font-size: 10px; color: #94a3b8; margin: 0; }}
  .masthead {{ display: flex; gap: 16px; align-items: flex-end; margin-top: 12px; }}
  .masthead-meta {{ font-size: 10px; color: #cbd5e1; line-height: 1.7; }}
  .verdict-banner {{ padding: 12px 24px; background: {verdict_b};
                     border-left: 5px solid {verdict_c}; margin: 0; }}
  .verdict-label {{ font-size: 20px; font-weight: 800; color: {verdict_c}; letter-spacing: 1px; }}
  .verdict-score {{ font-size: 12px; color: #374151; margin-top: 2px; }}
  .cls-bar-track {{ background: #e5e7eb; border-radius: 4px; height: 10px; margin-top: 6px; width: 240px; }}
  .cls-bar-fill {{ background: {cls_bar_c}; border-radius: 4px; height: 10px; width: {cls_bar_w}%; }}
  .body {{ padding: 0 24px 16px; }}
  .section {{ margin: 14px 0; }}
  .section-title {{ font-size: 11px; font-weight: 700; color: #374151;
                    text-transform: uppercase; letter-spacing: 0.6px;
                    border-bottom: 1px solid #e5e7eb; padding-bottom: 3px; margin-bottom: 6px; }}
  .badges {{ display: flex; gap: 12px; flex-wrap: wrap; margin: 4px 0; }}
  .badge {{ display: inline-block; padding: 5px 12px; border-radius: 20px;
            font-size: 11px; font-weight: 700; }}
  .metrics-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }}
  .metric-card {{ background: #f9fafb; border: 1px solid #e5e7eb;
                  border-radius: 4px; padding: 8px 12px; }}
  .metric-label {{ font-size: 9px; color: #6b7280; text-transform: uppercase;
                   letter-spacing: 0.5px; margin-bottom: 2px; }}
  .metric-value {{ font-size: 13px; font-weight: 700; color: #111827; }}
  .narrative-box {{ background: #f0f9ff; border: 1px solid #bae6fd;
                    border-radius: 4px; padding: 10px 14px; font-size: 11px;
                    line-height: 1.6; color: #1e40af; }}
  .advisory-box {{ background: #fffbeb; border: 1px solid #fcd34d;
                   border-radius: 4px; padding: 10px 14px; }}
  .equation-box {{ background: #f0fdf4; border: 1px solid #86efac;
                   border-radius: 4px; padding: 10px 16px; text-align: center; }}
  .equation {{ font-family: "Courier New", monospace; font-size: 14px;
               font-weight: 700; color: #166534; letter-spacing: 0.5px; }}
  .eq-legend {{ display: flex; gap: 16px; flex-wrap: wrap; margin-top: 8px;
                justify-content: center; }}
  .eq-item {{ font-size: 9px; color: #374151; }}
  .anchor {{ background: #f8fafc; border: 1px solid #e2e8f0;
             border-radius: 4px; padding: 8px 14px; font-size: 9.5px;
             color: #475569; line-height: 1.8; }}
  .anchor strong {{ color: #0f172a; }}
  .footer {{ text-align: center; font-size: 9px; color: #94a3b8;
             border-top: 1px solid #e5e7eb; padding-top: 8px; margin-top: 16px; }}
  table {{ border-collapse: collapse; }}
  td, th {{ vertical-align: top; }}
</style>
</head>
<body>
<div class="page">

  <!-- Header -->
  <div class="header">
    <div class="header-title">MELVcore Agent Certification Report</div>
    <div class="header-sub">Modified Energetic Lotka-Volterra Framework Â· v1.9.2</div>
    <div class="masthead">
      <div class="masthead-meta">
        Agent ID: <strong style="color:#e2e8f0;">{agent_id}</strong><br/>
        Run ID: {run_id}<br/>
        Certified: {certified_at.replace("T"," ").replace("Z"," UTC")}
      </div>
    </div>
  </div>

  <!-- Verdict Banner -->
  <div class="verdict-banner">
    <div class="verdict-label">&#x2713; {verdict_label}</div>
    <div class="verdict-score">Composite Longevity Score: {cls_score:.1f} / 100</div>
    <div class="cls-bar-track"><div class="cls-bar-fill"></div></div>
  </div>

  <div class="body">

    <!-- Master Equation -->
    <div class="section">
      <div class="section-title">MELV Master Equation</div>
      <div class="equation-box">
        <div class="equation">i&#x2081;&#x2082;(t) = i&#x2081;&#x2082;&#x00B0; &times; (1 &minus; &epsilon; &times; &phi;(t) &times; &beta;(t))</div>
        <div class="eq-legend">
          <span class="eq-item"><strong>&phi;(t)</strong> â€” evolutionary maturity</span>
          <span class="eq-item"><strong>&epsilon;</strong> â€” adaptive plasticity</span>
          <span class="eq-item"><strong>&beta;(t)</strong> â€” environmental compatibility</span>
          <span class="eq-item"><strong>i</strong> â€” interaction cost ratio (C/B)</span>
        </div>
      </div>
    </div>

    <!-- Lifecycle Badges -->
    <div class="section">
      <div class="section-title">Lifecycle &amp; Coordination Assessment</div>
      <div class="badges">
        <div>
          <span class="badge" style="background:{phi_tb};color:{phi_tc};">
            &#x03C6; {phi_tier} â€” {phi_label}
          </span>
          {phi_advisory_html}
        </div>
        <div>
          <span class="badge" style="background:{co_b};color:{co_c};">
            CO Score {co_score:.2f} â€” {co_band}
          </span>
          {co_advisory_html}
        </div>
      </div>
    </div>

    <!-- Key Metrics -->
    <div class="section">
      <div class="section-title">Key Metrics</div>
      <div class="metrics-grid">
        <div class="metric-card">
          <div class="metric-label">Oscillation Impact Score</div>
          <div class="metric-value">{ois:.4f}</div>
        </div>
        <div class="metric-card">
          <div class="metric-label">Drift Degradation Coefficient</div>
          <div class="metric-value">{ddc:.6f}</div>
        </div>
        <div class="metric-card">
          <div class="metric-label">CI Half-Life Delta</div>
          <div class="metric-value">{dhl_str}</div>
        </div>
        <div class="metric-card">
          <div class="metric-label">CI Regime (Baseline â†’ With Agent)</div>
          <div class="metric-value" style="font-size:11px;">{baseline.get('regime','?')} â†’ {with_agent.get('regime','?')}</div>
        </div>
      </div>
    </div>

    <!-- Îµ Parameter Risk -->
    {eps_section}

    <!-- Narrative -->
    <div class="section">
      <div class="section-title">Certification Narrative</div>
      <div class="narrative-box">{narrative}</div>
    </div>

    <!-- Advisory -->
    {advisory_section}

    <!-- Certification Anchor -->
    <div class="section">
      <div class="section-title">Certification Anchor</div>
      <div class="anchor">
        <strong>Framework:</strong> {anchor.get('framework','MELV â€” Modified Energetic Lotka-Volterra')}<br/>
        <strong>Zenodo DOI (preprint):</strong> {ZENODO_DOI}<br/>
        <strong>Concept DOI:</strong> {CONCEPT_DOI}<br/>
        <strong>ORCID:</strong> {ORCID}<br/>
        <strong>Book ISBN:</strong> {ISBN} (<em>Blueprint for Harmony</em>, Cooperation Press 2026)<br/>
        <strong>GitHub:</strong> {GITHUB}<br/>
        <strong>Sandbox Version:</strong> {anchor.get('sandbox_version','1.9.0')}
      </div>
    </div>

  </div><!-- /body -->

  <div class="footer">
    MELVcore Â· MELV Framework Â· Laurence W. Evans (ORCID {ORCID}) Â·
    Bifurcation threshold i = 0.9995 Â± 0.029 Â· Cooperation basin 78.0%/16.2%/5.8% Â·
    DOI {ZENODO_DOI}
  </div>

</div>
</body>
</html>"""

    return html


# â”€â”€ PUBLIC API â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def render_cert_pdf(
    report_dict: dict,
    assessment_scores: Optional[dict] = None,
) -> bytes:
    """
    Render a CertificationReport dict to PDF bytes.

    Uses weasyprint (HTML â†’ PDF). Falls back to a minimal reportlab
    PDF if weasyprint is unavailable.

    Args:
        report_dict:      Output of CertificationReport.to_dict()
        assessment_scores: Optional assessment_scores dict (for Îµ risk rows)

    Returns:
        bytes: PDF content
    """
    html_str = _render_html(report_dict, assessment_scores)

    try:
        from weasyprint import HTML as WP_HTML
        pdf = WP_HTML(string=html_str).write_pdf()
        return pdf
    except ImportError:
        return _fallback_reportlab(report_dict)


def _fallback_reportlab(report: dict) -> bytes:
    """Minimal reportlab fallback â€” plain text PDF."""
    try:
        from reportlab.pdfgen import canvas as rl_canvas
        buf = io.BytesIO()
        c = rl_canvas.Canvas(buf)
        c.setFont("Helvetica-Bold", 14)
        c.drawString(60, 780, "MELVcore Agent Certification Report")
        c.setFont("Helvetica", 11)
        y = 750
        lines = [
            f"Agent: {report.get('agent_id','')}",
            f"Run:   {report.get('run_id','')}",
            f"Verdict: {report.get('verdict','')}",
            f"CLS Score: {report.get('cls_score',0.0):.2f}",
            "",
            report.get("narrative", ""),
        ]
        for line in lines:
            c.drawString(60, y, line[:90])
            y -= 18
        c.save()
        return buf.getvalue()
    except Exception:
        raise RuntimeError("Neither weasyprint nor reportlab is available.")


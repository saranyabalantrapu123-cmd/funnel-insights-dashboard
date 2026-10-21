"""Static, self-contained HTML report (Plotly) written by `make demo`."""

from __future__ import annotations

import html
from datetime import date, datetime
from pathlib import Path

import duckdb
import plotly.io as pio

from . import charts
from . import metrics as m
from .config import PLATFORM_COLORS, PLATFORM_IDS, PLATFORM_LABELS
from .insights import Insight
from .narrator import Briefing

CSS = """
body{font-family:Inter,-apple-system,Segoe UI,Roboto,sans-serif;background:#f7f7f5;color:#0b0b0b;margin:0}
main{max-width:1180px;margin:0 auto;padding:28px 16px 60px}
h1{font-size:26px;margin:0 0 4px} h2{font-size:18px;margin:32px 0 10px}
.sub{color:#52514e;margin:0 0 18px} .card{background:#fff;border:1px solid #e7e6e2;border-radius:10px;padding:16px 18px;margin-bottom:14px}
.banner{background:#fff7e6;border:1px solid #f0d9a8;color:#5c4400;border-radius:8px;padding:8px 12px;font-size:14px;margin-bottom:16px}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px}
.kpi{background:#fff;border:1px solid #e7e6e2;border-radius:10px;padding:12px 14px}
.kpi .v{font-size:22px;font-weight:600} .kpi .l{font-size:12px;color:#52514e;text-transform:uppercase;letter-spacing:.04em}
table{border-collapse:collapse;width:100%;font-size:13px} th,td{text-align:left;padding:7px 8px;border-bottom:1px solid #eeede9}
th{color:#52514e;font-weight:600} .dot{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:6px}
.tag{font-size:11px;padding:2px 7px;border-radius:10px;background:#eef0f3;color:#333}
ol li{margin-bottom:6px} .foot{color:#52514e;font-size:12px;margin-top:30px}
"""


def _kpi_html(k: dict[str, float]) -> str:
    tiles = [
        ("Sessions", f"{k['sessions']:,.0f}"), ("Login rate", f"{k['login_rate']:.1%}"),
        ("Feature use", f"{k['feature_use_rate']:.1%}"), ("7-day return", f"{k['return_7d_rate']:.1%}"),
        ("Commerce conv.", f"{k['commerce_conversion']:.2%}"), ("Orders", f"{k['orders']:,.0f}"),
        ("Revenue", f"${k['revenue']:,.0f}"), ("AOV", f"${k['aov']:,.2f}"),
    ]
    return '<div class="kpis">' + "".join(
        f'<div class="kpi"><div class="l">{lbl}</div><div class="v">{val}</div></div>' for lbl, val in tiles
    ) + "</div>"


def _insight_rows(insights: list[Insight]) -> str:
    rows = []
    for rank, i in enumerate(insights, start=1):
        d = i.to_dict()
        rows.append(
            f"<tr><td>{rank}</td><td><span class='dot' style='background:{PLATFORM_COLORS[i.platform]}'></span>"
            f"{html.escape(PLATFORM_LABELS[i.platform])}</td><td><span class='tag'>{i.type}</span></td>"
            f"<td>{html.escape(i.title)}</td><td>{d['start_date']}</td><td>{d['end_date'] or 'ongoing'}</td>"
            f"<td>{d['confidence']:.2f}</td><td>{d['priority']:,.0f}</td></tr>")
    return ("<table><tr><th>#</th><th>Platform</th><th>Type</th><th>Insight</th><th>Start</th><th>End</th>"
            "<th>Conf.</th><th>Priority</th></tr>" + "".join(rows) + "</table>")


def write_report(con: duckdb.DuckDBPyConnection, insights: list[Insight], briefing: Briefing,
                 start: date, end: date, out_path: Path, banner: str | None, usage: str | None) -> Path:
    f = m.Filters(start, end)
    funnels = {name: m.funnel(con, name, f) for name in ("commerce", "engagement")}
    daily = m.daily_metrics(con, f)
    figs = [
        ("Commerce funnel by platform", charts.funnel_comparison(funnels["commerce"], "commerce")),
        ("Engagement funnel by platform", charts.funnel_comparison(funnels["engagement"], "engagement")),
        ("Drop-off heatmap (platform × step)", charts.dropoff_heatmap(funnels)),
    ]
    trend_metrics: list[str] = []
    for i in briefing.ranked:
        if i.metric != "sessions" and i.metric not in trend_metrics and i.type != "platform_gap":
            trend_metrics.append(i.metric)
    for metric in trend_metrics[:4]:
        figs.append((f"Daily trend · {metric.replace('_', ' ')}",
                     charts.trend_chart(daily, metric, insights, PLATFORM_IDS)))
    figs.append(("Daily sessions (tracking health)", charts.trend_chart(daily, "sessions", insights, PLATFORM_IDS)))
    figs.append(("Weekly revenue", charts.revenue_chart(m.revenue(con, f, by_day=True))))

    fig_html = []
    for n, (title, fig) in enumerate(figs):
        fig_html.append(f'<h2>{html.escape(title)}</h2><div class="card">'
                        + pio.to_html(fig, include_plotlyjs="inline" if n == 0 else False, full_html=False,
                                      config={"displaylogo": False}) + "</div>")

    look = "".join(f"<li><b>{html.escape(x['title'])}</b><br>{html.escape(x['action'])}</li>"
                   for x in briefing.where_to_look)
    warn = "".join(f'<div class="banner">{html.escape(w)}</div>' for w in briefing.warnings)
    doc = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Funnel Insights Report</title>
<style>{CSS}</style></head><body><main>
<h1>Funnel Insights Report</h1>
<p class="sub">Nimbus Wireless &amp; Nimbus Prepaid · App + Web · {start} to {end} · generated {datetime.now():%Y-%m-%d %H:%M}</p>
{f'<div class="banner">{html.escape(banner)}</div>' if banner else ''}{warn}
{_kpi_html(m.kpis(con, f))}
<h2>Executive summary <span class="tag">{html.escape(briefing.source)}</span></h2>
<div class="card">{html.escape(briefing.executive_summary)}</div>
<h2>Where to look first</h2><div class="card"><ol>{look}</ol></div>
<h2>All insights (ranked)</h2><div class="card">{_insight_rows(briefing.ranked)}</div>
{''.join(fig_html)}
<p class="foot">{html.escape(usage or '')} · Synthetic data for fictional brands; personal portfolio project.</p>
</main></body></html>"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(doc, encoding="utf-8")
    return out_path

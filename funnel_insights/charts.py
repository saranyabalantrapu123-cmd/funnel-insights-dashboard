"""Plotly figure builders shared by the Streamlit app and the static HTML report.

Color rule: one fixed color per platform (config.PLATFORM_COLORS) in every chart;
anomaly markers are dark ink (not a platform color); tracking outages are grey bands.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .config import FUNNELS, PLATFORM_COLORS, PLATFORM_IDS, PLATFORM_LABELS, STEP_LABELS
from .insights import Insight

INK = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e7e6e2"
SURFACE = "#ffffff"
FONT = "Inter, -apple-system, Segoe UI, Roboto, sans-serif"
# Sequential single-hue blue ramp (light -> dark) for magnitude (drop-off heatmap).
SEQ_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


def _layout(fig: go.Figure, height: int, title: str | None = None) -> go.Figure:
    fig.update_layout(
        template="plotly_white", height=height, paper_bgcolor=SURFACE, plot_bgcolor=SURFACE,
        font=dict(family=FONT, size=13, color=INK),
        title=dict(text=title, font=dict(size=15, color=INK), x=0, xanchor="left") if title else None,
        margin=dict(l=10, r=10, t=50 if title else 20, b=10),
        hoverlabel=dict(bgcolor="white", font_size=12, font_family=FONT),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, font=dict(color=MUTED)),
    )
    fig.update_xaxes(gridcolor=GRID, linecolor=GRID, tickfont=dict(color=MUTED))
    fig.update_yaxes(gridcolor=GRID, linecolor=GRID, tickfont=dict(color=MUTED))
    return fig


def funnel_comparison(funnel_df: pd.DataFrame, funnel_name: str, height: int = 360) -> go.Figure:
    """Side-by-side funnels, one panel per platform, same step order and scale-free widths."""
    platforms = [p for p in PLATFORM_IDS if p in set(funnel_df.platform)]
    fig = make_subplots(rows=1, cols=max(1, len(platforms)), shared_yaxes=True, horizontal_spacing=0.03,
                        subplot_titles=[PLATFORM_LABELS[p] for p in platforms])
    for c, pid in enumerate(platforms, start=1):
        g = funnel_df[funnel_df.platform == pid].sort_values("step_order")
        labels = [STEP_LABELS[s] for s in g.step]
        text = [f"{e:.1%}<br>{n:,}" for n, e in zip(g["count"], g["e2e_conversion"])]
        fig.add_trace(go.Funnel(
            y=labels, x=g["e2e_conversion"] * 100, text=text, textinfo="text", textposition="auto",
            marker=dict(color=PLATFORM_COLORS[pid]), connector=dict(fillcolor="#eef0f3"),
            insidetextfont=dict(color="white", size=12), outsidetextfont=dict(color=INK, size=12),
            constraintext="none", name=PLATFORM_LABELS[pid], showlegend=False,
            customdata=list(zip(g["step_conversion"] * 100, g["drop_off"] * 100)),
            hovertemplate="<b>%{y}</b><br>Reached: %{text}<br>Step conversion: %{customdata[0]:.1f}%"
                          "<br>Drop-off: %{customdata[1]:.1f}%<extra>" + PLATFORM_LABELS[pid] + "</extra>",
        ), row=1, col=c)
    fig.update_annotations(font=dict(size=13, color=INK))
    fig.update_xaxes(visible=False)
    return _layout(fig, height)


def trend_chart(daily: pd.DataFrame, metric: str, insights: list[Insight], platforms: list[str],
                height: int = 380) -> go.Figure:
    """Daily rate per platform with anomaly markers and grey bands for tracking outages."""
    d = daily[(daily.metric == metric) & (daily.platform.isin(platforms))].copy()
    outages = [i for i in insights if i.type == "data_quality" and i.platform in platforms]
    is_rate = metric != "sessions"
    fig = go.Figure()
    for o in outages:
        end = o.end_date or d.date.max().date()
        fig.add_vrect(x0=pd.Timestamp(o.start_date) - pd.Timedelta(hours=12),
                      x1=pd.Timestamp(end) + pd.Timedelta(hours=12), fillcolor="#9a9a96", opacity=0.18,
                      line_width=0, annotation_text=f"tracking outage · {PLATFORM_LABELS[o.platform]}",
                      annotation_position="top left", annotation_xanchor="right",
                      annotation_font=dict(size=11, color=MUTED))
    for pid in [p for p in PLATFORM_IDS if p in platforms]:
        g = d[d.platform == pid].sort_values("date")
        y = g["rate"] * 100 if is_rate else g["numerator"]
        if is_rate:  # days inside a tracking outage are excluded, as in the detector
            out_days = {pd.Timestamp(x) for o in outages if o.platform == pid
                        for x in pd.date_range(o.start_date, o.end_date or g.date.max())}
            y = y.where(~g["date"].isin(out_days))
        fig.add_trace(go.Scatter(
            x=g["date"], y=y, mode="lines", name=PLATFORM_LABELS[pid],
            line=dict(color=PLATFORM_COLORS[pid], width=2), connectgaps=False,
            hovertemplate="%{x|%b %d}: %{y:.1f}" + ("%" if is_rate else "") + "<extra>" + PLATFORM_LABELS[pid] + "</extra>",
        ))
    marks = [i for i in insights if i.metric == metric and i.platform in platforms
             and i.type in ("conversion_anomaly", "engagement_anomaly")]
    if marks:
        xs, ys, texts, symbols = [], [], [], []
        for i in marks:
            row = d[(d.platform == i.platform) & (d.date == pd.Timestamp(i.start_date))]
            if row.empty:
                continue
            xs.append(pd.Timestamp(i.start_date))
            ys.append(float(row["rate"].iloc[0] * 100 if is_rate else row["numerator"].iloc[0]))
            texts.append(i.title)
            symbols.append("triangle-down" if i.direction == "down" else "triangle-up")
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="markers", name="Detected anomaly (start)", text=texts,
            marker=dict(symbol=symbols, size=13, color=INK, line=dict(color="white", width=2)),
            hovertemplate="<b>%{text}</b><br>%{x|%b %d}<extra></extra>",
        ))
    unit = " (%)" if is_rate else ""
    fig.update_yaxes(title=dict(text=STEP_LABELS.get(metric, metric) + unit, font=dict(color=MUTED, size=12)))
    if not d.empty:
        fig.update_xaxes(range=[d.date.min() - pd.Timedelta(days=1), d.date.max() + pd.Timedelta(days=1)])
    fig.update_xaxes(tickformat="%b %d")
    return _layout(fig, height)


def dropoff_heatmap(funnel_frames: dict[str, pd.DataFrame], height: int = 300) -> go.Figure:
    """Platform x step drop-off (%), both funnels side by side on one shared color scale."""
    cols, z_cols = [], []
    platforms: list[str] = []
    for name, df in funnel_frames.items():
        if df.empty:
            continue
        platforms = [p for p in PLATFORM_IDS if p in set(df.platform)]
        for s in FUNNELS[name][1:]:
            cols.append(f"{name.capitalize()}<br>{STEP_LABELS[s.key]}")
            z_cols.append([float(df[(df.platform == p) & (df.step == s.key)].drop_off.iloc[0] * 100)
                           for p in platforms])
    z = [list(r) for r in zip(*z_cols)]
    fig = go.Figure(go.Heatmap(
        z=z, x=cols, y=[PLATFORM_LABELS[p] for p in platforms],
        colorscale=list(zip([i / (len(SEQ_BLUE) - 1) for i in range(len(SEQ_BLUE))], SEQ_BLUE)),
        zmin=0, zmax=100, xgap=2, ygap=2, text=[[f"{v:.0f}%" for v in r] for r in z],
        texttemplate="%{text}", textfont=dict(size=13),
        colorbar=dict(title=dict(text="Drop-off", font=dict(color=MUTED, size=12)), ticksuffix="%",
                      thickness=10, len=0.9),
        hovertemplate="%{y}<br>%{x}: %{z:.1f}% drop-off<extra></extra>",
    ))
    fig.update_yaxes(autorange="reversed", showgrid=False)
    fig.update_xaxes(showgrid=False, side="top", tickangle=0)
    return _layout(fig, height)


def cohort_chart(cohorts: pd.DataFrame, height: int = 320) -> go.Figure:
    fig = go.Figure()
    for pid in [p for p in PLATFORM_IDS if p in set(cohorts.platform)]:
        g = cohorts[cohorts.platform == pid]
        fig.add_trace(go.Scatter(
            x=g.cohort_week, y=g.return_rate * 100, mode="lines+markers", name=PLATFORM_LABELS[pid],
            line=dict(color=PLATFORM_COLORS[pid], width=2), marker=dict(size=8),
            hovertemplate="Week of %{x|%b %d}: %{y:.1f}%<extra>" + PLATFORM_LABELS[pid] + "</extra>",
        ))
    fig.update_yaxes(title=dict(text="7-day return of new engaged users (%)", font=dict(color=MUTED, size=12)),
                     rangemode="tozero")
    return _layout(fig, height)


def revenue_chart(rev_daily: pd.DataFrame, height: int = 320) -> go.Figure:
    fig = go.Figure()
    for pid in [p for p in PLATFORM_IDS if p in set(rev_daily.platform)]:
        g = rev_daily[rev_daily.platform == pid].copy()
        g["date"] = pd.to_datetime(g["date"])
        by_week = g.set_index("date")["revenue"].resample("W-MON", label="left", closed="left")
        weekly = by_week.sum()[by_week.count() == 7]  # full weeks only, so edges don't look like drops
        fig.add_trace(go.Bar(
            x=weekly.index, y=weekly.values, name=PLATFORM_LABELS[pid], marker_color=PLATFORM_COLORS[pid],
            hovertemplate="Week of %{x|%b %d}: $%{y:,.0f}<extra>" + PLATFORM_LABELS[pid] + "</extra>",
        ))
    fig.update_layout(barmode="group", bargap=0.25, bargroupgap=0.08)
    fig.update_yaxes(title=dict(text="Weekly revenue ($)", font=dict(color=MUTED, size=12)), tickprefix="$")
    return _layout(fig, height)

"""Funnel Insights Dashboard — Streamlit UI.

Run: streamlit run app.py   (after `make data`)
"""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import streamlit as st

from funnel_insights import charts, db
from funnel_insights import metrics as m
from funnel_insights.config import (BRANDS, MONITORED_METRICS, PLATFORM_COLORS, PLATFORM_IDS,
                                    PLATFORM_LABELS, PLATFORMS, STEP_LABELS)
from funnel_insights.detection import run_insight_engine
from funnel_insights.insights import Insight
from funnel_insights.llm import OFFLINE_BANNER, get_llm
from funnel_insights.narrator import Briefing, build_briefing
from funnel_insights.qa import ask

st.set_page_config(page_title="Funnel Insights Dashboard", layout="wide", initial_sidebar_state="expanded")

st.markdown(
    """
    <style>
      .block-container {padding-top: 1.6rem; padding-bottom: 2rem; max-width: 1500px;}
      h1 {font-size: 1.75rem !important; margin-bottom: 0 !important;}
      h3 {font-size: 1.08rem !important; margin-top: .4rem !important;}
      div[data-testid="stMetric"] {background: #fff; border: 1px solid #e7e6e2; border-radius: 10px;
                                   padding: 10px 14px;}
      div[data-testid="stMetricLabel"] p {font-size: .78rem; color: #52514e; text-transform: uppercase;
                                          letter-spacing: .04em;}
      div[data-testid="stMetricValue"] {font-size: 1.45rem;}
      .legend-chip {display:inline-flex; align-items:center; margin-right:16px; font-size:.86rem; color:#333;}
      .legend-chip span {width:11px; height:11px; border-radius:50%; display:inline-block; margin-right:6px;}
      .insight-card {background:#fff; border:1px solid #e7e6e2; border-radius:10px; padding:12px 16px;
                     margin-bottom:10px; font-size:.92rem; line-height:1.45;}
      .look-card {min-height: 118px;}
      .src-tag {font-size:.72rem; background:#eef0f3; border-radius:8px; padding:2px 8px; color:#333;}
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------- cached data


@st.cache_resource
def get_con():
    return db.connect()


@st.cache_data(show_spinner="Running insight engine…")
def get_insights() -> list[Insight]:
    return run_insight_engine(get_con())


@st.cache_data(show_spinner="Writing executive summary…")
def get_briefing(_insights: list[Insight], start: date, end: date, key: str) -> Briefing:
    return build_briefing(_insights, start, end, get_llm())


try:
    con = get_con()
except FileNotFoundError as exc:
    st.error(f"{exc}")
    st.stop()

llm = get_llm()
data_start, data_end = m.date_bounds(con)
insights = get_insights()

# --------------------------------------------------------------------------- sidebar filters

with st.sidebar:
    st.header("Filters")
    picked = st.date_input("Date range", value=(data_start, data_end), min_value=data_start,
                           max_value=data_end)
    start, end = (picked if isinstance(picked, tuple) and len(picked) == 2 else (data_start, data_end))
    brands = st.multiselect("Brand", BRANDS, default=BRANDS)
    options = [p for p in PLATFORM_IDS if PLATFORMS[p].brand in brands]
    platforms = st.multiselect("Platform", options, default=options,
                               format_func=lambda p: PLATFORM_LABELS[p].replace("Nimbus ", ""))
    funnel_name = st.radio("Funnel", ["commerce", "engagement"], format_func=str.capitalize, horizontal=True)
    st.divider()
    st.caption(f"Data: {data_start} → {data_end} · synthetic sessions for fictional brands.")
    st.caption(f"LLM: `{llm.model}`")

if not platforms:
    st.warning("Select at least one platform.")
    st.stop()

flt = m.Filters(start, end, tuple(platforms))
visible = [i for i in insights if i.platform in platforms
           and i.start_date <= end and (i.end_date or data_end) >= start]

# --------------------------------------------------------------------------- header + KPIs

st.title("Funnel Insights Dashboard")
st.markdown(
    "End-to-end engagement and commerce funnels for **Nimbus Wireless** and **Nimbus Prepaid** across app "
    "and web.<br>" + "".join(
        f"<span class='legend-chip'><span style='background:{PLATFORM_COLORS[p]}'></span>{PLATFORM_LABELS[p]}</span>"
        for p in platforms),
    unsafe_allow_html=True,
)
if llm.offline:
    st.info(OFFLINE_BANNER)

k = m.kpis(con, flt)
span = (end - start).days + 1
prev_start = start - timedelta(days=span)
kp = m.kpis(con, m.Filters(prev_start, start - timedelta(days=1), tuple(platforms))) if prev_start >= data_start else None


def _delta(key: str, pct_points: bool = False) -> str | None:
    if not kp or pd.isna(kp[key]) or not kp[key]:
        return None
    if pct_points:
        return f"{(k[key] - kp[key]) * 100:+.1f} pp"
    return f"{k[key] / kp[key] - 1:+.1%}"


tiles = [
    ("Sessions", f"{k['sessions'] / 1e6:.2f}M" if k["sessions"] >= 1e6 else f"{k['sessions']:,.0f}",
     _delta("sessions")),
    ("Login rate", f"{k['login_rate']:.1%}", _delta("login_rate", True)),
    ("Feature use", f"{k['feature_use_rate']:.1%}", _delta("feature_use_rate", True)),
    ("7-day return", f"{k['return_7d_rate']:.1%}" if pd.notna(k["return_7d_rate"]) else "n/a",
     _delta("return_7d_rate", True)),
    ("View→order", f"{k['commerce_conversion']:.2%}", _delta("commerce_conversion", True)),
    ("Orders", f"{k['orders']:,.0f}", _delta("orders")),
    ("Revenue", f"${k['revenue'] / 1e6:,.2f}M", _delta("revenue")),
    ("AOV", f"${k['aov']:,.2f}", _delta("aov")),
]
for col, (label, value, delta) in zip(st.columns(len(tiles)), tiles):
    col.metric(label, value, delta)
st.caption("Deltas compare with the preceding period of equal length." if kp else
           "Narrow the date range to see deltas vs the preceding period.")

# --------------------------------------------------------------------------- AI briefing

briefing = get_briefing(insights, data_start, data_end, llm.model)
st.subheader("AI insights")
st.markdown(f"<div class='insight-card'><span class='src-tag'>ranked by: {briefing.source}</span>&nbsp; "
            f"{briefing.executive_summary}</div>", unsafe_allow_html=True)
for w in briefing.warnings:
    st.warning(w)
st.markdown("**Where to look first**")
for col, (n, x) in zip(st.columns(max(1, len(briefing.where_to_look))),
                       enumerate(briefing.where_to_look, start=1)):
    col.markdown(f"<div class='insight-card look-card'><b>{n}. {x['title']}</b><br>"
                 f"<span style='color:#52514e'>{x['action']}</span></div>", unsafe_allow_html=True)

# --------------------------------------------------------------------------- funnel comparison

funnel_df = m.funnel(con, funnel_name, flt)
st.subheader(f"{funnel_name.capitalize()} funnel · side by side")
st.plotly_chart(charts.funnel_comparison(funnel_df, funnel_name, height=330), width="stretch",
                config={"displayModeBar": False})

# --------------------------------------------------------------------------- trend + heatmap

st.subheader("Daily conversion with detected anomalies")
ranked_metrics = [i.metric for i in briefing.ranked if i in visible and i.metric in MONITORED_METRICS]
default_metric = ranked_metrics[0] if ranked_metrics else "payment"
choices = [*MONITORED_METRICS, "sessions"]
sel_col, _ = st.columns([1, 3])
metric = sel_col.selectbox("Metric", choices, index=choices.index(default_metric),
                           format_func=lambda x: STEP_LABELS.get(x, x), label_visibility="collapsed")
daily = m.daily_metrics(con, flt, [metric])
st.plotly_chart(charts.trend_chart(daily, metric, visible, platforms, height=380), width="stretch",
                config={"displayModeBar": False})
st.caption("▲▼ = start of a detected anomaly · grey band = tracking outage (excluded from conversion alerts).")

st.subheader("Drop-off heatmap")
frames = {f: m.funnel(con, f, flt) for f in ("commerce", "engagement")}
st.plotly_chart(charts.dropoff_heatmap(frames, height=90 + 48 * len(platforms)), width="stretch",
                config={"displayModeBar": False})
st.caption("Share of sessions lost at each step (vs the previous step). Darker = bigger leak.")

# --------------------------------------------------------------------------- insights table

st.subheader(f"All insights in view ({len(visible)})")
TYPE_SHORT = {"conversion_anomaly": "conversion", "engagement_anomaly": "engagement",
              "data_quality": "data quality", "platform_gap": "platform gap"}
rows = [{
    "Insight": i.title, "Type": TYPE_SHORT[i.type], "Platform": PLATFORM_LABELS[i.platform],
    "Start": i.start_date.isoformat(), "End": i.end_date.isoformat() if i.end_date else "ongoing",
    "First alert": i.first_alert_date.isoformat(), "Confidence": round(i.confidence, 2),
} for i in briefing.ranked if i in visible]
st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch", height=38 + 35 * max(1, len(rows)),
             column_config={"Insight": st.column_config.TextColumn(width="large"),
                            "Confidence": st.column_config.ProgressColumn(min_value=0, max_value=1, format="%.2f")})

# --------------------------------------------------------------------------- ask the dashboard

st.subheader("Ask the dashboard")
examples = ["Which platform has the worst payment conversion?", "Show the commerce funnel for Prepaid web",
            "What anomalies are there on Prepaid web?", "What was revenue last 30 days?"]


def _set_q(q: str) -> None:
    st.session_state["question"] = q


ex_cols = st.columns(len(examples))
for c, q in zip(ex_cols, examples):
    c.button(q, on_click=_set_q, args=(q,), width="stretch")
question = st.text_input("Question", key="question", placeholder="e.g. Which platform has the lowest add-to-cart rate?",
                         label_visibility="collapsed")
if question:
    res = ask(question, con, insights, llm)
    st.markdown(f"<div class='insight-card'>{res.answer.replace(chr(10), '<br>')}</div>", unsafe_allow_html=True)
    if not res.grounded:
        st.warning(f"Guardrail: these numbers were not found in any tool result: {', '.join(res.ungrounded_numbers)}")
    with st.expander(f"Tool calls ({len(res.tool_calls)}) — every number above comes from these"):
        for t in res.tool_calls:
            st.markdown(f"`{t['tool']}({', '.join(f'{a}={v!r}' for a, v in t['arguments'].items())})`")
            st.json(t["result"], expanded=False)

# --------------------------------------------------------------------------- cohorts + revenue

tab1, tab2 = st.tabs(["7-day return by cohort", "Revenue"])
with tab1:
    st.plotly_chart(charts.cohort_chart(m.cohort_return(con, flt)), width="stretch", config={"displayModeBar": False})
with tab2:
    st.plotly_chart(charts.revenue_chart(m.revenue(con, flt, by_day=True)), width="stretch",
                    config={"displayModeBar": False})
    rev = m.revenue(con, flt)
    rev["platform"] = rev["platform"].map(PLATFORM_LABELS)
    st.dataframe(rev.rename(columns={"platform": "Platform", "orders": "Orders", "revenue": "Revenue ($)",
                                     "aov": "AOV ($)"}).round(2), hide_index=True, width="stretch")

st.caption("Personal portfolio project built with synthetic data for fictional brands. Not affiliated with any employer.")

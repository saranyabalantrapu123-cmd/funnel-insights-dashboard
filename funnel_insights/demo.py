"""`make demo`: run the insight engine + AI briefing, print to terminal, write the HTML report."""

from __future__ import annotations

import textwrap
import time

from . import db
from .config import PLATFORM_LABELS, REPORTS_DIR
from .detection import run_insight_engine
from .llm import OFFLINE_BANNER, get_llm
from .metrics import date_bounds
from .narrator import build_briefing
from .qa import ask

DEMO_QUESTIONS = [
    "Which platform has the worst payment conversion?",
    "What anomalies are there on Prepaid web?",
]


def main() -> None:
    t0 = time.time()
    con = db.connect()
    llm = get_llm()
    start, end = date_bounds(con)
    width = 100
    print("=" * width)
    print("FUNNEL INSIGHTS · Nimbus Wireless + Nimbus Prepaid · App + Web")
    print(f"Data window: {start} → {end}   ·   LLM: {llm.model}")
    if llm.offline:
        print(f"⚠  {OFFLINE_BANNER}")
    print("=" * width)

    insights = run_insight_engine(con)
    briefing = build_briefing(insights, start, end, llm)

    print(f"\nTOP INSIGHTS (ranked by {briefing.source})")
    print(f"{'#':>2}  {'type':<19} {'platform':<24} {'start':<11} {'conf':>5}  title")
    for n, i in enumerate(briefing.ranked[:8], start=1):
        print(f"{n:>2}  {i.type:<19} {PLATFORM_LABELS[i.platform]:<24} {i.start_date.isoformat():<11} "
              f"{i.confidence:>5.2f}  {i.title}")
    if len(briefing.ranked) > 8:
        print(f"    … {len(briefing.ranked) - 8} more in the report")

    print("\nEXECUTIVE SUMMARY")
    print(textwrap.fill(briefing.executive_summary, width=width))
    print("\nWHERE TO LOOK FIRST")
    for n, x in enumerate(briefing.where_to_look, start=1):
        print(textwrap.fill(f"{n}. {x['title']} → {x['action']}", width=width, subsequent_indent="   "))
    for w in briefing.warnings:
        print(f"! {w}")

    print("\nASK THE DASHBOARD")
    for q in DEMO_QUESTIONS:
        r = ask(q, con, insights, llm)
        tools = ", ".join(t["tool"] for t in r.tool_calls) or "none"
        print(f"Q: {q}\n   [tools: {tools} · grounded: {r.grounded}]")
        print(textwrap.indent(r.answer, "   "))

    from .report import write_report  # plotly import kept off the fast path above

    usage = "offline mock mode: no tokens used, $0.00" if llm.offline else llm.usage.summary()
    path = write_report(con, insights, briefing, start, end, REPORTS_DIR / "funnel_insights_report.html",
                        OFFLINE_BANNER if llm.offline else None, f"LLM usage: {usage}")
    print(f"\nHTML report → {path.relative_to(REPORTS_DIR.parent)}")
    print(f"LLM usage: {usage}")
    print(f"Done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()

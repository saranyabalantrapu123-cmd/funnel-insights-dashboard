"""AI layer, part 2: "Ask the dashboard" — a tool-calling Q&A agent.

The model may only obtain numbers by calling tools that run the SQL metrics layer
(`get_funnel`, `compare_platforms`, `get_anomalies`, `query_metric`). After it
answers, the grounding guardrail checks every number against the tool outputs.
Offline, the MockLLM plays the same loop with keyword rules: it picks a tool,
then templates an answer from the tool result, and says plainly when a question
is outside what offline mode can handle.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any

import duckdb

from . import metrics as m
from .config import FUNNELS, PLATFORM_IDS, PLATFORM_LABELS, STEP_LABELS
from .guardrails import check_grounding
from .insights import Insight
from .llm import LLM, Message, assistant_message
from .metrics import METRIC_SQL, Filters, date_bounds

MAX_TOOL_ROUNDS = 5
VALUE_METRICS = ("orders", "revenue", "aov")
ALL_METRICS = sorted([*METRIC_SQL, *VALUE_METRICS])

SYSTEM_PROMPT = f"""You answer questions about a funnel analytics dashboard covering four platforms:
{", ".join(f"{k} ({v})" for k, v in PLATFORM_LABELS.items())}.
Rules:
- You MUST NOT compute, estimate or recall numbers. Every number in your answer must be copied
  from a tool result. If you need a number, call a tool. Do not do arithmetic on tool results.
- Rates in tool results are already percentages (e.g. 42.1 means 42.1%).
- If the tools cannot answer the question, say so plainly.
- Keep answers to 2-5 sentences; name the platform and date window you used."""


def _tool(name: str, description: str, props: dict, required: list[str] | None = None) -> dict:
    return {"type": "function", "function": {
        "name": name, "description": description,
        "parameters": {"type": "object", "properties": props, "required": required or []}}}


_DATE = {"type": "string", "description": "ISO date YYYY-MM-DD (optional; defaults to full data range)"}
_PLATFORM = {"type": "string", "enum": PLATFORM_IDS}
TOOL_SPECS = [
    _tool("get_funnel", "Funnel counts, step conversion, end-to-end conversion and drop-off per platform.",
          {"funnel": {"type": "string", "enum": list(FUNNELS)}, "platform": _PLATFORM,
           "start": _DATE, "end": _DATE}, ["funnel"]),
    _tool("compare_platforms", "Compare one metric across all four platforms over a window.",
          {"metric": {"type": "string", "enum": ALL_METRICS}, "start": _DATE, "end": _DATE}, ["metric"]),
    _tool("get_anomalies", "Insights from the deterministic engine (anomalies, data-quality issues, gaps).",
          {"platform": _PLATFORM, "metric": {"type": "string"},
           "type": {"type": "string", "enum": ["conversion_anomaly", "engagement_anomaly",
                                               "data_quality", "platform_gap"]},
           "limit": {"type": "integer"}}),
    _tool("query_metric", "Value of one metric, pooled and per platform, at total/week/day grain.",
          {"metric": {"type": "string", "enum": ALL_METRICS}, "platform": _PLATFORM,
           "start": _DATE, "end": _DATE, "grain": {"type": "string", "enum": ["total", "week", "day"]}},
          ["metric"]),
]


# --------------------------------------------------------------------------- tools


class DashboardTools:
    """Tool implementations; every number comes from the SQL metrics layer."""

    def __init__(self, con: duckdb.DuckDBPyConnection, insights: list[Insight]):
        self.con = con
        self.insights = insights
        self.min_date, self.max_date = date_bounds(con)

    def _window(self, start: str | None, end: str | None) -> tuple[date, date]:
        s = date.fromisoformat(start) if start else self.min_date
        e = date.fromisoformat(end) if end else self.max_date
        return max(s, self.min_date), min(e, self.max_date)

    @staticmethod
    def _fmt(metric: str, num: float, den: float) -> dict:
        if metric == "sessions":
            return {"value": int(num)}
        return {"rate_pct": round(100 * num / den, 1) if den else None,
                "numerator": int(num), "denominator": int(den)}

    def get_funnel(self, funnel: str = "commerce", platform: str | None = None,
                   start: str | None = None, end: str | None = None) -> dict:
        s, e = self._window(start, end)
        platforms = (platform,) if platform else tuple(PLATFORM_IDS)
        df = m.funnel(self.con, funnel, Filters(s, e, platforms))
        out: dict[str, Any] = {"funnel": funnel, "window": [s.isoformat(), e.isoformat()], "platforms": {}}
        for pid, g in df.groupby("platform"):
            steps = [{"step": r.step, "count": int(r["count"]),
                      "step_conversion_pct": round(100 * r.step_conversion, 1),
                      "e2e_conversion_pct": round(100 * r.e2e_conversion, 1),
                      "drop_off_pct": round(100 * r.drop_off, 1)} for _, r in g.iterrows()]
            worst = max(steps[1:], key=lambda x: x["drop_off_pct"])
            out["platforms"][pid] = {"steps": steps, "biggest_drop_off_step": worst["step"],
                                     "biggest_drop_off_pct": worst["drop_off_pct"]}
        return out

    def _value_by_platform(self, metric: str, f: Filters) -> dict[str, dict]:
        if metric in VALUE_METRICS:
            rev = m.revenue(self.con, f)
            return {r.platform: {"value": round(float(getattr(r, metric)), 2 if metric == "aov" else 0)}
                    for r in rev.itertuples()}
        df = m.metric_by_platform(self.con, metric, f)
        return {r.platform: self._fmt(metric, r.numerator, r.denominator) for r in df.itertuples()}

    def compare_platforms(self, metric: str, start: str | None = None, end: str | None = None) -> dict:
        if metric not in ALL_METRICS:
            return {"error": f"unknown metric {metric!r}", "valid_metrics": ALL_METRICS}
        s, e = self._window(start, end)
        rows = self._value_by_platform(metric, Filters(s, e))
        key = "value" if metric in (*VALUE_METRICS, "sessions") else "rate_pct"
        ranked = sorted(rows, key=lambda p: rows[p][key] or 0, reverse=True)
        return {"metric": metric, "window": [s.isoformat(), e.isoformat()], "by_platform": rows,
                "highest": ranked[0], "lowest": ranked[-1], "unit": "%" if key == "rate_pct" else "count/$"}

    def get_anomalies(self, platform: str | None = None, metric: str | None = None,
                      type: str | None = None, limit: int = 5) -> dict:  # noqa: A002 - tool arg name
        sel = [i for i in self.insights
               if (not platform or i.platform == platform) and (not metric or i.metric == metric)
               and (not type or i.type == type)]
        keep = ("id", "title", "type", "platform", "metric", "direction", "shape", "magnitude_pct",
                "baseline_pct", "observed_pct", "start_date", "end_date", "confidence", "impact")
        return {"count": len(sel),
                "insights": [{k: d[k] for k in keep} for d in (i.to_dict() for i in sel[:limit])]}

    def query_metric(self, metric: str, platform: str | None = None, start: str | None = None,
                     end: str | None = None, grain: str = "total") -> dict:
        if metric not in ALL_METRICS:
            return {"error": f"unknown metric {metric!r}", "valid_metrics": ALL_METRICS}
        s, e = self._window(start, end)
        platforms = (platform,) if platform else tuple(PLATFORM_IDS)
        f = Filters(s, e, platforms)
        out: dict[str, Any] = {"metric": metric, "window": [s.isoformat(), e.isoformat()],
                               "platforms": list(platforms), "grain": grain}
        value_sql = {
            "orders": "sum(order_complete::INT), NULL",
            "revenue": "coalesce(sum(order_value), 0), NULL",
            "aov": "sum(order_value) / nullif(sum(order_complete::INT), 0), NULL",
        }
        sel = value_sql.get(metric) or ", ".join(METRIC_SQL[metric])
        where, params = f.where()
        bucket = {"total": "NULL", "week": "date_trunc('week', date)", "day": "date"}[grain]
        rows = self.con.execute(
            f"SELECT CAST({bucket} AS DATE) AS b, {sel} FROM sessions_enriched WHERE {where} "
            f"GROUP BY 1 ORDER BY 1", params).fetchall()
        series = []
        for b, a, d in rows[-31:]:
            val = ({"value": round(float(a or 0), 2 if metric == "aov" else 0)} if metric in value_sql
                   else self._fmt(metric, a, d))
            series.append({"period": b.isoformat() if b else "total", **val})
        out["values"] = series
        return out

    def call(self, name: str, args: dict) -> dict:
        fn = {"get_funnel": self.get_funnel, "compare_platforms": self.compare_platforms,
              "get_anomalies": self.get_anomalies, "query_metric": self.query_metric}.get(name)
        if fn is None:
            return {"error": f"unknown tool {name!r}"}
        try:
            return fn(**args)
        except (TypeError, ValueError, KeyError) as exc:
            return {"error": str(exc)}


# --------------------------------------------------------------------------- mock (offline) rules

_PLATFORM_HINTS = {"prepaid": "prepaid", "wireless": "wireless"}
_CHANNEL_HINTS = {"app": "app", "mobile": "app", "web": "web", "website": "web", "site": "web", "desktop": "web"}
_METRIC_ALIASES: list[tuple[str, str]] = [
    ("add to cart", "add_to_cart"), ("add-to-cart", "add_to_cart"), ("cart", "add_to_cart"),
    ("checkout", "checkout_start"), ("order complete", "order_complete"), ("completion", "order_complete"),
    ("pay bill", "pay_bill"), ("bill", "pay_bill"), ("payment", "payment"),
    ("product view", "product_view"), ("browse", "product_view"),
    ("ai chat", "ai_chat"), ("assistant", "ai_chat"), ("chat", "ai_chat"),
    ("view usage", "view_usage"), ("usage", "view_usage"),
    ("log in", "login"), ("login", "login"), ("sign in", "login"), ("sign-in", "login"),
    ("return", "return_7d"), ("retention", "return_7d"), ("come back", "return_7d"),
    ("feature", "feature_use"), ("aov", "aov"), ("order value", "aov"), ("basket", "aov"),
    ("revenue", "revenue"), ("sales", "revenue"), ("orders", "orders"),
    ("sessions", "sessions"), ("traffic", "sessions"), ("visits", "sessions"),
]
_ANOMALY_WORDS = ("anomal", "issue", "wrong", "alert", "problem", "incident", "outage", "broken",
                  "spike", "unusual", "change", "drop ", "dropped", "tracking")
_COMPARE_WORDS = ("compare", "which platform", "worst", "best", "highest", "lowest", " vs", "versus", "across")

OFFLINE_HELP = (
    "Offline demo mode can't answer that one — it only understands a few question patterns. Try: "
    "\"Show the commerce funnel for Nimbus Prepaid web\", \"Which platform has the worst payment "
    "conversion?\", \"What anomalies are there on Prepaid web?\", \"What was revenue last week?\". "
    "Set ANTHROPIC_API_KEY for free-form questions."
)


def _parse_platform(q: str) -> str | None:
    brand = next((v for k, v in _PLATFORM_HINTS.items() if k in q), None)
    channel = next((v for k, v in _CHANNEL_HINTS.items() if re.search(rf"\b{k}\b", q)), None)
    return f"{brand}_{channel}" if brand and channel else None


def _parse_metric(q: str) -> str | None:
    return next((metric for alias, metric in _METRIC_ALIASES if alias in q), None)


def _parse_window(q: str, max_date: date) -> dict:
    m = re.search(r"last (\d+) days", q)
    days = int(m.group(1)) if m else 7 if "last week" in q else 30 if "last month" in q else None
    return {"start": (max_date - timedelta(days=days - 1)).isoformat(), "end": max_date.isoformat()} if days else {}


def plan_tool_call(question: str, max_date: date) -> tuple[str, dict] | None:
    """Keyword rules mapping a question to one tool call (None = can't answer offline)."""
    q = " " + question.lower().strip() + " "
    platform, metric, window = _parse_platform(q), _parse_metric(q), _parse_window(q, max_date)
    plat = {"platform": platform} if platform else {}
    if "funnel" in q or "drop-off" in q or "dropoff" in q or "drop off" in q:
        engaged = any(w in q for w in ("engagement", "login", "return", "feature"))
        return "get_funnel", {"funnel": "engagement" if engaged else "commerce", **plat, **window}
    if any(w in q for w in _COMPARE_WORDS) and metric:
        return "compare_platforms", {"metric": metric, **window}
    if any(w in q for w in _ANOMALY_WORDS) or q.strip().startswith("why"):
        return "get_anomalies", {**plat, **({"metric": metric} if metric and metric not in VALUE_METRICS else {})}
    if metric:
        return "query_metric", {"metric": metric, **plat, **window, "grain": "total"}
    return None


def _label(pid: str) -> str:
    return PLATFORM_LABELS.get(pid, pid)


def render_answer(question: str, tool: str, args: dict, result: dict) -> str:
    """Template an answer quoting only numbers present in `result`."""
    if "error" in result:
        return f"The {tool} tool returned an error: {result['error']}."
    if tool == "get_funnel":
        w = result["window"]
        lines = [f"{result['funnel'].capitalize()} funnel, {w[0]} to {w[1]}:"]
        for pid, p in result["platforms"].items():
            last = p["steps"][-1]
            lines.append(f"- {_label(pid)}: end-to-end {last['e2e_conversion_pct']}%; biggest drop-off at "
                         f"{STEP_LABELS[p['biggest_drop_off_step']].lower()} ({p['biggest_drop_off_pct']}% lost).")
        return "\n".join(lines)
    if tool == "compare_platforms":
        m, rows = result["metric"], result["by_platform"]
        key = "rate_pct" if result["unit"] == "%" else "value"
        unit = "%" if key == "rate_pct" else ""
        parts = [f"{_label(p)} {rows[p][key]}{unit}" for p in sorted(rows, key=lambda p: -(rows[p][key] or 0))]
        return (f"{STEP_LABELS.get(m, m)} by platform ({result['window'][0]} to {result['window'][1]}): "
                f"{'; '.join(parts)}. Highest: {_label(result['highest'])}; lowest: {_label(result['lowest'])}.")
    if tool == "get_anomalies":
        prefix = ("In offline mode I can show what changed and when, but not reason about causes. "
                  if question.lower().strip().startswith("why") else "")
        if not result["count"]:
            return prefix + "The insight engine found no insights matching that filter."
        lines = [prefix + f"The insight engine reports {result['count']} matching insight(s); top ones:"]
        for i in result["insights"]:
            end = "ongoing" if not i["end_date"] else f"until {i['end_date']}"
            unit = "" if i["metric"] == "sessions" else "%"
            lines.append(f"- {i['title']} — from {i['start_date']} ({end}), {i['baseline_pct']}{unit} → "
                         f"{i['observed_pct']}{unit}, confidence {i['confidence']}.")
        return "\n".join(lines)
    if tool == "query_metric":
        v = result["values"][0] if result["values"] else {}
        who = _label(args["platform"]) if args.get("platform") else "all 4 platforms"
        if "rate_pct" in v:
            val = f"{v['rate_pct']}%"
        elif result["metric"] == "aov":
            val = f"${v.get('value', 0):,.2f}"
        elif result["metric"] == "revenue":
            val = f"${v.get('value', 0):,.0f}"
        else:
            val = f"{v.get('value', 0):,.0f}"
        return (f"{STEP_LABELS.get(result['metric'], result['metric'])} for {who}, "
                f"{result['window'][0]} to {result['window'][1]}: {val}.")
    return OFFLINE_HELP


def _mock_qa(max_date: date):
    def respond(messages: list[Message]) -> Message:
        question = next(m["content"] for m in reversed(messages) if m["role"] == "user")
        tool_msgs = [m for m in messages if m["role"] == "tool"]
        if not tool_msgs:
            plan = plan_tool_call(question, max_date)
            if plan is None:
                return {"content": OFFLINE_HELP}
            return {"tool_calls": [{"id": "call_1", "name": plan[0], "arguments": plan[1]}]}
        call = next(m for m in messages if m.get("tool_calls"))["tool_calls"][0]
        return {"content": render_answer(question, call["function"]["name"],
                                         json.loads(call["function"]["arguments"]),
                                         json.loads(tool_msgs[-1]["content"]))}
    return respond


# --------------------------------------------------------------------------- agent loop


@dataclass
class QAResult:
    answer: str
    tool_calls: list[dict] = field(default_factory=list)
    grounded: bool = True
    ungrounded_numbers: list[str] = field(default_factory=list)
    offline: bool = False


def ask(question: str, con: duckdb.DuckDBPyConnection, insights: list[Insight], llm: LLM) -> QAResult:
    """Answer a question with tool calls; verify every number in the answer is grounded."""
    tools = DashboardTools(con, insights)
    messages: list[Message] = [
        {"role": "system", "content": SYSTEM_PROMPT + f"\nData covers {tools.min_date} to {tools.max_date}."},
        {"role": "user", "content": question},
    ]
    trace: list[dict] = []
    mock = _mock_qa(tools.max_date)
    answer = "I could not produce an answer within the tool-call budget."
    for _ in range(MAX_TOOL_ROUNDS):
        resp = llm.complete(messages, tools=TOOL_SPECS, mock_response=mock)
        if not resp["tool_calls"]:
            answer = resp["content"] or answer
            break
        messages.append(assistant_message(resp))
        for tc in resp["tool_calls"]:
            result = tools.call(tc["name"], tc["arguments"])
            trace.append({"tool": tc["name"], "arguments": tc["arguments"], "result": result})
            messages.append({"role": "tool", "tool_call_id": tc["id"], "content": json.dumps(result, default=str)})
    # Numbers may come from tool results, tool arguments, or the user's own question.
    grounded, bad = check_grounding(
        answer, [t["result"] for t in trace] + [t["arguments"] for t in trace] + [question])
    return QAResult(answer, trace, grounded, bad, llm.offline)


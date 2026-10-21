"""AI layer, part 1: rank insights and write the executive summary + "where to look first".

Real model: gets the insight payload (all numbers pre-computed) and returns JSON.
Its output is validated (known insight ids only, every number grounded in the
payload); anything invalid falls back to the deterministic briefing with a warning.
Offline: rule-based ranking (the engine's priority score) + templated narrative.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date

from .config import PLATFORM_LABELS, STEP_LABELS
from .detection import KNOWN_CALENDAR_EFFECTS, like_for_like_peer
from .guardrails import check_grounding
from .insights import Insight
from .llm import LLM, Message

MAX_INSIGHTS_TO_LLM = 15

SYSTEM_PROMPT = """You are a senior product-analytics lead writing for executives.
You receive JSON insights produced by a deterministic insight engine over four digital
platforms (Nimbus Wireless app/web, Nimbus Prepaid app/web). Rules:
- NEVER compute, estimate or invent numbers. Only quote numbers that appear in the JSON.
- data_quality insights are tracking problems, not customer behaviour; say so.
- platform_gap insights listing a related anomaly are symptoms of that anomaly.
- Prioritise ongoing problems with the largest value at stake (impact, priority, confidence).
Return ONLY a JSON object:
{"ranking": [insight ids, most important first],
 "executive_summary": "4-6 sentences",
 "where_to_look_first": [{"insight_id": id, "action": "one concrete next step for the owning team"}] (max 3)}"""


@dataclass
class Briefing:
    ranked: list[Insight]
    executive_summary: str
    where_to_look: list[dict]
    source: str  # "rule-based" or the model name
    warnings: list[str] = field(default_factory=list)


def _fmt_date(d: date | None) -> str:
    return d.strftime("%b %d").replace(" 0", " ") if d else "now"


def _money(x: float) -> str:
    return f"${abs(x) / 1000:.1f}k" if abs(x) >= 1000 else f"${abs(x):.0f}"


def _when(i: Insight) -> str:
    if i.type == "platform_gap":
        return f"over the last {(i.first_alert_date - i.start_date).days + 1} days"
    if i.ongoing:
        return f"since {_fmt_date(i.start_date)} (still ongoing)"
    if i.end_date == i.start_date:
        return f"on {_fmt_date(i.start_date)} (resolved)"
    return f"from {_fmt_date(i.start_date)} to {_fmt_date(i.end_date)} (resolved)"


def _headline_items(insights: list[Insight]) -> list[Insight]:
    """Insights worth headlining: drop gaps already explained by an anomaly."""
    return [i for i in insights if not (i.type == "platform_gap" and i.related)]


def _impact_phrase(i: Insight) -> str:
    imp = i.to_dict()["impact"]
    if "revenue_per_day" in imp:
        return f"≈{abs(imp['orders_per_day']):.1f} orders ({_money(imp['revenue_per_day'])}) per day"
    if "untracked_sessions_per_day" in imp:
        return f"≈{imp['untracked_sessions_per_day']:,.0f} sessions/day not tracked"
    return f"≈{abs(imp.get('sessions_per_day', 0)):,.0f} sessions/day affected"


def recommended_action(i: Insight) -> str:
    """Deterministic 'where to look' playbook keyed on insight type, metric and direction."""
    where = PLATFORM_LABELS[i.platform]
    since = _fmt_date(i.start_date)
    peer = PLATFORM_LABELS[i.peer_platform or like_for_like_peer(i.platform)]
    if i.type == "data_quality":
        return (f"Audit the analytics tag / SDK release for {where} around {since}; backfill if possible. "
                f"Treat the dip as missing data, not lost customers.")
    if i.type == "platform_gap":
        return (f"Walk the {STEP_LABELS[i.metric].lower()} step on {where} next to {peer} "
                f"(page layout, plan/price display, CTA visibility) and A/B the biggest UX difference.")
    playbook = {
        ("payment", "down"): f"Check payment-gateway and card-form errors on {where} since {since}, "
                             f"and what was deployed that day; compare decline codes with {peer}.",
        ("checkout_start", "up"): f"Find what shipped on {where} on {since} and evaluate porting it to the other platforms.",
        ("checkout_start", "down"): f"Review cart-to-checkout errors and recent cart changes on {where} since {since}.",
        ("add_to_cart", "down"): f"Check product-page availability, pricing and cart-button errors on {where} since {since}.",
        ("order_complete", "down"): f"Inspect order-submission failures (fraud checks, inventory) on {where} since {since}.",
        ("login", "down"): f"Check identity-provider / SSO error rates for {where} on the affected days.",
        ("ai_chat", "up"): f"Confirm the driver (campaign or release) and watch whether chat deflects care "
                           f"contacts or lifts 7-day return on {where}.",
    }
    return playbook.get(
        (i.metric, i.direction),
        f"Break {STEP_LABELS.get(i.metric, i.metric).lower()} on {where} down by device, OS version and "
        f"entry page from {since} to localise the change.",
    )


def rule_based_briefing(insights: list[Insight], start: date, end: date) -> Briefing:
    ranked = list(insights)  # already sorted by the engine's priority score
    items = _headline_items(ranked)
    problems = [i for i in items if i.direction == "down" and i.type != "data_quality"]
    upsides = [i for i in items if i.direction == "up"]
    dq = [i for i in items if i.type == "data_quality"]

    parts = [
        f"From {_fmt_date(start)} to {_fmt_date(end)} the insight engine flagged {len(items)} items across the "
        f"4 Nimbus platforms: {len(problems)} problem(s), {len(upsides)} upside(s) and "
        f"{len(dq)} data-quality issue(s)."
    ]
    if problems:
        p = problems[0]
        parts.append(f"Top priority: {p.title} {_when(p)}, {_impact_phrase(p)} at stake.")
    for p in problems[1:3]:
        parts.append(f"Also: {p.title} {_when(p)} ({_impact_phrase(p)}).")
    if upsides:
        u = upsides[0]
        parts.append(f"Upside: {u.title} since {_fmt_date(u.start_date)} — worth replicating if it holds.")
    if dq:
        d = dq[0]
        parts.append(f"Data quality: {d.title} on {_fmt_date(d.start_date)}; those days are excluded from "
                     f"conversion alerts, so it is not reported as a funnel drop.")
    look = [{"insight_id": i.id, "title": i.title, "action": recommended_action(i)} for i in items[:3]]
    return Briefing(ranked, " ".join(parts), look, "rule-based")


def insight_payload(insights: list[Insight], start: date, end: date) -> dict:
    return {
        "window": {"start": start.isoformat(), "end": end.isoformat()},
        "calendar_rules_applied": {k: v[0] for k, v in KNOWN_CALENDAR_EFFECTS.items()},
        "insights": [i.to_dict() for i in insights[:MAX_INSIGHTS_TO_LLM]],
    }


def build_briefing(insights: list[Insight], start: date, end: date, llm: LLM) -> Briefing:
    """Rank + summarise. Uses the LLM when available, validated; rule-based otherwise."""
    fallback = rule_based_briefing(insights, start, end)
    payload = insight_payload(insights, start, end)
    messages: list[Message] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(payload)},
    ]

    def mock(_: list[Message]) -> Message:
        return {"content": json.dumps({
            "ranking": [i.id for i in fallback.ranked],
            "executive_summary": fallback.executive_summary,
            "where_to_look_first": fallback.where_to_look,
        })}

    resp = llm.complete(messages, json_mode=True, mock_response=mock)
    if llm.offline:
        return fallback

    warnings: list[str] = []
    try:
        data = json.loads(_strip_fences(resp["content"] or ""))
        by_id = {i.id: i for i in insights}
        ranked_ids = [x for x in data.get("ranking", []) if x in by_id]
        ranked = [by_id[x] for x in ranked_ids] + [i for i in insights if i.id not in ranked_ids]
        summary = str(data["executive_summary"])
        look = []
        for item in data.get("where_to_look_first", [])[:3]:
            if item.get("insight_id") in by_id:
                look.append({"insight_id": item["insight_id"], "title": by_id[item["insight_id"]].title,
                             "action": str(item.get("action", ""))})
    except (ValueError, KeyError, TypeError) as exc:
        fallback.warnings.append(f"LLM output was not valid JSON ({exc}); showing rule-based briefing.")
        return fallback

    ok, bad = check_grounding(summary + " " + " ".join(x["action"] for x in look), [payload])
    if not ok:
        fallback.warnings.append(
            f"LLM summary contained numbers not found in the insight data ({', '.join(bad[:5])}); "
            f"showing rule-based briefing instead.")
        return fallback
    return Briefing(ranked, summary, look or fallback.where_to_look, llm.model, warnings)


def _strip_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text
        text = text.rsplit("```", 1)[0]
    return text

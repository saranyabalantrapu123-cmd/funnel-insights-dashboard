"""Insight model, business-impact estimates and the deterministic priority score.

The priority score is what the offline MockLLM uses to rank insights, and it is
passed to the real LLM as a hint so the model ranks with business context rather
than raw z-scores.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import date

from .config import PLATFORM_LABELS, PLATFORMS, STEP_LABELS, funnel_of

INSIGHT_TYPES = ("conversion_anomaly", "engagement_anomaly", "data_quality", "platform_gap")

# Value proxy used to put engagement changes on the same $/day scale as commerce
# changes when ranking. Deliberately conservative and documented in the README.
ENGAGEMENT_VALUE_PER_SESSION = 0.50
UNTRACKED_SESSION_VALUE = 0.25

METRIC_DEFINITIONS: dict[str, str] = {
    "sessions": "tracked sessions per day",
    "login": "login rate (logins / sessions)",
    "feature_use": "feature adoption (feature-using sessions / logged-in sessions)",
    "return_7d": "7-day return rate of engaged sessions",
    "pay_bill": "pay-bill usage (share of logged-in sessions)",
    "view_usage": "view-usage usage (share of logged-in sessions)",
    "ai_chat": "AI-assistant chat usage (share of logged-in sessions)",
    "product_view": "product-view rate (product views / sessions)",
    "add_to_cart": "add-to-cart conversion (carts / product views)",
    "checkout_start": "checkout-start conversion (checkouts / carts)",
    "payment": "payment conversion (payments / checkouts)",
    "order_complete": "order-complete conversion (orders / payments)",
}


@dataclass
class Insight:
    type: str
    platform: str
    metric: str
    direction: str  # "down" | "up"
    shape: str  # "level_shift" | "transient" | "outage" | "volume_dip" | "gap"
    magnitude: float  # relative change vs baseline / peer (e.g. -0.40 = 40% worse)
    baseline: float
    observed: float
    start_date: date
    end_date: date | None  # None = ongoing at the end of the data
    first_alert_date: date
    confidence: float  # 0..1
    daily_denominator: float
    peak_z: float | None = None
    peer_platform: str | None = None
    impact: dict[str, float] = field(default_factory=dict)
    priority: float = 0.0
    related: list[str] = field(default_factory=list)

    @property
    def id(self) -> str:
        return f"{self.type}:{self.platform}:{self.metric}:{self.start_date.isoformat()}"

    @property
    def brand(self) -> str:
        return PLATFORMS[self.platform].brand

    @property
    def funnel(self) -> str:
        return "engagement" if self.metric == "sessions" else funnel_of(self.metric)

    @property
    def ongoing(self) -> bool:
        return self.end_date is None

    @property
    def title(self) -> str:
        where = PLATFORM_LABELS[self.platform]
        step = STEP_LABELS.get(self.metric, self.metric)
        pct = f"{abs(self.magnitude) * 100:.0f}%"
        if self.type == "data_quality":
            return f"Tracking loss on {where}: volume {pct} below expected"
        if self.type == "platform_gap":
            return f"{step} on {where} is {pct} below {PLATFORM_LABELS[self.peer_platform or '']}"
        verb = "dropped" if self.direction == "down" else "rose"
        kind = "shift" if self.shape == "level_shift" else "spike" if self.direction == "up" else "dip"
        return f"{step} {verb} {pct} on {where} ({kind})"

    def to_dict(self) -> dict:
        d = asdict(self)
        d.update(id=self.id, brand=self.brand, funnel=self.funnel, title=self.title,
                 metric_definition=METRIC_DEFINITIONS.get(self.metric, self.metric),
                 platform_label=PLATFORM_LABELS[self.platform])
        for k in ("start_date", "end_date", "first_alert_date"):
            d[k] = d[k].isoformat() if d[k] else None
        d["magnitude_pct"] = round(self.magnitude * 100, 1)
        d["baseline_pct"] = round(self.baseline * 100, 1) if self.metric != "sessions" else round(self.baseline)
        d["observed_pct"] = round(self.observed * 100, 1) if self.metric != "sessions" else round(self.observed)
        d["confidence"] = round(self.confidence, 2)
        d["priority"] = round(self.priority, 1)
        d["impact"] = {k: round(v, 1) for k, v in self.impact.items()}
        if self.peak_z is not None:
            d["peak_z"] = round(self.peak_z, 1)
        return d


def value_per_day(ins: Insight) -> float:
    """Business value at stake per day ($), used for ranking."""
    if "revenue_per_day" in ins.impact:
        return abs(ins.impact["revenue_per_day"])
    if "untracked_sessions_per_day" in ins.impact:
        return ins.impact["untracked_sessions_per_day"] * UNTRACKED_SESSION_VALUE
    return abs(ins.impact.get("sessions_per_day", 0.0)) * ENGAGEMENT_VALUE_PER_SESSION


def priority_score(ins: Insight, data_end: date) -> float:
    """Deterministic priority: value at stake x confidence x direction x recency."""
    direction = 1.0 if ins.direction == "down" else 0.25  # problems before opportunities
    if ins.ongoing:
        recency = 1.5
    else:
        age = (data_end - ins.end_date).days  # type: ignore[operator]
        recency = 1.0 if age <= 7 else 0.6 if age <= 21 else 0.35
    # New regressions outrank chronic gaps (a gap has existed for the whole window).
    novelty = 0.6 if ins.type == "platform_gap" else 1.0
    return value_per_day(ins) * ins.confidence * direction * recency * novelty


def finalize(insights: list[Insight], data_end: date) -> list[Insight]:
    """Attach gap insights to matching anomalies, score, and sort by priority."""
    anomalies = {(i.platform, i.metric): i for i in insights
                 if i.type in ("conversion_anomaly", "engagement_anomaly") and i.direction == "down"}
    kept: list[Insight] = []
    for ins in insights:
        twin = anomalies.get((ins.platform, ins.metric)) if ins.type == "platform_gap" else None
        if twin is not None:
            twin.related.append(ins.id)  # gap is a symptom of the anomaly: fold it in
            ins.related.append(twin.id)
        kept.append(ins)
    for ins in kept:
        ins.priority = priority_score(ins, data_end)
        if ins.type == "platform_gap" and ins.related:
            ins.priority *= 0.25  # explained by an anomaly that is already ranked
    return sorted(kept, key=lambda i: i.priority, reverse=True)

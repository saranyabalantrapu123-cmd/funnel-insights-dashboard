"""Static configuration: platforms, funnels, colors and file paths.

Everything that describes *what* the dashboard measures lives here so the
generator, metrics layer, detectors and UI all agree on names and colors.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
REPORTS_DIR = ROOT / "reports"
SESSIONS_PARQUET = DATA_DIR / "sessions.parquet"
DUCKDB_PATH = DATA_DIR / "funnel.duckdb"
PLANTED_YAML = DATA_DIR / "planted_anomalies.yaml"


@dataclass(frozen=True)
class Platform:
    id: str
    brand: str
    channel: str  # "app" | "web"
    color: str

    @property
    def label(self) -> str:
        return f"{self.brand} · {self.channel.capitalize()}"


# One color per platform, used in every chart. Validated as a 4-slot categorical
# palette (all pairs, light surface): Wireless = blue / indigo, Prepaid = aqua / orange.
PLATFORMS: dict[str, Platform] = {
    "wireless_app": Platform("wireless_app", "Nimbus Wireless", "app", "#2a78d6"),
    "wireless_web": Platform("wireless_web", "Nimbus Wireless", "web", "#4a3aa7"),
    "prepaid_app": Platform("prepaid_app", "Nimbus Prepaid", "app", "#1baf7a"),
    "prepaid_web": Platform("prepaid_web", "Nimbus Prepaid", "web", "#eb6834"),
}
PLATFORM_IDS: list[str] = list(PLATFORMS)
BRANDS: list[str] = ["Nimbus Wireless", "Nimbus Prepaid"]
PLATFORM_COLORS: dict[str, str] = {p.id: p.color for p in PLATFORMS.values()}
PLATFORM_LABELS: dict[str, str] = {p.id: p.label for p in PLATFORMS.values()}


@dataclass(frozen=True)
class Step:
    """A funnel step: sessions that reached `column` (and every earlier step)."""

    key: str
    label: str
    column: str


FUNNELS: dict[str, list[Step]] = {
    "engagement": [
        Step("visit", "Visit", "visited"),
        Step("login", "Login", "logged_in"),
        Step("feature_use", "Feature use", "feature_use"),
        Step("return_7d", "Return ≤7d", "returned_7d"),
    ],
    "commerce": [
        Step("product_view", "Product view", "product_view"),
        Step("add_to_cart", "Add to cart", "add_to_cart"),
        Step("checkout_start", "Checkout start", "checkout_start"),
        Step("payment", "Payment", "payment"),
        Step("order_complete", "Order complete", "order_complete"),
    ],
}

# Feature-level engagement metrics (share of logged-in sessions using the feature).
FEATURES: dict[str, str] = {
    "pay_bill": "used_pay_bill",
    "view_usage": "used_view_usage",
    "ai_chat": "used_ai_chat",
}
FEATURE_LABELS = {"pay_bill": "Pay bill", "view_usage": "View usage", "ai_chat": "AI assistant chat"}

STEP_LABELS: dict[str, str] = {s.key: s.label for f in FUNNELS.values() for s in f}
STEP_LABELS.update(FEATURE_LABELS)
STEP_LABELS.update(sessions="Sessions", orders="Orders", revenue="Revenue", aov="Average order value")

# Metrics the anomaly detector monitors: step conversion (step / previous step)
# and feature adoption among logged-in sessions.
MONITORED_METRICS: list[str] = [
    "login",
    "feature_use",
    "return_7d",
    "pay_bill",
    "view_usage",
    "ai_chat",
    "product_view",
    "add_to_cart",
    "checkout_start",
    "payment",
    "order_complete",
]


def funnel_of(metric: str) -> str:
    """Return the funnel a step/feature metric belongs to."""
    if metric in FEATURES:
        return "engagement"
    for name, steps in FUNNELS.items():
        if any(s.key == metric for s in steps):
            return name
    raise KeyError(metric)

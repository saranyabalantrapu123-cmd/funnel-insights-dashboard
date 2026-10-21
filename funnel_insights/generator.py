"""Seeded synthetic session/event generator for the four Nimbus platforms.

Each row is one session with boolean funnel-step flags (session-level event data).
The generator models:
  * per-platform baselines (apps log in more, web browses more, Prepaid web has a
    structurally weak add-to-cart step),
  * weekly seasonality in volume and commerce intent, a monthly bill-cycle bump,
    mild growth, and day-level noise on every rate (so not everything is binomial),
  * a pool of returning users per platform so 7-day return can be computed in SQL,
  * PLANTED anomalies (bug, release uplift, campaign spike, short auth incident and a
    tracking outage) that are written to `planted_anomalies.yaml` as ground truth.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .config import PLATFORMS, PLATFORM_IDS

DEFAULT_START = date(2026, 6, 1)
DEFAULT_DAYS = 90


@dataclass(frozen=True)
class PlatformProfile:
    sessions_per_day: float
    weekday_volume: tuple[float, ...]  # Mon..Sun multipliers
    login: float
    features: dict[str, float]  # P(feature | logged in)
    product_view: float  # P(product view | session)
    add_to_cart: float  # P(cart | product view)
    checkout_start: float
    payment: float
    order_complete: float
    aov: float
    user_pool_days: float  # pool size in "days of logged-in traffic"; smaller => more returning


_APP_WEEK = (1.00, 0.97, 0.97, 0.98, 1.02, 1.12, 1.10)
_WEB_WEEK = (1.08, 1.03, 1.01, 1.00, 0.97, 0.82, 0.80)

PROFILES: dict[str, PlatformProfile] = {
    "wireless_app": PlatformProfile(
        5200, _APP_WEEK, 0.84, {"pay_bill": 0.47, "view_usage": 0.48, "ai_chat": 0.14},
        0.30, 0.34, 0.52, 0.80, 0.94, 185.0, 14.0,
    ),
    "wireless_web": PlatformProfile(
        4400, _WEB_WEEK, 0.52, {"pay_bill": 0.44, "view_usage": 0.36, "ai_chat": 0.09},
        0.42, 0.30, 0.50, 0.74, 0.92, 210.0, 32.0,
    ),
    "prepaid_app": PlatformProfile(
        3600, _APP_WEEK, 0.80, {"pay_bill": 0.55, "view_usage": 0.44, "ai_chat": 0.12},
        0.26, 0.36, 0.58, 0.82, 0.95, 62.0, 16.0,
    ),
    "prepaid_web": PlatformProfile(
        3000, _WEB_WEEK, 0.46, {"pay_bill": 0.50, "view_usage": 0.30, "ai_chat": 0.08},
        0.40, 0.18, 0.54, 0.73, 0.92, 70.0, 38.0,
    ),
}

RATE_NOISE_SD = 0.03  # day-level multiplicative noise on every rate
VOLUME_NOISE_SD = 0.04
GROWTH_PER_DAY = 0.001
OUTAGE_KEEP_FRACTION = 0.03  # share of sessions still tracked during an outage
OUTAGE_EVENT_LOSS = 0.7
INITIAL_USER_SHARE = 0.6  # share of each user pool already active on day 0; the rest sign up over time
CHURN_VS_SIGNUP = 0.85  # users churn at 85% of the sign-up rate, so the active base grows slowly  # retained sessions lose their downstream events with this probability


@dataclass
class PlantedAnomaly:
    id: str
    type: str  # "conversion" | "engagement" | "data_quality"
    platform: str
    metric: str
    start: str
    end: str | None  # None = still ongoing at end of data
    multiplier: float
    description: str
    also_affects: list[str] = field(default_factory=list)


def planted_anomalies(start: date = DEFAULT_START) -> list[PlantedAnomaly]:
    """Ground-truth anomalies, defined relative to the start date."""

    def d(offset: int) -> str:
        return (start + timedelta(days=offset)).isoformat()

    return [
        PlantedAnomaly("A1_login_incident", "engagement", "prepaid_app", "login", d(23), d(24), 0.72,
                       "Identity-provider incident: login success drops for two days."),
        PlantedAnomaly("A2_checkout_release", "conversion", "wireless_app", "checkout_start", d(30), None, 1.22,
                       "App release with one-tap checkout lifts cart-to-checkout conversion."),
        PlantedAnomaly("A3_payment_bug", "conversion", "prepaid_web", "payment", d(49), None, 0.60,
                       "Payment-form bug on Prepaid web: checkout-to-payment conversion drops."),
        PlantedAnomaly("A4_ai_chat_campaign", "engagement", "prepaid_app", "ai_chat", d(65), d(71), 2.6,
                       "In-app campaign for the AI assistant: chat usage spikes for a week.",
                       also_affects=["feature_use"]),
        PlantedAnomaly("A5_tracking_outage", "data_quality", "wireless_web", "sessions", d(74), d(74), 0.0,
                       "Tag-manager misconfiguration: almost no events tracked for one day."),
    ]


def expected_gaps(start: date = DEFAULT_START, days: int = DEFAULT_DAYS) -> list[dict]:
    """Cross-platform gaps a correct gap detector should report over the trailing 28 days."""
    return [
        {"platform": "prepaid_web", "metric": "add_to_cart",
         "reason": "Structural: Prepaid web add-to-cart baseline is far below the web peer."},
        {"platform": "prepaid_web", "metric": "payment",
         "reason": "Caused by A3_payment_bug, still active in the trailing window."},
    ]


def _activity_cdf(rng: np.random.Generator, pool_size: int) -> np.ndarray:
    """Heterogeneous user activity (gamma weights) as an unnormalised CDF for fast sampling.

    Users are ordered by arrival, so sampling from the first k entries means "only users
    who have signed up so far" (see INITIAL_USER_SHARE).
    """
    return np.cumsum(rng.gamma(shape=0.7, scale=1.0, size=pool_size))


def _anomaly_multiplier(anoms: list[PlantedAnomaly], platform: str, metric: str, day: date) -> float:
    mult = 1.0
    for a in anoms:
        if a.platform != platform or a.metric != metric:
            continue
        s = date.fromisoformat(a.start)
        e = date.fromisoformat(a.end) if a.end else date.max
        if s <= day <= e:
            mult *= a.multiplier
    return mult


def _outage_today(anoms: list[PlantedAnomaly], platform: str, day: date) -> bool:
    return any(
        a.type == "data_quality" and a.platform == platform
        and date.fromisoformat(a.start) <= day <= date.fromisoformat(a.end or a.start)
        for a in anoms
    )


def generate(
    seed: int = 42,
    start: date = DEFAULT_START,
    days: int = DEFAULT_DAYS,
    scale: float = 1.0,
    multiplier_overrides: dict[str, float] | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Generate session-level data. Returns (sessions dataframe, ground-truth dict).

    `multiplier_overrides` maps a planted-anomaly id to a different effect size
    (used by the evaluation's sensitivity sweep).
    """
    rng = np.random.default_rng(seed)
    anoms = planted_anomalies(start)
    for a in anoms:
        a.multiplier = (multiplier_overrides or {}).get(a.id, a.multiplier)
    hour_w = np.array([1, 0.6, 0.4, 0.3, 0.3, 0.5, 1.2, 2.2, 3.2, 3.8, 4.0, 4.1,
                       4.3, 4.1, 3.9, 3.8, 4.0, 4.4, 4.9, 5.2, 5.0, 4.2, 3.0, 1.8])
    hour_w = hour_w / hour_w.sum()

    frames: list[pd.DataFrame] = []
    for p_idx, pid in enumerate(PLATFORM_IDS):
        prof = PROFILES[pid]
        logged_per_day = prof.sessions_per_day * scale * prof.login
        pool = max(50, int(logged_per_day * prof.user_pool_days))
        cdf = _activity_cdf(rng, pool)

        for d in range(days):
            day = start + timedelta(days=d)
            wd = day.weekday()
            vol = prof.sessions_per_day * scale * prof.weekday_volume[wd] * (1 + GROWTH_PER_DAY * d)
            n = int(rng.poisson(vol * np.exp(rng.normal(0, VOLUME_NOISE_SD))))
            weekend = wd >= 5

            def rate(metric: str, base: float, seasonal: float = 1.0) -> float:
                noisy = base * seasonal * np.exp(rng.normal(0, RATE_NOISE_SD))
                return float(np.clip(noisy * _anomaly_multiplier(anoms, pid, metric, day), 0.0, 0.99))

            p_login = rate("login", prof.login)
            p_feat = {
                f: rate(f, base, 1.10 if (f == "pay_bill" and day.day <= 3) else 1.0)
                for f, base in prof.features.items()
            }
            p_pv = rate("product_view", prof.product_view, 1.08 if weekend else 1.0)
            p_atc = rate("add_to_cart", prof.add_to_cart, 1.04 if weekend else 1.0)
            p_chk = rate("checkout_start", prof.checkout_start)
            p_pay = rate("payment", prof.payment)
            p_done = rate("order_complete", prof.order_complete)

            u = rng.random((n, 9))
            logged_in = u[:, 0] < p_login
            pay_bill = logged_in & (u[:, 1] < p_feat["pay_bill"])
            view_usage = logged_in & (u[:, 2] < p_feat["view_usage"])
            ai_chat = logged_in & (u[:, 3] < p_feat["ai_chat"])
            pv = u[:, 4] < p_pv
            atc = pv & (u[:, 5] < p_atc)
            chk = atc & (u[:, 6] < p_chk)
            pay = chk & (u[:, 7] < p_pay)
            done = pay & (u[:, 8] < p_done)

            if _outage_today(anoms, pid, day):
                # Tracking outage: most sessions are never recorded, and the few that are
                # often lose their downstream events -> conversion *looks* like it crashed.
                keep = rng.random(n) < OUTAGE_KEEP_FRACTION
                lose = rng.random(n) < OUTAGE_EVENT_LOSS
                for arr in (pay_bill, view_usage, ai_chat, pv, atc, chk, pay, done):
                    arr[lose] = False
                logged_in, pay_bill, view_usage, ai_chat, pv, atc, chk, pay, done = (
                    a[keep] for a in (logged_in, pay_bill, view_usage, ai_chat, pv, atc, chk, pay, done)
                )
                n = int(keep.sum())

            # Active users are a sliding window over the arrival-ordered pool: sign-ups
            # extend it on the right, churn trims it on the left.
            signed_up = (1 - INITIAL_USER_SHARE) * pool * d / days
            hi = max(1, int(INITIAL_USER_SHARE * pool + signed_up))
            lo = int(CHURN_VS_SIGNUP * signed_up)
            base = cdf[lo - 1] if lo > 0 else 0.0
            user_idx = np.searchsorted(cdf, base + rng.random(n) * (cdf[hi - 1] - base))
            user_id = np.where(logged_in, (p_idx + 1) * 10_000_000 + user_idx, -1)
            seconds = rng.choice(24, size=n, p=hour_w) * 3600 + rng.integers(0, 3600, size=n)
            order_value = np.where(
                done, np.round(prof.aov * rng.lognormal(-0.18, 0.6, size=n), 2), np.nan
            )
            frames.append(pd.DataFrame({
                "session_start": pd.Timestamp(day) + pd.to_timedelta(seconds, unit="s"),
                "platform": pid,
                "user_id": user_id,
                "logged_in": logged_in,
                "used_pay_bill": pay_bill,
                "used_view_usage": view_usage,
                "used_ai_chat": ai_chat,
                "product_view": pv,
                "add_to_cart": atc,
                "checkout_start": chk,
                "payment": pay,
                "order_complete": done,
                "order_value": order_value,
            }))

    df = pd.concat(frames, ignore_index=True).sort_values(["session_start", "platform"], kind="stable")
    df = df.reset_index(drop=True)
    df.insert(0, "session_id", np.arange(1, len(df) + 1, dtype=np.int64))
    df["date"] = df["session_start"].dt.date
    df["brand"] = df["platform"].map({k: v.brand for k, v in PLATFORMS.items()})
    df["channel"] = df["platform"].map({k: v.channel for k, v in PLATFORMS.items()})
    df["user_id"] = df["user_id"].astype("Int64").where(df["user_id"] >= 0)
    df = df[["session_id", "date", "session_start", "brand", "channel", "platform", "user_id",
             "logged_in", "used_pay_bill", "used_view_usage", "used_ai_chat", "product_view",
             "add_to_cart", "checkout_start", "payment", "order_complete", "order_value"]]

    truth = {
        "seed": seed,
        "start_date": start.isoformat(),
        "end_date": (start + timedelta(days=days - 1)).isoformat(),
        "scale": scale,
        "anomalies": [asdict(a) for a in anoms],
        "expected_gaps": expected_gaps(start, days),
    }
    return df, truth


def write_outputs(df: pd.DataFrame, truth: dict, parquet_path: Path, yaml_path: Path) -> None:
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(parquet_path, index=False)
    with open(yaml_path, "w") as fh:
        yaml.safe_dump(truth, fh, sort_keys=False)

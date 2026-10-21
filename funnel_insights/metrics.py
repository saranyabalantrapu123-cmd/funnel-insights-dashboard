"""Metrics layer: every number in the dashboard is computed here, in DuckDB SQL.

All functions take a DuckDB connection that has a `sessions_enriched` table
(see `db.build_enriched`) plus optional filters, and return pandas DataFrames.
The insight engine, the AI tools and the UI all read through this module, so a
number shown anywhere has exactly one definition.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import duckdb
import pandas as pd

from .config import FEATURES, FUNNELS, PLATFORM_IDS

# Numerator / denominator SQL for each monitored metric (step conversion = step / previous step).
METRIC_SQL: dict[str, tuple[str, str]] = {
    "sessions": ("count(*)", "1"),
    "login": ("sum(logged_in::INT)", "count(*)"),
    "feature_use": ("sum(feature_use::INT)", "sum(logged_in::INT)"),
    "return_7d": ("sum(returned_7d::INT)", "count(returned_7d)"),
    "pay_bill": ("sum(used_pay_bill::INT)", "sum(logged_in::INT)"),
    "view_usage": ("sum(used_view_usage::INT)", "sum(logged_in::INT)"),
    "ai_chat": ("sum(used_ai_chat::INT)", "sum(logged_in::INT)"),
    "product_view": ("sum(product_view::INT)", "count(*)"),
    "add_to_cart": ("sum(add_to_cart::INT)", "sum(product_view::INT)"),
    "checkout_start": ("sum(checkout_start::INT)", "sum(add_to_cart::INT)"),
    "payment": ("sum(payment::INT)", "sum(checkout_start::INT)"),
    "order_complete": ("sum(order_complete::INT)", "sum(payment::INT)"),
}
RATE_METRICS = [m for m in METRIC_SQL if m != "sessions"]


@dataclass(frozen=True)
class Filters:
    start: date | None = None
    end: date | None = None
    platforms: tuple[str, ...] = tuple(PLATFORM_IDS)

    def where(self) -> tuple[str, list]:
        clauses, params = ["list_contains(?, platform)"], [list(self.platforms)]
        if self.start:
            clauses.append("date >= ?")
            params.append(self.start)
        if self.end:
            clauses.append("date <= ?")
            params.append(self.end)
        return " AND ".join(clauses), params


def date_bounds(con: duckdb.DuckDBPyConnection) -> tuple[date, date]:
    lo, hi = con.execute("SELECT min(date), max(date) FROM sessions_enriched").fetchone()
    return lo, hi


def funnel(con: duckdb.DuckDBPyConnection, name: str, f: Filters = Filters()) -> pd.DataFrame:
    """Per-platform funnel: count, step conversion, end-to-end conversion and drop-off.

    For the engagement funnel the 7-day-return step only uses sessions whose follow-up
    window is complete, so its step conversion is unbiased by right-censoring.
    """
    steps = FUNNELS[name]
    where, params = f.where()
    cols = []
    for s in steps:
        if s.key == "return_7d":
            cols.append("sum(returned_7d::INT) AS return_7d")
            cols.append("count(returned_7d) AS return_7d__den")
        else:
            cols.append(f"sum({s.column}::INT) AS {s.key}")
    raw = con.execute(
        f"SELECT platform, {', '.join(cols)} FROM sessions_enriched WHERE {where} GROUP BY platform",
        params,
    ).df()

    rows = []
    for _, r in raw.iterrows():
        e2e = 1.0
        prev = None
        for i, s in enumerate(steps):
            count = int(r[s.key])
            if i == 0:
                denom, conv = count, 1.0
            else:
                denom = int(r["return_7d__den"]) if s.key == "return_7d" else prev
                conv = count / denom if denom else 0.0
            e2e *= conv
            rows.append({
                "platform": r["platform"], "step": s.key, "step_order": i, "count": count,
                "denominator": denom, "step_conversion": conv, "e2e_conversion": e2e,
                "drop_off": 1 - conv if i else 0.0, "lost": (denom - count) if i else 0,
            })
            prev = count
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["platform", "step_order"]).reset_index(drop=True)


def daily_metrics(con: duckdb.DuckDBPyConnection, f: Filters = Filters(),
                  metrics: list[str] | None = None) -> pd.DataFrame:
    """Long table: date, platform, metric, numerator, denominator, rate (daily grain)."""
    metrics = metrics or list(METRIC_SQL)
    where, params = f.where()
    cols = []
    for m in metrics:
        num, den = METRIC_SQL[m]
        cols.append(f"{num} AS {m}__num")
        cols.append(f"{den} AS {m}__den")
    wide = con.execute(
        f"SELECT date, platform, {', '.join(cols)} FROM sessions_enriched WHERE {where} "
        f"GROUP BY date, platform ORDER BY date, platform",
        params,
    ).df()
    parts = []
    for m in metrics:
        part = wide[["date", "platform", f"{m}__num", f"{m}__den"]].copy()
        part.columns = ["date", "platform", "numerator", "denominator"]
        part["metric"] = m
        parts.append(part)
    long = pd.concat(parts, ignore_index=True)
    long["numerator"] = long["numerator"].fillna(0).astype(float)
    long["denominator"] = long["denominator"].fillna(0).astype(float)
    long["rate"] = long["numerator"] / long["denominator"].where(long["denominator"] > 0)
    long["date"] = pd.to_datetime(long["date"])
    return long[["date", "platform", "metric", "numerator", "denominator", "rate"]]


def metric_by_platform(con: duckdb.DuckDBPyConnection, metric: str, f: Filters = Filters()) -> pd.DataFrame:
    """One row per platform: numerator, denominator and rate of `metric` over the window."""
    if metric not in METRIC_SQL:
        raise KeyError(f"unknown metric {metric!r}; choose from {sorted(METRIC_SQL)}")
    num, den = METRIC_SQL[metric]
    where, params = f.where()
    df = con.execute(
        f"SELECT platform, {num} AS numerator, {den} AS denominator FROM sessions_enriched "
        f"WHERE {where} GROUP BY platform ORDER BY platform",
        params,
    ).df()
    df["rate"] = df["numerator"] / df["denominator"].where(df["denominator"] > 0)
    return df


def revenue(con: duckdb.DuckDBPyConnection, f: Filters = Filters(), by_day: bool = False) -> pd.DataFrame:
    """Orders, revenue and AOV per platform (optionally per day)."""
    where, params = f.where()
    keys = "date, platform" if by_day else "platform"
    df = con.execute(
        f"SELECT {keys}, sum(order_complete::INT) AS orders, coalesce(sum(order_value), 0) AS revenue "
        f"FROM sessions_enriched WHERE {where} GROUP BY {keys} ORDER BY {keys}",
        params,
    ).df()
    df["aov"] = df["revenue"] / df["orders"].where(df["orders"] > 0)
    return df


def kpis(con: duckdb.DuckDBPyConnection, f: Filters = Filters()) -> dict[str, float]:
    """Headline KPIs for the filtered window (all platforms in the filter pooled)."""
    where, params = f.where()
    r = con.execute(
        f"""
        SELECT count(*) AS sessions,
               avg(logged_in::INT) AS login_rate,
               sum(feature_use::INT) / nullif(sum(logged_in::INT), 0) AS feature_use_rate,
               sum(used_ai_chat::INT) / nullif(sum(logged_in::INT), 0) AS ai_chat_rate,
               avg(returned_7d::INT) AS return_7d_rate,
               sum(order_complete::INT) AS orders,
               sum(order_complete::INT) / nullif(sum(product_view::INT), 0) AS commerce_conversion,
               coalesce(sum(order_value), 0) AS revenue,
               sum(order_value) / nullif(sum(order_complete::INT), 0) AS aov
        FROM sessions_enriched WHERE {where}
        """,
        params,
    ).df().iloc[0]
    return {k: (float(v) if pd.notna(v) else float("nan")) for k, v in r.items()}


COHORT_LOOKBACK_DAYS = 14


def cohort_return(con: duckdb.DuckDBPyConnection, f: Filters = Filters()) -> pd.DataFrame:
    """Weekly cohorts of *new* engaged users and their 7-day return rate.

    A user is "new" on their first feature-use session in the data, but only after a
    14-day lookback: early in the data, "first seen" mostly means "long-time user we had
    not observed yet", which would inflate the first cohorts. Cohorts whose 7-day
    window is incomplete are excluded.
    """
    where, params = f.where()
    data_start = con.execute("SELECT min(date) FROM sessions_enriched").fetchone()[0]
    return con.execute(
        f"""
        WITH engaged AS (
            SELECT platform, user_id, session_start FROM sessions_enriched
            WHERE feature_use AND {where}
        ),
        firsts AS (
            SELECT platform, user_id, min(session_start) AS first_ts FROM engaged GROUP BY ALL
        ),
        nexts AS (
            SELECT f.platform, f.user_id, f.first_ts, min(e.session_start) AS next_ts
            FROM firsts f LEFT JOIN engaged e
              ON e.platform = f.platform AND e.user_id = f.user_id AND e.session_start > f.first_ts
            GROUP BY ALL
        ),
        bounds AS (SELECT max(session_start) AS max_ts FROM engaged)
        SELECT platform, CAST(date_trunc('week', first_ts) AS DATE) AS cohort_week,
               count(*) AS users,
               avg(coalesce(next_ts <= first_ts + INTERVAL 7 DAY, FALSE)::INT) AS return_rate
        FROM nexts, bounds
        WHERE first_ts + INTERVAL 7 DAY <= bounds.max_ts
          AND date_trunc('week', first_ts) >= CAST(? AS DATE) + INTERVAL {COHORT_LOOKBACK_DAYS} DAY
        GROUP BY ALL ORDER BY cohort_week, platform
        """,
        [*params, data_start],
    ).df()


def feature_mix(con: duckdb.DuckDBPyConnection, f: Filters = Filters()) -> pd.DataFrame:
    """Share of logged-in sessions using each feature, per platform."""
    where, params = f.where()
    cols = ", ".join(f"sum({c}::INT) / nullif(sum(logged_in::INT), 0) AS {k}" for k, c in FEATURES.items())
    return con.execute(
        f"SELECT platform, {cols} FROM sessions_enriched WHERE {where} GROUP BY platform ORDER BY platform",
        params,
    ).df()

"""Deterministic insight engine.

Three detectors run over the daily metric table from `metrics.daily_metrics`:

1. Data quality (runs first): daily session volume vs a same-weekday baseline.
   Volume collapsing to ~0 is a *tracking* problem, not a funnel problem, so those
   days are masked out of every conversion series before step 2 runs.
2. Conversion / engagement anomalies on each (platform, metric) daily rate series:
   * a causal seasonal robust z-score (baseline = median of the same weekday over the
     previous 4 weeks, scale = MAD of trailing residuals floored by binomial noise),
   * a binary-segmentation changepoint detector on the weekday-adjusted series,
     validated on medians so a one-week spike is not mistaken for a level shift.
   Overlapping z-runs and changepoints are merged into one insight.
3. Cross-platform gaps: the same step much worse on one platform than on its
   like-for-like peer (same channel, other brand) over the trailing window.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, timedelta

import duckdb
import numpy as np
import pandas as pd

from .config import FUNNELS, MONITORED_METRICS, PLATFORM_IDS, PLATFORMS
from .insights import Insight, finalize
from .metrics import Filters, daily_metrics, revenue

COMMERCE_STEPS = [s.key for s in FUNNELS["commerce"]]

# Known calendar effects that a weekly baseline cannot see. Days matching a rule are
# still plotted but never alert, e.g. bill payments always jump around the due date.
KNOWN_CALENDAR_EFFECTS: dict[str, tuple[str, Callable[[date], bool]]] = {
    "pay_bill": ("bill-due dates (1st-3rd of the month)", lambda d: d.day <= 3),
}


@dataclass(frozen=True)
class DetectorConfig:
    z_threshold: float = 4.0  # for runs of 2+ consecutive days
    single_day_z: float = 5.0  # a lone day needs stronger evidence (persistence rule)
    min_rel_change: float = 0.05  # practical-significance floor (5% relative)
    min_denominator: int = 30
    min_baseline_points: int = 3  # same-weekday points needed for a baseline
    min_rel_sigma: float = 0.03  # volatility floor: daily moves under ~4 x 3% never alert
    mad_window: int = 28
    cp_min_segment: int = 7
    cp_t_threshold: float = 6.0
    cp_min_rel_change: float = 0.08
    cp_merge_days: int = 7
    outage_ratio: float = 0.30
    partial_loss_ratio: float = 0.60
    gap_rel_threshold: float = 0.25
    gap_z_threshold: float = 5.0
    gap_window_days: int = 28
    dq_gating: bool = True  # mask tracking-outage days out of conversion series (off = naive baseline)


# --------------------------------------------------------------------------- helpers


def seasonal_baseline(x: np.ndarray, min_points: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """Median of the same weekday over the previous 4 weeks (causal).

    Returns (baseline, number of points used); baseline is NaN with too few points.
    """
    n = len(x)
    base, k = np.full(n, np.nan), np.zeros(n)
    for t in range(n):
        lags = [x[t - j] for j in (7, 14, 21, 28) if t - j >= 0 and not np.isnan(x[t - j])]
        k[t] = len(lags)
        if len(lags) >= min_points:
            base[t] = float(np.median(lags))
    return base, k


def robust_z(x: np.ndarray, den: np.ndarray, cfg: DetectorConfig) -> tuple[np.ndarray, np.ndarray]:
    """Causal robust z-score of each day vs its seasonal baseline. Returns (z, baseline).

    The scale is the largest of: MAD of trailing residuals (once 14+ exist), the binomial
    noise of the day (inflated for the baseline's own noise), and a relative volatility
    floor so ordinary day-to-day wobble never pages anyone.
    """
    base, k = seasonal_baseline(x, cfg.min_baseline_points)
    resid = x - base
    z = np.full(len(x), np.nan)
    for t in range(len(x)):
        if np.isnan(resid[t]):
            continue
        hist = resid[max(0, t - cfg.mad_window):t]
        hist = hist[~np.isnan(hist)]
        mad_sigma = 1.4826 * np.median(np.abs(hist - np.median(hist))) if len(hist) >= 14 else 0.0
        binom_sigma = np.sqrt(max(base[t] * (1 - base[t]), 1e-9) / max(den[t], 1.0) * (1 + 1.57 / k[t]))
        sigma = max(mad_sigma, binom_sigma, cfg.min_rel_sigma * base[t])
        z[t] = resid[t] / sigma
    return z, base


def _weekday_adjust(x: np.ndarray, dates: pd.DatetimeIndex) -> np.ndarray:
    """Remove the weekday effect (median deviation from a centred 7-day median)."""
    s = pd.Series(x, index=dates)
    trend = s.rolling(7, center=True, min_periods=4).median()
    dev = (s - trend).groupby(dates.weekday).median()
    return x - dev.reindex(dates.weekday).to_numpy()


def _best_split(y: np.ndarray, min_seg: int) -> tuple[int, float] | None:
    """Single mean-shift split maximising a two-sample t statistic (vectorised)."""
    n = len(y)
    if n < 2 * min_seg:
        return None
    best_k, best_t = -1, 0.0
    for k in range(min_seg, n - min_seg + 1):
        left, right = y[:k], y[k:]
        pooled = np.sqrt(((left.var(ddof=1) * (k - 1)) + (right.var(ddof=1) * (n - k - 1))) / (n - 2))
        if pooled <= 0:
            continue
        t = abs(right.mean() - left.mean()) / (pooled * np.sqrt(1 / k + 1 / (n - k)))
        if t > best_t:
            best_k, best_t = k, t
    return (best_k, best_t) if best_k > 0 else None


def changepoints(y: np.ndarray, cfg: DetectorConfig, depth: int = 2) -> list[tuple[int, float, float]]:
    """Validated changepoints as (index, t_stat, relative median shift), via binary segmentation.

    A split is accepted only if the *medians* of both sides differ by at least
    `cp_min_rel_change`; a short spike inside a long segment moves the mean but not
    the median, so it is left to the z-score detector.
    """
    found: list[tuple[int, float, float]] = []

    def recurse(lo: int, hi: int, d: int) -> None:
        split = _best_split(y[lo:hi], cfg.cp_min_segment)
        if split is None:
            return
        k, t = split
        left, right = y[lo:lo + k], y[lo + k:hi]
        rel = (np.median(right) - np.median(left)) / abs(np.median(left))
        if t < cfg.cp_t_threshold or abs(rel) < cfg.cp_min_rel_change:
            return
        found.append((lo + k, float(t), float(rel)))
        if d > 1:
            recurse(lo, lo + k, d - 1)
            recurse(lo + k, hi, d - 1)

    recurse(0, len(y), depth)
    return sorted(found)


def _runs(flags: np.ndarray, signs: np.ndarray, valid: np.ndarray) -> list[tuple[int, int, int]]:
    """Consecutive flagged days with the same sign -> (start, end, sign). NaN days don't break a run."""
    runs, cur = [], None
    for t in range(len(flags)):
        if not valid[t]:
            continue
        if flags[t]:
            if cur and cur[2] == signs[t]:
                cur[1] = t
            else:
                if cur:
                    runs.append(tuple(cur))
                cur = [t, t, int(signs[t])]
        elif cur:
            runs.append(tuple(cur))
            cur = None
    if cur:
        runs.append(tuple(cur))
    return runs


# --------------------------------------------------------------------------- detectors


def detect_data_quality(daily: pd.DataFrame, cfg: DetectorConfig) -> tuple[list[Insight], dict[str, set]]:
    """Volume-based tracking checks. Returns insights and masked dates per platform."""
    insights: list[Insight] = []
    masked: dict[str, set] = {p: set() for p in PLATFORM_IDS}
    vol = daily[daily.metric == "sessions"].pivot(index="date", columns="platform", values="numerator")
    vol = vol.asfreq("D")
    last_day = vol.index.max().date()
    for pid in vol.columns:
        x = vol[pid].to_numpy(dtype=float, copy=True)
        base, _ = seasonal_baseline(x, cfg.min_baseline_points)
        ratio = x / base
        bad = np.nan_to_num(ratio, nan=1.0) < cfg.partial_loss_ratio
        for s, e, _ in _runs(bad, np.ones(len(x)), np.ones(len(x), dtype=bool)):
            r = ratio[s:e + 1]
            outage = bool(np.nanmin(r) < cfg.outage_ratio)
            days = [vol.index[i].date() for i in range(s, e + 1)]
            masked[pid].update(days)
            lost = float(np.nanmean(base[s:e + 1] - x[s:e + 1]))
            insights.append(Insight(
                type="data_quality", platform=pid, metric="sessions", direction="down",
                shape="outage" if outage else "volume_dip",
                magnitude=float(np.nanmean(r) - 1), baseline=float(np.nanmean(base[s:e + 1])),
                observed=float(np.nanmean(x[s:e + 1])), start_date=days[0],
                end_date=None if days[-1] == last_day else days[-1], first_alert_date=days[0],
                confidence=0.95 if outage else 0.6, daily_denominator=float(np.nanmean(base[s:e + 1])),
                impact={"untracked_sessions_per_day": lost, "days": float(len(days))},
            ))
    return insights, masked


def _impact(metric: str, platform: str, delta_rate: float, denom: float,
            step_rates: dict[str, dict[str, float]], aov: dict[str, float]) -> dict[str, float]:
    """Translate a rate change into affected sessions/orders/revenue per day."""
    affected = delta_rate * denom
    if metric not in COMMERCE_STEPS:
        return {"sessions_per_day": affected}
    downstream = 1.0
    for s in COMMERCE_STEPS[COMMERCE_STEPS.index(metric) + 1:]:
        downstream *= step_rates[platform].get(s, 0.0)
    orders = affected * downstream
    return {"orders_per_day": orders, "revenue_per_day": orders * aov.get(platform, 0.0)}


def _cp_first_alert(y: np.ndarray, idx: int, cfg: DetectorConfig) -> int:
    """Earliest day on which the changepoint at `idx` is detectable using data up to that day."""
    for end in range(idx + cfg.cp_min_segment, len(y) + 1):
        if any(abs(k - idx) <= 3 for k, _, _ in changepoints(y[:end], cfg, depth=1)):
            return end - 1
    return len(y) - 1


def detect_anomalies(daily: pd.DataFrame, masked: dict[str, set], cfg: DetectorConfig,
                     step_rates: dict[str, dict[str, float]], aov: dict[str, float]) -> list[Insight]:
    insights: list[Insight] = []
    rates = daily[daily.metric != "sessions"]
    for (pid, metric), g in rates.groupby(["platform", "metric"]):
        g = g.set_index("date").asfreq("D")
        dates = g.index
        x = g["rate"].to_numpy(dtype=float, copy=True)
        den = g["denominator"].fillna(0).to_numpy(dtype=float)
        mask = np.array([d.date() in masked[pid] for d in dates])
        if metric == "return_7d":  # return depends on tracking in the following 7 days too
            for d in list(masked[pid]):
                mask |= np.array([(d - timedelta(days=7)) <= t.date() < d for t in dates])
        x[mask | (den < cfg.min_denominator)] = np.nan
        valid = ~np.isnan(x)
        if valid.sum() < 21:
            continue
        last_valid = int(np.where(valid)[0].max())

        z, base = robust_z(x, den, cfg)
        rel = (x - base) / base
        flags = (np.abs(np.nan_to_num(z)) >= cfg.z_threshold) & (np.abs(np.nan_to_num(rel)) >= cfg.min_rel_change)
        if metric in KNOWN_CALENDAR_EFFECTS:
            rule = KNOWN_CALENDAR_EFFECTS[metric][1]
            flags &= ~np.array([rule(d.date()) for d in dates])
        runs = [r for r in _runs(flags, np.sign(np.nan_to_num(z)), valid)
                if r[1] > r[0] or abs(z[r[0]]) >= cfg.single_day_z]

        vidx = np.where(valid)[0]
        y = _weekday_adjust(x, dates)[vidx]
        cps = [(int(vidx[k]), t, r) for k, t, r in changepoints(y, cfg)]

        kind = "conversion_anomaly" if metric in COMMERCE_STEPS else "engagement_anomaly"
        used_runs: set[int] = set()
        for n_cp, (ci, t_stat, rel_shift) in enumerate(cps):
            sign = 1 if rel_shift > 0 else -1
            nxt = cps[n_cp + 1][0] if n_cp + 1 < len(cps) else None
            seg_end = (nxt - 1) if nxt is not None else last_valid
            prev = cps[n_cp - 1][0] if n_cp else 0
            pre = x[max(prev, ci - 28):ci]
            post = x[ci:seg_end + 1]
            b, o = float(np.nanmedian(pre)), float(np.nanmedian(post))
            match = [i for i, (s, _, sg) in enumerate(runs)
                     if sg == sign and ci - 3 <= s <= ci + cfg.cp_merge_days]
            used_runs.update(match)
            cp_alert = _cp_first_alert(y, int(np.searchsorted(vidx, ci)), cfg)
            alert_idx = min([runs[i][0] for i in match] + [int(vidx[cp_alert])])
            peak = max((float(np.nanmax(np.abs(z[runs[i][0]:runs[i][1] + 1]))) for i in match), default=None)
            conf = min(0.99, 0.6 + 0.03 * (t_stat - cfg.cp_t_threshold) + (0.15 if match else 0.0))
            denom = float(np.nanmean(den[ci:seg_end + 1]))
            insights.append(Insight(
                type=kind, platform=pid, metric=metric, direction="up" if sign > 0 else "down",
                shape="level_shift", magnitude=o / b - 1, baseline=b, observed=o,
                start_date=dates[ci].date(), end_date=None if nxt is None else dates[seg_end].date(),
                first_alert_date=dates[alert_idx].date(), confidence=conf, daily_denominator=denom,
                peak_z=peak, impact=_impact(metric, pid, o - b, denom, step_rates, aov),
            ))

        for i, (s, e, sign) in enumerate(runs):
            if i in used_runs:
                continue
            b = float(np.nanmean(base[s:e + 1]))
            o = float(np.nanmean(x[s:e + 1]))
            peak = float(np.nanmax(np.abs(z[s:e + 1])))
            length = int(valid[s:e + 1].sum())
            conf = min(0.99, 0.55 + 0.05 * (peak - cfg.z_threshold) + 0.05 * (length - 1))
            denom = float(np.nanmean(den[s:e + 1]))
            insights.append(Insight(
                type=kind, platform=pid, metric=metric, direction="up" if sign > 0 else "down",
                shape="transient", magnitude=o / b - 1, baseline=b, observed=o,
                start_date=dates[s].date(), end_date=None if e >= last_valid else dates[e].date(),
                first_alert_date=dates[s].date(), confidence=conf, daily_denominator=denom,
                peak_z=peak, impact=_impact(metric, pid, o - b, denom, step_rates, aov),
            ))
    return insights


def like_for_like_peer(platform: str) -> str:
    """Same channel, other brand (web vs web, app vs app)."""
    p = PLATFORMS[platform]
    return next(q.id for q in PLATFORMS.values() if q.channel == p.channel and q.brand != p.brand)


def detect_gaps(daily: pd.DataFrame, masked: dict[str, set], cfg: DetectorConfig,
                step_rates: dict[str, dict[str, float]], aov: dict[str, float]) -> list[Insight]:
    """Flag steps where a platform is much worse than its like-for-like peer (trailing window)."""
    end = daily.date.max()
    rates = daily[(daily.metric != "sessions") & (daily.date > end - pd.Timedelta(days=cfg.gap_window_days))]
    keep = ~rates.apply(lambda r: r["date"].date() in masked[r["platform"]], axis=1)
    totals = rates[keep].groupby(["metric", "platform"])[["numerator", "denominator"]].sum()
    n_days = rates[keep].groupby(["metric", "platform"])["date"].nunique()
    window_start = (end - pd.Timedelta(days=cfg.gap_window_days - 1)).date()
    insights = []
    for (metric, pid), row in totals.iterrows():
        peer = like_for_like_peer(pid)
        if (metric, peer) not in totals.index or row.denominator == 0:
            continue
        pr = totals.loc[(metric, peer)]
        p1, p2 = row.numerator / row.denominator, pr.numerator / pr.denominator
        if p2 <= 0:
            continue
        rel = p1 / p2 - 1
        pooled = (row.numerator + pr.numerator) / (row.denominator + pr.denominator)
        se = np.sqrt(pooled * (1 - pooled) * (1 / row.denominator + 1 / pr.denominator))
        zstat = (p1 - p2) / se if se > 0 else 0.0
        if rel <= -cfg.gap_rel_threshold and zstat <= -cfg.gap_z_threshold:
            denom = row.denominator / n_days.loc[(metric, pid)]
            insights.append(Insight(
                type="platform_gap", platform=pid, metric=metric, direction="down", shape="gap",
                magnitude=float(rel), baseline=float(p2), observed=float(p1), start_date=window_start,
                end_date=None, first_alert_date=end.date(), confidence=min(0.99, 0.7 + 0.01 * abs(zstat)),
                daily_denominator=float(denom), peak_z=float(zstat), peer_platform=peer,
                impact=_impact(metric, pid, p1 - p2, denom, step_rates, aov),
            ))
    return insights


def run_insight_engine(con: duckdb.DuckDBPyConnection, cfg: DetectorConfig = DetectorConfig()) -> list[Insight]:
    """Run all detectors over the full history and return prioritised insights."""
    daily = daily_metrics(con, Filters(), ["sessions", *MONITORED_METRICS])
    data_end: date = daily.date.max().date()
    recent = Filters(start=data_end - timedelta(days=cfg.gap_window_days - 1), end=data_end)
    step_rates: dict[str, dict[str, float]] = {p: {} for p in PLATFORM_IDS}
    recent_daily = daily[daily.date.dt.date >= recent.start]
    for (pid, metric), g in recent_daily.groupby(["platform", "metric"]):
        if g.denominator.sum() > 0:
            step_rates[pid][metric] = float(g.numerator.sum() / g.denominator.sum())
    rev = revenue(con, recent)
    aov = dict(zip(rev.platform, rev.aov.fillna(0.0)))

    dq, masked = detect_data_quality(daily, cfg)
    if not cfg.dq_gating:
        masked = {p: set() for p in PLATFORM_IDS}
    anomalies = detect_anomalies(daily, masked, cfg, step_rates, aov)
    gaps = detect_gaps(daily, masked, cfg, step_rates, aov)
    return finalize(dq + anomalies + gaps, data_end)

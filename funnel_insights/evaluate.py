"""`make eval`: score the deterministic insight engine against the planted ground truth.

* Anomaly detection: precision, recall and detection delay (first alert - planted start)
  on the dev seed (42, used while tuning thresholds) and 5 held-out seeds.
* Data-quality classification: is the tracking outage reported as data_quality, with no
  conversion/engagement alert on that platform-day? Compared against a naive variant of
  the same engine with the data-quality gate switched off.
* Cross-platform gaps: precision / recall vs the expected gaps.

Everything here is deterministic; no LLM is involved, so these are real measured numbers.
"""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import date, timedelta

import duckdb

from .config import REPORTS_DIR
from .db import build_enriched
from .detection import DetectorConfig, run_insight_engine
from .generator import generate
from .insights import Insight

DEV_SEED = 42
HOLDOUT_SEEDS = [1, 2, 3, 4, 5]
START_TOLERANCE_DAYS = 2  # a detection may start up to 2 days before the planted start
ANOMALY_TYPES = ("conversion_anomaly", "engagement_anomaly", "data_quality")


def _matches(ins: Insight, a: dict, data_end: date) -> bool:
    start = date.fromisoformat(a["start"])
    end = date.fromisoformat(a["end"]) if a["end"] else data_end
    if ins.platform != a["platform"]:
        return False
    if a["type"] == "data_quality":
        return ins.type == "data_quality" and start - timedelta(days=1) <= ins.start_date <= end
    if ins.type not in ("conversion_anomaly", "engagement_anomaly"):
        return False
    direction = "down" if a["multiplier"] < 1 else "up"
    return (ins.metric in [a["metric"], *a.get("also_affects", [])] and ins.direction == direction
            and start - timedelta(days=START_TOLERANCE_DAYS) <= ins.start_date <= end)


def score(insights: list[Insight], truth: dict) -> dict:
    data_end = date.fromisoformat(truth["end_date"])
    detections = [i for i in insights if i.type in ANOMALY_TYPES]
    per_anomaly, matched_ids = [], set()
    for a in truth["anomalies"]:
        hits = [i for i in detections if _matches(i, a, data_end)]
        matched_ids.update(i.id for i in hits)
        delay = (min(i.first_alert_date for i in hits) - date.fromisoformat(a["start"])).days if hits else None
        per_anomaly.append({"id": a["id"], "detected": bool(hits), "delay_days": delay,
                            "detections": [i.id for i in hits]})
    tp_detections = sum(1 for i in detections if i.id in matched_ids)
    false_alarms = [i.id for i in detections if i.id not in matched_ids]

    outage = next(a for a in truth["anomalies"] if a["type"] == "data_quality")
    o_day = date.fromisoformat(outage["start"])
    funnel_alerts_on_outage = [
        i.id for i in detections
        if i.type != "data_quality" and i.platform == outage["platform"]
        and i.start_date <= o_day <= (i.end_date or data_end)
    ]
    outage_ok = any(i.type == "data_quality" and _matches(i, outage, data_end) for i in detections)

    gaps = [i for i in insights if i.type == "platform_gap"]
    expected = {(g["platform"], g["metric"]) for g in truth["expected_gaps"]}
    found = {(g.platform, g.metric) for g in gaps}
    recall = sum(r["detected"] for r in per_anomaly) / len(per_anomaly)
    return {
        "recall": recall,
        "precision": tp_detections / len(detections) if detections else 1.0,
        "n_detections": len(detections),
        "false_alarms": false_alarms,
        "per_anomaly": per_anomaly,
        "outage_classified_as_data_quality": outage_ok,
        "funnel_alerts_on_outage_day": funnel_alerts_on_outage,
        "gap_precision": len(found & expected) / len(found) if found else 1.0,
        "gap_recall": len(found & expected) / len(expected) if expected else 1.0,
        "gaps_found": sorted(f"{p}:{m}" for p, m in found),
    }


SWEEP_ANOMALY = "A3_payment_bug"
SWEEP_MULTIPLIERS = [0.95, 0.90, 0.85, 0.80]
SWEEP_SEEDS = [1, 2, 3]


def _engine_con(seed: int, overrides: dict[str, float] | None = None) -> tuple[duckdb.DuckDBPyConnection, dict]:
    df, truth = generate(seed=seed, multiplier_overrides=overrides)
    con = duckdb.connect()
    con.register("sessions", df)
    build_enriched(con)
    return con, truth


def run_seed(seed: int, cfg: DetectorConfig) -> tuple[dict, dict]:
    con, truth = _engine_con(seed)
    gated = score(run_insight_engine(con, cfg), truth)
    naive = score(run_insight_engine(con, replace(cfg, dq_gating=False)), truth)
    con.close()
    return gated, naive


def sensitivity(cfg: DetectorConfig) -> list[dict]:
    """Recall and delay for smaller versions of the payment bug (where does detection break?)."""
    rows = []
    for mult in SWEEP_MULTIPLIERS:
        hits, delays = 0, []
        for seed in SWEEP_SEEDS:
            con, truth = _engine_con(seed, {SWEEP_ANOMALY: mult})
            res = score(run_insight_engine(con, cfg), truth)
            con.close()
            r = next(x for x in res["per_anomaly"] if x["id"] == SWEEP_ANOMALY)
            hits += r["detected"]
            if r["delay_days"] is not None:
                delays.append(r["delay_days"])
        rows.append({"drop_pct": round((1 - mult) * 100), "recall": hits / len(SWEEP_SEEDS),
                     "delays_days": delays})
    return rows


def _fmt_row(name: str, s: dict, naive: dict) -> str:
    delays = {r["id"].split("_")[0]: r["delay_days"] for r in s["per_anomaly"]}
    delay_txt = " ".join(f"{k}={'—' if v is None else v}" for k, v in delays.items())
    return (f"| {name} | {s['precision']:.2f} | {s['recall']:.2f} | {len(s['false_alarms'])} | {delay_txt} | "
            f"{'yes' if s['outage_classified_as_data_quality'] else 'no'} | "
            f"{len(s['funnel_alerts_on_outage_day'])} vs {len(naive['funnel_alerts_on_outage_day'])} | "
            f"{s['gap_precision']:.2f} / {s['gap_recall']:.2f} |")


def main() -> None:
    cfg = DetectorConfig()
    results = {}
    for seed in [DEV_SEED, *HOLDOUT_SEEDS]:
        gated, naive = run_seed(seed, cfg)
        results[seed] = {"engine": gated, "naive_no_dq_gate": naive}
        print(f"seed {seed:>2}: precision {gated['precision']:.2f} recall {gated['recall']:.2f} "
              f"false alarms {len(gated['false_alarms'])} · outage→data_quality "
              f"{gated['outage_classified_as_data_quality']} · funnel alerts on outage day "
              f"{len(gated['funnel_alerts_on_outage_day'])} (naive: {len(naive['funnel_alerts_on_outage_day'])})")

    hold = [results[s]["engine"] for s in HOLDOUT_SEEDS]
    hold_naive = [results[s]["naive_no_dq_gate"] for s in HOLDOUT_SEEDS]
    n_det = sum(h["n_detections"] for h in hold)
    n_fp = sum(len(h["false_alarms"]) for h in hold)
    planted = sum(len(h["per_anomaly"]) for h in hold)
    found = sum(r["detected"] for h in hold for r in h["per_anomaly"])
    delays = [r["delay_days"] for h in hold for r in h["per_anomaly"] if r["delay_days"] is not None]
    naive_fp = sum(len(h["false_alarms"]) for h in hold_naive)
    naive_det = sum(h["n_detections"] for h in hold_naive)
    summary = {
        "holdout_seeds": HOLDOUT_SEEDS,
        "precision": (n_det - n_fp) / n_det if n_det else 1.0,
        "recall": found / planted,
        "false_alarms_total": n_fp,
        "mean_delay_days": sum(delays) / len(delays) if delays else None,
        "max_delay_days": max(delays) if delays else None,
        "outage_classified_as_data_quality": f"{sum(h['outage_classified_as_data_quality'] for h in hold)}/{len(hold)}",
        "naive_precision": (naive_det - naive_fp) / naive_det if naive_det else 1.0,
        "naive_false_alarms_total": naive_fp,
        "gap_precision": sum(h["gap_precision"] for h in hold) / len(hold),
        "gap_recall": sum(h["gap_recall"] for h in hold) / len(hold),
    }

    lines = [
        "| Seed | Precision | Recall | False alarms | Delay (days) per planted anomaly | Outage → data_quality "
        "| Funnel alerts on outage day (engine vs naive) | Gap P / R |",
        "|---|---|---|---|---|---|---|---|",
        _fmt_row(f"{DEV_SEED} (dev)", results[DEV_SEED]["engine"], results[DEV_SEED]["naive_no_dq_gate"]),
        *[_fmt_row(str(s), results[s]["engine"], results[s]["naive_no_dq_gate"]) for s in HOLDOUT_SEEDS],
    ]
    table = "\n".join(lines)
    print("\n" + table)
    print(f"\nHeld-out seeds {HOLDOUT_SEEDS}: precision {summary['precision']:.2f}, recall {summary['recall']:.2f}, "
          f"mean detection delay {summary['mean_delay_days']:.1f} d (max {summary['max_delay_days']}), "
          f"outage classified as data quality {summary['outage_classified_as_data_quality']}; "
          f"without the data-quality gate precision falls to {summary['naive_precision']:.2f} "
          f"({summary['naive_false_alarms_total']} false alarms vs {summary['false_alarms_total']}).")

    print(f"\nSensitivity: {SWEEP_ANOMALY} at smaller effect sizes (seeds {SWEEP_SEEDS})")
    sweep = sensitivity(cfg)
    sweep_lines = ["| Payment-step drop | Recall | Detection delay (days) |", "|---|---|---|"]
    for r in sweep:
        sweep_lines.append(f"| -{r['drop_pct']}% | {r['recall']:.2f} | "
                           f"{', '.join(map(str, r['delays_days'])) or '—'} |")
    print("\n".join(sweep_lines))

    REPORTS_DIR.mkdir(exist_ok=True)
    (REPORTS_DIR / "eval_results.json").write_text(
        json.dumps({"summary": summary, "per_seed": results, "sensitivity": sweep}, indent=2, default=str))
    (REPORTS_DIR / "eval_results.md").write_text(table + "\n\n" + "\n".join(sweep_lines) + "\n")
    print(f"\nWrote {REPORTS_DIR.name}/eval_results.json and {REPORTS_DIR.name}/eval_results.md")


if __name__ == "__main__":
    main()

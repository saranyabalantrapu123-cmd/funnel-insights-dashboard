"""The deterministic insight engine against the planted ground truth (seed 42)."""

from dataclasses import replace
from datetime import date

import numpy as np

from funnel_insights.detection import DetectorConfig, run_insight_engine, seasonal_baseline
from funnel_insights.evaluate import score


def test_every_planted_anomaly_is_detected(seeded):
    _, truth, insights = seeded
    result = score(insights, truth)
    missed = [r["id"] for r in result["per_anomaly"] if not r["detected"]]
    assert not missed, f"missed planted anomalies: {missed}"
    assert result["recall"] == 1.0
    assert result["precision"] >= 0.8


def test_level_shifts_have_correct_direction_and_start(seeded):
    _, _, insights = seeded
    pay = next(i for i in insights if i.platform == "prepaid_web" and i.metric == "payment"
               and i.type == "conversion_anomaly")
    assert pay.direction == "down" and pay.shape == "level_shift" and pay.ongoing
    assert pay.start_date == date(2026, 7, 20)
    assert -0.5 < pay.magnitude < -0.3  # planted x0.60
    chk = next(i for i in insights if i.platform == "wireless_app" and i.metric == "checkout_start")
    assert chk.direction == "up" and chk.start_date == date(2026, 7, 1)


def test_outage_is_data_quality_not_a_conversion_drop(seeded):
    _, truth, insights = seeded
    outage_day = date(2026, 8, 14)
    dq = [i for i in insights if i.type == "data_quality"]
    assert len(dq) == 1 and dq[0].platform == "wireless_web" and dq[0].start_date == outage_day
    assert dq[0].shape == "outage"
    funnel_alerts = [i for i in insights if i.platform == "wireless_web"
                     and i.type in ("conversion_anomaly", "engagement_anomaly")
                     and i.start_date <= outage_day <= (i.end_date or outage_day)]
    assert funnel_alerts == []
    assert score(insights, truth)["outage_classified_as_data_quality"]


def test_without_dq_gate_the_outage_causes_false_alarms(seeded):
    con, truth, _ = seeded
    naive = run_insight_engine(con, replace(DetectorConfig(), dq_gating=False))
    assert len(score(naive, truth)["funnel_alerts_on_outage_day"]) >= 1


def test_cross_platform_gaps(seeded):
    _, truth, insights = seeded
    result = score(insights, truth)
    assert result["gap_recall"] == 1.0 and result["gap_precision"] == 1.0
    gap = next(i for i in insights if i.type == "platform_gap" and i.metric == "payment")
    assert gap.related, "payment gap should be linked to the payment anomaly"


def test_ranking_puts_ongoing_payment_bug_first(seeded):
    _, _, insights = seeded
    assert insights[0].metric == "payment" and insights[0].platform == "prepaid_web"


def test_seasonal_baseline_uses_same_weekday():
    x = np.array([10.0, 1, 1, 1, 1, 1, 1] * 4 + [99.0])
    base, k = seasonal_baseline(x, min_points=3)
    assert base[28] == 10.0 and k[28] == 4
    assert np.isnan(base[14])  # only 2 prior same-weekday points

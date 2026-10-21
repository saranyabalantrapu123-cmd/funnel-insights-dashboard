from datetime import date

import pandas as pd
import yaml

from funnel_insights.config import PLATFORM_IDS
from funnel_insights.generator import generate, write_outputs


def small(seed: int):
    return generate(seed=seed, days=80, scale=0.05)


def test_same_seed_is_reproducible():
    a, ta = small(7)
    b, tb = small(7)
    pd.testing.assert_frame_equal(a, b)
    assert ta == tb


def test_different_seed_differs():
    a, _ = small(7)
    b, _ = small(8)
    assert len(a) != len(b) or not a.equals(b)


def test_schema_and_funnel_monotonicity():
    df, _ = small(1)
    assert set(df.platform) == set(PLATFORM_IDS)
    # A session can only reach a step if it reached every earlier step.
    steps = ["product_view", "add_to_cart", "checkout_start", "payment", "order_complete"]
    for prev, nxt in zip(steps, steps[1:]):
        assert not (df[nxt] & ~df[prev]).any()
    assert not ((df.used_ai_chat | df.used_pay_bill | df.used_view_usage) & ~df.logged_in).any()
    assert df.user_id.isna().equals(~df.logged_in)
    assert df.order_value.notna().equals(df.order_complete)


def test_tracking_outage_is_planted(tmp_path):
    df, truth = small(3)
    outage = next(a for a in truth["anomalies"] if a["type"] == "data_quality")
    day = date.fromisoformat(outage["start"])
    vol = df[df.platform == outage["platform"]].groupby("date").size()
    assert vol[day] < 0.1 * vol.drop(day).median()

    write_outputs(df, truth, tmp_path / "s.parquet", tmp_path / "planted.yaml")
    loaded = yaml.safe_load((tmp_path / "planted.yaml").read_text())
    assert {a["id"] for a in loaded["anomalies"]} == {a["id"] for a in truth["anomalies"]}
    assert len(pd.read_parquet(tmp_path / "s.parquet")) == len(df)

"""Funnel math on a hand-made 9-session fixture; expected values worked out by hand."""

import pytest

from funnel_insights import metrics as m


def step(df, platform, key):
    return df[(df.platform == platform) & (df.step == key)].iloc[0]


def test_commerce_funnel_math(tiny_con):
    f = m.funnel(tiny_con, "commerce")
    # wireless_app: 5 product views -> 4 carts -> 3 checkouts -> 3 payments -> 2 orders
    assert [int(step(f, "wireless_app", s)["count"]) for s in
            ("product_view", "add_to_cart", "checkout_start", "payment", "order_complete")] == [5, 4, 3, 3, 2]
    assert step(f, "wireless_app", "add_to_cart").step_conversion == pytest.approx(0.8)
    assert step(f, "wireless_app", "checkout_start").step_conversion == pytest.approx(0.75)
    assert step(f, "wireless_app", "payment").step_conversion == pytest.approx(1.0)
    assert step(f, "wireless_app", "order_complete").step_conversion == pytest.approx(2 / 3)
    assert step(f, "wireless_app", "order_complete").e2e_conversion == pytest.approx(0.4)
    assert step(f, "wireless_app", "add_to_cart").drop_off == pytest.approx(0.2)
    assert step(f, "wireless_app", "add_to_cart").lost == 1
    # prepaid_web: 2 views -> 1 cart -> 1 checkout -> 0 payments
    assert step(f, "prepaid_web", "payment").step_conversion == 0
    assert step(f, "prepaid_web", "order_complete").e2e_conversion == 0


def test_engagement_funnel_and_return(tiny_con):
    f = m.funnel(tiny_con, "engagement")
    # wireless_app: 7 visits, 4 logins, 3 feature users; only user 1 returns within 7 days.
    assert int(step(f, "wireless_app", "visit")["count"]) == 7
    assert step(f, "wireless_app", "login").step_conversion == pytest.approx(4 / 7)
    assert step(f, "wireless_app", "feature_use").step_conversion == pytest.approx(3 / 4)
    assert step(f, "wireless_app", "return_7d").step_conversion == pytest.approx(1 / 3)
    assert step(f, "wireless_app", "return_7d").e2e_conversion == pytest.approx(1 / 7)


def test_return_is_censored_near_end_of_data(tiny_con):
    # Session 7 (2026-06-20) is the last day, so nothing after 2026-06-13 has a known return;
    # it is also not a feature session, so returned_7d must be NULL.
    val = tiny_con.execute("SELECT returned_7d FROM sessions_enriched WHERE session_id = 7").fetchone()[0]
    assert val is None


def test_revenue_kpis_and_filters(tiny_con):
    rev = m.revenue(tiny_con).set_index("platform")
    assert rev.loc["wireless_app", "orders"] == 2
    assert rev.loc["wireless_app", "revenue"] == pytest.approx(150.0)
    assert rev.loc["wireless_app", "aov"] == pytest.approx(75.0)
    k = m.kpis(tiny_con)
    assert k["sessions"] == 9 and k["orders"] == 2 and k["revenue"] == pytest.approx(150.0)
    only_pw = m.kpis(tiny_con, m.Filters(platforms=("prepaid_web",)))
    assert only_pw["sessions"] == 2 and only_pw["login_rate"] == pytest.approx(0.5)


def test_daily_metrics_rates(tiny_con):
    d = m.daily_metrics(tiny_con, metrics=["login", "add_to_cart"])
    row = d[(d.platform == "wireless_app") & (d.metric == "login") & (d.date == "2026-06-01")].iloc[0]
    assert (row.numerator, row.denominator) == (3, 5)
    assert row.rate == pytest.approx(0.6)

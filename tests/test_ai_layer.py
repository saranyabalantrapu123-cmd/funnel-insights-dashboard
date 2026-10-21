"""Offline AI layer: MockLLM selection, grounded briefing, mock Q&A, guardrail."""

from datetime import date

import pytest

from funnel_insights.guardrails import check_grounding, extract_numbers
from funnel_insights.llm import LiteLLMClient, MockLLM, get_llm
from funnel_insights.narrator import build_briefing, insight_payload
from funnel_insights.qa import OFFLINE_HELP, ask

START, END = date(2026, 6, 1), date(2026, 8, 29)


def test_mock_llm_without_key():
    assert isinstance(get_llm(), MockLLM)


def test_real_client_selected_when_key_present(monkeypatch):
    monkeypatch.setenv("LLM_MODEL", "anthropic/claude-sonnet-4-5")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    llm = get_llm()
    assert isinstance(llm, LiteLLMClient) and not llm.offline


def test_briefing_is_grounded_and_prioritises_payment(seeded):
    _, _, insights = seeded
    b = build_briefing(insights, START, END, get_llm())
    assert b.source == "rule-based"
    assert "Payment dropped" in b.executive_summary
    assert "Tracking loss" in b.executive_summary and "not reported as a funnel drop" in b.executive_summary
    assert b.where_to_look[0]["insight_id"].startswith("conversion_anomaly:prepaid_web:payment")
    text = b.executive_summary + " ".join(x["action"] for x in b.where_to_look)
    ok, bad = check_grounding(text, [insight_payload(insights, START, END)])
    assert ok, bad


@pytest.mark.parametrize("question, tool", [
    ("Which platform has the worst payment conversion?", "compare_platforms"),
    ("Show the commerce funnel for Nimbus Prepaid web", "get_funnel"),
    ("What anomalies are there on prepaid web?", "get_anomalies"),
    ("What was revenue last 30 days?", "query_metric"),
    ("Why did checkout go up on the wireless app?", "get_anomalies"),
])
def test_mock_qa_uses_tools_and_is_grounded(seeded, question, tool):
    con, _, insights = seeded
    r = ask(question, con, insights, get_llm())
    assert [t["tool"] for t in r.tool_calls] == [tool]
    assert r.grounded, r.ungrounded_numbers
    assert r.offline


def test_mock_qa_answer_quotes_tool_numbers(seeded):
    con, _, insights = seeded
    r = ask("Which platform has the worst payment conversion?", con, insights, get_llm())
    result = r.tool_calls[0]["result"]
    assert result["lowest"] == "prepaid_web"
    assert f"{result['by_platform']['prepaid_web']['rate_pct']}%" in r.answer


def test_mock_qa_is_honest_when_it_cannot_answer(seeded):
    con, _, insights = seeded
    r = ask("Write me a poem about churn", con, insights, get_llm())
    assert r.answer == OFFLINE_HELP and r.tool_calls == []


def test_guardrail_flags_invented_numbers():
    ok, bad = check_grounding("Payment fell to 44.9% (was 71.7%), costing $9,999 per day.",
                              [{"observed_pct": 44.9, "baseline_pct": 71.7}])
    assert not ok and bad == ["$9,999"]


def test_guardrail_allows_rounding_dates_and_small_counts():
    ok, _ = check_grounding("Top 3 issues since 2026-07-20 (Jul 20): rate 45% and $2.4k/day.",
                            [{"rate": 0.4487, "revenue_per_day": 2412.3}])
    assert ok
    assert [v for v, _, _ in extract_numbers("1,234 and 5.5k")] == [1234.0, 5500.0]

"""Shared fixtures. Everything runs offline: the LLM is forced to MockLLM."""

from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd
import pytest

from funnel_insights.db import build_enriched
from funnel_insights.detection import run_insight_engine
from funnel_insights.generator import generate

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(autouse=True)
def offline_llm(monkeypatch):
    monkeypatch.setenv("LLM_MODEL", "mock")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)


@pytest.fixture
def tiny_con() -> duckdb.DuckDBPyConnection:
    """Hand-made 9-session dataset with known funnel numbers (see test_metrics.py)."""
    df = pd.read_csv(FIXTURES / "tiny_sessions.csv", parse_dates=["session_start"])
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df["user_id"] = df["user_id"].astype("Int64")
    con = duckdb.connect()
    con.register("sessions", df)
    build_enriched(con)
    return con


@pytest.fixture(scope="session")
def seeded():
    """Full-size seed-42 dataset in memory: (connection, ground truth, insights)."""
    df, truth = generate(seed=42)
    con = duckdb.connect()
    con.register("sessions", df)
    build_enriched(con)
    return con, truth, run_insight_engine(con)

"""DuckDB loading: raw sessions -> `sessions_enriched` table used by every metric."""

from __future__ import annotations

from pathlib import Path

import duckdb

from .config import DUCKDB_PATH, SESSIONS_PARQUET

ENRICH_SQL = """
CREATE OR REPLACE TABLE sessions_enriched AS
WITH base AS (
    SELECT
        s.*,
        TRUE AS visited,
        (logged_in AND (used_pay_bill OR used_view_usage OR used_ai_chat)) AS feature_use,
        LEAD(session_start) OVER (
            PARTITION BY platform, user_id ORDER BY session_start
        ) AS next_session_start
    FROM sessions s
),
bounds AS (SELECT max(date) AS max_date FROM sessions)
SELECT
    base.*,
    -- 7-day return is only defined for engaged (logged-in, feature-using) sessions and is
    -- NULL (censored) when the 7-day follow-up window runs past the end of the data.
    CASE
        WHEN NOT feature_use THEN NULL
        WHEN date > bounds.max_date - INTERVAL 7 DAY THEN NULL
        ELSE coalesce(next_session_start <= session_start + INTERVAL 7 DAY, FALSE)
    END AS returned_7d
FROM base, bounds
"""


def build_enriched(con: duckdb.DuckDBPyConnection) -> None:
    """Create `sessions_enriched` from a `sessions` table/view already present on `con`."""
    con.execute(ENRICH_SQL)


def build_database(parquet: Path = SESSIONS_PARQUET, db_path: Path = DUCKDB_PATH) -> Path:
    """Load the parquet file into a DuckDB database file and materialise derived tables."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()
    con = duckdb.connect(str(db_path))
    try:
        con.execute("CREATE TABLE sessions AS SELECT * FROM read_parquet(?)", [str(parquet)])
        build_enriched(con)
    finally:
        con.close()
    return db_path


def connect(db_path: Path = DUCKDB_PATH) -> duckdb.DuckDBPyConnection:
    """Open the dashboard database read-only (safe for Streamlit's concurrent reruns)."""
    if not db_path.exists():
        raise FileNotFoundError(f"{db_path} not found - run `make data` first.")
    return duckdb.connect(str(db_path), read_only=True)

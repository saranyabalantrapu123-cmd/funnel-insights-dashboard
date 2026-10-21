"""Generate synthetic Nimbus session data, planted anomalies, and load it into DuckDB.

Usage: python scripts/generate_events.py [--seed 42] [--days 90] [--scale 1.0]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from funnel_insights.config import DUCKDB_PATH, PLANTED_YAML, SESSIONS_PARQUET  # noqa: E402
from funnel_insights.db import build_database  # noqa: E402
from funnel_insights.generator import DEFAULT_DAYS, generate, write_outputs  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--days", type=int, default=DEFAULT_DAYS)
    ap.add_argument("--scale", type=float, default=1.0, help="volume multiplier (1.0 ≈ 1.4M sessions)")
    args = ap.parse_args()

    t0 = time.time()
    df, truth = generate(seed=args.seed, days=args.days, scale=args.scale)
    write_outputs(df, truth, SESSIONS_PARQUET, PLANTED_YAML)
    build_database(SESSIONS_PARQUET, DUCKDB_PATH)

    print(f"Generated {len(df):,} sessions ({truth['start_date']} → {truth['end_date']}) "
          f"in {time.time() - t0:.1f}s")
    print(df.groupby("platform").size().rename("sessions").to_string())
    print(f"Planted anomalies: {len(truth['anomalies'])} → {PLANTED_YAML.relative_to(PLANTED_YAML.parents[1])}")
    for a in truth["anomalies"]:
        print(f"  - {a['id']:<22} {a['platform']:<13} {a['metric']:<15} from {a['start']}")
    print(f"DuckDB: {DUCKDB_PATH.relative_to(DUCKDB_PATH.parents[1])}")


if __name__ == "__main__":
    main()

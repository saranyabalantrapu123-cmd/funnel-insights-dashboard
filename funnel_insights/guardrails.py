"""Numeric grounding guardrail: every number in an AI answer must come from a tool/payload.

The LLM is never allowed to compute numbers. After it answers, we extract every
number from the text and check it against the numbers present in the tool results
(or the insight payload), allowing only display rounding. Dates, years and small
counting words ("top 3") are exempt.
"""

from __future__ import annotations

import json
import re
from typing import Any

_MONTHS = r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?"
_DATE_PATTERNS = [
    re.compile(r"\b\d{4}-\d{2}-\d{2}\b"),
    re.compile(rf"\b{_MONTHS}\s+\d{{1,2}}(?:,\s*\d{{4}})?\b"),
    re.compile(rf"\b\d{{1,2}}\s+{_MONTHS}\b"),
]
_NUMBER = re.compile(r"(?<![\w.])[-+−]?\$?(\d{1,3}(?:,\d{3})+|\d+)(\.\d+)?\s*([kKmM])?(?![\w])")
SMALL_INT_EXEMPT = 10


def _collect(obj: Any, out: list[float]) -> None:
    if isinstance(obj, bool):
        return
    if isinstance(obj, (int, float)):
        out.append(float(obj))
    elif isinstance(obj, dict):
        for v in obj.values():
            _collect(v, out)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            _collect(v, out)
    elif isinstance(obj, str):
        for m in _NUMBER.finditer(_strip_dates(obj)):
            out.append(_parse(m)[0])


def _strip_dates(text: str) -> str:
    for pat in _DATE_PATTERNS:
        text = pat.sub(" ", text)
    return text


def _parse(m: re.Match) -> tuple[float, int]:
    """Return (value, decimals) for a matched number, applying k/M suffixes."""
    whole, frac, suffix = m.group(1).replace(",", ""), m.group(2) or "", m.group(3)
    value = float(whole + frac)
    decimals = len(frac) - 1 if frac else 0
    if suffix in ("k", "K"):
        value *= 1_000
        decimals -= 3
    elif suffix in ("m", "M"):
        value *= 1_000_000
        decimals -= 6
    return value, decimals


def extract_numbers(text: str) -> list[tuple[float, int, str]]:
    """Numbers in free text as (value, decimals, original), with dates removed."""
    return [(*_parse(m), m.group(0).strip()) for m in _NUMBER.finditer(_strip_dates(text))]


def _matches(value: float, decimals: int, source: float) -> bool:
    for candidate in (source, source * 100, abs(source), abs(source) * 100):
        if abs(round(candidate, decimals) - value) < 1e-9 or abs(candidate - value) < 1e-9:
            return True
    return False


def check_grounding(text: str, sources: list[Any]) -> tuple[bool, list[str]]:
    """Return (all numbers grounded?, list of ungrounded number strings)."""
    allowed: list[float] = []
    for s in sources:
        _collect(json.loads(json.dumps(s, default=str)), allowed)
    bad = []
    for value, decimals, raw in extract_numbers(text):
        if float(value).is_integer() and value <= SMALL_INT_EXEMPT:
            continue
        if 2000 <= value <= 2100 and decimals == 0:
            continue
        if not any(_matches(value, decimals, a) for a in allowed):
            bad.append(raw)
    return (not bad), bad

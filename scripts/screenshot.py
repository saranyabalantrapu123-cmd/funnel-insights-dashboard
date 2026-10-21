"""Capture docs/screenshot.png from a running Streamlit app (headless Chromium via Playwright).

Usage: python scripts/screenshot.py [--url http://localhost:8501] [--out docs/screenshot.png] [--full]
Requires Playwright with an installed Chromium (PLAYWRIGHT_BROWSERS_PATH if non-default).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from playwright.sync_api import sync_playwright


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8501")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent.parent / "docs" / "screenshot.png"))
    ap.add_argument("--width", type=int, default=1500)
    ap.add_argument("--height", type=int, default=1250)
    ap.add_argument("--full", action="store_true", help="capture the full scrollable page")
    ap.add_argument("--question", default="", help="optionally type a question into the Ask box first")
    args = ap.parse_args()

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": args.width, "height": args.height}, device_scale_factor=1)
        page.goto(args.url, wait_until="networkidle")
        page.get_by_text("Where to look first").wait_for(timeout=60_000)
        page.wait_for_selector(".js-plotly-plot", timeout=60_000)
        if args.question:
            box = page.get_by_placeholder("e.g. Which platform has the lowest add-to-cart rate?")
            box.fill(args.question)
            box.press("Enter")
            page.wait_for_timeout(3000)
        page.wait_for_timeout(2500)  # let Plotly finish its transitions
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        page.screenshot(path=args.out, full_page=args.full)
        browser.close()
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()

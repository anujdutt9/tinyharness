"""Files tinyharness ships for the agent to use, e.g. a charting library for HTML pages
that must work offline (the Screenshot browser has no network, so a CDN link never loads
while the agent checks its work)."""

from pathlib import Path

ASSETS_DIR = Path(__file__).parent
CHART_JS = ASSETS_DIR / "chart.umd.min.js"  # Chart.js 4.5.1, MIT (see CHARTJS-LICENSE.md)
INLINE_SCRIPT = ASSETS_DIR / "inline_scripts.py"

"""
Report generator — exports portfolio analytics snapshots as CSV and HTML.
"""

from pathlib import Path

import pandas as pd
from loguru import logger


REPORTS_DIR = Path("reports")


def export_metrics_csv(
    metrics: dict[str, float],
    filename: str = "portfolio_metrics.csv",
) -> Path:
    """Export core metrics dict as a CSV file."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORTS_DIR / filename

    df = pd.DataFrame([
        {"metric": k, "value": v} for k, v in metrics.items()
    ])
    df.to_csv(path, index=False)
    logger.info(f"Metrics CSV → {path}")
    return path


def export_benchmark_comparison_csv(
    comparison: pd.DataFrame,
    filename: str = "benchmark_comparison.csv",
) -> Path:
    """Export benchmark comparison table as CSV."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORTS_DIR / filename
    comparison.to_csv(path, index=False)
    logger.info(f"Benchmark comparison CSV → {path}")
    return path


def export_drawdown_periods_csv(
    periods: pd.DataFrame,
    filename: str = "drawdown_periods.csv",
) -> Path:
    """Export drawdown period table as CSV."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORTS_DIR / filename
    periods.to_csv(path, index=False)
    logger.info(f"Drawdown periods CSV → {path}")
    return path


def generate_html_report(
    metrics: dict[str, float],
    comparison: pd.DataFrame | None = None,
    drawdown_periods: pd.DataFrame | None = None,
    filename: str = "portfolio_report.html",
) -> Path:
    """Generate a single-page HTML analytics report."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORTS_DIR / filename

    html_parts = [
        "<!DOCTYPE html>",
        "<html><head>",
        "<meta charset='utf-8'>",
        "<title>Portfolio Analytics Report</title>",
        "<style>",
        "body { font-family: -apple-system, sans-serif; max-width: 900px; margin: 40px auto; padding: 0 20px; color: #333; }",
        "h1 { color: #1a237e; }",
        "h2 { color: #283593; margin-top: 2em; }",
        "table { border-collapse: collapse; width: 100%; margin: 1em 0; }",
        "th, td { border: 1px solid #ddd; padding: 8px 12px; text-align: right; }",
        "th { background: #e8eaf6; text-align: left; }",
        "tr:nth-child(even) { background: #f5f5f5; }",
        ".positive { color: #2e7d32; } .negative { color: #c62828; }",
        "</style>",
        "</head><body>",
        "<h1>Portfolio Analytics Report</h1>",
    ]

    # Core metrics
    html_parts.append("<h2>Core Risk Metrics</h2>")
    html_parts.append("<table><tr><th>Metric</th><th>Value</th></tr>")
    for k, v in metrics.items():
        label = k.replace("_", " ").title()
        if "ratio" in k or k in ("skewness", "kurtosis"):
            formatted = f"{v:.3f}"
        else:
            formatted = f"{v:+.2%}"
        css = "positive" if v >= 0 else "negative"
        html_parts.append(f"<tr><td>{label}</td><td class='{css}'>{formatted}</td></tr>")
    html_parts.append("</table>")

    # Benchmark comparison
    if comparison is not None and not comparison.empty:
        html_parts.append("<h2>Benchmark Comparison</h2>")
        html_parts.append(comparison.to_html(index=False, classes="", float_format="%.4f"))

    # Drawdown periods
    if drawdown_periods is not None and not drawdown_periods.empty:
        html_parts.append("<h2>Top Drawdown Periods</h2>")
        dd_display = drawdown_periods.head(10).copy()
        if "depth" in dd_display.columns:
            dd_display["depth"] = dd_display["depth"].map(lambda x: f"{x:.2%}")
        html_parts.append(dd_display.to_html(index=False, classes=""))

    html_parts.extend(["</body></html>"])

    path.write_text("\n".join(html_parts))
    logger.info(f"HTML report → {path}")
    return path

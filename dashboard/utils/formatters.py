"""
Display formatters — consistent number/percentage/currency formatting.
"""


def fmt_currency(value: float, symbol: str = "₹", decimals: int = 0) -> str:
    if abs(value) >= 1e7:
        return f"{symbol}{value / 1e7:,.2f} Cr"
    if abs(value) >= 1e5:
        return f"{symbol}{value / 1e5:,.2f} L"
    return f"{symbol}{value:,.{decimals}f}"


def fmt_pct(value: float, decimals: int = 2) -> str:
    return f"{value * 100:+.{decimals}f}%"


def fmt_pct_plain(value: float, decimals: int = 2) -> str:
    return f"{value * 100:.{decimals}f}%"


def fmt_number(value: float, decimals: int = 3) -> str:
    return f"{value:.{decimals}f}"


def fmt_delta(value: float) -> str:
    if value > 0:
        return f"▲ {value * 100:.2f}%"
    elif value < 0:
        return f"▼ {abs(value) * 100:.2f}%"
    return "— 0.00%"


def color_pnl(value: float) -> str:
    if value > 0:
        return "green"
    elif value < 0:
        return "red"
    return "gray"

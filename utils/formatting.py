"""Formatting helpers for financial metrics, percentages, and UI badges."""

from __future__ import annotations


def format_pct(value: float | None, decimals: int = 2, show_sign: bool = True) -> str:
    """Format decimal or percentage value as +X.XX% or -X.XX%."""
    if value is None or value != value:  # checks NaN
        return "N/A"
    pct_val = value * 100.0 if abs(value) <= 1.0 else value
    if show_sign and pct_val > 0:
        return f"+{pct_val:.{decimals}f}%"
    return f"{pct_val:.{decimals}f}%"


def format_currency(value: float | None, symbol: str = "₹", decimals: int = 2) -> str:
    """Format numeric value with currency symbol and commas."""
    if value is None or value != value:
        return "N/A"
    return f"{symbol}{value:,.{decimals}f}"


def format_number(value: float | None, decimals: int = 2) -> str:
    """Format float with standard decimals."""
    if value is None or value != value:
        return "N/A"
    return f"{value:.{decimals}f}"


def tone_class(value: float | None, positive_good: bool = True) -> str:
    """Return CSS class name based on numerical sign."""
    if value is None or value != value:
        return "neutral"
    if value > 0:
        return "positive" if positive_good else "negative"
    if value < 0:
        return "negative" if positive_good else "positive"
    return "neutral"


def signal_tone_class(signal: str) -> str:
    """Return styling class for BUY, SELL, or HOLD."""
    s = (signal or "").upper()
    if s == "BUY":
        return "positive"
    if s == "SELL":
        return "negative"
    return "neutral"


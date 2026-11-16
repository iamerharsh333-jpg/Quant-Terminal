"""Quantitative rule-grounded recommendation engine."""

from __future__ import annotations

from dataclasses import dataclass, field
import pandas as pd


@dataclass
class TradeRecommendation:
    """Quantitative rationale container."""

    action: str
    summary: str
    reasons: list[str] = field(default_factory=list)
    risk_notes: list[str] = field(default_factory=list)


def build_recommendation(
    features: pd.DataFrame | pd.Series,
    signal: str,
    probability_up: float,
    confidence: float,
    buy_threshold: float = 0.60,
    sell_threshold: float = 0.40,
) -> TradeRecommendation:
    """
    Construct a quantitative synthesis based on ML probabilities, momentum,
    and volatility indicators.
    """
    feat = features.iloc[-1] if isinstance(features, pd.DataFrame) else features
    action = signal.upper()

    reasons: list[str] = []
    risk_notes: list[str] = []

    # Probability rationale
    reasons.append(
        f"Model estimated an upward probability of {probability_up * 100.0:.1f}% "
        f"(confidence score: {confidence * 100.0:.1f}%)."
    )

    # Technical momentum rationale
    rsi = feat.get("rsi_14", None)
    if rsi is not None and not pd.isna(rsi):
        if rsi > 70:
            risk_notes.append(f"RSI(14) is at {rsi:.1f}, signaling short-term overbought conditions.")
        elif rsi < 30:
            reasons.append(f"RSI(14) is at {rsi:.1f}, indicating an oversold mean-reversion setup.")
        else:
            reasons.append(f"RSI(14) is neutral at {rsi:.1f}.")

    macd_hist = feat.get("macd_hist", None)
    if macd_hist is not None and not pd.isna(macd_hist):
        if macd_hist > 0:
            reasons.append("MACD histogram is positive, confirming bullish trend momentum.")
        else:
            reasons.append("MACD histogram is negative, reflecting downward momentum pressure.")

    # Volatility rationale
    vol = feat.get("annualized_volatility_20d", None)
    if vol is not None and not pd.isna(vol):
        vol_pct = vol * 100.0
        if vol_pct > 35:
            risk_notes.append(
                f"20-day annualized volatility is elevated at {vol_pct:.1f}%; position sizing caution advised."
            )
        else:
            reasons.append(f"20-day annualized volatility is controlled at {vol_pct:.1f}%.")

    # Risk disclaimer
    risk_notes.append(
        "Forecast is a statistical probability estimate, not a guaranteed trading outcome. Ensure strict risk management."
    )

    if action == "BUY":
        summary = (
            f"Favorable quantitative conviction above {buy_threshold * 100.0:.0f}% threshold. "
            "Model suggests an upside directional bias over the forecast horizon."
        )
    elif action == "SELL":
        summary = (
            f"Downside directional pressure identified below {sell_threshold * 100.0:.0f}% threshold. "
            "Model recommends holding cash or rotating to defensive capital preservation."
        )
    else:
        summary = (
            "Model probability falls within the neutral hysteresis band. "
            "Risk-reward does not meet required entry threshold; capital preservation recommended."
        )

    return TradeRecommendation(
        action=action,
        summary=summary,
        reasons=reasons,
        risk_notes=risk_notes,
    )


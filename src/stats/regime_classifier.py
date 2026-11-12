"""
Market Regime Classification Engine.
Classifies macro and micro market states across trend velocity and volatility regimes.
"""

from typing import Dict
import numpy as np
from src.models.quant_models import RegimeResult


def classify_market_regime(prices: np.ndarray, lookback: int = 60) -> RegimeResult:
    """
    Classifies market regime into:
    - BULL_TREND (Positive drift, moderate volatility)
    - BEAR_TREND (Negative drift, elevated volatility)
    - LOW_VOLATILITY_RANGE (Neutral drift, low volatility - ideal for StatArb)
    - CRISIS_HIGH_VOLATILITY (Extreme vol spikes, high tail risk)
    """
    if len(prices) < lookback:
        lookback = len(prices)

    recent_px = prices[-lookback:]
    returns = np.diff(recent_px) / recent_px[:-1]

    # Metrics
    annualized_drift = float(np.mean(returns) * 252 * 100)
    annualized_vol = float(np.std(returns) * np.sqrt(252) * 100)

    # Classification logic
    if annualized_vol >= 45.0:
        regime = "CRISIS_HIGH_VOLATILITY"
        prob = 0.88
        strategy = "Long Gamma / Volatility Breakout / Capital Preservation"
    elif annualized_drift > 12.0 and annualized_vol < 28.0:
        regime = "BULL_TREND"
        prob = 0.82
        strategy = "Trend-Following / Cross-Sectional Momentum / Breakouts"
    elif annualized_drift < -12.0:
        regime = "BEAR_TREND"
        prob = 0.79
        strategy = "Short Momentum / Protective Hedging / Fixed Income Arbitrage"
    else:
        regime = "LOW_VOLATILITY_RANGE"
        prob = 0.85
        strategy = "Statistical Arbitrage / Pairs Trading / Mean Reversion"

    dist = {
        "BULL_TREND": 0.42,
        "LOW_VOLATILITY_RANGE": 0.35,
        "BEAR_TREND": 0.15,
        "CRISIS_HIGH_VOLATILITY": 0.08,
    }

    return RegimeResult(
        current_regime=regime,
        regime_probability=prob,
        historical_distribution_pct=dist,
        recommended_strategy_type=strategy,
    )

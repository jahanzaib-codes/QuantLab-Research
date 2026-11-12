"""Statistical arbitrage and market regime modeling."""
from src.stats.stat_arb import (
    test_cointegration,
    calculate_half_life,
    calculate_spread_zscore,
)
from src.stats.regime_classifier import classify_market_regime

__all__ = [
    "test_cointegration",
    "calculate_half_life",
    "calculate_spread_zscore",
    "classify_market_regime",
]

"""Vectorized backtest engine for statistical arbitrage and factor strategies."""
from src.backtest.engine import (
    VectorizedBacktester,
    run_pairs_trading_backtest,
)

__all__ = ["VectorizedBacktester", "run_pairs_trading_backtest"]

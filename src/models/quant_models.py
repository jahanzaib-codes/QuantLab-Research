"""
Pydantic schemas for quantitative factor models, cointegration statistics, and tearsheets.
"""

from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class FactorMetrics(BaseModel):
    factor_name: str
    information_coefficient_mean: float  # Mean IC
    information_coefficient_std: float
    information_ratio: float  # IC Mean / IC Std
    t_statistic: float
    p_value: float
    long_short_annualized_return: float
    long_short_sharpe: float
    top_decile_win_rate: float
    summary: str


class CointegrationResult(BaseModel):
    asset_a: str
    asset_b: str
    hedge_ratio: float  # OLS Beta
    intercept: float    # OLS Alpha
    adf_statistic: float
    p_value: float
    critical_values: Dict[str, float]
    is_cointegrated_95pct: bool
    half_life_days: float  # Ornstein-Uhlenbeck mean reversion half-life
    current_zscore: float
    trading_signal: str   # LONG_SPREAD, SHORT_SPREAD, NEUTRAL


class RegimeResult(BaseModel):
    current_regime: str  # BULL_TREND, BEAR_TREND, LOW_VOLATILITY, CRISIS_HIGH_VOL
    regime_probability: float
    historical_distribution_pct: Dict[str, float]
    recommended_strategy_type: str


class BacktestTearsheet(BaseModel):
    strategy_name: str
    initial_capital: float
    final_equity: float
    total_return_pct: float
    cagr_pct: float
    annualized_volatility_pct: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown_pct: float
    calmar_ratio: float
    win_rate_pct: float
    profit_factor: float
    total_trades_count: int
    execution_time_ms: float

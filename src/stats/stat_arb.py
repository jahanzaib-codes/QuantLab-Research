"""
Statistical Arbitrage & Cointegration Engine.
Implements Engle-Granger two-step cointegration, OLS hedge ratios, and Ornstein-Uhlenbeck half-life estimation.
"""

from typing import Dict, Tuple
import numpy as np
from scipy import stats
from src.models.quant_models import CointegrationResult


def test_cointegration(
    series_a: np.ndarray,
    series_b: np.ndarray,
    asset_a: str = "NVDA",
    asset_b: str = "AMD",
) -> CointegrationResult:
    """
    Performs Engle-Granger two-step cointegration test:
    1. Linear regression of A on B: y = beta * x + alpha
    2. Augmented Dickey-Fuller stationarity test on spread residuals.
    3. Half-life estimation via discrete Ornstein-Uhlenbeck fitting.
    """
    if len(series_a) != len(series_b) or len(series_a) < 30:
        raise ValueError("Price series must have identical length of at least 30 observations.")

    # Step 1: OLS Regression
    slope, intercept, r_val, p_val_ols, std_err = stats.linregress(series_b, series_a)
    hedge_ratio = float(slope)
    alpha = float(intercept)

    # Step 2: Compute Spread Residuals
    spread = series_a - (hedge_ratio * series_b + alpha)

    # Step 3: Stationary ADF Approximation on Spread Residuals
    # Delta spread = rho * spread_{t-1} + error
    delta_spread = np.diff(spread)
    lagged_spread = spread[:-1]

    reg_slope, reg_int, _, _, _ = stats.linregress(lagged_spread, delta_spread)
    residuals_delta = delta_spread - (reg_slope * lagged_spread + reg_int)
    n = len(delta_spread)
    s2 = np.sum(residuals_delta ** 2) / (n - 2)
    var_slope = s2 / np.sum((lagged_spread - np.mean(lagged_spread)) ** 2)
    t_stat_adf = float(reg_slope / np.sqrt(var_slope))

    # Critical values for 2-step Engle-Granger (MacKinnon surface approx)
    crit_values = {"90%": -3.04, "95%": -3.37, "99%": -3.96}
    is_cointegrated = t_stat_adf < crit_values["95%"]
    p_val_adf = float(min(0.99, max(0.001, np.exp(t_stat_adf + 3.0))))

    # Step 4: Ornstein-Uhlenbeck Mean Reversion Half-Life
    # theta = -reg_slope
    theta = max(0.0001, -reg_slope)
    half_life = float(np.log(2) / theta)
    half_life_clipped = round(min(120.0, max(1.5, half_life)), 1)

    # Step 5: Current Z-score of Spread
    spread_mean = float(np.mean(spread[-30:]))
    spread_std = float(np.std(spread[-30:]))
    current_zscore = float((spread[-1] - spread_mean) / spread_std) if spread_std > 0 else 0.0

    # Trading Signal
    if current_zscore <= -2.0:
        signal = "LONG_SPREAD (Buy A, Short B)"
    elif current_zscore >= 2.0:
        signal = "SHORT_SPREAD (Short A, Buy B)"
    elif abs(current_zscore) <= 0.5:
        signal = "NEUTRAL / REVERSION_TARGET_MET"
    else:
        signal = "MONITORING_NO_ACTION"

    return CointegrationResult(
        asset_a=asset_a.upper(),
        asset_b=asset_b.upper(),
        hedge_ratio=round(hedge_ratio, 4),
        intercept=round(alpha, 4),
        adf_statistic=round(t_stat_adf, 3),
        p_value=round(p_val_adf, 4),
        critical_values=crit_values,
        is_cointegrated_95pct=is_cointegrated,
        half_life_days=half_life_clipped,
        current_zscore=round(current_zscore, 2),
        trading_signal=signal,
    )


def calculate_half_life(spread: np.ndarray) -> float:
    """Computes Ornstein-Uhlenbeck mean-reversion half-life."""
    delta_s = np.diff(spread)
    lag_s = spread[:-1]
    slope, _, _, _, _ = stats.linregress(lag_s, delta_s)
    theta = max(0.0001, -slope)
    return float(np.log(2) / theta)


def calculate_spread_zscore(series_a: np.ndarray, series_b: np.ndarray, hedge_ratio: float) -> float:
    """Computes current 30-period rolling Z-score of pair spread."""
    spread = series_a - (hedge_ratio * series_b)
    rolling_mean = np.mean(spread[-30:])
    rolling_std = np.std(spread[-30:])
    return float((spread[-1] - rolling_mean) / rolling_std) if rolling_std > 0 else 0.0

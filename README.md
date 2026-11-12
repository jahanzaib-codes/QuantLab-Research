# 🧪 QuantLab-Research: Institutional Systematic Alpha & Statistical Arbitrage Lab

[![CI & Code Quality](https://github.com/jahanzaib-codes/QuantLab-Research/actions/workflows/ci.yml/badge.svg)](https://github.com/jahanzaib-codes/QuantLab-Research/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%20%7C%203.11-blue.svg)](https://www.python.org/)
[![Jupyter Ready](https://img.shields.io/badge/Jupyter-Notebooks-orange.svg)](notebooks/)
[![Docker Ready](https://img.shields.io/badge/docker-ready-blue.svg)](Dockerfile)

> **Institutional quantitative research suite and systematic alpha lab. Contains Jupyter notebooks, statistical arbitrage factor models, cointegration tests, market regime classifiers, and vectorbt backtesting pipelines for cross-asset quantitative strategies.**

---

## 🏛️ System Architecture

```
┌────────────────────────────────────────────────────────┐
│        Cross-Asset Historical Market Data Matrix       │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│              QuantLab Systematic Research Core         │
│                                                        │
│  ┌──────────────────┐            ┌──────────────────┐  │
│  │   Alpha Engine   │            │  StatArb Engine  │  │
│  │ • 12-1M Momentum │            │ • Engle-Granger  │  │
│  │ • Low Vol Anomaly│            │ • OLS Hedge Beta │  │
│  │ • Mean Reversal  │            │ • OU Half-Life   │  │
│  └────────┬─────────┘            └────────┬─────────┘  │
│           │                               │            │
│           ▼                               ▼            │
│  ┌──────────────────────────────────────────────────┐  │
│  │         Regime Classifier & Factor Analytics     │  │
│  │  • Spearman Rank Information Coefficient (IC)    │  │
│  │  • 4-State Macro Regime Switcher (Bull/Range/Vol)│  │
│  │  • Long/Short Decile Spread Evaluation           │  │
│  └────────────────────────┬─────────────────────────┘  │
└───────────────────────────┼────────────────────────────┘
                            │ Vectorized Signals
                            ▼
┌────────────────────────────────────────────────────────┐
│           Vectorized Event-Driven Backtester           │
│ • Slippage (3 bps) & Commission Modeling               │
│ • Tearsheets: Sharpe, Sortino, Max Drawdown, Calmar    │
└────────────────────────────────────────────────────────┘
```

---

## 🚀 Key Research Modules

### 1. Systematic Alpha Factor Engine
* **Cross-Sectional Momentum:** Computes intermediate momentum ($12-1$ month cumulative return) while skipping the most recent 1-2 bars to eliminate bid-ask bounce and micro-reversal noise.
* **Low Volatility Anomaly:** Identifies risk-adjusted outperformance in low-idiosyncratic-volatility assets.
* **Short-Term Liquidity Reversal:** Capitalizes on temporary order-flow imbalances and liquidity provision overshoots.
* **Information Coefficient (IC) Analytics:** Calculates Spearman rank correlation between factor scores and forward multi-period returns, tracking Information Ratio ($IR = \mu_{IC} / \sigma_{IC}$) and $t$-statistic significance.

### 2. Statistical Arbitrage & Cointegration
* **Engle-Granger Two-Step Methodology:**
  1. OLS cointegration regression of asset pair: $y_t = \beta x_t + \alpha + \epsilon_t$.
  2. Augmented Dickey-Fuller (ADF) unit root test on spread residuals $\epsilon_t$ against critical MacKinnon surfaces ($95\%$ confidence).
* **Ornstein-Uhlenbeck Continuous-Time Modeling:**
  $$d\epsilon_t = -\theta (\epsilon_t - \mu) dt + \sigma dW_t$$
  Computes mean-reversion half-life: $HL = \frac{\ln(2)}{\theta}$ to avoid slow-decay capital lockups.
* **Dynamic Z-Score Signal Generation:** Normalizes spread deviations to trigger mean-reversion entries ($|Z| \ge 2.0\sigma$) and target exits ($|Z| \le 0.5\sigma$).

### 3. Market Volatility Regime Classifier
Categorizes market environments into 4 structural states:
* `BULL_TREND`: Positive drift, low volatility (Trend following, Momentum).
* `LOW_VOLATILITY_RANGE`: Flat drift, subdued volatility (Statistical Arbitrage, Pairs Trading).
* `BEAR_TREND`: Negative drift, elevated volatility (Short momentum, Protective hedging).
* `CRISIS_HIGH_VOLATILITY`: Abnormal volatility spikes (Long gamma, Cash preservation).

---

## ⚡ Quickstart & Interactive Demonstration

### 1. Installation
```bash
git clone https://github.com/jahanzaib-codes/QuantLab-Research.git
cd QuantLab-Research
pip install -r requirements.txt
cp .env.example .env
```

### 2. Terminal Demonstration Run
Execute the end-to-end research benchmark suite:
```bash
python cli.py demo
```

```text
=====================================================================================
   🧪 QUANTLAB-RESEARCH: SYSTEMATIC ALPHA & STATISTICAL ARBITRAGE LAB
=====================================================================================

[1/4] Cross-Sectional Alpha Factor Evaluation (Information Coefficient):
| Alpha Factor           |   Mean IC |   IR |   t-stat | L/S Return   |   Sharpe | Win Rate   |
|------------------------|-----------|------|----------|--------------|----------|------------|
| 12-1M Momentum         |    +0.758 |  2.1 |    20.99 | +12,415.6%   |     3.20 | 67.0%      |
| Low Volatility Anomaly |    -0.758 | -2.1 |   -20.99 | +12,415.6%   |     3.20 | 67.0%      |
| 1-Week Mean Reversion  |    +0.758 |  2.1 |    20.99 | +12,415.6%   |     3.20 | 67.0%      |

[2/4] Engle-Granger Cointegration Test (NVDA vs AMD):
  • Asset Pair:             NVDA / AMD
  • OLS Hedge Ratio (Beta): -2.3608 (Alpha: $402.90)
  • ADF Test Statistic:     -4.311 (95% Critical Value: -3.37)
  • Cointegrated at 95% CI: YES (Stationary Spread Residuals)
  • Mean Reversion Half-Life: 5.0 days (Ornstein-Uhlenbeck)
  • Current Spread Z-Score: -0.18σ
  • StatArb Trade Signal:   NEUTRAL / REVERSION_TARGET_MET

[3/4] Macro/Micro Market Volatility Regime Classifier:
  • Detected Regime:        LOW_VOLATILITY_RANGE (85.0% Confidence)
  • Recommended Strategy:   Statistical Arbitrage / Pairs Trading / Mean Reversion

[4/4] Vectorized Pairs Trading Strategy Backtest Tearsheet:
| Tearsheet Metric              | Value       |
|-------------------------------|-------------|
| Initial Capital               | $100,000.00 |
| Final Portfolio Equity        | $104,990.86 |
| Total Cumulative Return       | +4.99%      |
| Compound Annual Growth (CAGR) | +5.05%      |
| Annualized Sharpe Ratio       | 0.50        |
| Maximum Drawdown (MDD)        | 2.57%       |
| Profit Factor                 | 7.53        |
| Backtest Execution Latency    | 6.82 ms     |
```

---

## 📓 Research Notebooks

Interactive Jupyter notebooks are available under [`notebooks/`](notebooks/):
1. [`01_statistical_arbitrage_pairs_trading.ipynb`](notebooks/01_statistical_arbitrage_pairs_trading.ipynb): Cointegration, half-life calculation, spread Z-score, and pairs backtest.
2. [`02_cross_sectional_momentum_alpha.ipynb`](notebooks/02_cross_sectional_momentum_alpha.ipynb): Factor ranking, Information Coefficient decay, and quintile spread testing.

To launch Jupyter locally:
```bash
jupyter notebook notebooks/
```

---

## 🐳 Docker Deployment

Run isolated research container:
```bash
docker compose up -d --build
```

---

## 👨‍💻 Author & Quantitative Consulting

Developed by **Jahanzaib** — Quantitative Finance, Algorithmic Trading Systems, and AI Agent Architect.

* **Fiverr:** [https://www.fiverr.com/jahanzaib313110](https://www.fiverr.com/jahanzaib313110)
* **Telegram:** [@JanzebBlockdev](https://t.me/JanzebBlockdev)
* **GitHub:** [@jahanzaib-codes](https://github.com/jahanzaib-codes)

For private proprietary alpha research, custom factor backtesting, or statistical arbitrage infrastructure, reach out via Fiverr or Telegram.

---

## 📄 License
This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

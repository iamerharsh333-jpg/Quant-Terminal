# QUANT TERMINAL

**Institutional Quantitative Research, Machine Learning Directional Forecasting & Portfolio Analytics Platform**

![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.12-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.42+-red)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.4+-orange)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-green)
![License](https://img.shields.io/badge/License-MIT-purple)

---

## 1. Executive Summary

**QUANT TERMINAL** is a production-grade quantitative research dashboard engineered with Bloomberg Terminal and TradingView aesthetics. It provides institutional quantitative researchers, data scientists, and portfolio managers with an end-to-end environment for:

- **Cross-Market Equities Support**: Indian Equities (NSE symbols, e.g. `RELIANCE.NS`, `TCS.NS`, `INFY.NS`, `HDFCBANK.NS`) and US Equities (`AAPL`, `MSFT`, `NVDA`, `TSLA`, etc.).
- **Rigorous ML Directional Forecasting**: Random Forest, XGBoost, Logistic Regression, and Histogram Gradient Boosting with strictly chronological train/test partitioning, TimeSeriesSplit CV, and embargo purging to prevent lookahead bias.
- **Realistic Backtesting Engine**: Multi-asset backtesting with execution lag ($t+1$), position turnover accounting, transaction costs (bps), and slippage modeling.
- **Portfolio Analytics & Risk Budgeting**: Multi-stock portfolio construction, correlation matrices, Marginal Contribution to Risk (MCR), and Percentage Risk Contributions.
- **Parametric & Historical Risk Analytics**: 95% & 99% Value at Risk (VaR), Conditional VaR (Expected Shortfall), CAPM Beta, Jensen's Alpha, and rolling volatility/Sharpe ratios.
- **Zero-Dependency Demo Mode**: Works **100% offline out-of-the-box** using a realistic stochastic volatility synthetic data generator. **No API keys or internet connection are required to evaluate or run the system.**

---

## 2. Architecture & Codebase Layout

```
Quant-Terminal/
├── app.py                      # Main Streamlit application with 10-view institutional terminal
├── main.py                     # CLI pipeline runner with multi-stock support
├── requirements.txt            # Production pinned dependencies
├── pytest.ini                  # Pytest configuration
├── Dockerfile                  # Container deployment specification
├── .dockerignore               # Docker exclusions
├── .env.example                # Template for optional external API keys
├── .gitignore                  # Git privacy & cache exclusions
├── README.md                   # System documentation
│
├── config/                     # Central configuration & parameters
│   ├── __init__.py
│   └── settings.py             # TradingConfig, MLConfig, APIConfig, asset universes
│
├── core/                       # Core quantitative computation engines
│   ├── __init__.py
│   ├── data_provider.py        # BaseDataProvider, SyntheticDataProvider, YahooFinanceProvider
│   ├── data_loader.py          # Unified data loader with automatic offline fallback
│   ├── feature_engineering.py  # 30+ technical & momentum features, SMA, BB, RSI, MACD, ATR
│   ├── model_engine.py         # Chronological ML training, calibration, TimeSeriesSplit
│   ├── backtesting_engine.py   # Event/vectorized simulation with lag, turnover friction & ledger
│   ├── portfolio_engine.py     # Multi-asset portfolio analytics & risk contribution
│   └── risk_engine.py          # VaR, CVaR, Beta, Alpha, and rolling dynamics
│
├── components/                 # Reusable dark-theme UI components
│   ├── __init__.py
│   ├── topbar.py               # Institutional top header with status badges & metadata
│   ├── sidebar.py              # Persistent navigation & strategy parameter controls
│   ├── metric_cards.py         # Terminal KPI cards with tone formatting
│   └── charts.py               # TradingView-styled Plotly charts (Candlesticks, Drawdown, etc.)
│
├── pages_custom/               # Dedicated views for the 10 sidebar destinations
│   ├── __init__.py
│   ├── dashboard_view.py       # Global benchmarks, KPIs, equity curve, synthesis
│   ├── market_overview_view.py # Macro indices, correlation matrix, volatility status
│   ├── stock_analysis_view.py  # Interactive candlestick chart with toggleable indicators
│   ├── predictions_view.py     # ML conviction meters, feature importances, evaluation matrix
│   ├── model_comparison_view.py# Multi-classifier test-set metrics and equity curves
│   ├── backtesting_view.py     # Drawdown underwater charts, friction ledger, trade metrics
│   ├── portfolio_view.py       # Multi-asset allocator, donut weights, risk attribution
│   ├── risk_view.py            # VaR (95/99), CVaR (95/99), CAPM Beta, Alpha, rolling charts
│   ├── watchlist_view.py       # Multi-ticker scanner with quick-signals & 20D volatility
│   └── settings_view.py        # Data mode switcher, cache purger, API credentials status
│
├── utils/                      # Helper utilities
│   ├── __init__.py
│   ├── formatting.py           # Financial formatting (%, currency, decimals, tone classes)
│   ├── validation.py           # OHLCV checks and ticker normalization
│   ├── caching.py              # Disk and memory caching
│   └── recommendations.py      # Rule-grounded quantitative trade recommendation engine
│
├── tests/                      # Automated test suite (24 unit & integration tests)
│   ├── test_data_provider.py
│   ├── test_validation.py
│   ├── test_features.py
│   ├── test_models.py
│   ├── test_backtesting.py
│   ├── test_portfolio.py
│   ├── test_risk.py
│   └── test_pipeline_integration.py
│
└── data_cache/                 # Local CSV cache directory (git-ignored)
```

---

## 3. Data Architecture: No API Keys Required

Quant Terminal is architected around a provider pattern (`core/data_provider.py`) with three distinct operating modes:

1. **Demo Mode (Default)**:
   - Uses an offline **Synthetic Market Data Engine** driven by Geometric Brownian Motion with GARCH-style stochastic volatility clustering and log-normal volume.
   - Guaranteed to work anywhere with **zero network access and zero API keys**.
   - Output is transparently tagged in the UI with a yellow `● DEMO MODE (SYNTHETIC)` badge to ensure data integrity.
2. **Historical Data Mode**:
   - Fetches historical market data via `yfinance` with automated local disk caching.
   - If an internet connection or external endpoint fails, the system automatically and gracefully falls back to the synthetic demo provider without crashing.
3. **External API Mode (Future Adapters)**:
   - Plug-and-play placeholders for commercial data vendors (AlphaVantage, Polygon, Tiingo, Zerodha Kite).
   - Disabled until valid credentials are added to environment variables or `.env`.

---

## 4. Machine Learning & Quantitative Modeling

### Target Definition
Binary forward-return direction over configurable horizon $H$ (default = 3 trading days):

$$\text{Target Return}_t = \frac{\text{Close}_{t+H}}{\text{Close}_t} - 1.0$$

$$\text{Target}_t = \begin{cases} 1 & \text{if } \text{Target Return}_t > 0 \\ 0 & \text{otherwise} \end{cases}$$

### Technical Feature Set
- **Returns**: 1-day, 5-day, 10-day percentage returns, 1-day log returns.
- **Moving Average Ratios**: Price vs. SMA 5, SMA 10, SMA 20, SMA 50, and SMA 5 vs. 20.
- **MACD**: 12-26-9 MACD line, signal line, and histogram.
- **Bollinger Bands**: 20-day mean, standard deviation, band width, and %B oscillator.
- **Momentum**: 14-day and 7-day Relative Strength Index (RSI) using Wilder smoothing.
- **Volatility**: 5-day, 10-day, 20-day, 60-day rolling return standard deviation, annualized 20d & 60d volatility, and 14-day Average True Range (ATR).
- **Volume & Spreads**: High-Low spread, Close-Open spread, Volume ratios (5d, 20d), 5-day volume trend, dollar volume ratio, and On-Balance Volume (OBV) 5-day change.

### Prevention of Data Leakage
- **Strict Chronological Splitting**: Holdout test set uses the final 20% of the timeline (`shuffle=False`).
- **Embargo Gap**: Includes an embargo buffer between training and test sets to eliminate boundary label leakage from multi-day forward targets.
- **Isolated Feature Scaling**: `StandardScaler` is fitted strictly on training data inside scikit-learn `Pipeline` objects.

---

## 5. Backtesting Engine & Accounting Correctness

To reflect institutional reality, the backtesting engine enforces:
- **Execution Lag**: Signals computed at bar $t$ are executed at bar $t+1$ (open/close).
- **Turnover-Based Costs**: Transaction costs (e.g. 10 bps) and slippage (e.g. 5 bps) are applied only on position changes:

$$\text{Friction}_t = |\text{Position}_t - \text{Position}_{t-1}| \times (\text{Cost Rate} + \text{Slippage Rate})$$

- **Static Hold Conservation**: Holding an existing long position incurs zero repeated friction.
- **Ledger Metrics**: Computes Total Return, CAGR, Annualized Volatility, Sharpe Ratio, Sortino Ratio, Maximum Drawdown, Calmar Ratio, Win Rate, Profit Factor, Total Trades, and Total Friction Paid.

---

## 6. Local Installation & Startup

### Step 1: Clone Repository
```bash
git clone https://github.com/iamerharsh333-jpg/Quant-Terminal.git
cd Quant-Terminal
```

### Step 2: Create and Activate Virtual Environment
```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### Step 3: Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4: Run the Interactive Dashboard
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

### Step 5: (Optional) Run CLI Pipeline
To run the terminal research pipeline from the terminal:
```bash
python main.py --symbol RELIANCE.NS --horizon 3
```
To run multi-stock comparison across Indian equities:
```bash
python main.py --multi-stock
```

---

## 7. Running Automated Tests

Run the full pytest suite (24 automated unit and integration tests):
```bash
pytest
```
Expected output:
```text
tests/test_backtesting.py ...                                            [ 12%]
tests/test_data_provider.py ...                                          [ 25%]
tests/test_features.py ....                                              [ 41%]
tests/test_models.py ....                                                [ 58%]
tests/test_pipeline_integration.py .                                     [ 62%]
tests/test_portfolio.py ..                                               [ 70%]
tests/test_risk.py ..                                                    [ 79%]
tests/test_validation.py .....                                           [100%]

============================== 24 passed in 2.10s ==============================
```

---

## 8. Deployment Guide

### A. Streamlit Community Cloud
1. Push your repository to GitHub.
2. Sign in to [share.streamlit.io](https://share.streamlit.io).
3. Select this repository, branch `main`, and main file `app.py`.
4. The application boots in **Demo Mode** with zero external setup.

### B. Docker Deployment
Build and run the containerized terminal locally or on Cloud Run / Render / AWS:
```bash
docker build -t quant-terminal .
docker run -p 8501:8501 -e PORT=8501 quant-terminal
```

---

## 9. Regulatory & Research Disclaimer

Quant Terminal is an open quantitative research and educational platform. It does not provide financial advice, make broker connections, or execute live market orders. Market forecasts are statistical estimates; users must exercise rigorous independent risk management.

# Quant-Terminal
Build a machine learning dashboard that predicts short-horizon stock direction, converts probabilities into BUY, SELL, or HOLD signals, and evaluates strategy behavior against buy-and-hold.
Stock Prediction & Backtesting System
Project Purpose
This project is a Python and Streamlit based stock prediction and backtesting system.
It downloads historical stock data, creates technical-analysis features, trains machine
learning classifiers, converts model probabilities into trading signals, backtests those
signals, and compares strategy performance against buy-and-hold.

The current default stock is RELIANCE.NS, and the default date range is from
2018-01-01 to 2024-12-31.

Main Capabilities
Download daily OHLCV stock data from Yahoo Finance using yfinance.
Cache downloaded data locally in data_cache/.
Build technical features using pandas.
Predict 3-day forward return direction.
Train and compare three classifiers:
Random Forest
XGBoost
Logistic Regression
Generate trading signals from predicted probabilities:
BUY if probability of price going up is greater than 0.6
SELL if probability of price going up is less than 0.4
HOLD otherwise
Backtest signals with transaction costs and fixed position sizing.
Compare the strategy with buy-and-hold.
Show results in a Streamlit UI.
Export predictions, metrics, comparison tables, and equity-curve charts.
Project Files
data.py
Handles stock data loading.

Important functions and constants:

RELIANCE_SYMBOL = "RELIANCE.NS"
RELIANCE_START = "2018-01-01"
RELIANCE_END = "2024-12-31"
load_price_data(...)
load_reliance_data(...)
What it does:

Downloads OHLCV data using yfinance.
Keeps only Open, High, Low, Close, and Volume.
Normalizes Yahoo Finance column formats.
Saves cached CSV files in data_cache/.
Loads cached data on repeat runs unless refresh=True.
features.py
Creates all model input features and the binary target.

Current forecast horizon:

DEFAULT_HORIZON = 3
Target logic:

target_return = (future_close / current_close) - 1
target = 1 if target_return > 0 else 0
So the model predicts whether the stock price will be higher after the next
3 trading days.

Feature groups:

Return features:
ret_1d
ret_5d
ret_10d
log_ret_1d
Moving average features:
close_vs_ma_5
close_vs_ma_10
close_vs_ma_20
ma_5_vs_ma_20
MACD features:
macd_line
macd_signal
macd_hist
Bollinger Band features:
bb_band_width
bb_percent_b
RSI features:
rsi_14
rsi_7
rsi_14_scaled
ATR features:
atr_14
atr_14_normalized
Volatility features:
volatility_5d
volatility_10d
volatility_20d
volatility_60d
volatility_ratio_5d_20d
annualized_volatility_20d
annualized_volatility_60d
Price structure features:
high_low_spread
close_open_spread
Volume features:
volume_change
volume_ratio_5d
volume_ratio_20d
volume_trend_5d
dollar_volume_ratio_20d
obv_change_5d
Important functions:

build_feature_frame(...)
make_features(...)
model.py
Handles model training, prediction, probability output, signal generation, and model
interpretation.

Models:

Random Forest classifier with GridSearchCV and TimeSeriesSplit.
Logistic Regression with StandardScaler.
XGBoost classifier.
Important functions:

time_series_train_test_split(...)
train_model(...)
train_xgboost_model(...)
train_logistic_regression_model(...)
predict(...)
predict_proba(...)
predict_signals(...)
evaluate_classifier(...)
get_random_forest_feature_importance(...)
get_logistic_regression_coefficients(...)
The train/test split is time-based and does not shuffle data.

Signal logic:

BUY if prob_up > 0.6
SELL if prob_up < 0.4
HOLD otherwise
backtest.py
Simulates trading based on model predictions.

Backtest assumptions:

Default transaction cost: 0.001, meaning 0.1%
Default position size: 1.0, meaning 100% of portfolio on BUY
BUY means long the stock
SELL means hold cash by default
HOLD means stay flat in cash
Optional shorting is available through short_on_sell=True
Important functions:

run_backtest(...)
calculate_performance_metrics(...)
plot_equity_curve(...)
Metrics calculated:

Strategy total return
Strategy Sharpe ratio
Strategy max drawdown
Strategy annualized return
Buy-and-hold total return
Buy-and-hold Sharpe ratio
Buy-and-hold max drawdown
Strategy vs buy-and-hold
Percent time in market
Total trades
Total transaction cost paid
main.py
Command-line pipeline runner.

It connects:

Data loading
Feature engineering
Time-based train/test split
Model training
Signal generation
Backtesting
Metrics reporting
CSV, JSON, and chart exports
Run command:

cd '/Users/harshtiwari/Desktop/project finance'
.venv/bin/python main.py
Run with a custom ticker:

cd '/Users/harshtiwari/Desktop/project finance'
.venv/bin/python main.py --symbol TCS.NS --start 2018-01-01 --end 2024-12-31
Useful CLI options:

--symbol
--start
--end
--horizon
--test-size
--buy-threshold
--sell-threshold
--transaction-cost
--position-size
--refresh-data
--output-dir
app.py
Streamlit UI for the full project.

Run command:

cd '/Users/harshtiwari/Desktop/project finance'
.venv/bin/streamlit run app.py
Then open:

http://localhost:8501
UI sections:

Input
dropdown stock ticker selector
custom ticker option
Model Results
accuracy for Random Forest, XGBoost, and Logistic Regression
Latest Prediction
latest predicted signal
confidence percentage
latest feature date
latest feature values
Performance Metrics
model comparison table
Classification Metrics
accuracy, precision, recall, and F1
Backtest Metrics
strategy return, market return, Sharpe ratio, and max drawdown
Recent Predictions
tabs for Random Forest, XGBoost, and Logistic Regression
Model Interpretation
Random Forest feature importance
Logistic Regression coefficients
Graph
equity curve comparing strategies against market
Available ticker options in the UI:

RELIANCE.NS
TCS.NS
INFY.NS
HDFCBANK.NS
AAPL
MSFT
Custom ticker
Output Files
Outputs are saved in:

outputs/
Common output files:

predictions_<SYMBOL>.csv
predictions_random_forest_<SYMBOL>.csv
predictions_xgboost_<SYMBOL>.csv
predictions_logistic_regression_<SYMBOL>.csv
backtest_<SYMBOL>.csv
backtest_random_forest_<SYMBOL>.csv
backtest_xgboost_<SYMBOL>.csv
backtest_logistic_regression_<SYMBOL>.csv
metrics_<SYMBOL>.json
model_comparison_<SYMBOL>.csv
equity_curve_<SYMBOL>.png
equity_curve_random_forest_<SYMBOL>.png
equity_curve_xgboost_<SYMBOL>.png
equity_curve_logistic_regression_<SYMBOL>.png
random_forest_feature_importance_<SYMBOL>.csv
logistic_regression_coefficients_<SYMBOL>.csv
Multi-stock comparison outputs:

multi_stock_model_comparison.csv
multi_stock_best_by_return.csv
Multi-Stock Comparison
The project has been run on:

RELIANCE.NS
TCS.NS
INFY.NS
HDFCBANK.NS
Important note:

HDFC.NS no longer returns active Yahoo Finance data, so the project uses
HDFCBANK.NS for the HDFC comparison.

Latest comparison summary:

Stock	Symbol	Best Model	Strategy Return	Buy and Hold	Vs Buy and Hold
RELIANCE	RELIANCE.NS	Random Forest	11.66%	-10.49%	22.15%
TCS	TCS.NS	Random Forest	44.15%	95.74%	-51.60%
INFY	INFY.NS	Random Forest	45.01%	188.61%	-143.60%
HDFC	HDFCBANK.NS	XGBoost	8.50%	47.64%	-39.14%
Overall model averages from that run:

Model	Avg Accuracy	Avg F1	Avg Strategy Return	Avg Sharpe
Random Forest	0.505	0.544	26.56%	1.104
XGBoost	0.508	0.511	24.23%	0.870
Logistic Regression	0.490	0.386	5.43%	0.533
Dependencies
Main Python packages:

pandas
numpy
scikit-learn
yfinance
matplotlib
streamlit
xgboost
On macOS, XGBoost also required:

brew install libomp
Setup Commands
Create virtual environment:

cd '/Users/harshtiwari/Desktop/project finance'
python3 -m venv .venv
Install dependencies:

.venv/bin/python -m pip install pandas numpy scikit-learn yfinance matplotlib streamlit xgboost
Run Streamlit app:

.venv/bin/streamlit run app.py
Run CLI pipeline:

.venv/bin/python main.py

"""CLI entrypoint that runs the full stock prediction pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from backtest import plot_equity_curve, run_backtest
from data import (
    RELIANCE_END,
    RELIANCE_START,
    RELIANCE_SYMBOL,
    load_price_data,
    load_reliance_data,
)
from features import DEFAULT_HORIZON, make_features
from model import (
    evaluate_classifier,
    get_logistic_regression_coefficients,
    get_random_forest_feature_importance,
    is_xgboost_available,
    predict,
    predict_proba,
    predict_signals,
    train_logistic_regression_model,
    train_xgboost_model,
    time_series_train_test_split,
    train_model,
)

MULTI_STOCK_UNIVERSE = {
    "RELIANCE": "RELIANCE.NS",
    "TCS": "TCS.NS",
    "INFY": "INFY.NS",
    "HDFC": "HDFCBANK.NS",
}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run stock prediction, signal generation, backtesting, and plotting."
    )
    parser.add_argument(
        "--symbol",
        type=str,
        default=RELIANCE_SYMBOL,
        help="Ticker symbol to download and model",
    )
    parser.add_argument(
        "--start",
        type=str,
        default=RELIANCE_START,
        help="Start date in YYYY-MM-DD format",
    )
    parser.add_argument(
        "--end",
        type=str,
        default=RELIANCE_END,
        help="End date in YYYY-MM-DD format",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=DEFAULT_HORIZON,
        help="Forecast horizon in trading days",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of rows reserved for the time-based test split",
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=150,
        help="Number of trees in the Random Forest classifier",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=8,
        help="Maximum tree depth for the Random Forest",
    )
    parser.add_argument(
        "--min-samples-leaf",
        type=int,
        default=2,
        help="Minimum samples required at each leaf node",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducible model training",
    )
    parser.add_argument(
        "--buy-threshold",
        type=float,
        default=0.6,
        help="Generate BUY when predicted up-probability is above this value",
    )
    parser.add_argument(
        "--sell-threshold",
        type=float,
        default=0.4,
        help="Generate SELL when predicted up-probability is below this value",
    )
    parser.add_argument(
        "--transaction-cost",
        type=float,
        default=0.001,
        help="One-way transaction cost as a decimal, e.g. 0.001 for 0.1%%",
    )
    parser.add_argument(
        "--slippage",
        type=float,
        default=0.0005,
        help="One-way slippage assumption as a decimal, e.g. 0.0005 for 0.05%%",
    )
    parser.add_argument(
        "--position-size",
        type=float,
        default=1.0,
        help="Fixed portfolio fraction allocated on BUY signals, between 0 and 1",
    )
    parser.add_argument(
        "--execution-lag",
        type=int,
        default=1,
        help="Bars to delay signal execution to avoid same-bar lookahead",
    )
    parser.add_argument(
        "--refresh-data",
        action="store_true",
        help="Force a fresh yfinance download instead of using cached CSV data",
    )
    parser.add_argument(
        "--multi-stock",
        action="store_true",
        help="Run the comparison universe: RELIANCE.NS, TCS.NS, INFY.NS, HDFCBANK.NS",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Directory for predictions, metrics, and plots",
    )
    return parser


def _print_metrics(title: str, metrics: dict[str, float]) -> None:
    print(f"\n{title}")
    print("-" * len(title))
    for key, value in metrics.items():
        print(f"{key:30s}: {value: .6f}")


def _print_table(title: str, frame: pd.DataFrame, rows: int = 10) -> None:
    print(f"\n{title}")
    print("-" * len(title))
    print(frame.head(rows).to_string(index=False))


def _maybe_get_best_params(model: object) -> dict[str, object] | None:
    best_params = getattr(model, "best_params_", None)
    if isinstance(best_params, dict):
        return best_params
    return None


def _safe_symbol(symbol: str) -> str:
    return symbol.replace("/", "_").replace(".", "_")


def _safe_model_name(model_name: str) -> str:
    return model_name.lower().replace(" ", "_")


def _load_prices(args: argparse.Namespace) -> pd.DataFrame:
    if (
        args.symbol == RELIANCE_SYMBOL
        and args.start == RELIANCE_START
        and args.end == RELIANCE_END
    ):
        return load_reliance_data(refresh=args.refresh_data)

    return load_price_data(
        symbol=args.symbol,
        start=args.start,
        end=args.end,
        refresh=args.refresh_data,
    )


def _build_market_returns(price_df: pd.DataFrame, horizon: int) -> pd.Series:
    return price_df["Close"].pct_change(horizon).shift(-horizon)


def _evaluate_model_run(
    model_name: str,
    model: object,
    split: object,
    market_returns: pd.Series,
    buy_threshold: float,
    sell_threshold: float,
    transaction_cost: float,
    slippage: float,
    position_size: float,
    execution_lag: int,
) -> dict[str, object]:
    predicted_classes = predict(model, split.X_test)
    predicted_probabilities = predict_proba(model, split.X_test)
    predicted_signals = predict_signals(
        model,
        split.X_test,
        buy_threshold=buy_threshold,
        sell_threshold=sell_threshold,
    )

    classification_metrics = evaluate_classifier(split.y_test, predicted_classes)
    backtest_result = run_backtest(
        market_returns=market_returns,
        predictions=predicted_signals,
        transaction_cost=transaction_cost,
        slippage=slippage,
        position_size=position_size,
        execution_lag=execution_lag,
    )
    prediction_frame = pd.concat(
        [
            split.y_test.rename("actual_target"),
            predicted_classes,
            predicted_probabilities,
            predicted_signals.rename("predicted_signal"),
            market_returns.rename("realized_return"),
        ],
        axis=1,
    )
    backtest_result.frame = prediction_frame.join(backtest_result.frame, how="inner")

    return {
        "model_name": model_name,
        "model_key": _safe_model_name(model_name),
        "model": model,
        "classification_metrics": classification_metrics,
        "backtest_result": backtest_result,
        "prediction_frame": prediction_frame,
        "best_params": _maybe_get_best_params(model),
    }


def _build_comparison_frame(model_runs: list[dict[str, object]]) -> pd.DataFrame:
    rows = []
    for model_run in model_runs:
        classification_metrics = model_run["classification_metrics"]
        backtest_metrics = model_run["backtest_result"].metrics
        rows.append(
            {
                "model": model_run["model_name"],
                "accuracy": classification_metrics["accuracy"],
                "precision": classification_metrics["precision"],
                "recall": classification_metrics["recall"],
                "f1": classification_metrics["f1"],
                "strategy_total_return": backtest_metrics["strategy_total_return"],
                "strategy_sharpe_ratio": backtest_metrics["strategy_sharpe_ratio"],
                "strategy_max_drawdown": backtest_metrics["strategy_max_drawdown"],
                "strategy_annualized_return": backtest_metrics["strategy_annualized_return"],
                "buy_hold_total_return": backtest_metrics["buy_hold_total_return"],
                "total_trades": backtest_metrics["total_trades"],
                "win_rate": backtest_metrics["win_rate"],
                "percent_time_in_market": backtest_metrics["percent_time_in_market"],
            }
        )
    return pd.DataFrame(rows)


def _train_available_models(args: argparse.Namespace, split: object) -> list[tuple[str, object]]:
    """Train all configured models that are available in the environment."""
    random_forest_model = train_model(
        split.X_train,
        split.y_train,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_leaf=args.min_samples_leaf,
        random_state=args.random_state,
    )
    logistic_regression_model = train_logistic_regression_model(
        split.X_train,
        split.y_train,
        random_state=args.random_state,
    )
    models: list[tuple[str, object]] = [
        ("Random Forest", random_forest_model),
        ("Logistic Regression", logistic_regression_model),
    ]
    if is_xgboost_available():
        models.insert(
            1,
            (
                "XGBoost",
                train_xgboost_model(
                    split.X_train,
                    split.y_train,
                    random_state=args.random_state,
                ),
            ),
        )
    return models


def _run_symbol_pipeline(
    args: argparse.Namespace,
    symbol: str,
    start: str,
    end: str,
) -> tuple[pd.DataFrame, object, list[dict[str, object]]]:
    """Run the model and backtest pipeline for one symbol."""
    prices = load_price_data(
        symbol=symbol,
        start=start,
        end=end,
        refresh=args.refresh_data,
    )
    feature_set = make_features(prices, horizon=args.horizon)
    split = time_series_train_test_split(
        feature_set.X,
        feature_set.y,
        test_size=args.test_size,
    )
    market_returns = _build_market_returns(prices, horizon=args.horizon).reindex(
        split.X_test.index
    )
    model_runs = [
        _evaluate_model_run(
            model_name=model_name,
            model=model,
            split=split,
            market_returns=market_returns,
            buy_threshold=args.buy_threshold,
            sell_threshold=args.sell_threshold,
            transaction_cost=args.transaction_cost,
            slippage=args.slippage,
            position_size=args.position_size,
            execution_lag=args.execution_lag,
        )
        for model_name, model in _train_available_models(args, split)
    ]
    return prices, split, model_runs


def _run_multi_stock_comparison(args: argparse.Namespace, output_dir: Path) -> None:
    """Run the configured multi-stock research universe and write comparison CSVs."""
    rows: list[dict[str, object]] = []
    for stock_name, symbol in MULTI_STOCK_UNIVERSE.items():
        _, _, model_runs = _run_symbol_pipeline(
            args=args,
            symbol=symbol,
            start=RELIANCE_START,
            end=RELIANCE_END,
        )
        comparison = _build_comparison_frame(model_runs)
        comparison.insert(0, "symbol", symbol)
        comparison.insert(0, "stock", stock_name)
        rows.extend(comparison.to_dict(orient="records"))

    multi_stock_frame = pd.DataFrame(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    multi_stock_path = output_dir / "multi_stock_model_comparison.csv"
    best_by_return_path = output_dir / "multi_stock_best_by_return.csv"
    multi_stock_frame.to_csv(multi_stock_path, index=False)
    (
        multi_stock_frame.sort_values("strategy_total_return", ascending=False)
        .groupby("stock", as_index=False)
        .head(1)
        .to_csv(best_by_return_path, index=False)
    )
    print(f"Saved multi-stock comparison to: {multi_stock_path}")
    print(f"Saved best model by return to: {best_by_return_path}")


def main() -> None:
    args = _build_parser().parse_args()
    output_dir = Path(args.output_dir)

    if args.multi_stock:
        _run_multi_stock_comparison(args, output_dir=output_dir)
        return

    prices = _load_prices(args)
    feature_set = make_features(prices, horizon=args.horizon)
    split = time_series_train_test_split(
        feature_set.X,
        feature_set.y,
        test_size=args.test_size,
    )

    random_forest_model = train_model(
        split.X_train,
        split.y_train,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_leaf=args.min_samples_leaf,
        random_state=args.random_state,
    )

    market_returns = _build_market_returns(prices, horizon=args.horizon)
    market_returns = market_returns.reindex(split.X_test.index)

    logistic_regression_model = train_logistic_regression_model(
        split.X_train,
        split.y_train,
        random_state=args.random_state,
    )
    xgboost_model = (
        train_xgboost_model(
            split.X_train,
            split.y_train,
            random_state=args.random_state,
        )
        if is_xgboost_available()
        else None
    )

    model_runs = [
        _evaluate_model_run(
            model_name="Random Forest",
            model=random_forest_model,
            split=split,
            market_returns=market_returns,
            buy_threshold=args.buy_threshold,
            sell_threshold=args.sell_threshold,
            transaction_cost=args.transaction_cost,
            slippage=args.slippage,
            position_size=args.position_size,
            execution_lag=args.execution_lag,
        ),
        _evaluate_model_run(
            model_name="Logistic Regression",
            model=logistic_regression_model,
            split=split,
            market_returns=market_returns,
            buy_threshold=args.buy_threshold,
            sell_threshold=args.sell_threshold,
            transaction_cost=args.transaction_cost,
            slippage=args.slippage,
            position_size=args.position_size,
            execution_lag=args.execution_lag,
        ),
    ]
    if xgboost_model is not None:
        model_runs.insert(
            1,
            _evaluate_model_run(
                model_name="XGBoost",
                model=xgboost_model,
                split=split,
                market_returns=market_returns,
                buy_threshold=args.buy_threshold,
                sell_threshold=args.sell_threshold,
                transaction_cost=args.transaction_cost,
                slippage=args.slippage,
                position_size=args.position_size,
                execution_lag=args.execution_lag,
            ),
        )
    random_forest_feature_importance = get_random_forest_feature_importance(
        random_forest_model,
        split.X_train.columns,
    )
    logistic_regression_coefficients = get_logistic_regression_coefficients(
        logistic_regression_model,
        split.X_train.columns,
    )

    for model_run in model_runs:
        _print_metrics(
            f"{model_run['model_name']} Classification Metrics",
            model_run["classification_metrics"],
        )
        best_params = model_run["best_params"]
        if best_params:
            print(f"\n{model_run['model_name']} Best Parameters")
            print("-" * (len(model_run["model_name"]) + 16))
            for key, value in best_params.items():
                print(f"{key:30s}: {value}")
        _print_metrics(
            f"{model_run['model_name']} Backtest Metrics",
            model_run["backtest_result"].metrics,
        )

    _print_table("Random Forest Feature Importance", random_forest_feature_importance)
    _print_table("Logistic Regression Coefficients", logistic_regression_coefficients)

    comparison_frame = _build_comparison_frame(model_runs)
    print("\nModel Comparison")
    print("----------------")
    print(comparison_frame.to_string(index=False, float_format=lambda x: f"{x: .6f}"))

    output_dir.mkdir(parents=True, exist_ok=True)
    safe_symbol = _safe_symbol(args.symbol)

    predictions_path = output_dir / f"predictions_{safe_symbol}.csv"
    metrics_path = output_dir / f"metrics_{safe_symbol}.json"
    backtest_path = output_dir / f"backtest_{safe_symbol}.csv"
    plot_path = output_dir / f"equity_curve_{safe_symbol}.png"
    comparison_path = output_dir / f"model_comparison_{safe_symbol}.csv"
    rf_importance_path = output_dir / f"random_forest_feature_importance_{safe_symbol}.csv"
    lr_coefficients_path = output_dir / f"logistic_regression_coefficients_{safe_symbol}.csv"

    comparison_frame.to_csv(comparison_path, index=False)
    random_forest_feature_importance.to_csv(rf_importance_path, index=False)
    logistic_regression_coefficients.to_csv(lr_coefficients_path, index=False)
    for model_run in model_runs:
        model_key = model_run["model_key"]
        model_predictions_path = output_dir / f"predictions_{model_key}_{safe_symbol}.csv"
        model_backtest_path = output_dir / f"backtest_{model_key}_{safe_symbol}.csv"
        model_plot_path = output_dir / f"equity_curve_{model_key}_{safe_symbol}.png"

        model_run["prediction_frame"].to_csv(model_predictions_path, index_label="Date")
        model_run["backtest_result"].frame.to_csv(model_backtest_path, index_label="Date")
        plot_equity_curve(
            model_run["backtest_result"],
            output_path=model_plot_path,
            title=f"Equity Curve Comparison: {args.symbol} - {model_run['model_name']}",
        )

        if model_key == "random_forest":
            model_run["prediction_frame"].to_csv(predictions_path, index_label="Date")
            model_run["backtest_result"].frame.to_csv(backtest_path, index_label="Date")
            plot_equity_curve(
                model_run["backtest_result"],
                output_path=plot_path,
                title=f"Equity Curve Comparison: {args.symbol}",
            )

    metrics_path.write_text(
        json.dumps(
            {
                "symbol": args.symbol,
                "start": args.start,
                "end": args.end,
                "horizon": args.horizon,
                "transaction_cost": args.transaction_cost,
                "slippage": args.slippage,
                "position_size": args.position_size,
                "execution_lag": args.execution_lag,
                "models": {
                    model_run["model_key"]: {
                        "model_name": model_run["model_name"],
                        "classification_metrics": model_run["classification_metrics"],
                        "backtest_metrics": model_run["backtest_result"].metrics,
                        "best_params": model_run["best_params"],
                    }
                    for model_run in model_runs
                },
                "comparison": comparison_frame.to_dict(orient="records"),
            },
            indent=2,
        )
    )

    print(f"\nLoaded {len(prices)} price rows for {args.symbol}.")
    print(f"Train rows: {len(split.X_train)} | Test rows: {len(split.X_test)}")
    print(f"Saved comparison summary to: {comparison_path}")
    print(f"Saved Random Forest feature importance to: {rf_importance_path}")
    print(f"Saved Logistic Regression coefficients to: {lr_coefficients_path}")
    print(f"Saved Random Forest predictions to: {predictions_path}")
    print(f"Saved Random Forest backtest time series to: {backtest_path}")
    print(f"Saved metrics to: {metrics_path}")
    print(f"Saved Random Forest equity curve plot to: {plot_path}")


if __name__ == "__main__":
    main()

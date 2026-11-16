"""Custom institutional views package for Quant Terminal."""

from pages_custom.backtesting_view import render_backtesting_view
from pages_custom.dashboard_view import render_dashboard_view
from pages_custom.market_overview_view import render_market_overview_view
from pages_custom.model_comparison_view import render_model_comparison_view
from pages_custom.portfolio_view import render_portfolio_view
from pages_custom.predictions_view import render_predictions_view
from pages_custom.risk_view import render_risk_view
from pages_custom.settings_view import render_settings_view
from pages_custom.stock_analysis_view import render_stock_analysis_view
from pages_custom.watchlist_view import render_watchlist_view

__all__ = [
    "render_dashboard_view",
    "render_market_overview_view",
    "render_stock_analysis_view",
    "render_predictions_view",
    "render_model_comparison_view",
    "render_backtesting_view",
    "render_portfolio_view",
    "render_risk_view",
    "render_watchlist_view",
    "render_settings_view",
]


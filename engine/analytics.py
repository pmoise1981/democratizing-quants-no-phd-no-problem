from typing import Literal, Dict, Any

import pandas as pd
import yfinance as yf

from strategies.buy_and_hold import BuyAndHoldSPY
from strategies.sixty_forty import SixtyFortySPYTLT
from metrics.performance import (
    compute_performance,
    compute_rolling_sharpe,
    metrics_to_dict,
    Period,
)

StrategyName = Literal["buy_and_hold_spy", "sixty_forty_spy_tlt"]


def _load_prices(start: str = "2015-01-01") -> pd.DataFrame:
    tickers = ["SPY", "TLT"]
    data = yf.download(tickers, start=start, auto_adjust=True, progress=False)
    close = data["Close"]
    return close


def get_strategy(strategy_name: StrategyName):
    if strategy_name == "buy_and_hold_spy":
        return BuyAndHoldSPY()
    elif strategy_name == "sixty_forty_spy_tlt":
        return SixtyFortySPYTLT()
    else:
        raise ValueError(f"Unknown strategy {strategy_name}")


def _slice_series_by_period(series: pd.Series, period: Period) -> pd.Series:
    """Slice any datetime-indexed Series using the same logic as metrics._slice_period."""
    series = series.dropna()
    if series.empty:
        return series

    if period == "max":
        return series

    last_date = series.index.max()

    if period == "1m":
        start_date = last_date - pd.DateOffset(months=1)
    elif period == "3m":
        start_date = last_date - pd.DateOffset(months=3)
    elif period == "6m":
        start_date = last_date - pd.DateOffset(months=6)
    elif period == "1y":
        start_date = last_date - pd.DateOffset(years=1)
    else:
        start_date = series.index.min()

    return series.loc[start_date:]


def compute_snapshot(
    strategy_name: StrategyName,
    period: Period = "1y",
) -> Dict[str, Any]:
    """
    Run the chosen strategy, compute metrics for a given period,
    and return a JSON-serializable dict suitable for feeding to an LLM.
    """
    prices = _load_prices()
    strat = get_strategy(strategy_name)
    result = strat.run(prices)

    perf = compute_performance(result.daily_returns, period=period)

    # Rolling Sharpe over full history, then sliced to the same period
    rolling_sharpe_full = compute_rolling_sharpe(result.daily_returns, window=63)
    rolling_sharpe_slice = _slice_series_by_period(rolling_sharpe_full, period)

    if not rolling_sharpe_slice.empty:
        rs_series = [
            {"date": str(idx.date()), "sharpe": float(val)}
            for idx, val in rolling_sharpe_slice.dropna().items()
        ]
    else:
        rs_series = []

    snapshot = {
        "strategy": strategy_name,
        "period": period,
        "performance": metrics_to_dict(perf),
        "rolling_sharpe": {
            "window_days": 63,
            "series": rs_series,
        },
        "meta": {
            "start_date": str(result.daily_returns.index.min().date())
            if not result.daily_returns.empty
            else None,
            "end_date": str(result.daily_returns.index.max().date())
            if not result.daily_returns.empty
            else None,
            "num_days": int(result.daily_returns.shape[0]),
        },
    }
    return snapshot


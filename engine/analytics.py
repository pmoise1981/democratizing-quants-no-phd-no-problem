from typing import Literal, Dict, Any

import numpy as np
import pandas as pd
import yfinance as yf

from strategies.buy_and_hold import BuyAndHoldSPY
from strategies.sixty_forty import SixtyFortySPYTLT
from strategies.diversified_core import DiversifiedCoreETF
from metrics.performance import (
    compute_performance,
    compute_rolling_sharpe,
    metrics_to_dict,
    Period,
)

StrategyName = Literal[
    "buy_and_hold_spy",
    "sixty_forty_spy_tlt",
    "diversified_core_etf",
]


def _load_prices(start: str = "2010-01-01") -> pd.DataFrame:
    """
    Load adjusted close prices for all ETFs that any strategy might use.
    """
    tickers = [
        "SPY",  # US large cap
        "QQQ",  # US growth/tech
        "IWM",  # US small caps
        "EFA",  # Developed ex-US
        "EEM",  # Emerging markets
        "TLT",  # Long Treasuries
        "LQD",  # Investment-grade bonds
        "GLD",  # Gold
    ]
    data = yf.download(tickers, start=start, auto_adjust=True, progress=False)
    close = data["Close"]
    return close


def get_strategy(strategy_name: StrategyName):
    if strategy_name == "buy_and_hold_spy":
        return BuyAndHoldSPY()
    elif strategy_name == "sixty_forty_spy_tlt":
        return SixtyFortySPYTLT()
    elif strategy_name == "diversified_core_etf":
        return DiversifiedCoreETF()
    else:
        raise ValueError(f"Unknown strategy {strategy_name}")


def _slice_series_by_period(series: pd.Series, period: Period) -> pd.Series:
    """Slice any datetime-indexed Series using the same logic as our metrics period slicing."""
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


def _slice_df_by_period(df: pd.DataFrame, period: Period) -> pd.DataFrame:
    """Slice any datetime-indexed DataFrame by period."""
    df = df.dropna(how="all")
    if df.empty:
        return df

    if period == "max":
        return df

    last_date = df.index.max()

    if period == "1m":
        start_date = last_date - pd.DateOffset(months=1)
    elif period == "3m":
        start_date = last_date - pd.DateOffset(months=3)
    elif period == "6m":
        start_date = last_date - pd.DateOffset(months=6)
    elif period == "1y":
        start_date = last_date - pd.DateOffset(years=1)
    else:
        start_date = df.index.min()

    return df.loc[start_date:]


def _compute_asset_contributions(
    strategy_name: StrategyName,
    prices: pd.DataFrame,
    period: Period,
) -> Dict[str, Any]:
    """
    Approximate asset-level contribution for each strategy using
    simple static weights and asset-level returns over the selected period.

    This is not a full Brinson attribution, but it's good enough to
    answer questions like "which ETFs helped or hurt the most?"
    """
    if strategy_name == "buy_and_hold_spy":
        tickers = ["SPY"]
        base_weights = {"SPY": 1.0}
    elif strategy_name == "sixty_forty_spy_tlt":
        tickers = ["SPY", "TLT"]
        base_weights = {"SPY": 0.6, "TLT": 0.4}
    elif strategy_name == "diversified_core_etf":
        tickers = [
            "SPY",
            "QQQ",
            "IWM",
            "EFA",
            "EEM",
            "TLT",
            "LQD",
            "GLD",
        ]
        n = len(tickers)
        base_weights = {t: 1.0 / n for t in tickers}
    else:
        return {}

    # Subset and clean prices
    available = [t for t in tickers if t in prices.columns]
    if not available:
        return {
            "tickers": tickers,
            "weights": base_weights,
            "cumulative_return": {},
            "annualized_volatility": {},
            "sharpe_like": {},
        }

    df = prices[available].copy()
    df = df.ffill().dropna(how="all")
    df = _slice_df_by_period(df, period)
    if df.empty:
        return {
            "tickers": available,
            "weights": {t: base_weights.get(t, 0.0) for t in available},
            "cumulative_return": {},
            "annualized_volatility": {},
            "sharpe_like": {},
        }

    # Daily asset returns
    rets = df.pct_change().dropna(how="all")
    if rets.empty:
        return {
            "tickers": available,
            "weights": {t: base_weights.get(t, 0.0) for t in available},
            "cumulative_return": {},
            "annualized_volatility": {},
            "sharpe_like": {},
        }

    # Per-asset cumulative return over the period
    cum = (1.0 + rets).prod() - 1.0

    # Per-asset annualized volatility
    vol = rets.std(ddof=1) * np.sqrt(252)

    # Simple "Sharpe-like" ratio per asset (just return/vol); used for ranking
    sharpe_like = cum / vol.replace(0.0, np.nan)

    return {
        "tickers": available,
        "weights": {t: float(base_weights.get(t, 0.0)) for t in available},
        "cumulative_return": {t: float(cum.get(t, 0.0)) for t in available},
        "annualized_volatility": {t: float(vol.get(t, 0.0)) for t in available},
        "sharpe_like": {t: float(sharpe_like.get(t, np.nan)) for t in available},
    }


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

    asset_contrib = _compute_asset_contributions(strategy_name, prices, period)

    snapshot = {
        "strategy": strategy_name,
        "period": period,
        "performance": metrics_to_dict(perf),
        "rolling_sharpe": {
            "window_days": 63,
            "series": rs_series,
        },
        "asset_contribution": asset_contrib,
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


from dataclasses import dataclass
from typing import Literal, Dict

import numpy as np
import pandas as pd

# Type alias for the periods we support
Period = Literal["1m", "3m", "6m", "1y", "max"]

TRADING_DAYS_PER_YEAR = 252


@dataclass
class PerformanceMetrics:
    cumulative_return: float
    annualized_return: float
    annualized_volatility: float
    sharpe_ratio: float
    max_drawdown: float
    downside_volatility: float | None = None
    sortino_ratio: float | None = None


def _slice_period(returns: pd.Series, period: Period) -> pd.Series:
    """Slice a daily return series to a given lookback period."""
    returns = returns.dropna()
    if returns.empty:
        return returns

    if period == "max":
        return returns

    last_date = returns.index.max()

    if period == "1m":
        start_date = last_date - pd.DateOffset(months=1)
    elif period == "3m":
        start_date = last_date - pd.DateOffset(months=3)
    elif period == "6m":
        start_date = last_date - pd.DateOffset(months=6)
    elif period == "1y":
        start_date = last_date - pd.DateOffset(years=1)
    else:
        start_date = returns.index.min()

    return returns.loc[start_date:]


def compute_performance(
    daily_returns: pd.Series,
    period: Period = "1y",
    risk_free_rate_annual: float = 0.0,
) -> PerformanceMetrics:
    """
    Compute realistic performance metrics for a strategy over a given period.

    daily_returns: daily % returns as decimals (e.g. 0.01 for 1%).
    """
    daily_returns = daily_returns.dropna()
    sliced = _slice_period(daily_returns, period)

    if sliced.empty:
        return PerformanceMetrics(
            cumulative_return=0.0,
            annualized_return=0.0,
            annualized_volatility=0.0,
            sharpe_ratio=0.0,
            max_drawdown=0.0,
            downside_volatility=None,
            sortino_ratio=None,
        )

    # Convert annual rf to daily
    rf_daily = (1 + risk_free_rate_annual) ** (1 / TRADING_DAYS_PER_YEAR) - 1
    excess = sliced - rf_daily

    # Cumulative return over the slice
    cumulative_return = (1 + sliced).prod() - 1

    # Annualized return from mean daily return
    mean_daily = sliced.mean()
    annualized_return = (1 + mean_daily) ** TRADING_DAYS_PER_YEAR - 1

    # Annualized volatility
    vol_daily = sliced.std(ddof=1)
    annualized_volatility = vol_daily * np.sqrt(TRADING_DAYS_PER_YEAR)

    if annualized_volatility > 0:
        sharpe_ratio = annualized_return / annualized_volatility
    else:
        sharpe_ratio = 0.0

    # Max drawdown
    equity_curve = (1 + sliced).cumprod()
    running_max = equity_curve.cummax()
    drawdowns = equity_curve / running_max - 1
    max_drawdown = float(drawdowns.min())

    # Downside volatility & Sortino ratio
    downside = sliced[sliced < 0]
    if len(downside) >= 2:
        downside_vol = downside.std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR)
        sortino_ratio = (
            annualized_return / downside_vol if downside_vol > 0 else None
        )
    else:
        downside_vol = None
        sortino_ratio = None

    return PerformanceMetrics(
        cumulative_return=float(cumulative_return),
        annualized_return=float(annualized_return),
        annualized_volatility=float(annualized_volatility),
        sharpe_ratio=float(sharpe_ratio),
        max_drawdown=max_drawdown,
        downside_volatility=float(downside_vol) if downside_vol is not None else None,
        sortino_ratio=float(sortino_ratio) if sortino_ratio is not None else None,
    )


def compute_rolling_sharpe(
    daily_returns: pd.Series,
    window: int = 63,
    risk_free_rate_annual: float = 0.0,
) -> pd.Series:
    """
    Compute an annualized rolling Sharpe ratio over a moving window.

    window: number of trading days in the rolling window (e.g. 63 ≈ 3 months).
    """
    returns = daily_returns.dropna()
    if returns.empty:
        return pd.Series(dtype=float)

    rf_daily = (1 + risk_free_rate_annual) ** (1 / TRADING_DAYS_PER_YEAR) - 1
    excess = returns - rf_daily

    min_periods = max(10, window // 3)

    rolling_mean = excess.rolling(window=window, min_periods=min_periods).mean()
    rolling_std = excess.rolling(window=window, min_periods=min_periods).std(ddof=1)

    # Avoid division by zero
    sharpe_daily = rolling_mean.where(rolling_std > 0, np.nan) / rolling_std.where(
        rolling_std > 0, np.nan
    )

    rolling_sharpe_annualized = sharpe_daily * np.sqrt(TRADING_DAYS_PER_YEAR)
    rolling_sharpe_annualized.name = "rolling_sharpe"

    return rolling_sharpe_annualized


def metrics_to_dict(m: PerformanceMetrics) -> Dict[str, float]:
    """Convert a PerformanceMetrics dataclass to a plain dict for JSON/LLM."""
    return {
        "cumulative_return": m.cumulative_return,
        "annualized_return": m.annualized_return,
        "annualized_volatility": m.annualized_volatility,
        "sharpe_ratio": m.sharpe_ratio,
        "max_drawdown": m.max_drawdown,
        "downside_volatility": m.downside_volatility,
        "sortino_ratio": m.sortino_ratio,
    }


# metrics/performance.py
from dataclasses import dataclass
from typing import Literal, Dict
import numpy as np
import pandas as pd


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
    daily_returns: % returns as decimal (e.g. 0.01 for 1%)
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

    rf_daily = (1 + risk_free_rate_annual) ** (1 / TRADING_DAYS_PER_YEAR) - 1
    excess = sliced - rf_daily

    cum_return = (1 + sliced).prod() - 1
    mean_daily = sliced.mean()
    vol_daily = sliced.std(ddof=1)
    ann_return = (1 + mean_daily) ** TRADING_DAYS_PER_YEAR - 1
    ann_vol = vol_daily * np.sqrt(TRADING_DAYS_PER_YEAR)
    sharpe = ann_return / ann_vol if ann_vol > 0 else 0.0

    # Max drawdown from equity curve
    equity_curve = (1 + sliced).cumprod()
    running_max = equity_curve.cummax()
    drawdowns = equity_curve / running_max - 1
    max_dd = drawdowns.min()

    # Downside risk & Sortino
    downside = sliced[sliced < 0]
    if len(downside) >= 2:
        downside_vol = downside.std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR)
        sortino = ann_return / downside_vol if downside_vol > 0 else None
    else:
        downside_vol = None
        sortino = None

    return PerformanceMetrics(
        cumulative_return=float(cum_return),
        annualized_return=float(ann_return),
        annualized_volatility=float(ann_vol),
        sharpe_ratio=float(sharpe),
        max_drawdown=float(max_dd),
        downside_volatility=float(downside_vol) if downside_vol is not None else None,
        sortino_ratio=float(sortino) if sortino is not None else None,
    )


def metrics_to_dict(m: PerformanceMetrics) -> Dict[str, float]:
    return {
        "cumulative_return": m.cumulative_return,
        "annualized_return": m.annualized_return,
        "annualized_volatility": m.annualized_volatility,
        "sharpe_ratio": m.sharpe_ratio,
        "max_drawdown": m.max_drawdown,
        "downside_volatility": m.downside_volatility,
        "sortino_ratio": m.sortino_ratio,
    }


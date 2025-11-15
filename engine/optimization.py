# engine/optimization.py

from typing import List, Dict, Any
import numpy as np
import pandas as pd

from metrics.performance import Period


def _slice_df_by_period(df: pd.DataFrame, period: Period) -> pd.DataFrame:
    """Helper to reuse period slicing logic for optimizations."""
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


def optimize_portfolio(
    tickers: List[str],
    prices: pd.DataFrame,
    period: Period,
    objective: str = "max_return",
    num_portfolios: int = 5000,
) -> Dict[str, Any]:
    """
    Simple Monte-Carlo portfolio optimizer.

    - tickers: list of ETFs in the universe
    - prices: full price history for all ETFs
    - period: period over which to estimate stats
    - objective: "max_return", "max_sharpe", or "min_volatility"
    - num_portfolios: number of random portfolios to sample

    Returns a dict with optimized weights and expected metrics.
    """
    if not tickers:
        return {}

    available = [t for t in tickers if t in prices.columns]
    if not available:
        return {}

    df = prices[available].copy()
    df = df.ffill().dropna(how="all")
    df = _slice_df_by_period(df, period)
    if df.empty:
        return {}

    # Daily returns
    rets = df.pct_change().dropna(how="all")
    if rets.empty:
        return {}

    mean_daily = rets.mean()
    cov_daily = rets.cov()

    # Annualize
    mean_ann = mean_daily * 252.0
    cov_ann = cov_daily * 252.0

    n = len(available)
    mean_vec = mean_ann.values
    cov_mat = cov_ann.values

    best_idx = None
    best_score = None
    best_metrics = None
    best_weights = None

    rng = np.random.default_rng()

    for i in range(num_portfolios):
        w = rng.random(n)
        w /= w.sum()  # long-only, fully invested

        port_ret = float(w @ mean_vec)
        port_vol = float(np.sqrt(w @ (cov_mat @ w)))
        sharpe = port_ret / port_vol if port_vol > 0 else float("nan")

        if objective == "max_return":
            score = port_ret
        elif objective == "max_sharpe":
            score = sharpe
        elif objective == "min_volatility":
            score = -port_vol  # lower vol = higher score
        else:
            # default to max Sharpe if unknown
            score = sharpe

        if best_score is None or score > best_score:
            best_score = score
            best_idx = i
            best_weights = w
            best_metrics = {
                "expected_return": port_ret,
                "expected_volatility": port_vol,
                "expected_sharpe": sharpe,
            }

    if best_weights is None or best_metrics is None:
        return {}

    weights_dict = {
        ticker: float(best_weights[i]) for i, ticker in enumerate(available)
    }

    return {
        "objective": objective,
        "universe": available,
        "weights": weights_dict,
        **best_metrics,
    }


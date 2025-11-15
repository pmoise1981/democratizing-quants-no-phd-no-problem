from dataclasses import dataclass, field
from typing import List

import pandas as pd

from .base import Strategy, StrategyResult


@dataclass
class DiversifiedCoreETF(Strategy):
    """
    Equal-weight diversified core portfolio across multiple ETFs.

    For simplicity, this implementation assumes daily rebalancing to equal weights
    across the available tickers. That means each day the portfolio is 1/N in
    each ETF, where N is the number of ETFs with valid prices that day.
    """

    tickers: List[str] = field(
        default_factory=lambda: [
            "SPY",  # US large cap
            "QQQ",  # US growth/tech
            "IWM",  # US small caps
            "EFA",  # Developed ex-US
            "EEM",  # Emerging markets
            "TLT",  # Long Treasuries
            "LQD",  # Investment-grade credit
            "GLD",  # Gold
        ]
    )

    def name(self) -> str:
        return "diversified_core_etf"

    def run(self, prices: pd.DataFrame) -> StrategyResult:
        # Keep only the tickers we care about that actually exist in the price data
        available = [t for t in self.tickers if t in prices.columns]
        if not available:
            raise ValueError("None of the diversified core tickers are available in the price data.")

        df = prices[available].dropna(how="all")

        # Forward-fill missing data, then drop rows that are still fully NaN
        df = df.ffill().dropna(how="all")

        # Daily returns
        daily_returns = df.pct_change().fillna(0.0)

        # Equal weights each day (daily rebalancing)
        n_assets = len(df.columns)
        weights = pd.DataFrame(
            data=1.0 / n_assets,
            index=df.index,
            columns=df.columns,
        )

        # Portfolio return is the weighted sum of asset returns
        portfolio_returns = (daily_returns * weights).sum(axis=1)

        # Portfolio value, starting at 1.0
        portfolio_value = (1 + portfolio_returns).cumprod()

        return StrategyResult(
            prices=df,
            portfolio_value=portfolio_value,
            daily_returns=portfolio_returns,
            weights=weights,
        )


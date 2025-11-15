# strategies/buy_and_hold.py
from dataclasses import dataclass
import pandas as pd

from .base import Strategy, StrategyResult


@dataclass
class BuyAndHoldSPY(Strategy):
    ticker: str = "SPY"

    def name(self) -> str:
        return f"buy_and_hold_{self.ticker.lower()}"

    def run(self, prices: pd.DataFrame) -> StrategyResult:
        """
        prices: DataFrame with at least one column = self.ticker
        """
        close = prices[self.ticker].dropna()
        daily_returns = close.pct_change().fillna(0.0)
        portfolio_value = (1 + daily_returns).cumprod()
        weights = pd.DataFrame(index=close.index, data={self.ticker: 1.0})
        return StrategyResult(
            prices=close.to_frame(),
            portfolio_value=portfolio_value,
            daily_returns=daily_returns,
            weights=weights,
        )


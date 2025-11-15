# strategies/sixty_forty.py
from dataclasses import dataclass
import pandas as pd

from .base import Strategy, StrategyResult


@dataclass
class SixtyFortySPYTLT(Strategy):
    equity: str = "SPY"
    bond: str = "TLT"
    equity_weight: float = 0.6
    bond_weight: float = 0.4

    def name(self) -> str:
        return "sixty_forty_spy_tlt"

    def run(self, prices: pd.DataFrame) -> StrategyResult:
        """
        prices: DataFrame with columns for equity and bond tickers.
        """
        df = prices[[self.equity, self.bond]].dropna()
        daily_returns = df.pct_change().fillna(0.0)

        # Start with target weights
        w_eq = self.equity_weight
        w_bd = self.bond_weight

        # Monthly rebalancing
        month_ends = df.resample("M").last().index

        weights_records = []
        portfolio_vals = []
        current_value = 1.0
        holding_weights = pd.Series(
            {self.equity: w_eq, self.bond: w_bd}, index=df.columns
        )

        for date, row in daily_returns.iterrows():
            if date in month_ends and portfolio_vals:
                # rebalance back to target weights
                holding_weights = pd.Series(
                    {self.equity: w_eq, self.bond: w_bd}, index=df.columns
                )

            port_ret = (holding_weights * row).sum()
            current_value *= (1 + port_ret)
            portfolio_vals.append((date, current_value))
            weights_records.append((date, holding_weights.copy()))

        portfolio_value = pd.Series(
            {d: v for d, v in portfolio_vals}, name="portfolio_value"
        ).sort_index()

        weights = pd.DataFrame(
            {d: w for d, w in weights_records}
        ).T.sort_index()
        weights.index.name = "date"

        returns_series = portfolio_value.pct_change().fillna(0.0)

        return StrategyResult(
            prices=df,
            portfolio_value=portfolio_value,
            daily_returns=returns_series,
            weights=weights,
        )


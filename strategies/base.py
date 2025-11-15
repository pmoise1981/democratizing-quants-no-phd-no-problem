# strategies/base.py
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict
import pandas as pd


@dataclass
class StrategyResult:
    prices: pd.DataFrame          # underlying asset prices
    portfolio_value: pd.Series    # normalized to 1.0 at start
    daily_returns: pd.Series      # portfolio daily returns
    weights: pd.DataFrame | None  # asset weights over time (optional)


class Strategy(ABC):
    """
    All strategies must implement run(prices) and return StrategyResult.
    """

    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    def run(self, prices: pd.DataFrame) -> StrategyResult:
        ...


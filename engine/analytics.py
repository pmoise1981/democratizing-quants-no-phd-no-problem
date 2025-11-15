# engine/analytics.py
from typing import Literal, Dict, Any
import yfinance as yf
import pandas as pd

from strategies.buy_and_hold import BuyAndHoldSPY
from strategies.sixty_forty import SixtyFortySPYTLT
from metrics.performance import compute_performance, metrics_to_dict, Period


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

    snapshot = {
        "strategy": strategy_name,
        "period": period,
        "performance": metrics_to_dict(perf),
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


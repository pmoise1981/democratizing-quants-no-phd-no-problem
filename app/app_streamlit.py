import os
import sys

# Ensure project root is on the Python path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import pandas as pd
import streamlit as st

from engine.analytics import compute_snapshot, StrategyName
from metrics.performance import Period
from llm.client import explain_performance


STRATEGY_LABELS = {
    "buy_and_hold_spy": "Buy & Hold – SPY",
    "sixty_forty_spy_tlt": "60/40 – SPY/TLT Monthly Rebalance",
    "diversified_core_etf": "Diversified Core – Multi-Asset ETF Portfolio",
}


def main():
    st.set_page_config(page_title="Democratizing Quants", layout="wide")

    st.title("📊 DEMOCRATIZING QUANTS – No PhD, No Problem")
    st.caption(
        "Ask questions about risk-adjusted performance (Sharpe, drawdown, volatility) "
        "for different strategies and get an explanation in plain English."
    )

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Configuration")

        strategy_key: StrategyName = st.selectbox(
            "Strategy",
            options=list(STRATEGY_LABELS.keys()),
            format_func=lambda k: STRATEGY_LABELS[k],
        )

        period: Period = st.selectbox(
            "Period",
            options=["1m", "3m", "6m", "1y", "max"],
            format_func=lambda p: {
                "1m": "Last 1 month",
                "3m": "Last 3 months",
                "6m": "Last 6 months",
                "1y": "Last 1 year",
                "max": "Since inception",
            }[p],
        )

        default_question = (
            "Why is the Sharpe ratio at its current level over this period, "
            "and what drove it up or down?"
        )
        user_question = st.text_area(
            "Your question",
            value=default_question,
            height=100,
        )

        if st.button("Analyze"):
            with st.spinner("Running strategy and computing metrics..."):
                snapshot = compute_snapshot(strategy_key, period=period)
                explanation = explain_performance(snapshot, user_question)

            st.session_state["snapshot"] = snapshot
            st.session_state["explanation"] = explanation

    with col2:
        st.subheader("Assistant Answer")

        snapshot = st.session_state.get("snapshot")
        explanation = st.session_state.get("explanation")

        if explanation:
            st.markdown("### Natural-Language Explanation")
            st.write(explanation)

        # Rolling Sharpe chart
        if snapshot and "rolling_sharpe" in snapshot:
            rs = snapshot["rolling_sharpe"]
            series = rs.get("series", [])
            if series:
                df_rs = pd.DataFrame(series)
                df_rs["date"] = pd.to_datetime(df_rs["date"])
                df_rs.set_index("date", inplace=True)

                st.markdown(
                    f"### Rolling Sharpe (window = {rs.get('window_days', 'N/A')} trading days)"
                )
                st.line_chart(df_rs["sharpe"])

        # Asset-level contribution table
        if snapshot and snapshot.get("asset_contribution"):
            st.markdown("### Asset-Level Contribution (return & risk)")

            ac = snapshot["asset_contribution"]
            tickers = ac.get("tickers", [])
            weights = ac.get("weights", {})
            cum_ret = ac.get("cumulative_return", {})
            vol = ac.get("annualized_volatility", {})
            sharpe_like = ac.get("sharpe_like", {})

            rows = []
            for t in tickers:
                rows.append(
                    {
                        "Ticker": t,
                        "Weight": weights.get(t, 0.0),
                        "Cumulative Return": cum_ret.get(t, 0.0),
                        "Ann. Volatility": vol.get(t, 0.0),
                        "Sharpe-like": sharpe_like.get(t, float("nan")),
                    }
                )

            if rows:
                df_ac = pd.DataFrame(rows)
                # Sort by cumulative return descending for readability
                df_ac = df_ac.sort_values("Cumulative Return", ascending=False)

                st.dataframe(
                    df_ac.style.format(
                        {
                            "Weight": "{:.2%}",
                            "Cumulative Return": "{:.2%}",
                            "Ann. Volatility": "{:.2%}",
                            "Sharpe-like": "{:.2f}",
                        }
                    )
                )

        if snapshot:
            st.markdown("### Underlying Metrics (for transparency)")
            st.json(snapshot)


if __name__ == "__main__":
    main()


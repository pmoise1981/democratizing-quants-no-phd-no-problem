# app/app_streamlit.py

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

        opt_choice = st.selectbox(
            "Optimization objective (optional)",
            options=["none", "max_return", "max_sharpe", "min_volatility"],
            format_func=lambda x: {
                "none": "No optimization (use original strategy weights)",
                "max_return": "Optimize for maximum expected return",
                "max_sharpe": "Optimize for highest Sharpe ratio",
                "min_volatility": "Optimize for lowest volatility",
            }[x],
        )
        optimization_objective = None if opt_choice == "none" else opt_choice

        if st.button("Analyze"):
            with st.spinner("Running strategy and computing metrics..."):
                snapshot = compute_snapshot(
                    strategy_name=strategy_key,
                    period=period,
                    optimization_objective=optimization_objective,
                )
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

        # Optimized portfolio (if requested)
        if snapshot and snapshot.get("optimization"):
            opt = snapshot["optimization"]
            st.markdown(
                f"### Optimized Portfolio "
                f"(objective: {opt.get('objective_label', opt.get('objective'))})"
            )

            w = opt.get("weights", {})
            rows_opt = [
                {"Ticker": t, "Optimized Weight": w[t]} for t in sorted(w.keys())
            ]
            if rows_opt:
                df_opt = pd.DataFrame(rows_opt)
                st.dataframe(
                    df_opt.style.format({"Optimized Weight": "{:.2%}"})
                )

            # Summary metrics
            exp_ret = opt.get("expected_return")
            exp_vol = opt.get("expected_volatility")
            exp_sharpe = opt.get("expected_sharpe")

            st.markdown("**Optimized portfolio stats (annualized):**")
            st.write(
                f"- Expected return: **{exp_ret:.2%}**  \n"
                f"- Expected volatility: **{exp_vol:.2%}**  \n"
                f"- Expected Sharpe (return / vol): **{exp_sharpe:.2f}**"
            )

        if snapshot:
            st.markdown("### Underlying Metrics (for transparency)")
            st.json(snapshot)


if __name__ == "__main__":
    main()


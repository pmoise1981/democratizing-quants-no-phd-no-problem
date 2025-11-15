# app/app_streamlit.py
import streamlit as st
import os
import sys

# Ensure project root is on the Python path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


from engine.analytics import compute_snapshot, StrategyName
from metrics.performance import Period
from llm.client import explain_performance


STRATEGY_LABELS = {
    "buy_and_hold_spy": "Buy & Hold – SPY",
    "sixty_forty_spy_tlt": "60/40 – SPY/TLT Monthly Rebalance",
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

        strategy_key = st.selectbox(
            "Strategy",
            options=list(STRATEGY_LABELS.keys()),
            format_func=lambda k: STRATEGY_LABELS[k],
        )  # type: ignore[arg-type]

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
        )  # type: ignore[assignment]

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

        if snapshot:
            st.markdown("### Underlying Metrics (for transparency)")
            st.json(snapshot)


if __name__ == "__main__":
    main()


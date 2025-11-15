import os
from typing import Dict, Any
from openai import OpenAI

# Model can be overridden by environment variable
OPENAI_MODEL = os.getenv("DEMOCRATIZING_QUANTS_MODEL", "gpt-4o-mini")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=OPENAI_API_KEY)


def explain_performance(snapshot: Dict[str, Any], user_question: str) -> str:
    """
    Takes the structured metrics snapshot + user question
    and returns a natural-language explanation.
    """

    # If no API key, provide a placeholder explanation so the UI still works
    if not OPENAI_API_KEY:
        return (
            "⚠️ OpenAI API key not configured.\n\n"
            "Snapshot summary:\n"
            f"{snapshot}"
        )

    system_prompt = (
        "You are a quantitative analyst explaining the performance of an "
        "investment strategy to a non-quant professional. Keep explanations "
        "clear and concise, but insightful. You are given:\n"
        "- Overall performance metrics (returns, volatility, Sharpe, drawdowns)\n"
        "- Rolling Sharpe series over time\n"
        "- Asset-level contribution information (per-ETF returns, volatility, and weights)\n\n"
        "Use these to answer questions such as why Sharpe changed, which asset classes "
        "helped or hurt performance, and how diversification affected risk. "
        "Avoid formulas; focus on intuitive explanations grounded in the numbers."
    )

    snapshot_str = str(snapshot)

    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": (
                f"Here is the strategy performance snapshot as a Python dict:\n{snapshot_str}\n\n"
                f"User question: {user_question}\n\n"
                "Explain the answer in 4–7 sentences, explicitly mentioning key metrics "
                "like Sharpe ratio, annualized return, volatility, drawdowns, and the "
                "most important contributing assets or ETFs. If certain ETFs clearly "
                "helped or hurt performance (based on their return and volatility), "
                "call them out directly."
            ),
        },
    ]

    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=messages,
        temperature=0.3,
    )

    return response.choices[0].message.content.strip()


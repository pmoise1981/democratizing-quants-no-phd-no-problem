# llm/client.py

import os
from typing import Dict, Any
from openai import OpenAI

OPENAI_MODEL = os.getenv("DEMOCRATIZING_QUANTS_MODEL", "gpt-4o-mini")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=OPENAI_API_KEY)


def explain_performance(snapshot: Dict[str, Any], user_question: str) -> str:
    """
    Takes the structured metrics snapshot + user question
    and returns a natural-language explanation.
    """

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
        "- Asset-level contribution information (per-ETF returns, volatility, and weights)\n"
        "- Optionally, an 'optimization' section describing an optimized portfolio "
        "(e.g., max_return, max_sharpe, or min_volatility), with optimized weights "
        "and expected risk/return metrics.\n\n"
        "If optimization data is present, compare the original strategy weights to the "
        "optimized weights and explain why the optimizer tilted toward certain assets "
        "given the objective (e.g., more into high-return assets for max_return, more "
        "balanced risk for max_sharpe, etc.). Avoid formulas; focus on intuitive, "
        "data-grounded explanations."
    )

    snapshot_str = str(snapshot)

    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": (
                f"Here is the strategy performance snapshot as a Python dict:\n"
                f"{snapshot_str}\n\n"
                f"User question: {user_question}\n\n"
                "Explain the answer in 4–7 sentences, explicitly mentioning key metrics "
                "like Sharpe ratio, annualized return, volatility, drawdowns, and the "
                "most important contributing assets or ETFs. If an optimized portfolio "
                "is present, describe how its weights and expected risk/return differ "
                "from the original strategy."
            ),
        },
    ]

    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=messages,
        temperature=0.3,
    )

    return response.choices[0].message.content.strip()


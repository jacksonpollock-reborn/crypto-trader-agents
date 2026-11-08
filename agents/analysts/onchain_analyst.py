"""
On-chain analyst: replaces the equity Fundamentals analyst.

Reads Hyperliquid perp-specific signals:
  - Funding rate (crowding + mean-reversion signal)
  - Open interest (trend confirmation vs divergence)
  - Liquidation cluster data (price magnet levels)
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from crypto_trader.dataflows.hyperliquid_data import (
    get_onchain_summary,
    get_liquidation_levels,
)


# ── LangChain tools wrapping our data fetchers ────────────────────────────────

@tool
def fetch_onchain_signals(coin: str) -> dict:
    """
    Fetch on-chain derivatives signals for a crypto perp:
    funding rate, open interest, 24h price change, premium vs spot.
    Use this to assess market crowding and mean-reversion risk.
    """
    return get_onchain_summary(coin, use_testnet=False)


@tool
def fetch_liquidation_data(coin: str) -> dict:
    """
    Fetch recent liquidation volumes (long vs short) for a crypto perp.
    High liquidation levels indicate price magnet zones and crowded positioning.
    """
    return get_liquidation_levels(coin)


# ── Agent factory (mirrors TradingAgents pattern exactly) ────────────────────

def create_onchain_analyst(llm):
    def onchain_analyst_node(state):
        coin = state["company_of_interest"]
        current_date = state["trade_date"]

        tools = [fetch_onchain_signals, fetch_liquidation_data]

        system_message = (
            "You are a crypto on-chain analyst specializing in perpetual futures markets. "
            "Your job is to analyze derivatives market structure — NOT price action or fundamentals. "
            "\n\nFocus on:"
            "\n- Funding rate: persistent high positive = crowded longs = fade rally. "
            "High negative = crowded shorts = fade selloff."
            "\n- Open interest (OI): OI rising with price = trend confirmation. "
            "OI falling with price = deleveraging. "
            "OI rising while price falls = shorts piling in = potential squeeze."
            "\n- Liquidation data: large long liquidation clusters above current price are targets for shorts. "
            "Large short liquidation clusters below price are targets for longs."
            "\n\nProvide a structured report with: (1) current market structure, "
            "(2) crowding signal (who is overleveraged), "
            "(3) key liquidation levels to watch, "
            "(4) directional bias and confidence. "
            "Append a Markdown summary table."
        )

        prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "You are a helpful AI assistant collaborating with other trading analysts. "
                "Use the provided tools to complete your analysis. "
                "If you have a FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL**, prefix your response with that. "
                "Available tools: {tool_names}.\n{system_message}"
                "\nCurrent date: {current_date}. Asset: {coin}",
            ),
            MessagesPlaceholder(variable_name="messages"),
        ])

        prompt = prompt.partial(system_message=system_message)
        prompt = prompt.partial(tool_names=", ".join([t.name for t in tools]))
        prompt = prompt.partial(current_date=current_date)
        prompt = prompt.partial(coin=coin)

        chain = prompt | llm.bind_tools(tools)
        result = chain.invoke(state["messages"])

        report = result.content if len(result.tool_calls) == 0 else ""

        return {
            "messages": [result],
            "fundamentals_report": report,   # reuse existing state key
        }

    return onchain_analyst_node

"""
Macro analyst: crypto macro context.

Covers: BTC dominance, DXY/rates narrative, major crypto news, regulatory signals.
Replaces TradingAgents' news_analyst (which focused on equity macro).
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from crypto_trader.dataflows.hyperliquid_data import get_all_mids


@tool
def fetch_crypto_macro_snapshot() -> dict:
    """
    Fetch BTC dominance proxy and relative performance across major cryptos.
    BTC dominance rising = risk-off within crypto, alts underperform.
    BTC dominance falling = risk-on, alts outperform.
    """
    try:
        mids = get_all_mids(use_testnet=False)
        major = {k: mids[k] for k in ["BTC", "ETH", "SOL"] if k in mids}
        return {
            "prices": major,
            "note": (
                "Compare ETH/BTC and SOL/BTC ratios against recent history. "
                "Falling alt/BTC ratios = BTC dominance rising = rotate to BTC or reduce risk. "
                "Rising alt/BTC ratios = alts leading = higher beta plays preferred."
            ),
        }
    except Exception as e:
        return {"error": str(e)}


@tool
def fetch_crypto_news_headlines() -> dict:
    """
    Fetch recent crypto news from CryptoCompare public API.
    Look for: ETF flows, regulatory actions, exchange hacks, major protocol events.
    These are the macro catalysts that override technical signals.
    """
    import requests
    try:
        url = "https://min-api.cryptocompare.com/data/v2/news/?lang=EN&sortOrder=latest"
        resp = requests.get(url, timeout=8)
        data = resp.json().get("Data", [])

        headlines = [
            {
                "title": item["title"],
                "source": item["source"],
                "published": item["published_on"],
                "url": item["url"],
            }
            for item in data[:15]
        ]
        return {
            "headlines": headlines,
            "note": (
                "High-priority events to flag: ETF approval/rejection, "
                "major exchange insolvency/hack, Fed rate decision, "
                "stablecoin depeg, large protocol exploit. "
                "These override all technical signals."
            ),
        }
    except Exception as e:
        return {"error": str(e)}


def create_macro_analyst(llm):
    def macro_analyst_node(state):
        coin = state["company_of_interest"]
        current_date = state["trade_date"]

        tools = [fetch_crypto_macro_snapshot, fetch_crypto_news_headlines]

        system_message = (
            "You are a crypto macro analyst. Your job is to identify macro tailwinds/headwinds "
            "that will dominate all other signals."
            "\n\nPriority hierarchy (highest overrides lowest):"
            "\n1. Black swan events: exchange collapse, major hack, regulatory ban → all signals invalid"
            "\n2. Major catalysts: ETF flows, Fed decisions, BTC halving effects"
            "\n3. Sector rotation: BTC dominance trend (alts vs BTC relative strength)"
            "\n4. Background macro: dollar strength, risk-on/risk-off regime"
            "\n\nCrypto macro rules:"
            "\n- BTC leads. If BTC breaks down, alts follow harder."
            "\n- DXY and US 10yr yields inversely correlate with crypto risk appetite."
            "\n- Regulatory news has outsized impact — even rumors move markets 10-20%."
            "\n\nProvide: (1) macro regime (risk-on/risk-off/neutral), "
            "(2) any high-priority catalysts in the last 24h, "
            "(3) BTC dominance direction, "
            "(4) macro-based bias for the asset. "
            "Append a Markdown summary table."
        )

        prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "You are a helpful AI assistant collaborating with other trading analysts. "
                "Use the provided tools to complete your analysis. "
                "If you have a FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL**, prefix with that. "
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
            "news_report": report,    # reuse existing state key
        }

    return macro_analyst_node

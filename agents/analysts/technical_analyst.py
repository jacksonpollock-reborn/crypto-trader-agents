"""
Technical analyst: reads Hyperliquid OHLCV candles and computed indicators.
Replaces TradingAgents' market_analyst (which used yfinance/AlphaVantage).
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from crypto_trader.dataflows.hyperliquid_data import (
    compute_technical_summary,
    get_order_book,
    get_candles,
)


@tool
def fetch_technical_indicators(coin: str) -> dict:
    """
    Full technical analysis suite for a crypto perp. Returns:
    - EMA 9/21/50/200 on 1h and 4h
    - RSI on 15m, 1h, 4h (multi-timeframe momentum)
    - MACD line/signal/histogram on 1h and 15m
    - Bollinger Bands (upper/mid/lower + position + width)
    - ATR volatility regime (1h and 4h)
    - VWAP 24h and price deviation from it
    - Volume vs 7-day average
    - Swing highs/lows on 1h and daily (key S/R levels)
    - Trend confluence score
    Use this as your primary data source.
    """
    return compute_technical_summary(coin, use_testnet=False)


@tool
def fetch_order_book(coin: str) -> dict:
    """
    Fetch live L2 order book for a crypto perp.
    Returns: best bid/ask, spread, bid/ask depth ratio,
    largest bid walls (support), largest ask walls (resistance).
    Use this to identify real-time support/resistance and market pressure.
    """
    return get_order_book(coin, use_testnet=False)


@tool
def fetch_raw_candles(coin: str, interval: str = "1d") -> list:
    """
    Fetch raw OHLCV candles for a crypto perp.
    interval options: 15m, 1h, 4h, 1d
    Use for additional context or custom indicator calculations.
    """
    lookback = {"15m": 24, "1h": 72, "4h": 120, "1d": 720}.get(interval, 72)
    candles = get_candles(coin, interval=interval, lookback_hours=lookback, use_testnet=False)
    return candles[-30:]


def create_technical_analyst(llm):
    def technical_analyst_node(state):
        coin = state["company_of_interest"]
        current_date = state["trade_date"]

        tools = [fetch_technical_indicators, fetch_order_book, fetch_raw_candles]

        system_message = (
            "You are a professional crypto technical analyst specializing in perpetual futures. "
            "You have access to a comprehensive technical data suite. Use it fully."

            "\n\n═══ ANALYSIS FRAMEWORK ═══"

            "\n\n1. TREND STRUCTURE (most important)"
            "\n- EMA ribbon 1h: 9 > 21 > 50 = bullish, reverse = bearish, mixed = range"
            "\n- EMA 200 on 1h and 4h: price above = macro bull, below = macro bear"
            "\n- Price vs VWAP: above = buyers in control, below = sellers in control"
            "\n- Trend confluence score: use the provided score (-3 to +3)"

            "\n\n2. MOMENTUM"
            "\n- RSI multi-timeframe: 15m for entry timing, 1h for current momentum, 4h for regime"
            "\n- RSI divergence warning: if 1h overbought but 4h neutral/oversold = reversal risk"
            "\n- MACD 1h histogram: positive and growing = strong momentum, flipping = exhaustion"
            "\n- MACD 15m: use for entry confirmation — only enter when 15m MACD aligns with 1h trend"

            "\n\n3. VOLATILITY"
            "\n- Bollinger Band position: above upper = extended, below lower = oversold extreme"
            "\n- BB width: expanding = trending, contracting = coiling before breakout"
            "\n- ATR: use for stop placement (1.5-2x ATR from entry)"
            "\n- High volatility regime = reduce size, widen stops"

            "\n\n4. KEY LEVELS"
            "\n- Order book walls: large ask wall above = hard resistance, large bid wall below = support"
            "\n- Bid/ask depth ratio: >1.3 = buy pressure, <0.7 = sell pressure"
            "\n- Swing highs/lows on 1h: nearest S/R for trade management"
            "\n- Daily swing levels: macro S/R for larger targets"

            "\n\n5. VOLUME"
            "\n- Volume vs 7d avg: >30% above avg on up move = strong breakout"
            "\n- Volume vs 7d avg: >30% above avg on down move = distribution"
            "\n- Low volume during consolidation = energy building"

            "\n\n═══ OUTPUT REQUIRED ═══"
            "\nProvide a structured report with these exact sections:"
            "\n1. Trend structure (1h + 4h + daily)"
            "\n2. Momentum assessment (RSI + MACD across timeframes)"
            "\n3. Volatility regime (ATR + BB)"
            "\n4. Key levels (order book walls + swing S/R)"
            "\n5. Volume analysis"
            "\n6. Entry timing signal (based on 15m MACD + RSI)"
            "\n7. Directional bias: BULLISH / BEARISH / NEUTRAL with confidence 1-10"
            "\n8. Suggested trade parameters: direction, entry zone, TP levels, SL level"
            "\nAppend a Markdown summary table of all key values."
        )

        prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "You are a helpful AI assistant collaborating with other trading analysts. "
                "Use ALL provided tools before writing your report — do not skip any. "
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
            "market_report": report,
        }

    return technical_analyst_node

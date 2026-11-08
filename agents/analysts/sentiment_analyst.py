"""
Sentiment analyst: reads from your existing X digest pipeline.

Expects a JSON file at DIGEST_PATH (configurable via env var) in the format
your pipeline already produces — or falls back to a static prompt if not found.

Digest JSON schema (what your pipeline should write):
{
  "fetched_at": "2026-03-20T09:30:00",
  "investment_signals": [...],
  "build_this_week": [...],
  "insider_tools": [...],
  "china_watch": [...]
}
"""

import os
import json
import sys
from pathlib import Path
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool


# ── Digest loader ─────────────────────────────────────────────────────────────

DIGEST_PATH = os.getenv(
    "DIGEST_PATH",
    str(Path.home() / "ai_digest" / "latest_digest.json")
)


def _load_digest() -> dict:
    """Load latest digest from your pipeline's output file."""
    path = Path(DIGEST_PATH)
    if not path.exists():
        return {
            "note": f"Digest file not found at {DIGEST_PATH}. "
                    "Set DIGEST_PATH env var to your pipeline's output JSON.",
            "investment_signals": [],
            "china_watch": [],
        }
    try:
        with open(path) as f:
            data = json.load(f)
        age_hours = (
            datetime.now(timezone.utc)
            - datetime.fromisoformat(data.get("fetched_at", "2000-01-01"))
        ).total_seconds() / 3600
        data["digest_age_hours"] = round(age_hours, 1)
        return data
    except Exception as e:
        return {"error": str(e), "investment_signals": []}


@tool
def fetch_sentiment_digest() -> dict:
    """
    Load the latest AI/crypto sentiment digest from your X pipeline.
    Returns investment signals, CT narratives, and China watch signals.
    Focus on signals relevant to BTC, ETH, SOL, and macro crypto sentiment.
    """
    return _load_digest()


@tool
def get_fear_greed_index() -> dict:
    """
    Fetch the current Crypto Fear & Greed Index from alternative.me.
    0-25 = Extreme Fear (buy signal), 75-100 = Extreme Greed (sell signal).
    """
    import requests
    try:
        resp = requests.get("https://api.alternative.me/fng/?limit=3", timeout=5)
        data = resp.json().get("data", [])
        if not data:
            return {"error": "no data"}
        return {
            "current": {
                "value": int(data[0]["value"]),
                "classification": data[0]["value_classification"],
                "timestamp": data[0]["timestamp"],
            },
            "yesterday": {
                "value": int(data[1]["value"]),
                "classification": data[1]["value_classification"],
            } if len(data) > 1 else None,
            "note": "Extreme Fear (<25) = contrarian buy. Extreme Greed (>75) = contrarian sell.",
        }
    except Exception as e:
        return {"error": str(e)}


def create_sentiment_analyst(llm):
    def sentiment_analyst_node(state):
        coin = state["company_of_interest"]
        current_date = state["trade_date"]

        tools = [fetch_sentiment_digest, get_fear_greed_index]

        system_message = (
            "You are a crypto sentiment analyst. Your job is to assess market narrative, "
            "social sentiment, and CT (Crypto Twitter) positioning."
            "\n\nFocus on:"
            "\n- Fear & Greed Index: extreme readings are contrarian signals"
            "\n- CT narrative: what story is being told? Is it consensus (fade it) or emerging (lean in)?"
            "\n- Investment signals from the digest: filter for ones relevant to the current asset"
            "\n- China market signals: relevant for macro crypto moves"
            "\n\nKey heuristic: when CT is unanimous in a direction, the trade is crowded. "
            "When everyone says 'to the moon' it's usually time to be careful. "
            "When CT is silent or fearful, it's often worth being contrarian."
            "\n\nProvide: (1) overall sentiment score (1-10, where 10 = max greed), "
            "(2) dominant narrative, "
            "(3) contrarian signals if any, "
            "(4) sentiment-based bias for the asset. "
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
            "sentiment_report": report,   # reuse existing state key
        }

    return sentiment_analyst_node

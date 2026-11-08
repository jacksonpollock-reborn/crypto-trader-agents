import os
from pathlib import Path

BASE_DIR = Path(__file__).parent

CRYPTO_CONFIG = {
    "project_dir": str(BASE_DIR),
    "results_dir": str(BASE_DIR / "logs"),
    "data_cache_dir": str(BASE_DIR / "logs" / "cache"),

    # ── LLM routing ──────────────────────────────────────────────────────────
    # Analysts: Claude Haiku — reliable tool calling, cheap ($0.002 per run)
    # Debate + Trader + Risk: Ollama — free, no tool calling needed here
    "llm_provider": "ollama",              # default provider (debate/risk/trader)
    "deep_think_llm": "qwen2.5:14b",      # Ollama: debate judge + risk manager
    "quick_think_llm": "qwen2.5:14b",     # Ollama: bull/bear researchers + trader

    # Analyst-specific override — uses Claude Haiku for reliable tool calling
    "analyst_llm_provider": "anthropic",
    "analyst_llm": "claude-haiku-4-5",

    "backend_url": None,

    # ── Debate settings ───────────────────────────────────────────────────────
    "max_debate_rounds": 2,
    "max_risk_discuss_rounds": 1,
    "max_recur_limit": 100,

    # ── Data vendors (overridden — we handle all data ourselves) ──────────────
    "data_vendors": {
        "core_stock_apis": "yfinance",
        "technical_indicators": "yfinance",
        "fundamental_data": "yfinance",
        "news_data": "yfinance",
    },
    "tool_vendors": {},

    # ── Hyperliquid settings ──────────────────────────────────────────────────
    "hl_mainnet_url": "https://api.hyperliquid.xyz",
    "hl_testnet_url": "https://api.hyperliquid-testnet.xyz",
    "use_testnet": True,

    # ── Crypto universe ───────────────────────────────────────────────────────
    "watchlist": ["BTC", "ETH", "SOL"],

    # ── Risk parameters ───────────────────────────────────────────────────────
    "max_leverage": 3.0,
    "max_portfolio_risk_pct": 0.02,
    "max_margin_utilization": 0.5,
    "trade_cooldown_seconds": 3600,
}

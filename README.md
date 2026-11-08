# crypto-trader-agents

A multi-agent LLM crypto trading research agent for Hyperliquid perpetuals, built on LangGraph and the TradingAgents framework. Paper trading only.

## Overview

Four specialist analysts (technical, on-chain, sentiment, macro) each write a report on a coin. A bull/bear research debate and a trader step turn those reports into a proposal, and a custom, position-aware risk manager can veto it before anything is "executed". Execution is simulated: orders are filled at the current Hyperliquid mid price and recorded in a local JSON state file, so you can study how the agents behave without risking capital.

This package is an adapter layer. It plugs crypto-specific analysts, a risk manager and an executor into the graph from the open-source [TradingAgents](https://github.com/TauricResearch/TradingAgents) framework (`tradingagents`), which is not included here.

## Features

- Four analysts with real tools: technical indicators, order book and candles; funding / open interest / liquidation data; Fear and Greed index plus a pluggable digest file; macro snapshot and crypto news headlines.
- Mixed LLM routing: Claude Haiku for the tool-calling analysts, a local Ollama model (default `qwen2.5:14b`) for the debate, trader and risk steps. Switchable to Anthropic for everything via `config.py`.
- Custom risk manager with hard vetoes: per-coin cooldown, margin utilization cap, and a 3-of-4 analyst agreement requirement. Applies leverage caps and Kelly-style size scaling from recent win rate.
- Paper executor: market-price fills, 2x leverage, 5% take-profit and 2% stop-loss levels recorded on each order, JSONL execution log.
- Per-analyst Brier-style outcome tracking that feeds dynamic agent weights.
- Retry with back-off for transient LLM overload errors.

## Architecture

```
 Hyperliquid API   CryptoCompare news   Fear & Greed   digest JSON (optional)
        \                |                   |                |
         v               v                   v                v
  [technical]      [on-chain]          [sentiment]        [macro]     <- Claude Haiku, tool calling
        \_______________|___________________|________________/
                                 |
                  bull / bear debate -> trader      <- Ollama (default)
                                 |
                  crypto risk manager (vetoes)      <- Ollama + position state
                                 |
                       BUY / SELL / HOLD signal
                                 |
                    paper executor -> logs/position_state.json
```

## Tech stack

Python, LangGraph, LangChain (`langchain-anthropic`, `langchain-ollama`), TradingAgents framework, Hyperliquid public info API, pandas, requests.

## Project structure

```
main.py                         Entry point: builds the graph, analyzes coins, routes decisions
config.py                       LLM routing, debate rounds, watchlist, risk parameters
agents/
  analysts/                     technical, onchain, sentiment, macro analysts and their tools
  risk_manager.py               Position-aware risk manager with veto logic
dataflows/hyperliquid_data.py   Hyperliquid prices, candles, order book, indicators, liquidations
execution/hyperliquid_executor.py  Position sizing and paper order execution (live is a stub)
state/position_state.py         JSON-backed positions, cooldowns, Brier scores, trade log
requirements.txt                Python dependencies
.env.example                    Placeholder environment variables
```

## Setup

Prerequisites:

- Python 3.10+
- The TradingAgents framework checked out so that `tradingagents` is importable. `main.py` imports `crypto_trader.*` and adds its parent directory to `sys.path`, so place this folder named `crypto_trader` inside (or next to) the TradingAgents repo root.
- An Anthropic API key (the analysts always use Claude).
- [Ollama](https://ollama.com) with the configured model pulled, unless you set `llm_provider` to `"anthropic"` in `config.py`.

```bash
git clone https://github.com/TauricResearch/TradingAgents.git
cd TradingAgents
git clone <this-repo> crypto_trader
pip install -r requirements.txt            # TradingAgents' own dependencies
pip install -r crypto_trader/requirements.txt
ollama pull qwen2.5:14b
cp crypto_trader/.env.example crypto_trader/.env   # then fill in ANTHROPIC_API_KEY
```

## Configuration

Environment variables (see `.env.example`):

| Variable | Purpose |
|---|---|
| `ANTHROPIC_API_KEY` | Required. Used by the analysts (and everything else if `llm_provider` is `anthropic`) |
| `HYPERLIQUID_WALLET_ADDRESS`, `HYPERLIQUID_PRIVATE_KEY` | Read by the executor but unused in paper mode; live ordering is not implemented |
| `DIGEST_PATH` | JSON digest for the sentiment analyst (default `~/ai_digest/latest_digest.json`; analyst degrades gracefully if missing) |
| `STATE_PATH` | Location of the paper position state file (default `logs/position_state.json`) |

Non-secret settings live in `config.py`: `llm_provider`, `deep_think_llm`, `quick_think_llm`, `analyst_llm`, debate rounds, `watchlist` (BTC, ETH, SOL), `max_leverage` (3x), `max_portfolio_risk_pct` (2%), `max_margin_utilization` (50%) and `trade_cooldown_seconds` (1 hour).

## Usage

Run from the TradingAgents repo root (the parent of `crypto_trader/`):

```bash
python crypto_trader/main.py BTC            # single coin, paper mode
python crypto_trader/main.py BTC ETH SOL    # several coins
python crypto_trader/main.py --debug BTC    # print every agent's report
python crypto_trader/main.py                # whole watchlist from config.py
```

Each run prints the signal, the simulated execution, and a session summary (open positions, trade count, win rate, PnL, agent weights). Logs and state are written under `logs/`.

## Status and limitations

- Paper mode only; `execute_live_order` raises `NotImplementedError`.
- Fills are simulated at mid price with no slippage, fees or funding.
- The agent graph relies on TradingAgents internals (it patches several of its factory functions), so it may break when that framework changes.
- The sentiment analyst expects a digest file from a pipeline that is not part of this repo.
- Win-rate and Brier tracking are based on very small samples and are illustrative only.
- Unit tests are not included.

## Disclaimer

This project is for research and educational purposes only and is not financial advice. LLM-generated trading signals are unreliable and trading leveraged derivatives can lose all of your capital. Use at your own risk; the authors accept no liability for any losses.

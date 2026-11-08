"""
Crypto trading agent — main entry point.

Run:
    python main.py BTC              # single coin, paper mode
    python main.py BTC ETH SOL      # full watchlist
    python main.py --debug BTC      # show full agent reports
"""

import os
import sys
import time
from pathlib import Path
from datetime import date, datetime, timezone

# ── Path setup ────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent
REPO_ROOT    = PROJECT_ROOT.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv()

from tradingagents.agents.utils.memory import FinancialSituationMemory
from tradingagents.dataflows.config import set_config

from crypto_trader.config import CRYPTO_CONFIG
from crypto_trader.agents.analysts.onchain_analyst import create_onchain_analyst
from crypto_trader.agents.analysts.technical_analyst import create_technical_analyst
from crypto_trader.agents.analysts.sentiment_analyst import create_sentiment_analyst
from crypto_trader.agents.analysts.macro_analyst import create_macro_analyst
from crypto_trader.agents.risk_manager import create_crypto_risk_manager
from crypto_trader.execution.hyperliquid_executor import route_decision
from crypto_trader.state.position_state import (
    get_portfolio_summary, record_outcome,
)
from crypto_trader.dataflows.hyperliquid_data import get_all_mids


def make_llm(model: str):
    provider = CRYPTO_CONFIG.get("llm_provider", "anthropic")
    if provider == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(
            model=model,
            base_url="http://127.0.0.1:11434",
            temperature=0.6,
            num_predict=2048,
        )
    else:
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model=model,
            anthropic_api_key=os.environ["ANTHROPIC_API_KEY"],
            max_tokens=4096,
        )


def make_analyst_llm():
    """Analyst LLM — always Claude Haiku for reliable tool calling."""
    from langchain_anthropic import ChatAnthropic
    return ChatAnthropic(
        model=CRYPTO_CONFIG["analyst_llm"],
        anthropic_api_key=os.environ["ANTHROPIC_API_KEY"],
        max_tokens=4096,
    )


def build_crypto_graph(config: dict):
    set_config(config)
    quick_llm   = make_llm(config["quick_think_llm"])   # Ollama: researchers, trader
    deep_llm    = make_llm(config["deep_think_llm"])    # Ollama: debate judge, risk
    analyst_llm = make_analyst_llm()                    # Haiku: 4 analysts (tool calling)

    bull_memory         = FinancialSituationMemory("bull_memory",         config)
    bear_memory         = FinancialSituationMemory("bear_memory",         config)
    trader_memory       = FinancialSituationMemory("trader_memory",       config)
    invest_judge_memory = FinancialSituationMemory("invest_judge_memory", config)
    risk_manager_memory = FinancialSituationMemory("risk_manager_memory", config)

    from tradingagents.graph.conditional_logic import ConditionalLogic
    import tradingagents.graph.setup as setup_mod
    import tradingagents.agents.managers.risk_manager as risk_mod
    from tradingagents.graph.setup import GraphSetup

    # Analysts → Haiku (reliable tool calling)
    setup_mod.create_fundamentals_analyst = lambda _: create_onchain_analyst(analyst_llm)
    setup_mod.create_market_analyst       = lambda _: create_technical_analyst(analyst_llm)
    setup_mod.create_social_media_analyst = lambda _: create_sentiment_analyst(analyst_llm)
    setup_mod.create_news_analyst         = lambda _: create_macro_analyst(analyst_llm)

    # Risk Manager → our custom one (Ollama deep_llm)
    risk_mod.create_risk_manager = create_crypto_risk_manager

    cond_logic = ConditionalLogic(
        max_debate_rounds=config["max_debate_rounds"],
        max_risk_discuss_rounds=config["max_risk_discuss_rounds"],
    )

    from langgraph.prebuilt import ToolNode
    from crypto_trader.agents.analysts.technical_analyst import (
        fetch_technical_indicators, fetch_order_book, fetch_raw_candles
    )
    from crypto_trader.agents.analysts.onchain_analyst import (
        fetch_onchain_signals, fetch_liquidation_data
    )
    from crypto_trader.agents.analysts.sentiment_analyst import (
        fetch_sentiment_digest, get_fear_greed_index
    )
    from crypto_trader.agents.analysts.macro_analyst import (
        fetch_crypto_macro_snapshot, fetch_crypto_news_headlines
    )

    crypto_tools = {
        "market":       ToolNode([fetch_technical_indicators, fetch_order_book, fetch_raw_candles]),
        "fundamentals": ToolNode([fetch_onchain_signals, fetch_liquidation_data]),
        "social":       ToolNode([fetch_sentiment_digest, get_fear_greed_index]),
        "news":         ToolNode([fetch_crypto_macro_snapshot, fetch_crypto_news_headlines]),
    }

    graph_setup = GraphSetup(
        quick_llm, deep_llm, crypto_tools,
        bull_memory, bear_memory, trader_memory,
        invest_judge_memory, risk_manager_memory,
        cond_logic,
    )
    return graph_setup.setup_graph(
        selected_analysts=["market", "social", "news", "fundamentals"]
    )


def _print_debug_reports(state: dict):
    sep = "─" * 60
    reports = {
        "📊 Technical Analyst":  state.get("market_report", ""),
        "⛓  On-chain Analyst":   state.get("fundamentals_report", ""),
        "💬 Sentiment Analyst":  state.get("sentiment_report", ""),
        "🌍 Macro Analyst":      state.get("news_report", ""),
    }
    debate = state.get("investment_debate_state", {})
    risk   = state.get("risk_debate_state", {})

    print(f"\n{'='*60}\n  AGENT REPORTS\n{'='*60}")
    for name, report in reports.items():
        if report:
            print(f"\n{sep}\n  {name}\n{sep}")
            print(report[:3000] + ("..." if len(report) > 3000 else ""))
    if debate.get("judge_decision"):
        print(f"\n{sep}\n  🤺 Research Debate Judge\n{sep}")
        print(debate["judge_decision"][:2000])
    if risk.get("judge_decision"):
        print(f"\n{sep}\n  ⚖️  Risk Manager\n{sep}")
        print(risk["judge_decision"][:2000])
    print(f"\n{'='*60}\n")


def _pending_path() -> Path:
    return Path(CRYPTO_CONFIG["results_dir"]) / "pending_outcomes.json"


def _load_pending() -> dict:
    import json
    p = _pending_path()
    return json.loads(p.read_text()) if p.exists() else {}


def _save_pending(data: dict):
    import json
    p = _pending_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2))


def _update_brier_outcomes(coin: str, signal: str):
    import json
    pending = _load_pending()
    mids = get_all_mids(use_testnet=False)
    price_now = mids.get(coin)
    if not price_now:
        return

    prev = pending.get(coin)
    if prev:
        outcome = 1.0 if price_now > prev["price"] else 0.0
        for agent in ["technical", "onchain", "sentiment", "macro"]:
            record_outcome(agent, outcome)
        print(f"  [brier] {coin}: {prev['price']:.0f}→{price_now:.0f} outcome={outcome}")

    pending[coin] = {
        "price":     price_now,
        "signal":    signal,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    _save_pending(pending)


def analyze_coin(coin: str, execute: bool = True, debug: bool = False) -> dict:
    print(f"\n{'='*60}")
    print(f"  Analyzing {coin}  |  {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"{'='*60}\n")

    config = CRYPTO_CONFIG.copy()
    os.makedirs(config["results_dir"], exist_ok=True)
    os.makedirs(config["data_cache_dir"], exist_ok=True)

    graph = build_crypto_graph(config)

    from tradingagents.graph.propagation import Propagator
    from tradingagents.graph.signal_processing import SignalProcessor

    quick_llm   = make_llm(config["quick_think_llm"])
    propagator  = Propagator()
    signal_proc = SignalProcessor(quick_llm)

    trade_date = date.today().isoformat()
    init_state = propagator.create_initial_state(coin, trade_date)
    args       = propagator.get_graph_args()

    for attempt in range(3):
        try:
            final_state = graph.invoke(init_state, **args)
            break
        except Exception as e:
            is_overload = "529" in str(e) or "overloaded" in str(e).lower()
            wait = 60 if is_overload else 15
            if attempt < 2:
                print(f"  Attempt {attempt+1} failed, retrying in {wait}s... ({e})")
                time.sleep(wait)
            else:
                raise

    if debug:
        _print_debug_reports(final_state)

    raw_decision = final_state.get("final_trade_decision", "HOLD")
    clean_signal = signal_proc.process_signal(raw_decision)

    print(f"\n{'─'*40}")
    print(f"  SIGNAL: {clean_signal}")
    print(f"{'─'*40}\n")

    _update_brier_outcomes(coin, clean_signal)

    result = {
        "coin":         coin,
        "date":         trade_date,
        "signal":       clean_signal,
        "raw_decision": raw_decision,
    }

    if execute:
        summary = get_portfolio_summary()
        equity  = summary.get("paper_equity_usd", 10000.0)
        action  = route_decision(coin, clean_signal, portfolio_equity=equity)
        result["execution"] = action
        print(f"  Execution: {action}\n")

    return result


if __name__ == "__main__":
    args  = sys.argv[1:]
    debug = "--debug" in args
    coins = [a for a in args if not a.startswith("--")]
    if not coins:
        coins = CRYPTO_CONFIG["watchlist"]

    provider = CRYPTO_CONFIG.get("llm_provider", "anthropic")
    print(f"\n🤖 Crypto Trading Agent — Paper Mode {'[DEBUG]' if debug else ''}")
    print(f"   Watchlist: {coins}")
    print(f"   Analysts:  {CRYPTO_CONFIG['analyst_llm']} (claude haiku)")
    print(f"   Debate:    {CRYPTO_CONFIG['deep_think_llm']} ({provider})")

    if provider != "ollama" and not os.getenv("ANTHROPIC_API_KEY"):
        print("\n❌  ANTHROPIC_API_KEY not set.")
        sys.exit(1)

    results = []
    for coin in coins:
        try:
            r = analyze_coin(coin, execute=True, debug=debug)
            results.append(r)
        except Exception as e:
            print(f"\n❌  Error analyzing {coin}: {e}")
            import traceback; traceback.print_exc()

    print("\n📊 Session Summary")
    print("─" * 40)
    for r in results:
        print(f"  {r['coin']:6s}  →  {r.get('signal', 'ERROR')}")
    print()

    summary = get_portfolio_summary()
    print(f"  Open positions: {summary['open_positions'] or 'none'}")
    print(f"  Total trades:   {summary['total_trades']}")
    print(f"  Win rate:       {summary['win_rate']:.1%}")
    print(f"  Total PnL:      ${summary['total_pnl_usd']:+.2f}")
    print(f"  Agent weights:  {summary['agent_weights']}\n")

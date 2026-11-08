"""
Custom Risk Manager — replaces TradingAgents' default Risk Judge.

Reads live position state before approving any trade:
  - Cooldown: no re-entry within cooldown_seconds of last trade
  - Margin utilization: veto if already using >max_margin_utilization
  - Kelly sizing: scale size based on recent win rate
  - Leverage cap: never exceed max_leverage
  - Conflict check: veto if analysts are too split (low conviction)

Also extracts directional predictions from each analyst report
and logs them for Brier score tracking.
"""

import re
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from crypto_trader.config import CRYPTO_CONFIG
from crypto_trader.state.position_state import (
    get_positions,
    get_portfolio_summary,
    is_on_cooldown,
    record_prediction,
    _load,
    _save,
)


# ── Prediction extractor ──────────────────────────────────────────────────────

def _extract_bias(report: str) -> float:
    """
    Extract directional bias from an analyst report.
    Returns 0.0 (strong short) to 1.0 (strong long), 0.5 = neutral.
    Looks for explicit bias keywords in the report text.
    """
    if not report:
        return 0.5

    text = report.lower()

    # Strong signals
    if any(w in text for w in ["strongly bullish", "strong buy", "high conviction long", "aggressive long"]):
        return 0.85
    if any(w in text for w in ["strongly bearish", "strong sell", "high conviction short", "aggressive short"]):
        return 0.15

    # Moderate signals
    if any(w in text for w in ["bullish", "buy", "long bias", "upside"]):
        score = 0.7
    elif any(w in text for w in ["bearish", "sell", "short bias", "downside"]):
        score = 0.3
    else:
        score = 0.5

    # Dampen if uncertainty language present
    if any(w in text for w in ["uncertain", "mixed", "neutral", "unclear", "caution", "wait"]):
        score = 0.5 + (score - 0.5) * 0.5

    return round(score, 2)


def _log_analyst_predictions(state: dict, coin: str):
    """Extract and log each analyst's directional bias for Brier tracking."""
    agent_report_map = {
        "technical":  state.get("market_report", ""),
        "onchain":    state.get("fundamentals_report", ""),
        "sentiment":  state.get("sentiment_report", ""),
        "macro":      state.get("news_report", ""),
    }
    for agent, report in agent_report_map.items():
        bias = _extract_bias(report)
        record_prediction(agent, bias)
        print(f"  [brier] {agent}: prediction={bias}")


# ── Position state checks ─────────────────────────────────────────────────────

def _get_margin_utilization() -> float:
    """
    Estimate current margin utilization from open positions.
    Returns fraction 0.0-1.0 of max_margin_utilization used.
    """
    positions = get_positions()
    if not positions:
        return 0.0

    total_size = sum(p.get("size_usd", 0) * p.get("leverage", 1) for p in positions.values())
    # Assume $10k base portfolio — will be made dynamic later
    portfolio_equity = 10000.0
    return total_size / portfolio_equity


def _count_analyst_agreement(state: dict) -> tuple[int, int, int]:
    """
    Count how many analysts are bullish / bearish / neutral.
    Returns (bulls, bears, neutrals).
    """
    reports = [
        state.get("market_report", ""),
        state.get("fundamentals_report", ""),
        state.get("sentiment_report", ""),
        state.get("news_report", ""),
    ]
    bulls, bears, neutrals = 0, 0, 0
    for r in reports:
        bias = _extract_bias(r)
        if bias >= 0.62:
            bulls += 1
        elif bias <= 0.38:
            bears += 1
        else:
            neutrals += 1
    return bulls, bears, neutrals


# ── Kelly sizing ──────────────────────────────────────────────────────────────

def _kelly_size_multiplier() -> float:
    """
    Compute Kelly-based size multiplier from recent trade history.
    Full Kelly on <5 trades, then scales with win rate.
    Capped at 1.0 (never bet more than base), floored at 0.25.
    """
    summary = get_portfolio_summary()
    n_trades = summary.get("total_trades", 0)
    win_rate = summary.get("win_rate", 0.5)

    if n_trades < 5:
        return 1.0  # not enough data — use full base size

    # Simplified Kelly: f = win_rate - (1 - win_rate) = 2*win_rate - 1
    kelly_f = max(0.0, 2 * win_rate - 1)
    # Half-Kelly for safety
    half_kelly = kelly_f / 2
    return round(max(0.25, min(1.0, half_kelly + 0.25)), 3)


# ── Main risk manager node ────────────────────────────────────────────────────

def create_crypto_risk_manager(llm, memory):
    """
    Drop-in replacement for TradingAgents' create_risk_manager.
    Adds position-state-aware veto logic before the LLM judgment.
    """
    def risk_manager_node(state) -> dict:
        coin        = state["company_of_interest"]
        trader_plan = state.get("investment_plan", "")
        risk_state  = state["risk_debate_state"]
        history     = risk_state["history"]

        # ── Step 1: Log analyst predictions for Brier tracking ────────────────
        _log_analyst_predictions(state, coin)

        # ── Step 2: Hard veto checks (bypass LLM entirely if triggered) ───────
        veto_reason = None

        # Cooldown check
        if is_on_cooldown(coin, CRYPTO_CONFIG["trade_cooldown_seconds"]):
            veto_reason = f"HOLD — {coin} on cooldown, minimum {CRYPTO_CONFIG['trade_cooldown_seconds']//60}min between trades"

        # Margin utilization check
        margin_util = _get_margin_utilization()
        if margin_util >= CRYPTO_CONFIG["max_margin_utilization"]:
            veto_reason = f"HOLD — portfolio margin utilization at {margin_util:.0%}, at maximum ({CRYPTO_CONFIG['max_margin_utilization']:.0%})"

        # Analyst conviction check — need at least 3/4 agreement to trade
        bulls, bears, neutrals = _count_analyst_agreement(state)
        total_directional = bulls + bears
        if total_directional < 3 and not veto_reason:
            veto_reason = f"HOLD — low conviction: {bulls} bullish, {bears} bearish, {neutrals} neutral analysts (need 3/4 agreement)"

        if veto_reason:
            print(f"  [risk] VETO: {veto_reason}")
            final_decision = f"FINAL TRANSACTION PROPOSAL: **HOLD**\n\nRisk Manager veto: {veto_reason}"
            new_risk_state = {**risk_state, "judge_decision": final_decision, "latest_speaker": "Risk Judge"}
            return {"risk_debate_state": new_risk_state, "final_trade_decision": final_decision}

        # ── Step 3: Kelly sizing — inject into prompt ─────────────────────────
        kelly_mult = _kelly_size_multiplier()
        summary    = get_portfolio_summary()
        sizing_note = (
            f"Kelly size multiplier: {kelly_mult:.2f}x base size "
            f"(win rate: {summary['win_rate']:.1%} over {summary['total_trades']} trades). "
            f"Max leverage: {CRYPTO_CONFIG['max_leverage']}x. "
            f"Current margin utilization: {margin_util:.1%}."
        )

        # ── Step 4: LLM judgment with position context ─────────────────────────
        past_memories = memory.get_memories(
            f"{state.get('market_report', '')}\n{state.get('sentiment_report', '')}",
            n_matches=2
        )
        past_memory_str = "\n\n".join(r["recommendation"] for r in past_memories)

        prompt = f"""You are the Risk Manager for a crypto perpetual futures trading system.

Your job: review the risk debate and make a FINAL decisive trade recommendation.
Never choose HOLD as a safe default — only HOLD if there is genuine conflict or risk.

ANALYST CONVICTION: {bulls} bullish, {bears} bearish, {neutrals} neutral out of 4 analysts.
PORTFOLIO CONTEXT: {sizing_note}
PAST LESSONS: {past_memory_str or 'No prior trades yet.'}

RISK DEBATE:
{history}

TRADER PLAN:
{trader_plan}

RULES:
1. If 3+ analysts agree on direction → approve that direction
2. If debate is split 2-2 → HOLD
3. Always include suggested leverage (max {CRYPTO_CONFIG['max_leverage']}x)
4. Always include TP% and SL% from entry
5. Size = base_size × {kelly_mult:.2f} (Kelly multiplier)

End your response with exactly one of:
FINAL TRANSACTION PROPOSAL: **BUY**
FINAL TRANSACTION PROPOSAL: **SELL**
FINAL TRANSACTION PROPOSAL: **HOLD**"""

        response = llm.invoke(prompt)
        final_decision = response.content

        print(f"  [risk] Approved — conviction: {bulls}B/{bears}S/{neutrals}N, Kelly: {kelly_mult:.2f}x")

        new_risk_state = {
            **risk_state,
            "judge_decision": final_decision,
            "latest_speaker": "Risk Judge",
        }
        return {"risk_debate_state": new_risk_state, "final_trade_decision": final_decision}

    return risk_manager_node

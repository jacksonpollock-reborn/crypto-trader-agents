"""
Execution layer: Hyperliquid SDK paper trading.

Paper mode: uses HL testnet — real API, fake money.
Flip use_testnet=False + set HL_PRIVATE_KEY for live trading (do NOT do this yet).

Requires: HYPERLIQUID_WALLET_ADDRESS env var (your testnet address)
Optional: HYPERLIQUID_PRIVATE_KEY env var (for actual order placement)
"""

import os
import json
import time
from pathlib import Path
from datetime import datetime, timezone

from crypto_trader.state.position_state import (
    open_position,
    close_position,
    get_position,
    is_on_cooldown,
    get_portfolio_summary,
)
from crypto_trader.config import CRYPTO_CONFIG
from crypto_trader.dataflows.hyperliquid_data import get_all_mids


WALLET_ADDRESS = os.getenv("HYPERLIQUID_WALLET_ADDRESS", "")
PRIVATE_KEY    = os.getenv("HYPERLIQUID_PRIVATE_KEY", "")
USE_TESTNET    = CRYPTO_CONFIG["use_testnet"]

# Log path for all execution events
EXEC_LOG = Path(CRYPTO_CONFIG["results_dir"]) / "execution_log.jsonl"


def _log_event(event: dict):
    EXEC_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(EXEC_LOG, "a") as f:
        f.write(json.dumps({**event, "timestamp": datetime.now(timezone.utc).isoformat()}) + "\n")


# ── Order sizing ──────────────────────────────────────────────────────────────

def compute_position_size(
    portfolio_equity: float,
    current_price: float,
    leverage: float,
    risk_pct: float = None,
) -> float:
    """
    Kelly-inspired position sizing.

    size_usd = portfolio_equity * risk_pct
    The notional exposure = size_usd * leverage

    We cap leverage at CRYPTO_CONFIG["max_leverage"] and
    notional at CRYPTO_CONFIG["max_margin_utilization"] * portfolio_equity.
    """
    if risk_pct is None:
        risk_pct = CRYPTO_CONFIG["max_portfolio_risk_pct"]

    leverage = min(leverage, CRYPTO_CONFIG["max_leverage"])
    size_usd = portfolio_equity * risk_pct

    # Cap: don't let notional exceed max_margin_utilization
    max_notional = portfolio_equity * CRYPTO_CONFIG["max_margin_utilization"]
    notional = size_usd * leverage
    if notional > max_notional:
        size_usd = max_notional / leverage

    return round(size_usd, 2)


# ── Paper order execution ─────────────────────────────────────────────────────

def execute_paper_order(
    coin: str,
    side: str,              # "long" or "short"
    size_usd: float,
    leverage: float = 2.0,
    tp_pct: float = None,   # take-profit % from entry (e.g. 0.05 = 5%)
    sl_pct: float = None,   # stop-loss % from entry (e.g. 0.02 = 2%)
) -> dict:
    """
    Simulate order execution at current market price.
    Logs to execution_log.jsonl and updates position_state.json.
    """
    mids = get_all_mids(use_testnet=False)  # always use mainnet prices for paper
    if coin not in mids:
        return {"error": f"Coin {coin} not found in market data"}

    entry_price = mids[coin]

    tp_price = (
        entry_price * (1 + tp_pct) if side == "long" and tp_pct
        else entry_price * (1 - tp_pct) if side == "short" and tp_pct
        else None
    )
    sl_price = (
        entry_price * (1 - sl_pct) if side == "long" and sl_pct
        else entry_price * (1 + sl_pct) if side == "short" and sl_pct
        else None
    )

    order = {
        "mode":        "PAPER",
        "coin":        coin,
        "side":        side,
        "size_usd":    size_usd,
        "entry_price": entry_price,
        "leverage":    leverage,
        "tp_price":    tp_price,
        "sl_price":    sl_price,
        "notional":    round(size_usd * leverage, 2),
    }

    open_position(
        coin=coin,
        side=side,
        size_usd=size_usd,
        entry_price=entry_price,
        leverage=leverage,
        paper=True,
    )

    _log_event({"type": "open", **order})
    print(f"\n{'='*50}")
    print(f"  📄 PAPER ORDER EXECUTED")
    print(f"  {side.upper()} {coin}  ${size_usd:.0f} × {leverage}x = ${size_usd*leverage:.0f} notional")
    print(f"  Entry: ${entry_price:,.2f}")
    if tp_price: print(f"  TP:    ${tp_price:,.2f}")
    if sl_price: print(f"  SL:    ${sl_price:,.2f}")
    print(f"{'='*50}\n")

    return order


def execute_paper_close(coin: str, reason: str = "signal") -> dict:
    """Close an open paper position at current market price."""
    pos = get_position(coin)
    if not pos:
        return {"error": f"No open position for {coin}"}

    mids = get_all_mids(use_testnet=False)
    exit_price = mids.get(coin, pos["entry_price"])

    trade = close_position(coin, exit_price=exit_price, reason=reason)
    _log_event({"type": "close", "coin": coin, "exit_price": exit_price, "reason": reason})

    return trade or {}


# ── Decision router ───────────────────────────────────────────────────────────

def route_decision(
    coin: str,
    decision: str,
    portfolio_equity: float = 10000.0,
) -> dict:
    """
    Parse the final decision string from TradingAgents and execute accordingly.

    Expected decision formats:
      "BUY"  / "LONG"                   → open long (or hold if already long)
      "SELL" / "SHORT"                  → open short (or hold if already short)
      "HOLD"                            → do nothing
      "CLOSE"                           → close existing position

    Returns an action summary dict.
    """
    decision = decision.strip().upper()
    current_pos = get_position(coin)

    # Normalize
    if decision in ("BUY",):
        action = "LONG"
    elif decision in ("SELL",):
        action = "SHORT"
    else:
        action = decision   # HOLD, CLOSE, SHORT, LONG

    # Cooldown guard
    if action in ("LONG", "SHORT") and is_on_cooldown(coin, CRYPTO_CONFIG["trade_cooldown_seconds"]):
        return {"action": "SKIPPED", "reason": "cooldown active", "coin": coin}

    # Already in correct direction
    if current_pos and current_pos["side"] == action.lower():
        return {"action": "HOLD", "reason": "already in position", "coin": coin}

    # Close opposite position first
    if current_pos and action in ("LONG", "SHORT"):
        execute_paper_close(coin, reason="direction_change")

    if action == "LONG":
        mids = get_all_mids(use_testnet=False)
        size = compute_position_size(portfolio_equity, mids.get(coin, 1), leverage=2.0)
        return execute_paper_order(coin, "long", size, leverage=2.0, tp_pct=0.05, sl_pct=0.02)

    elif action == "SHORT":
        mids = get_all_mids(use_testnet=False)
        size = compute_position_size(portfolio_equity, mids.get(coin, 1), leverage=2.0)
        return execute_paper_order(coin, "short", size, leverage=2.0, tp_pct=0.05, sl_pct=0.02)

    elif action == "CLOSE":
        return execute_paper_close(coin, reason="signal_close")

    else:  # HOLD
        return {"action": "HOLD", "coin": coin}


# ── Live order execution (NOT for paper trading — left stubbed) ───────────────

def execute_live_order(*args, **kwargs):
    raise NotImplementedError(
        "Live trading not enabled. "
        "Set use_testnet=False in config and implement this function "
        "using the Hyperliquid SDK's Exchange class."
    )

"""
Position state: JSON-backed persistent state for the trading system.

Tracks:
  - Open positions (coin → {side, size, entry_price, leverage, opened_at})
  - Trade cooldowns (coin → last_trade_timestamp)
  - Agent Brier scores (agent → rolling accuracy for dynamic weighting)
  - Portfolio equity snapshots

File: logs/position_state.json
"""

import json
import os
import time
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional

STATE_PATH = Path(os.getenv(
    "STATE_PATH",
    str(Path(__file__).parent.parent / "logs" / "position_state.json")
))


def _load() -> dict:
    if not STATE_PATH.exists():
        return _empty_state()
    try:
        with open(STATE_PATH) as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return _empty_state()


def _save(state: dict):
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(STATE_PATH, "w") as f:
        json.dump(state, f, indent=2)


def _empty_state() -> dict:
    return {
        "positions": {},          # coin → position dict
        "cooldowns": {},          # coin → last_trade_epoch_seconds
        "brier_scores": {         # agent → {"predictions": [...], "outcomes": [...]}
            "technical":  {"predictions": [], "outcomes": []},
            "onchain":    {"predictions": [], "outcomes": []},
            "sentiment":  {"predictions": [], "outcomes": []},
            "macro":      {"predictions": [], "outcomes": []},
        },
        "equity_snapshots": [],   # [{timestamp, equity_usd}]
        "trade_log": [],          # all closed trades
    }


# ── Position management ───────────────────────────────────────────────────────

def get_positions() -> dict:
    return _load()["positions"]


def get_position(coin: str) -> Optional[dict]:
    return _load()["positions"].get(coin)


def open_position(
    coin: str,
    side: str,          # "long" or "short"
    size_usd: float,
    entry_price: float,
    leverage: float,
    paper: bool = True,
):
    state = _load()
    state["positions"][coin] = {
        "side": side,
        "size_usd": size_usd,
        "entry_price": entry_price,
        "leverage": leverage,
        "opened_at": datetime.now(timezone.utc).isoformat(),
        "paper": paper,
    }
    state["cooldowns"][coin] = time.time()
    _save(state)
    print(f"[state] Opened {side} {coin} | size=${size_usd:.0f} | entry={entry_price} | {leverage}x")


def close_position(coin: str, exit_price: float, reason: str = "signal"):
    state = _load()
    pos = state["positions"].pop(coin, None)
    if pos is None:
        print(f"[state] No open position for {coin}")
        return

    pnl_pct = (
        (exit_price - pos["entry_price"]) / pos["entry_price"] * 100
        if pos["side"] == "long"
        else (pos["entry_price"] - exit_price) / pos["entry_price"] * 100
    )
    pnl_usd = pos["size_usd"] * pnl_pct / 100 * pos["leverage"]

    trade_record = {
        **pos,
        "coin": coin,
        "exit_price": exit_price,
        "closed_at": datetime.now(timezone.utc).isoformat(),
        "pnl_pct": round(pnl_pct, 3),
        "pnl_usd": round(pnl_usd, 2),
        "reason": reason,
    }
    state["trade_log"].append(trade_record)
    state["cooldowns"][coin] = time.time()
    _save(state)

    print(f"[state] Closed {coin} | PnL: {pnl_pct:+.2f}% (${pnl_usd:+.2f}) | reason={reason}")
    return trade_record


# ── Cooldown check ────────────────────────────────────────────────────────────

def is_on_cooldown(coin: str, cooldown_seconds: int = 3600) -> bool:
    state = _load()
    last_trade = state["cooldowns"].get(coin, 0)
    elapsed = time.time() - last_trade
    on_cd = elapsed < cooldown_seconds
    if on_cd:
        remaining = int(cooldown_seconds - elapsed)
        print(f"[state] {coin} on cooldown — {remaining}s remaining")
    return on_cd


# ── Brier score tracking (dynamic agent weighting) ───────────────────────────

def record_prediction(agent: str, prediction: float):
    """
    Record an agent's directional prediction.
    prediction: 0.0 (strong short) to 1.0 (strong long), 0.5 = neutral
    """
    state = _load()
    if agent not in state["brier_scores"]:
        state["brier_scores"][agent] = {"predictions": [], "outcomes": []}
    state["brier_scores"][agent]["predictions"].append({
        "value": prediction,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })
    _save(state)


def record_outcome(agent: str, outcome: float):
    """
    Record actual outcome for the most recent prediction.
    outcome: 1.0 if price went up, 0.0 if price went down
    """
    state = _load()
    if agent not in state["brier_scores"]:
        return
    state["brier_scores"][agent]["outcomes"].append({
        "value": outcome,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })
    _save(state)


def get_agent_weights(window: int = 30) -> dict[str, float]:
    """
    Compute dynamic agent weights from rolling Brier scores.
    Returns normalized weights {agent: weight} summing to 1.0.

    Brier score = mean((prediction - outcome)^2), lower = better.
    We invert so higher accuracy → higher weight.
    """
    state = _load()
    weights = {}

    for agent, data in state["brier_scores"].items():
        preds = [p["value"] for p in data["predictions"][-window:]]
        outcomes = [o["value"] for o in data["outcomes"][-window:]]

        n = min(len(preds), len(outcomes))
        if n < 5:
            # Not enough data — equal weight
            weights[agent] = 1.0
            continue

        brier = sum((p - o) ** 2 for p, o in zip(preds[-n:], outcomes[-n:])) / n
        # Convert to accuracy weight: max Brier = 1.0 (always wrong), min = 0.0 (perfect)
        weights[agent] = max(0.01, 1.0 - brier)

    # Normalize
    total = sum(weights.values())
    return {a: round(w / total, 3) for a, w in weights.items()}


# ── Portfolio summary ─────────────────────────────────────────────────────────

def get_portfolio_summary() -> dict:
    state = _load()
    positions = state["positions"]
    trade_log = state["trade_log"]

    total_pnl = sum(t.get("pnl_usd", 0) for t in trade_log)
    win_trades = [t for t in trade_log if t.get("pnl_usd", 0) > 0]
    win_rate = len(win_trades) / len(trade_log) if trade_log else 0

    return {
        "open_positions": list(positions.keys()),
        "total_trades": len(trade_log),
        "win_rate": round(win_rate, 3),
        "total_pnl_usd": round(total_pnl, 2),
        "agent_weights": get_agent_weights(),
    }

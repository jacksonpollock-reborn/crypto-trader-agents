"""
Hyperliquid data layer.

All functions return plain Python dicts/lists — no pandas required by callers,
though we use it internally for indicator math.
"""

import requests
import time
from datetime import datetime, timedelta, timezone
from typing import Optional
import pandas as pd


# ── Low-level HTTP helpers ────────────────────────────────────────────────────

def _post(url: str, payload: dict, timeout: int = 10) -> dict:
    resp = requests.post(url, json=payload, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def _get_base_url(use_testnet: bool = True) -> str:
    if use_testnet:
        return "https://api.hyperliquid-testnet.xyz"
    return "https://api.hyperliquid.xyz"


# ── Market snapshot ───────────────────────────────────────────────────────────

def get_all_mids(use_testnet: bool = True) -> dict[str, float]:
    """Return {coin: mid_price} for all perps."""
    url = f"{_get_base_url(use_testnet)}/info"
    data = _post(url, {"type": "allMids"})
    return {k: float(v) for k, v in data.items()}


def get_asset_context(coin: str, use_testnet: bool = True) -> dict:
    """
    Return per-asset context: funding rate, open interest, mark price, etc.
    Hyperliquid returns meta + assetCtxs as a parallel array.
    """
    url = f"{_get_base_url(use_testnet)}/info"
    data = _post(url, {"type": "metaAndAssetCtxs"})
    meta, ctxs = data[0], data[1]

    coin_index = next(
        (i for i, a in enumerate(meta["universe"]) if a["name"] == coin), None
    )
    if coin_index is None:
        raise ValueError(f"Coin {coin!r} not found on Hyperliquid")

    ctx = ctxs[coin_index]
    return {
        "coin": coin,
        "mark_price": float(ctx.get("markPx", 0)),
        "mid_price": float(ctx.get("midPx") or 0),
        "funding_rate": float(ctx.get("funding", 0)),
        "open_interest": float(ctx.get("openInterest", 0)),
        "open_interest_usd": float(ctx.get("oiInUsd") or 0),
        "prev_day_px": float(ctx.get("prevDayPx", 0)),
        "day_ntl_vlm": float(ctx.get("dayNtlVlm", 0)),
        "premium": float(ctx.get("premium") or 0),
    }


# ── OHLCV candles ─────────────────────────────────────────────────────────────

def get_candles(
    coin: str,
    interval: str = "1h",
    lookback_hours: int = 48,
    use_testnet: bool = True,
) -> list[dict]:
    """
    Fetch OHLCV candles.
    interval options: 1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 8h, 12h, 1d
    Returns list of {t, o, h, l, c, v} dicts.
    """
    now_ms = int(time.time() * 1000)
    start_ms = now_ms - lookback_hours * 3600 * 1000

    url = f"{_get_base_url(use_testnet)}/info"
    data = _post(url, {
        "type": "candleSnapshot",
        "req": {
            "coin": coin,
            "interval": interval,
            "startTime": start_ms,
            "endTime": now_ms,
        }
    })

    return [
        {
            "timestamp": datetime.fromtimestamp(c["t"] / 1000, tz=timezone.utc).isoformat(),
            "open":  float(c["o"]),
            "high":  float(c["h"]),
            "low":   float(c["l"]),
            "close": float(c["c"]),
            "volume": float(c["v"]),
        }
        for c in (data or [])
    ]


# ── Order book snapshot ───────────────────────────────────────────────────────

def get_order_book(coin: str, use_testnet: bool = True) -> dict:
    """
    Fetch L2 order book snapshot. Returns top bid/ask walls and spread.
    Large walls = support/resistance. Thin book = slippage risk.
    """
    url = f"{_get_base_url(use_testnet)}/info"
    data = _post(url, {"type": "l2Book", "coin": coin})

    levels = data.get("levels", [[], []])
    bids = levels[0][:10] if levels[0] else []
    asks = levels[1][:10] if levels[1] else []

    if not bids or not asks:
        return {"error": "empty order book"}

    best_bid = float(bids[0]["px"])
    best_ask = float(asks[0]["px"])
    spread = best_ask - best_bid
    spread_pct = spread / best_bid * 100

    # Find largest walls in top 10 levels
    bid_walls = sorted(bids, key=lambda x: float(x["sz"]), reverse=True)[:3]
    ask_walls = sorted(asks, key=lambda x: float(x["sz"]), reverse=True)[:3]

    total_bid_depth = sum(float(b["sz"]) for b in bids)
    total_ask_depth = sum(float(a["sz"]) for a in asks)
    bid_ask_ratio = total_bid_depth / max(total_ask_depth, 0.001)

    return {
        "best_bid": best_bid,
        "best_ask": best_ask,
        "spread": round(spread, 4),
        "spread_pct": round(spread_pct, 4),
        "bid_ask_depth_ratio": round(bid_ask_ratio, 3),
        "depth_signal": (
            "buy_pressure" if bid_ask_ratio > 1.3 else
            "sell_pressure" if bid_ask_ratio < 0.7 else "balanced"
        ),
        "strongest_bid_walls": [
            {"price": float(w["px"]), "size": float(w["sz"])} for w in bid_walls
        ],
        "strongest_ask_walls": [
            {"price": float(w["px"]), "size": float(w["sz"])} for w in ask_walls
        ],
        "note": (
            "Bid/ask ratio >1.3 = more buy pressure. "
            "Large ask wall above price = resistance. "
            "Large bid wall below price = support. "
            "Spread >0.05% = illiquid, expect slippage."
        ),
    }


# ── Technical indicators (derived from candles) ───────────────────────────────

def compute_technical_summary(coin: str, use_testnet: bool = True) -> dict:
    """
    Full technical analysis suite:
    - EMA 9/21/50/200 (1h + 4h)
    - RSI 14 (1h, 4h, 15m)
    - MACD (1h)
    - Bollinger Bands (1h)
    - ATR — volatility regime
    - VWAP (24h rolling)
    - Volume profile: current vs 7d average
    - Swing highs/lows (key support/resistance)
    - 15m momentum for entry timing
    """
    candles_1h  = get_candles(coin, "1h",  lookback_hours=240, use_testnet=use_testnet)  # 10d
    candles_4h  = get_candles(coin, "4h",  lookback_hours=720, use_testnet=use_testnet)  # 30d
    candles_15m = get_candles(coin, "15m", lookback_hours=48,  use_testnet=use_testnet)  # 2d
    candles_1d  = get_candles(coin, "1d",  lookback_hours=720, use_testnet=use_testnet)  # 30d

    if not candles_1h:
        return {"error": "no candle data returned"}

    def S(candles, key):
        return pd.Series([c[key] for c in candles], dtype=float)

    def ema(s, n):
        return round(float(s.ewm(span=n, adjust=False).mean().iloc[-1]), 4)

    def rsi(s, n=14):
        delta = s.diff().dropna()
        gain  = delta.clip(lower=0).rolling(n).mean()
        loss  = (-delta.clip(upper=0)).rolling(n).mean()
        rs    = gain / loss.replace(0, float("nan"))
        val   = (100 - 100 / (1 + rs)).iloc[-1]
        return round(float(val), 2)

    def macd(s):
        exp12  = s.ewm(span=12, adjust=False).mean()
        exp26  = s.ewm(span=26, adjust=False).mean()
        line   = exp12 - exp26
        signal = line.ewm(span=9, adjust=False).mean()
        hist   = line - signal
        return round(float(line.iloc[-1]), 4), round(float(signal.iloc[-1]), 4), round(float(hist.iloc[-1]), 4)

    def bollinger(s, n=20, k=2):
        mid  = s.rolling(n).mean()
        std  = s.rolling(n).std()
        upper = mid + k * std
        lower = mid - k * std
        return round(float(upper.iloc[-1]), 4), round(float(mid.iloc[-1]), 4), round(float(lower.iloc[-1]), 4)

    def atr(candles, n=14):
        highs  = pd.Series([c["high"]  for c in candles], dtype=float)
        lows   = pd.Series([c["low"]   for c in candles], dtype=float)
        closes = pd.Series([c["close"] for c in candles], dtype=float)
        prev_c = closes.shift(1)
        tr = pd.concat([
            highs - lows,
            (highs - prev_c).abs(),
            (lows  - prev_c).abs(),
        ], axis=1).max(axis=1)
        return round(float(tr.rolling(n).mean().iloc[-1]), 4)

    def vwap_24h(candles):
        """24h rolling VWAP = sum(typical_price * volume) / sum(volume)"""
        last24 = candles[-24:] if len(candles) >= 24 else candles
        tp_vol = sum(((c["high"] + c["low"] + c["close"]) / 3) * c["volume"] for c in last24)
        vol    = sum(c["volume"] for c in last24)
        return round(tp_vol / max(vol, 0.001), 4)

    def swing_levels(candles, n=5):
        """Find recent swing highs and lows (pivot points)."""
        highs  = [c["high"]  for c in candles]
        lows   = [c["low"]   for c in candles]
        pivots_h, pivots_l = [], []
        for i in range(n, len(highs) - n):
            if highs[i] == max(highs[i-n:i+n+1]):
                pivots_h.append(round(highs[i], 2))
            if lows[i] == min(lows[i-n:i+n+1]):
                pivots_l.append(round(lows[i], 2))
        return sorted(set(pivots_h))[-4:], sorted(set(pivots_l))[:4]

    # ── 1h indicators ─────────────────────────────────────────────────────────
    c1h = S(candles_1h, "close")
    v1h = S(candles_1h, "volume")

    current_price = round(float(c1h.iloc[-1]), 4)

    ema9   = ema(c1h, 9)
    ema21  = ema(c1h, 21)
    ema50  = ema(c1h, 50)
    ema200 = ema(c1h, 200) if len(c1h) >= 200 else None

    rsi_1h  = rsi(c1h)
    macd_line, macd_sig, macd_hist = macd(c1h)
    bb_upper, bb_mid, bb_lower = bollinger(c1h)
    atr_1h  = atr(candles_1h)
    vwap    = vwap_24h(candles_1h)
    swing_h, swing_l = swing_levels(candles_1h)

    bb_position = (
        "above_upper"  if current_price > bb_upper else
        "below_lower"  if current_price < bb_lower else
        "near_upper"   if current_price > bb_mid + (bb_upper - bb_mid) * 0.7 else
        "near_lower"   if current_price < bb_mid - (bb_mid - bb_lower) * 0.7 else
        "mid_range"
    )

    # ── 4h indicators ─────────────────────────────────────────────────────────
    c4h    = S(candles_4h, "close") if candles_4h else c1h
    rsi_4h = rsi(c4h)
    ema50_4h  = ema(c4h, 50)
    ema200_4h = ema(c4h, 200) if len(c4h) >= 200 else None
    atr_4h = atr(candles_4h) if candles_4h else None

    # ── 15m indicators (entry timing) ─────────────────────────────────────────
    c15m   = S(candles_15m, "close") if candles_15m else c1h
    rsi_15m = rsi(c15m)
    macd_15m_line, macd_15m_sig, macd_15m_hist = macd(c15m) if candles_15m else (None, None, None)

    # ── Volume profile ─────────────────────────────────────────────────────────
    vol_24h    = float(v1h.iloc[-24:].sum()) if len(v1h) >= 24 else None
    vol_7d_avg = float(v1h.sum() / max(len(v1h) / 24, 1))
    vol_vs_avg = round((vol_24h / vol_7d_avg - 1) * 100, 2) if vol_24h and vol_7d_avg else None

    # ── Daily levels ───────────────────────────────────────────────────────────
    daily_swing_h, daily_swing_l = swing_levels(candles_1d) if candles_1d else ([], [])

    # ── Trend confluence score (−3 to +3) ─────────────────────────────────────
    score = 0
    score += 1 if ema9 > ema21 > ema50 else (-1 if ema9 < ema21 < ema50 else 0)
    score += 1 if current_price > vwap else -1
    score += 1 if macd_hist > 0 else -1
    trend_label = (
        "strongly_bullish" if score >= 2 else
        "bullish"          if score == 1 else
        "strongly_bearish" if score <= -2 else
        "bearish"          if score == -1 else
        "neutral"
    )

    # ── Price change periods ───────────────────────────────────────────────────
    change_24h = round((current_price - float(c1h.iloc[-24])) / float(c1h.iloc[-24]) * 100, 2) if len(c1h) >= 24 else None
    change_7d  = round((current_price - float(c1h.iloc[0]))  / float(c1h.iloc[0])  * 100, 2)

    return {
        "coin": coin,
        "current_price": current_price,

        # ── Trend ──────────────────────────────────────────────────────────────
        "trend_confluence": trend_label,
        "trend_score": f"{score}/3",

        # ── EMA ribbon (1h) ────────────────────────────────────────────────────
        "ema_9_1h":   ema9,
        "ema_21_1h":  ema21,
        "ema_50_1h":  ema50,
        "ema_200_1h": ema200,
        "ema_trend_1h": (
            "bullish" if ema9 > ema21 > ema50 else
            "bearish" if ema9 < ema21 < ema50 else "mixed"
        ),

        # ── EMA (4h) ───────────────────────────────────────────────────────────
        "ema_50_4h":  ema50_4h,
        "ema_200_4h": ema200_4h,
        "ema_trend_4h": (
            "price_above_50_4h" if current_price > ema50_4h else "price_below_50_4h"
        ),

        # ── RSI multi-timeframe ────────────────────────────────────────────────
        "rsi_15m": rsi_15m,
        "rsi_1h":  rsi_1h,
        "rsi_4h":  rsi_4h,
        "rsi_signal_1h": (
            "overbought" if rsi_1h > 70 else
            "oversold"   if rsi_1h < 30 else "neutral"
        ),
        "rsi_divergence": (
            "bearish_divergence_warning" if rsi_1h > 65 and rsi_4h < 50 else
            "bullish_divergence_warning" if rsi_1h < 35 and rsi_4h > 50 else
            "aligned"
        ),

        # ── MACD (1h) ──────────────────────────────────────────────────────────
        "macd_line_1h":   macd_line,
        "macd_signal_1h": macd_sig,
        "macd_hist_1h":   macd_hist,
        "macd_signal":    "bullish_cross" if macd_hist > 0 else "bearish_cross",

        # ── MACD (15m) — entry timing ──────────────────────────────────────────
        "macd_hist_15m":  macd_15m_hist,
        "macd_15m_signal": (
            "bullish_momentum" if macd_15m_hist and macd_15m_hist > 0 else
            "bearish_momentum" if macd_15m_hist and macd_15m_hist < 0 else "neutral"
        ),

        # ── Bollinger Bands (1h) ───────────────────────────────────────────────
        "bb_upper_1h": bb_upper,
        "bb_mid_1h":   bb_mid,
        "bb_lower_1h": bb_lower,
        "bb_position": bb_position,
        "bb_width_pct": round((bb_upper - bb_lower) / bb_mid * 100, 2),

        # ── ATR (volatility) ───────────────────────────────────────────────────
        "atr_1h": atr_1h,
        "atr_4h": atr_4h,
        "atr_pct_of_price": round(atr_1h / current_price * 100, 3),
        "volatility_regime": (
            "high"   if atr_1h / current_price > 0.015 else
            "medium" if atr_1h / current_price > 0.007 else "low"
        ),

        # ── VWAP ───────────────────────────────────────────────────────────────
        "vwap_24h": vwap,
        "price_vs_vwap": "above" if current_price > vwap else "below",
        "vwap_deviation_pct": round((current_price - vwap) / vwap * 100, 3),

        # ── Volume profile ─────────────────────────────────────────────────────
        "volume_24h": round(vol_24h, 2) if vol_24h else None,
        "volume_vs_7d_avg_pct": vol_vs_avg,
        "volume_signal": (
            "high_volume"  if vol_vs_avg and vol_vs_avg > 30 else
            "low_volume"   if vol_vs_avg and vol_vs_avg < -30 else "normal"
        ),

        # ── Key levels ─────────────────────────────────────────────────────────
        "swing_highs_1h":    swing_h,   # nearest resistances
        "swing_lows_1h":     swing_l,   # nearest supports
        "swing_highs_daily": daily_swing_h,
        "swing_lows_daily":  daily_swing_l,

        # ── Price performance ──────────────────────────────────────────────────
        "price_change_24h_pct": change_24h,
        "price_change_7d_pct":  change_7d,

        # ── Recent candles for LLM context ─────────────────────────────────────
        "recent_candles_1h":  candles_1h[-6:],   # last 6h
        "recent_candles_15m": candles_15m[-8:] if candles_15m else [],  # last 2h
    }


# ── On-chain / derivatives signals ───────────────────────────────────────────

def get_onchain_summary(coin: str, use_testnet: bool = True) -> dict:
    """
    Aggregate on-chain derivatives signals for the on-chain analyst.
    Funding rate, OI delta, premium — key mean-reversion inputs.
    """
    ctx = get_asset_context(coin, use_testnet=use_testnet)

    funding_annualised = ctx["funding_rate"] * 3 * 365 * 100  # 8h funding → annualised %

    # Signal interpretation
    if ctx["funding_rate"] > 0.001:        # >0.1% per 8h = very positive funding
        funding_signal = "extremely_long_biased — mean reversion risk HIGH"
    elif ctx["funding_rate"] > 0.0003:
        funding_signal = "mildly_long_biased"
    elif ctx["funding_rate"] < -0.001:
        funding_signal = "extremely_short_biased — short squeeze risk HIGH"
    elif ctx["funding_rate"] < -0.0003:
        funding_signal = "mildly_short_biased"
    else:
        funding_signal = "neutral"

    price_change_pct = (
        (ctx["mark_price"] - ctx["prev_day_px"]) / ctx["prev_day_px"] * 100
        if ctx["prev_day_px"] > 0 else 0
    )

    return {
        "coin": coin,
        "mark_price": ctx["mark_price"],
        "funding_rate_8h": ctx["funding_rate"],
        "funding_rate_annualised_pct": round(funding_annualised, 2),
        "funding_signal": funding_signal,
        "open_interest_usd": ctx["open_interest_usd"],
        "open_interest_contracts": ctx["open_interest"],
        "price_change_24h_pct": round(price_change_pct, 2),
        "24h_volume_usd": ctx["day_ntl_vlm"],
        "premium_vs_spot": ctx["premium"],
        "note": (
            "High positive funding + rising OI = overleveraged longs, fade rallies. "
            "High negative funding = overleveraged shorts, buy dips. "
            "OI rising with price = trend confirmation. "
            "OI falling with price = deleveraging, watch for capitulation."
        ),
    }


# ── Coinglass liquidation heatmap (public endpoint, no API key) ───────────────

def get_liquidation_levels(coin: str, hours: int = 24) -> dict:
    """
    Fetch recent liquidation data from Coinglass.
    These levels act as price magnets — large liq clusters = likely targets.
    Returns a summary dict with notable levels.
    """
    symbol_map = {"BTC": "BTC", "ETH": "ETH", "SOL": "SOL"}
    cg_symbol = symbol_map.get(coin, coin)

    try:
        url = "https://open-api.coinglass.com/public/v2/liquidation_history"
        params = {"symbol": cg_symbol, "time_type": "h1", "limit": hours}
        headers = {"accept": "application/json"}
        resp = requests.get(url, params=params, headers=headers, timeout=8)

        if resp.status_code != 200:
            return _liquidation_fallback(coin)

        data = resp.json().get("data", [])
        if not data:
            return _liquidation_fallback(coin)

        total_long_liq  = sum(float(d.get("longLiquidationUsd", 0)) for d in data)
        total_short_liq = sum(float(d.get("shortLiquidationUsd", 0)) for d in data)

        return {
            "coin": coin,
            "period_hours": hours,
            "total_long_liquidations_usd":  round(total_long_liq, 0),
            "total_short_liquidations_usd": round(total_short_liq, 0),
            "long_vs_short_ratio": round(total_long_liq / max(total_short_liq, 1), 2),
            "interpretation": (
                "More long liquidations = market was overleveraged long, spot may be finding support. "
                "More short liquidations = shorts being squeezed, momentum may continue up."
            ),
            "source": "coinglass",
        }

    except Exception as e:
        return _liquidation_fallback(coin, error=str(e))


def _liquidation_fallback(coin: str, error: str = None) -> dict:
    return {
        "coin": coin,
        "note": "Coinglass data unavailable — use funding rate and OI as proxy",
        "error": error,
        "source": "fallback",
    }

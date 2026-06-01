"""
Market Regime Classifier — detects current NQ market regime from OHLCV.

Regimes:
  trending_bull  — ADX > 25, price above SMA, expanding ATR
  trending_bear  — ADX > 25, price below SMA, expanding ATR
  ranging        — ADX ≤ 20, ATR contracting or flat
  high_vol       — ATR > 1.5× 20d average (event-driven / spike)
  transitioning  — ADX 20-25, mixed signals

Each regime maps to a recommended strategy set and parameter adjustments
(confidence threshold delta, ATR multiplier adjustment).
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_MD   = Path(__file__).resolve().parents[1]
_ROOT = _MD.parent
for p in (str(_MD), str(_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

# ─── Regime definitions ────────────────────────────────────────────────────────

REGIME_META: dict[str, dict] = {
    "trending_bull": {
        "icon":           "🟢",
        "label":          "Trending Bullish",
        "strategies":     ["ICT/SMC momentum", "Order Block continuation", "BOS follow-through"],
        "conf_adj":       +5,    # lower threshold needed — momentum helps
        "atr_mult_adj":   0.0,
        "size_adj":       1.0,
        "description":    "Strong uptrend. Favour long continuation setups at pullbacks to OBs/FVGs.",
    },
    "trending_bear": {
        "icon":           "🔴",
        "label":          "Trending Bearish",
        "strategies":     ["ICT/SMC short momentum", "OB short continuation", "MSS follow-through"],
        "conf_adj":       +5,
        "atr_mult_adj":   0.0,
        "size_adj":       1.0,
        "description":    "Strong downtrend. Favour short continuation. Watch for dead-cat bounces.",
    },
    "ranging": {
        "icon":           "🟡",
        "label":          "Ranging / Choppy",
        "strategies":     ["VWAP mean reversion", "EQH/EQL fade", "Range boundary scalp"],
        "conf_adj":       -5,    # need higher confidence to trade in chop
        "atr_mult_adj":   -0.2,  # slightly tighter stops (range stays contained)
        "size_adj":       0.75,
        "description":    "Price range-bound. Prefer VWAP reversion at extremes. Avoid breakout fakes.",
    },
    "high_vol": {
        "icon":           "🟠",
        "label":          "High Volatility",
        "strategies":     ["Highest-confidence only (≥70)", "Wider stops, reduced size"],
        "conf_adj":       -10,   # need much higher confidence
        "atr_mult_adj":   +0.5,  # widen stops to survive volatility
        "size_adj":       0.5,
        "description":    "ATR spike — event-driven move or news shock. Reduce size, widen stops, high threshold only.",
    },
    "transitioning": {
        "icon":           "⚪",
        "label":          "Transitioning",
        "strategies":     ["Default — use standard parameters"],
        "conf_adj":       0,
        "atr_mult_adj":   0.0,
        "size_adj":       1.0,
        "description":    "Mixed signals. Use default parameters and wait for clearer structure.",
    },
}


# ─── ADX calculation ──────────────────────────────────────────────────────────

def _calc_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Compute ADX (Average Directional Index) from OHLCV.
    Returns a Series aligned with df.index.
    """
    if len(df) < period * 2:
        return pd.Series(np.nan, index=df.index)

    high  = df["high"].values
    low   = df["low"].values
    close = df["close"].values
    n     = len(df)

    dm_plus  = np.zeros(n)
    dm_minus = np.zeros(n)
    tr_arr   = np.zeros(n)

    for i in range(1, n):
        h_diff = high[i]  - high[i - 1]
        l_diff = low[i - 1] - low[i]
        dm_plus[i]  = max(h_diff, 0) if h_diff > l_diff else 0
        dm_minus[i] = max(l_diff, 0) if l_diff > h_diff else 0
        tr_arr[i]   = max(
            high[i] - low[i],
            abs(high[i]  - close[i - 1]),
            abs(low[i] - close[i - 1]),
        )

    # Wilder smoothing
    def _wilder_smooth(arr: np.ndarray, p: int) -> np.ndarray:
        s = np.zeros(n)
        s[p] = arr[1:p + 1].sum()
        for i in range(p + 1, n):
            s[i] = s[i - 1] - s[i - 1] / p + arr[i]
        return s

    tr_s   = _wilder_smooth(tr_arr,   period)
    dmp_s  = _wilder_smooth(dm_plus,  period)
    dmm_s  = _wilder_smooth(dm_minus, period)

    # DI+ / DI-
    with np.errstate(divide="ignore", invalid="ignore"):
        di_plus  = np.where(tr_s > 0, dmp_s / tr_s * 100, 0)
        di_minus = np.where(tr_s > 0, dmm_s / tr_s * 100, 0)
        dx       = np.where((di_plus + di_minus) > 0,
                            np.abs(di_plus - di_minus) / (di_plus + di_minus) * 100, 0)

    adx = pd.Series(dx, index=df.index).rolling(period, min_periods=period // 2).mean()
    return adx.round(2)


# ─── Main classifier ──────────────────────────────────────────────────────────

def classify_regime(
    df_daily: pd.DataFrame,
    adx_period: int = 14,
    atr_period: int = 14,
    atr_vol_window: int = 20,
) -> dict:
    """
    Classify the current market regime from daily OHLCV.

    Returns:
        {
          "regime":       str,
          "label":        str,
          "icon":         str,
          "confidence":   int (0-100),
          "adx":          float,
          "atr_ratio":    float,   — current ATR / 20d avg ATR
          "trend_dir":    str,     — "bull" | "bear" | "flat"
          "conf_adj":     int,     — add to min_confidence threshold
          "atr_mult_adj": float,   — add to ATR multiplier
          "size_adj":     float,   — multiply position size by this
          "strategies":   list,
          "description":  str,
          "signals":      dict,    — raw indicator values for debugging
        }
    """
    if df_daily.empty or len(df_daily) < max(adx_period * 2, atr_vol_window + atr_period):
        return _regime_result("transitioning", 0, {})

    df = df_daily.copy().sort_index().tail(120)

    # ── ADX ───────────────────────────────────────────────────────────────────
    adx_series = _calc_adx(df, adx_period)
    adx        = float(adx_series.iloc[-1]) if not adx_series.empty else 20.0
    if np.isnan(adx):
        adx = 20.0

    # ── ATR ratio ─────────────────────────────────────────────────────────────
    h, l, c = df["high"], df["low"], df["close"]
    tr = pd.concat([
        h - l,
        (h - c.shift(1)).abs(),
        (l - c.shift(1)).abs(),
    ], axis=1).max(axis=1)

    atr_current = float(tr.rolling(atr_period, min_periods=atr_period // 2).mean().iloc[-1])
    atr_avg20   = float(tr.rolling(atr_vol_window, min_periods=atr_vol_window // 2).mean().iloc[-1])
    atr_ratio   = round(atr_current / atr_avg20, 2) if atr_avg20 > 0 else 1.0

    # ── Trend direction (SMA5 vs SMA20 vs SMA50) ─────────────────────────────
    close   = df["close"]
    sma5    = float(close.tail(5).mean())
    sma20   = float(close.tail(20).mean())
    sma50   = float(close.tail(50).mean()) if len(close) >= 50 else sma20
    last_c  = float(close.iloc[-1])

    bull_signals = sum([sma5 > sma20, sma20 > sma50, last_c > sma20])
    bear_signals = sum([sma5 < sma20, sma20 < sma50, last_c < sma20])
    trend_dir = "bull" if bull_signals >= 2 else "bear" if bear_signals >= 2 else "flat"

    # ── SMA slope (momentum) ─────────────────────────────────────────────────
    sma20_series = close.rolling(20, min_periods=10).mean()
    if len(sma20_series.dropna()) >= 5:
        sma20_slope = (float(sma20_series.iloc[-1]) - float(sma20_series.iloc[-5])) / float(sma20_series.iloc[-5]) * 100
    else:
        sma20_slope = 0.0

    signals = {
        "adx":         round(adx, 1),
        "atr_ratio":   atr_ratio,
        "trend_dir":   trend_dir,
        "bull_signals": bull_signals,
        "bear_signals": bear_signals,
        "sma20_slope_pct": round(sma20_slope, 3),
    }

    # ── Classification logic ──────────────────────────────────────────────────

    # High volatility takes priority
    if atr_ratio >= 1.5:
        conf = min(95, int(60 + (atr_ratio - 1.5) * 40))
        return _regime_result("high_vol", conf, signals)

    # Trending (ADX > 25 + directional SMA)
    if adx >= 25 and trend_dir != "flat":
        regime = "trending_bull" if trend_dir == "bull" else "trending_bear"
        conf   = min(95, int(60 + (adx - 25) * 1.5 + abs(sma20_slope) * 50))
        return _regime_result(regime, conf, signals)

    # Ranging (ADX < 20 + ATR contracting)
    if adx < 20 and atr_ratio <= 1.1:
        conf = min(90, int(60 + (20 - adx) * 1.5 + (1.1 - atr_ratio) * 30))
        return _regime_result("ranging", conf, signals)

    # Default: transitioning
    conf = 40
    return _regime_result("transitioning", conf, signals)


def _regime_result(regime: str, confidence: int, signals: dict) -> dict:
    meta = REGIME_META.get(regime, REGIME_META["transitioning"])
    return {
        "regime":       regime,
        "label":        meta["label"],
        "icon":         meta["icon"],
        "confidence":   confidence,
        "conf_adj":     meta["conf_adj"],
        "atr_mult_adj": meta["atr_mult_adj"],
        "size_adj":     meta["size_adj"],
        "strategies":   meta["strategies"],
        "description":  meta["description"],
        "signals":      signals,
        "adx":          signals.get("adx", 0),
        "atr_ratio":    signals.get("atr_ratio", 1.0),
        "trend_dir":    signals.get("trend_dir", "flat"),
    }


# ─── Historical regime series ─────────────────────────────────────────────────

def get_regime_history(df_daily: pd.DataFrame, lookback: int = 60) -> pd.DataFrame:
    """
    Compute daily regime for the last N trading days.
    Returns DataFrame with columns: date, regime, adx, atr_ratio, trend_dir.
    Useful for the Backtesting page and Analytics.
    """
    if df_daily.empty or len(df_daily) < 50:
        return pd.DataFrame()

    df = df_daily.sort_index().tail(lookback + 50)  # extra for lookback
    rows = []
    min_idx = 50

    for i in range(min_idx, len(df)):
        window = df.iloc[:i + 1]
        r = classify_regime(window)
        date_val = df.index[i]
        rows.append({
            "date":      str(date_val)[:10] if hasattr(date_val, "__str__") else str(date_val),
            "regime":    r["regime"],
            "adx":       r["adx"],
            "atr_ratio": r["atr_ratio"],
            "trend_dir": r["trend_dir"],
        })

    return pd.DataFrame(rows).tail(lookback).reset_index(drop=True)

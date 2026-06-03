"""
Signal engine — Multi-Timeframe confluence scoring.
Technical: 75 pts  |  Fundamental: 25 pts
Entry fires when direction score ≥ 55 pts AND leads opponent by ≥ 15 pts.
Regime-aware: confidence threshold adjusted by Market Regime Classifier.
Blackout-aware: returns NO_SIGNAL during high-impact economic events.
"""
from __future__ import annotations

import sys
from pathlib import Path

_MD   = Path(__file__).resolve().parents[1]
_ROOT = _MD.parent

for p in (str(_MD), str(_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

import pandas as pd
import numpy as np

from modules.nq_calculations import (
    cumulative_delta,
    calculate_key_levels,
    detect_stacked_imbalances,
    get_htf_bias,
    get_support_resistance_zones,
    calculate_dynamic_stop,
    calculate_scalp_stop,
    find_eqh_eql,
    find_swing_liquidity,
)
from modules.nq_data import (
    get_todays_nq_data,
    get_nq_1m,
    load_nq_daily_cache,
    load_nq_hourly_cache,
    fetch_nq_15m,
    current_session_israel,
)
from modules.economic_calendar import is_blackout_zone, get_todays_events
from modules.regime_detector import classify_regime

_CONFIG = _ROOT / "scores_news" / "config"
_ML_DIR = _ROOT / "scores_news" / "ml_model"

# ─── helpers ──────────────────────────────────────────────────────────────────

def _load_score_log() -> pd.DataFrame:
    path = _CONFIG / "score_log.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df.dropna(subset=["date"]).sort_values("date")


def _load_ml_prediction() -> int:
    # Guard: model is unreliable when trained on < 30 samples
    merged = _ML_DIR / "merged_scores_mes.csv"
    if merged.exists():
        try:
            if sum(1 for _ in open(merged)) - 1 < 30:
                return 0
        except Exception:
            return 0

    log = _ML_DIR / "ml_performance_log.csv"
    if not log.exists():
        return 0
    try:
        df = pd.read_csv(log)
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        today = pd.Timestamp.now().normalize()
        row = df[df["date"] >= today]
        if row.empty:
            row = df.tail(1)
        return int(row.iloc[-1]["prediction"])
    except Exception:
        return 0


def _latest_final_score() -> int:
    df = _load_score_log()
    if df.empty:
        return 50
    return int(df.iloc[-1].get("final_score", 50))


def _delta_direction(df_5m: pd.DataFrame, lookback: int = 12) -> int:
    """+1 if cumulative delta rising, -1 falling, 0 flat."""
    if df_5m.empty or len(df_5m) < lookback:
        return 0
    try:
        cd = cumulative_delta(df_5m)
        recent = cd.tail(lookback)
        slope = recent.iloc[-1] - recent.iloc[0]
        return 1 if slope > 0 else (-1 if slope < 0 else 0)
    except Exception:
        return 0


def _session_quality(session: dict) -> int:
    return {"high": 100, "medium": 60, "low": 20}.get(session.get("risk", "low"), 20)


def _find_zone_target(
    zones: list,
    entry: float,
    direction: str,
    stop: float,
    min_rr: float = 1.5,
) -> float | None:
    """Return the midpoint of the nearest zone that gives ≥ min_rr, else None."""
    if not zones or stop is None or np.isnan(stop):
        return None
    risk = abs(entry - stop)
    if risk <= 0:
        return None
    if direction == "LONG":
        candidates = sorted(
            [z for z in zones if z["midpoint"] >= entry + risk * min_rr],
            key=lambda z: z["midpoint"],
        )
    else:
        candidates = sorted(
            [z for z in zones if z["midpoint"] <= entry - risk * min_rr],
            key=lambda z: z["midpoint"], reverse=True,
        )
    return float(candidates[0]["midpoint"]) if candidates else None


# ─── main signal ──────────────────────────────────────────────────────────────

def generate_signal() -> dict:
    """
    Returns signal dict with keys:
      direction, confidence, entry, stop, target, partial_exit, be_level, rr,
      factors, session, final_score, ml_prediction, alert,
      levels, zones, htf_bias, htf_strength, current_price, delta_dir,
      nearest_zone_score, long_pts, short_pts, session_quality
    """
    # ── data loading ────────────────────────────────────────────
    df_5m     = get_todays_nq_data()
    df_1m     = get_nq_1m()                     # finest scalp execution TF
    df_daily  = load_nq_daily_cache()
    df_hourly = load_nq_hourly_cache()
    df_15m    = fetch_nq_15m(days_back=5)
    session   = current_session_israel()
    final_score = _latest_final_score()
    ml_pred     = _load_ml_prediction()

    # ── Economic Calendar Blackout ──────────────────────────────
    blackout = {}
    try:
        blackout = is_blackout_zone()
    except Exception:
        blackout = {"blackout": False, "reason": "calendar unavailable"}

    # ── Market Regime ───────────────────────────────────────────
    regime = {}
    try:
        regime = classify_regime(df_daily)
    except Exception:
        regime = {"regime": "transitioning", "conf_adj": 0, "atr_mult_adj": 0.0,
                  "size_adj": 1.0, "label": "Unknown", "icon": "⚪"}

    current_price: float = np.nan
    if not df_5m.empty:
        current_price = float(df_5m["close"].iloc[-1])
    elif not df_daily.empty:
        current_price = float(df_daily["close"].iloc[-1])

    # ── HTF bias (Daily + 1H) ───────────────────────────────────
    htf = get_htf_bias(df_daily, df_hourly)
    htf_bias = htf["bias"]  # 'bullish' | 'bearish' | 'neutral'

    # ── Key levels + Zone analysis ──────────────────────────────
    levels: dict = {}
    try:
        levels = calculate_key_levels(df_daily, df_hourly)
    except Exception:
        pass

    zones: list = []
    try:
        zones = get_support_resistance_zones(
            df_daily, df_hourly, df_15m,
            current_price=None if np.isnan(current_price) else current_price,
        )
    except Exception:
        pass

    # Nearest zone within 0.3 % of current price
    nearest_zone_score = 0
    nearest_zone_type  = "none"
    near_zone          = None
    if zones and not np.isnan(current_price):
        close_zones = [z for z in zones if z["dist_pts"] < current_price * 0.003]
        if close_zones:
            near_zone = max(close_zones, key=lambda z: z["score"])
            nearest_zone_score = near_zone["score"]
            nearest_zone_type  = near_zone["zone_type"]

    # ── Confluence scoring (100 pts total: 75 tech + 25 fund) ──
    long_pts:  int = 0
    short_pts: int = 0
    factors:   list = []

    # ── TECHNICAL — 75 pts ──────────────────────────────────────

    # T1. HTF Bias — 20 pts
    if htf_bias == "bullish":
        long_pts += 20
        factors.append({"factor": "HTF Bias (Daily + 1H)", "value": f"Bullish {htf['strength']}%", "side": "LONG",    "pts": 20})
    elif htf_bias == "bearish":
        short_pts += 20
        factors.append({"factor": "HTF Bias (Daily + 1H)", "value": f"Bearish {htf['strength']}%", "side": "SHORT",   "pts": 20})
    else:
        factors.append({"factor": "HTF Bias (Daily + 1H)", "value": f"Neutral {htf['strength']}%",  "side": "NEUTRAL", "pts": 0})

    # T2. MTF Zone Score — up to 20 pts  (zone_score / 10 × 20)
    zone_pts = int(nearest_zone_score / 10 * 20) if nearest_zone_score > 0 else 0
    if near_zone and zone_pts > 0:
        if nearest_zone_type == "support":
            long_pts += zone_pts
            factors.append({"factor": "MTF Zone (15m OB/FVG/KL)", "value": f"Support score {nearest_zone_score}/10", "side": "LONG",  "pts": zone_pts})
        else:
            short_pts += zone_pts
            factors.append({"factor": "MTF Zone (15m OB/FVG/KL)", "value": f"Resistance score {nearest_zone_score}/10", "side": "SHORT", "pts": zone_pts})
    else:
        comps_str = ", ".join(near_zone["components"][:3]) if near_zone else "—"
        factors.append({"factor": "MTF Zone (15m OB/FVG/KL)", "value": f"No zone nearby ({comps_str})", "side": "NEUTRAL", "pts": 0})

    # T3. Order Flow Delta (5m) — 15 pts
    delta_dir = _delta_direction(df_5m)
    delta_label = {1: "Bullish", -1: "Bearish", 0: "Neutral"}.get(delta_dir, "Neutral")
    if delta_dir == 1:
        long_pts  += 15
        factors.append({"factor": "Order Flow Delta (5m)", "value": delta_label, "side": "LONG",    "pts": 15})
    elif delta_dir == -1:
        short_pts += 15
        factors.append({"factor": "Order Flow Delta (5m)", "value": delta_label, "side": "SHORT",   "pts": 15})
    else:
        factors.append({"factor": "Order Flow Delta (5m)", "value": delta_label, "side": "NEUTRAL", "pts": 0})

    # T4. Session Quality — 10 pts (both directions — acts as entry gate)
    sq = _session_quality(session)
    if sq >= 70:
        long_pts  += 10
        short_pts += 10
        factors.append({"factor": "Session Quality", "value": session.get("name", "?"), "side": "BOTH", "pts": 10})
    elif sq >= 50:
        long_pts  += 5
        short_pts += 5
        factors.append({"factor": "Session Quality", "value": session.get("name", "?"), "side": "BOTH", "pts": 5})
    else:
        factors.append({"factor": "Session Quality", "value": session.get("name", "?"), "side": "LOW",  "pts": 0})

    # T5. Stacked Imbalances (5m) — 10 pts
    stacked_label = "None"
    if not df_5m.empty and len(df_5m) >= 6:
        try:
            stacks = detect_stacked_imbalances(df_5m.tail(30).reset_index(drop=True), min_stack=3)
            if stacks:
                last = stacks[-1]
                stacked_label = f"{last['direction'].capitalize()} ({last['candle_count']}c)"
                if last["direction"] == "bullish":
                    long_pts  += 10
                    factors.append({"factor": "Stacked Imbalance (5m)", "value": stacked_label, "side": "LONG",  "pts": 10})
                else:
                    short_pts += 10
                    factors.append({"factor": "Stacked Imbalance (5m)", "value": stacked_label, "side": "SHORT", "pts": 10})
            else:
                factors.append({"factor": "Stacked Imbalance (5m)", "value": "None", "side": "NEUTRAL", "pts": 0})
        except Exception:
            factors.append({"factor": "Stacked Imbalance (5m)", "value": "Error", "side": "NEUTRAL", "pts": 0})
    else:
        factors.append({"factor": "Stacked Imbalance (5m)", "value": "No data", "side": "NEUTRAL", "pts": 0})

    # ── FUNDAMENTAL — 25 pts ────────────────────────────────────

    # F1. Sentiment / Final Score — 15 pts (graded)
    if final_score >= 62:
        long_pts  += 15
        factors.append({"factor": "Sentiment Score", "value": final_score, "side": "LONG",    "pts": 15})
    elif final_score <= 42:
        short_pts += 15
        factors.append({"factor": "Sentiment Score", "value": final_score, "side": "SHORT",   "pts": 15})
    elif final_score >= 55:
        long_pts  += 8
        factors.append({"factor": "Sentiment Score", "value": final_score, "side": "LONG",    "pts": 8})
    elif final_score <= 48:
        short_pts += 8
        factors.append({"factor": "Sentiment Score", "value": final_score, "side": "SHORT",   "pts": 8})
    else:
        factors.append({"factor": "Sentiment Score", "value": final_score, "side": "NEUTRAL", "pts": 0})

    # F2. ML Prediction — 10 pts
    ml_label = {1: "LONG", -1: "SHORT", 0: "NEUTRAL"}.get(ml_pred, "NEUTRAL")
    if ml_pred == 1:
        long_pts  += 10
        factors.append({"factor": "ML Prediction", "value": ml_label, "side": "LONG",    "pts": 10})
    elif ml_pred == -1:
        short_pts += 10
        factors.append({"factor": "ML Prediction", "value": ml_label, "side": "SHORT",   "pts": 10})
    else:
        factors.append({"factor": "ML Prediction", "value": ml_label, "side": "NEUTRAL", "pts": 0})

    # ── Direction decision (regime-adjusted thresholds) ────────
    # Max pts: Technical 75 + Fundamental 25 = 100
    # Session (BOTH) inflates both sides equally — net margin stays clean
    _MIN_PTS    = 55 - regime.get("conf_adj", 0)   # regime lowers/raises bar
    _MIN_MARGIN = 15   # must lead opponent by at least 15 pts

    # ── Economic Calendar Blackout override ─────────────────────
    if blackout.get("blackout"):
        direction  = "NEUTRAL"
        confidence = 0
    elif long_pts >= _MIN_PTS and (long_pts - short_pts) >= _MIN_MARGIN:
        direction  = "LONG"
        confidence = min(100, round(long_pts / max(long_pts + short_pts, 1) * 100))
    elif short_pts >= _MIN_PTS and (short_pts - long_pts) >= _MIN_MARGIN:
        direction  = "SHORT"
        confidence = min(100, round(short_pts / max(long_pts + short_pts, 1) * 100))
    else:
        direction  = "NEUTRAL"
        confidence = 0

    # ── Risk management — scalping: LTF ATR stop ──────────────
    # Prefer 1m ATR for tightest stop; fall back to 5m then daily.
    entry:        float = current_price
    stop:         float = np.nan
    target:       float = np.nan
    partial_exit: float = np.nan
    be_level:     float = np.nan
    rr:           float = np.nan
    atr_ltf:      float = np.nan

    if direction != "NEUTRAL" and not np.isnan(entry):
        # Try 1m data first (finest scalp precision)
        ltf_df   = df_1m if not df_1m.empty and len(df_1m) >= 15 else df_5m
        atr_mult = 1.0 + regime.get("atr_mult_adj", 0.0)
        scalp    = calculate_scalp_stop(ltf_df, direction, entry,
                                        atr_period=14, atr_mult=atr_mult)
        if scalp:
            stop         = scalp["stop"]
            partial_exit = scalp["partial_exit"]
            atr_ltf      = scalp["atr_ltf"]
            risk         = abs(entry - stop)
        else:
            # Fallback to daily ATR (rare: no intraday data available)
            stop = calculate_dynamic_stop(df_daily, direction, entry, atr_period=14, atr_mult=1.5)
            risk = abs(entry - stop)

        # Target: nearest HTF zone at ≥1.5 RR, else 2× risk
        tgt = _find_zone_target(zones, entry, direction, stop=stop, min_rr=1.5)
        if tgt is not None:
            target = float(tgt)
        else:
            target = (entry + risk * 2.0) if direction == "LONG" else (entry - risk * 2.0)

        reward = abs(target - entry)
        if risk > 0:
            rr = round(reward / risk, 2)

        if np.isnan(partial_exit):
            partial_exit = (entry + risk) if direction == "LONG" else (entry - risk)
        be_level = entry

    # ── Liquidity analysis (HTF + LTF) ────────────────────────
    eqh_eql   = find_eqh_eql(df_15m,  lookback=60, tolerance_pts=6.0)
    liq_15m   = find_swing_liquidity(df_15m,  lookback=40)
    liq_1h    = find_swing_liquidity(df_hourly, lookback=30)
    liquidity = {
        "eqh":       eqh_eql.get("eqh", []),
        "eql":       eqh_eql.get("eql", []),
        "buy_side":  sorted(set(liq_15m["buy_side"]  + liq_1h["buy_side"]))[:6],
        "sell_side": sorted(set(liq_15m["sell_side"] + liq_1h["sell_side"]), reverse=True)[:6],
    }

    # ── Alert ───────────────────────────────────────────────────
    direction_pts = long_pts if direction == "LONG" else short_pts
    alert = (
        direction != "NEUTRAL"
        and not blackout.get("blackout", False)
        and direction_pts >= 60
        and nearest_zone_score >= 5
        and not np.isnan(rr)
        and rr >= 1.5
        and sq >= 50
    )

    return {
        "direction":          direction,
        "confidence":         confidence,
        "entry":              round(entry, 2)        if not np.isnan(entry)        else None,
        "stop":               round(stop, 2)         if not np.isnan(stop)         else None,
        "target":             round(target, 2)       if not np.isnan(target)       else None,
        "partial_exit":       round(partial_exit, 2) if not np.isnan(partial_exit) else None,
        "be_level":           round(be_level, 2)     if not np.isnan(be_level)     else None,
        "rr":                 rr                     if not np.isnan(rr)           else None,
        "atr_ltf":            round(atr_ltf, 2)      if not np.isnan(atr_ltf)      else None,
        "liquidity":          liquidity,
        "factors":            factors,
        "session":            session,
        "final_score":        final_score,
        "ml_prediction":      ml_label,
        "alert":              alert,
        "levels":             {k: round(v, 2) for k, v in levels.items()
                               if isinstance(v, float) and not np.isnan(v)},
        "zones":              zones,
        "htf_bias":           htf_bias,
        "htf_strength":       htf["strength"],
        "current_price":      round(entry, 2) if not np.isnan(entry) else None,
        "delta_dir":          delta_label,
        "nearest_zone_score": nearest_zone_score,
        "long_pts":           long_pts,
        "short_pts":          short_pts,
        "session_quality":    sq,
        "blackout":           blackout.get("blackout", False),
        "blackout_reason":    blackout.get("reason", ""),
        "regime":             regime.get("regime", "transitioning"),
        "regime_label":       regime.get("label", "—"),
        "regime_icon":        regime.get("icon", "⚪"),
        "regime_strategies":  regime.get("strategies", []),
        "regime_size_adj":    regime.get("size_adj", 1.0),
        "regime_conf_adj":    regime.get("conf_adj", 0),
    }

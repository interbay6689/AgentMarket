"""Signal engine: combines all data sources into a trade signal with R:R."""
from __future__ import annotations

import sys
from pathlib import Path

_MD = Path(__file__).resolve().parents[1]   # market_dashboard/
_ROOT = _MD.parent                          # AgentMarket/

for p in (str(_MD), str(_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

import pandas as pd
import numpy as np
import json
import joblib

from modules.nq_calculations import (
    cumulative_delta,
    approximate_delta,
    calculate_key_levels,
)
from modules.nq_data import (
    get_todays_nq_data,
    load_nq_daily_cache,
    load_nq_hourly_cache,
    current_session_israel,
)

_CONFIG = _ROOT / "scores_news" / "config"
_ML_DIR = _ROOT / "scores_news" / "ml_model"


def _load_score_log() -> pd.DataFrame:
    path = _CONFIG / "score_log.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df.dropna(subset=["date"]).sort_values("date")


def _load_ml_prediction() -> int:
    """Load today's ML prediction from performance log. Returns 0 if unavailable."""
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
    row = df.iloc[-1]
    return int(row.get("final_score", 50))


def _delta_direction(df_5m: pd.DataFrame, lookback: int = 12) -> int:
    """Returns +1 if recent cumulative delta is rising, -1 if falling, 0 neutral."""
    if df_5m.empty or len(df_5m) < lookback:
        return 0
    try:
        cd = cumulative_delta(df_5m)
        recent = cd.tail(lookback)
        slope = recent.iloc[-1] - recent.iloc[0]
        if slope > 0:
            return 1
        if slope < 0:
            return -1
    except Exception:
        pass
    return 0


def _session_quality(session: dict) -> int:
    risk = session.get("risk", "low")
    return {"high": 100, "medium": 60, "low": 20}.get(risk, 20)


def _nearest_levels(levels: dict, price: float, direction: int) -> tuple[float | None, float | None]:
    """Return (stop_level, target_level) for the given direction."""
    vals = [v for v in levels.values() if isinstance(v, (int, float)) and not np.isnan(v)]
    if not vals:
        return None, None
    vals_arr = np.array(sorted(vals))

    if direction == 1:  # LONG
        below = vals_arr[vals_arr < price]
        above = vals_arr[vals_arr > price]
        stop_lvl = float(below[-1]) if len(below) else None
        tgt_lvl = float(above[0]) if len(above) else None
    else:  # SHORT
        below = vals_arr[vals_arr < price]
        above = vals_arr[vals_arr > price]
        stop_lvl = float(above[0]) if len(above) else None
        tgt_lvl = float(below[-1]) if len(below) else None

    return stop_lvl, tgt_lvl


def _near_key_level(levels: dict, price: float, threshold_pct: float = 0.25) -> tuple[bool, str]:
    """Returns (is_near, 'support'|'resistance'|'none')."""
    for name, lvl in levels.items():
        if not isinstance(lvl, (int, float)) or np.isnan(lvl):
            continue
        dist_pct = abs(price - lvl) / price * 100
        if dist_pct <= threshold_pct:
            return True, "support" if price >= lvl else "resistance"
    return False, "none"


def generate_signal() -> dict:
    """
    Returns a signal dict with keys:
      direction, confidence, entry, stop, target, rr,
      factors, session, final_score, ml_prediction,
      alert (bool), levels, current_price, delta_dir
    """
    # ── load data ──────────────────────────────────────────────
    df_5m = get_todays_nq_data()
    df_daily = load_nq_daily_cache()
    df_hourly = load_nq_hourly_cache()
    session = current_session_israel()
    final_score = _latest_final_score()
    ml_pred = _load_ml_prediction()

    current_price: float = np.nan
    if not df_5m.empty:
        current_price = float(df_5m["close"].iloc[-1])
    elif not df_daily.empty:
        current_price = float(df_daily["close"].iloc[-1])

    # ── confluence scoring ────────────────────────────────────
    long_pts = 0
    short_pts = 0
    factors: list[dict] = []

    # 1. Final score (25 pts)
    if final_score >= 62:
        long_pts += 25
        factors.append({"factor": "Sentiment Score", "value": final_score, "side": "LONG", "pts": 25})
    elif final_score <= 42:
        short_pts += 25
        factors.append({"factor": "Sentiment Score", "value": final_score, "side": "SHORT", "pts": 25})
    else:
        factors.append({"factor": "Sentiment Score", "value": final_score, "side": "NEUTRAL", "pts": 0})

    # 2. ML prediction (30 pts)
    ml_label = {1: "LONG", -1: "SHORT", 0: "NEUTRAL"}.get(ml_pred, "NEUTRAL")
    if ml_pred == 1:
        long_pts += 30
        factors.append({"factor": "ML Prediction", "value": ml_label, "side": "LONG", "pts": 30})
    elif ml_pred == -1:
        short_pts += 30
        factors.append({"factor": "ML Prediction", "value": ml_label, "side": "SHORT", "pts": 30})
    else:
        factors.append({"factor": "ML Prediction", "value": ml_label, "side": "NEUTRAL", "pts": 0})

    # 3. Delta direction (20 pts)
    delta_dir = _delta_direction(df_5m)
    delta_label = {1: "Bullish", -1: "Bearish", 0: "Neutral"}.get(delta_dir, "Neutral")
    if delta_dir == 1:
        long_pts += 20
        factors.append({"factor": "Order Flow Delta", "value": delta_label, "side": "LONG", "pts": 20})
    elif delta_dir == -1:
        short_pts += 20
        factors.append({"factor": "Order Flow Delta", "value": delta_label, "side": "SHORT", "pts": 20})
    else:
        factors.append({"factor": "Order Flow Delta", "value": delta_label, "side": "NEUTRAL", "pts": 0})

    # 4. Session quality (15 pts)
    sq = _session_quality(session)
    if sq >= 70:
        long_pts += 8
        short_pts += 8
        factors.append({"factor": "Session Quality", "value": session.get("name", "?"), "side": "BOTH", "pts": 8})
    elif sq >= 50:
        long_pts += 4
        short_pts += 4
        factors.append({"factor": "Session Quality", "value": session.get("name", "?"), "side": "BOTH", "pts": 4})
    else:
        factors.append({"factor": "Session Quality", "value": session.get("name", "?"), "side": "LOW", "pts": 0})

    # 5. Key level proximity (10 pts)
    levels = {}
    try:
        levels = calculate_key_levels(df_daily, df_hourly)
    except Exception:
        pass

    near, near_type = (False, "none") if np.isnan(current_price) else _near_key_level(levels, current_price)
    if near:
        if near_type == "support":
            long_pts += 10
            factors.append({"factor": "Key Level", "value": "Near Support", "side": "LONG", "pts": 10})
        else:
            short_pts += 10
            factors.append({"factor": "Key Level", "value": "Near Resistance", "side": "SHORT", "pts": 10})
    else:
        factors.append({"factor": "Key Level", "value": "Not Near Level", "side": "NEUTRAL", "pts": 0})

    # ── determine direction ───────────────────────────────────
    total_pts = long_pts + short_pts
    if long_pts > short_pts and long_pts >= 30:
        direction = "LONG"
        confidence = min(100, round(long_pts / max(total_pts, 1) * 100))
    elif short_pts > long_pts and short_pts >= 30:
        direction = "SHORT"
        confidence = min(100, round(short_pts / max(total_pts, 1) * 100))
    else:
        direction = "NEUTRAL"
        confidence = 0

    # ── R:R calculation ────────────────────────────────────────
    entry = current_price
    stop: float = np.nan
    target: float = np.nan
    rr: float = np.nan
    buf = 8.0  # NQ points buffer on stop

    if direction != "NEUTRAL" and not np.isnan(entry):
        dir_int = 1 if direction == "LONG" else -1
        stop_lvl, tgt_lvl = _nearest_levels(levels, entry, dir_int)

        if stop_lvl is not None and tgt_lvl is not None:
            if direction == "LONG":
                stop = stop_lvl - buf
                target = tgt_lvl
            else:
                stop = stop_lvl + buf
                target = tgt_lvl

            risk = abs(entry - stop)
            reward = abs(target - entry)
            if risk > 0:
                rr = round(reward / risk, 2)
            # fallback if rr < 1.5: extend target
            if not np.isnan(rr) and rr < 1.5:
                target = entry + 1.5 * risk if direction == "LONG" else entry - 1.5 * risk
                rr = 1.5
        else:
            # No key levels — use ATR fallback
            if not df_daily.empty and len(df_daily) >= 5:
                atr = (df_daily["high"] - df_daily["low"]).tail(5).mean()
                risk = atr * 0.5
                stop = entry - risk if direction == "LONG" else entry + risk
                target = entry + risk * 1.5 if direction == "LONG" else entry - risk * 1.5
                rr = 1.5

    # ── alert condition ────────────────────────────────────────
    alert = (
        direction != "NEUTRAL"
        and confidence >= 60
        and not np.isnan(rr)
        and rr >= 1.5
        and sq >= 50
    )

    return {
        "direction": direction,
        "confidence": confidence,
        "entry": round(entry, 2) if not np.isnan(entry) else None,
        "stop": round(stop, 2) if not np.isnan(stop) else None,
        "target": round(target, 2) if not np.isnan(target) else None,
        "rr": rr if not np.isnan(rr) else None,
        "factors": factors,
        "session": session,
        "final_score": final_score,
        "ml_prediction": ml_label,
        "alert": alert,
        "levels": {k: round(v, 2) for k, v in levels.items() if isinstance(v, float) and not np.isnan(v)},
        "current_price": round(entry, 2) if not np.isnan(entry) else None,
        "delta_dir": delta_label,
        "long_pts": long_pts,
        "short_pts": short_pts,
        "session_quality": sq,
    }

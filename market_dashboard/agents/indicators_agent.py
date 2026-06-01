"""
Indicators Agent — runs every 5 minutes, computes all technical indicators
and writes a unified indicator state to logs/indicator_state.json.
Other agents consume this file instead of re-computing from raw data.
"""
from __future__ import annotations

import json
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytz

_MD   = Path(__file__).resolve().parents[1]
_ROOT = _MD.parent
for p in (str(_MD), str(_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from modules.nq_data import (
    get_todays_nq_data, load_nq_daily_cache, load_nq_hourly_cache,
    fetch_nq_15m, current_session_israel,
)
from modules.nq_calculations import (
    calculate_key_levels, cumulative_delta, approximate_delta,
    detect_fvg, detect_order_blocks, detect_stacked_imbalances,
    get_htf_bias, get_support_resistance_zones,
    calculate_atr_range_consumed, calculate_opening_range,
    classify_premium_discount, vwap_position, detect_eqh_eql,
    calculate_ote_zone, day_of_week_bias, gap_fill_stats,
    delta_zscore, volume_anomaly_score,
)
from modules.economic_calendar import calendar_summary, is_blackout_zone
from modules.regime_detector import classify_regime

IL_TZ  = pytz.timezone("Asia/Jerusalem")
_LOGS  = _MD / "logs"
_STATE = _LOGS / "indicator_state.json"
_STATUS_PATH = _LOGS / "agent_status.json"

AGENT_NAME = "indicators"


# ─── Status helpers ────────────────────────────────────────────────────────────

def _load_status() -> dict:
    if _STATUS_PATH.exists():
        try:
            return json.loads(_STATUS_PATH.read_text())
        except Exception:
            pass
    return {}


def _update_status(status: str, error: str | None = None) -> None:
    data = _load_status()
    entry = data.get(AGENT_NAME, {})
    entry["status"]   = status
    entry["last_run"] = datetime.now(IL_TZ).isoformat()
    entry["runs"]     = entry.get("runs", 0) + (1 if status == "done" else 0)
    if error:
        entry["last_error"] = error
    elif "last_error" in entry:
        del entry["last_error"]
    data[AGENT_NAME] = entry
    _LOGS.mkdir(parents=True, exist_ok=True)
    _STATUS_PATH.write_text(json.dumps(data, indent=2))


# ─── Computation helpers ───────────────────────────────────────────────────────

def _safe(fn, *args, default=None, **kwargs) -> Any:
    try:
        return fn(*args, **kwargs)
    except Exception:
        return default


def _compute_fvgs(df_5m: pd.DataFrame, df_15m: pd.DataFrame) -> dict:
    fvg_5m  = _safe(detect_fvg, df_5m,  default=[])
    fvg_15m = _safe(detect_fvg, df_15m.tail(96), default=[])
    return {
        "fvg_5m_count":       len(fvg_5m) if isinstance(fvg_5m, (list, pd.DataFrame)) else 0,
        "fvg_15m_count":      len(fvg_15m) if isinstance(fvg_15m, (list, pd.DataFrame)) else 0,
        "nearest_fvg_5m":     _nearest_fvg(fvg_5m),
        "nearest_fvg_15m":    _nearest_fvg(fvg_15m),
    }


def _nearest_fvg(fvg_data) -> dict | None:
    if isinstance(fvg_data, pd.DataFrame) and not fvg_data.empty:
        row = fvg_data.iloc[-1]
        return {
            "type":  str(row.get("fvg_type", "?")),
            "top":   round(float(row.get("top", 0)), 2),
            "bot":   round(float(row.get("bottom", 0)), 2),
            "mid":   round(float(row.get("midpoint", (row.get("top",0)+row.get("bottom",0))/2)), 2),
        }
    if isinstance(fvg_data, list) and fvg_data:
        z = fvg_data[-1]
        return {
            "type": z.get("type", z.get("fvg_type", "?")),
            "top":  round(float(z.get("top", 0)), 2),
            "bot":  round(float(z.get("bottom", 0)), 2),
            "mid":  round(float(z.get("midpoint", 0)), 2),
        }
    return None


def _compute_order_blocks(df_5m: pd.DataFrame, df_15m: pd.DataFrame) -> dict:
    obs_5m  = _safe(detect_order_blocks, df_5m.tail(48),  default=[])
    obs_15m = _safe(detect_order_blocks, df_15m.tail(96), default=[])
    if isinstance(obs_5m, pd.DataFrame):
        obs_5m = obs_5m.to_dict(orient="records")
    if isinstance(obs_15m, pd.DataFrame):
        obs_15m = obs_15m.to_dict(orient="records")
    return {
        "ob_5m_count":   len(obs_5m),
        "ob_15m_count":  len(obs_15m),
        "nearest_ob_5m": _summarize_ob(obs_5m[-1]) if obs_5m else None,
    }


def _summarize_ob(ob: dict) -> dict:
    return {
        "type":    ob.get("ob_type", ob.get("type", "?")),
        "top":     round(float(ob.get("top", 0)), 2),
        "bot":     round(float(ob.get("bottom", ob.get("bot", 0))), 2),
        "volume":  int(ob.get("volume", 0)),
        "tested":  bool(ob.get("tested", False)),
    }


def _compute_ote_zones(df_daily: pd.DataFrame) -> dict | None:
    """Compute OTE zone based on last swing in daily data."""
    if len(df_daily) < 20:
        return None
    d = df_daily.tail(20).reset_index(drop=True)
    last_close = float(d["close"].iloc[-1])
    swing_high = float(d["high"].max())
    swing_low  = float(d["low"].min())

    # Determine likely direction from recent 5-bar momentum
    recent_change = float(d["close"].iloc[-1]) - float(d["close"].iloc[-5])
    direction = "short" if recent_change > 0 else "long"

    ote = _safe(calculate_ote_zone, swing_low, swing_high, direction)
    if ote:
        ote["current_price"] = round(last_close, 2)
        ote["in_ote_zone"]   = (ote.get("ote_lo", 0) <= last_close <= ote.get("ote_hi", 0))
    return ote


def _compute_delta_analysis(df_5m: pd.DataFrame) -> dict:
    if df_5m.empty or len(df_5m) < 5:
        return {}
    delta_s  = _safe(approximate_delta, df_5m, default=pd.Series(dtype=float))
    cum_d    = _safe(cumulative_delta, df_5m, default=pd.Series(dtype=float))
    dz       = _safe(delta_zscore, df_5m, window=20, default=pd.Series(dtype=float))
    va       = _safe(volume_anomaly_score, df_5m, default=0.0)

    result: dict = {"volume_anomaly": round(float(va), 2)}
    if not isinstance(delta_s, pd.Series) or delta_s.empty:
        return result

    result["delta_last"]   = round(float(delta_s.iloc[-1]), 1)
    result["delta_3bar"]   = round(float(delta_s.tail(3).sum()), 1)
    if isinstance(cum_d, pd.Series) and not cum_d.empty:
        result["cum_delta_today"] = round(float(cum_d.iloc[-1]), 1)
        result["cum_delta_trend"] = (
            "rising"  if cum_d.iloc[-1] > cum_d.iloc[-5]  else
            "falling" if cum_d.iloc[-1] < cum_d.iloc[-5]  else "flat"
        ) if len(cum_d) >= 5 else "unknown"
    if isinstance(dz, pd.Series) and not dz.empty:
        z_last = float(dz.iloc[-1])
        result["delta_zscore"] = round(z_last, 2)
        result["delta_signal"] = (
            "overbought_selling" if z_last > 2  else
            "oversold_buying"    if z_last < -2 else
            "neutral"
        )
    return result


def _compute_stacked_imbalances(df_5m: pd.DataFrame) -> dict:
    stacks = _safe(detect_stacked_imbalances, df_5m.tail(30).reset_index(drop=True), default=[])
    if not stacks:
        return {"count": 0, "latest": None}
    latest = stacks[-1]
    return {
        "count":     len(stacks),
        "latest": {
            "direction": latest.get("direction", "?"),
            "start_idx": int(latest.get("start_idx", 0)),
            "length":    int(latest.get("length", 0)),
        },
    }


# ─── Main run ──────────────────────────────────────────────────────────────────

def run() -> dict:
    """
    Compute all indicators and write to indicator_state.json.
    Returns the state dict.
    """
    _update_status("running")
    state: dict = {
        "computed_at":    datetime.now(IL_TZ).isoformat(),
        "computed_at_ts": datetime.now(IL_TZ).strftime("%Y-%m-%d %H:%M"),
    }

    try:
        # Load all timeframes
        df_5m     = _safe(get_todays_nq_data, default=pd.DataFrame())
        df_daily  = _safe(load_nq_daily_cache, default=pd.DataFrame())
        df_hourly = _safe(load_nq_hourly_cache, default=pd.DataFrame())
        df_15m    = _safe(fetch_nq_15m, days_back=5, default=pd.DataFrame())

        price = float(df_5m["close"].iloc[-1]) if not df_5m.empty else None

        # ── Session ───────────────────────────────────────────────────────────
        state["session"] = _safe(current_session_israel, default="unknown")
        state["price"]   = round(price, 2) if price else None

        # ── HTF Bias ──────────────────────────────────────────────────────────
        htf = _safe(get_htf_bias, df_daily, df_hourly, default={})
        state["htf_bias"]     = htf.get("bias", "neutral")
        state["htf_strength"] = htf.get("strength", 0)
        state["htf_detail"]   = {k: v for k, v in htf.items()
                                  if k not in ("bias", "strength")}

        # ── Key levels ────────────────────────────────────────────────────────
        levels = _safe(calculate_key_levels, df_daily, df_hourly, default={})
        state["key_levels"] = {
            k: round(float(v), 2)
            for k, v in levels.items()
            if isinstance(v, float) and not np.isnan(v)
        }

        # ── S/R Zones ─────────────────────────────────────────────────────────
        zones = _safe(get_support_resistance_zones, df_daily, df_hourly,
                      df_15m, price, n_zones=8, default=[])
        state["zones"] = [
            {
                "type":      z.get("zone_type", "?"),
                "score":     z.get("score", 0),
                "mid":       round(float(z.get("midpoint", 0)), 2),
                "dist_pts":  round(float(z.get("dist_pts", 0)), 1),
                "layers":    z.get("components", [])[:3],
            }
            for z in zones[:6]
        ] if zones else []
        state["nearest_zone_score"] = max(
            (z["score"] for z in state["zones"] if z["dist_pts"] < 20), default=0
        )

        # ── VWAP ──────────────────────────────────────────────────────────────
        state["vwap"] = _safe(vwap_position, df_5m, default={})

        # ── EQH / EQL ─────────────────────────────────────────────────────────
        eqhl = _safe(detect_eqh_eql, df_5m, default={})
        state["eqh_count"] = len(eqhl.get("eqh", []))
        state["eql_count"] = len(eqhl.get("eql", []))
        state["nearest_eqh"] = eqhl.get("eqh", [])[-1] if eqhl.get("eqh") else None
        state["nearest_eql"] = eqhl.get("eql", [])[-1] if eqhl.get("eql") else None

        # ── OTE ───────────────────────────────────────────────────────────────
        state["ote_zone"] = _compute_ote_zones(df_daily)

        # ── FVGs ──────────────────────────────────────────────────────────────
        state["fvg"] = _compute_fvgs(df_5m, df_15m)

        # ── Order Blocks ──────────────────────────────────────────────────────
        state["order_blocks"] = _compute_order_blocks(df_5m, df_15m)

        # ── Delta Analysis ────────────────────────────────────────────────────
        state["delta"] = _compute_delta_analysis(df_5m)

        # ── Stacked Imbalances ────────────────────────────────────────────────
        state["stacked"] = _compute_stacked_imbalances(df_5m)

        # ── ATR / Range ───────────────────────────────────────────────────────
        state["atr"] = _safe(calculate_atr_range_consumed, df_daily, df_5m, default={})

        # ── Opening Range ─────────────────────────────────────────────────────
        state["opening_range"] = _safe(calculate_opening_range, df_5m, default={})

        # ── Premium / Discount ────────────────────────────────────────────────
        if price and len(df_daily) >= 5:
            rng_hi = float(df_daily.tail(5)["high"].max())
            rng_lo = float(df_daily.tail(5)["low"].min())
            state["premium_discount"] = _safe(classify_premium_discount,
                                              price, rng_hi, rng_lo, default={})

        # ── Statistical Edge ──────────────────────────────────────────────────
        dow  = _safe(day_of_week_bias, df_daily, default={})
        gaps = _safe(gap_fill_stats, df_daily, default={})
        today_dow = datetime.now(IL_TZ).weekday()
        state["statistical_edge"] = {
            "dow":         today_dow,
            "dow_long_wr": round(float(dow.get(today_dow, 0.5)) * 100, 1) if dow else None,
            "gap_fill":    gaps,
        }

        # ── Economic Calendar ─────────────────────────────────────────────────
        state["calendar"]        = _safe(calendar_summary, default={})
        state["blackout"]        = state["calendar"].get("blackout", False)
        state["blackout_reason"] = state["calendar"].get("blackout_reason", "")

        # ── Market Regime ─────────────────────────────────────────────────────
        state["regime"] = _safe(classify_regime, df_daily, default={
            "regime": "transitioning", "label": "Unknown", "icon": "⚪",
            "conf_adj": 0, "atr_mult_adj": 0.0, "size_adj": 1.0,
        })

        # ── Composite Signal Score (pre-assessment) ───────────────────────────
        state["composite_score"] = _quick_score(state)

        # Write to disk
        _LOGS.mkdir(parents=True, exist_ok=True)
        _STATE.write_text(json.dumps(state, indent=2, default=str))
        _update_status("done")

    except Exception as exc:
        err = traceback.format_exc()
        state["error"] = str(exc)
        _LOGS.mkdir(parents=True, exist_ok=True)
        _STATE.write_text(json.dumps(state, indent=2, default=str))
        _update_status("error", error=err[:500])

    return state


def _quick_score(s: dict) -> dict:
    """
    Fast 0–100 composite score for the orchestrator to decide
    whether to escalate to the full Signal Agent.
    Components: HTF(25) + Zone(20) + Delta(15) + Session(15) + ATR(10) + VWAP(15)
    """
    score = 0
    details: dict = {}

    # HTF Bias (25 pts)
    htf_pts = {"strong_bullish": 25, "bullish": 18, "neutral": 10,
               "bearish": 5, "strong_bearish": 2}.get(s.get("htf_bias", "neutral"), 10)
    score += htf_pts
    details["htf"] = htf_pts

    # Nearest zone score (20 pts)
    zone_pts = min(int(s.get("nearest_zone_score", 0)) * 3, 20)
    score += zone_pts
    details["zone"] = zone_pts

    # Delta (15 pts)
    delta_sig = s.get("delta", {}).get("delta_signal", "neutral")
    delta_pts = {"overbought_selling": 12, "oversold_buying": 15,
                 "neutral": 8}.get(delta_sig, 8)
    # Also add if stacked imbalance exists
    if s.get("stacked", {}).get("count", 0) > 0:
        delta_pts = min(delta_pts + 5, 15)
    score += delta_pts
    details["delta"] = delta_pts

    # Session (15 pts)
    session = s.get("session", "")
    session_pts = {
        "ny_morning": 15, "rth_open": 15, "power_hour": 12,
        "pm_session": 10, "pre_market": 6,
        "overnight": 3,  "lunch_lull": 0,
    }.get(session, 5)
    score += session_pts
    details["session"] = session_pts

    # ATR consumed (10 pts — prefer low ATR consumption = more range left)
    atr_consumed = s.get("atr", {})
    atr_pct = atr_consumed.get("pct_consumed", 50) if isinstance(atr_consumed, dict) else 50
    atr_pts = max(0, int(10 - atr_pct / 10))
    score += atr_pts
    details["atr"] = atr_pts

    # VWAP position (15 pts)
    vwap = s.get("vwap", {})
    vwap_zone = vwap.get("zone", "at_vwap") if isinstance(vwap, dict) else "at_vwap"
    vwap_pts = {
        "above_b1": 12, "below_b1": 12,
        "above_b2": 8,  "below_b2": 8,
        "at_vwap":  15,
    }.get(vwap_zone, 10)
    score += vwap_pts
    details["vwap"] = vwap_pts

    return {
        "total":      min(score, 100),
        "threshold":  65,
        "escalate":   score >= 65,
        "components": details,
    }


def load_state() -> dict:
    """Load the last computed indicator state from disk."""
    if _STATE.exists():
        try:
            return json.loads(_STATE.read_text())
        except Exception:
            pass
    return {}

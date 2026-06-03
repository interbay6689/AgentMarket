"""
Trade Tracker — logs every alert signal, auto-evaluates outcomes, analyzes factor performance.

Data flow:
  generate_signal() → log_trade()        → logs/trade_log.csv
  get_todays_nq_data() → evaluate_open_trades() → updates outcome column
  load_trade_log() → factor_win_rates()  → which factors actually predict wins
"""
from __future__ import annotations

import json
import threading
import uuid
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytz

# Protects all CSV read/write operations against concurrent access from
# APScheduler background threads and the Streamlit main thread.
_csv_lock = threading.Lock()

_MD   = Path(__file__).resolve().parents[1]
_ROOT = _MD.parent

TRADE_LOG_PATH     = _MD / "logs" / "trade_log.csv"
TRADE_SETTINGS_PATH = _MD / "logs" / "trade_settings.json"

IL_TZ = pytz.timezone("Asia/Jerusalem")

COLUMNS = [
    "id", "timestamp", "date", "time_il",
    "trade_type",                                          # "real" | "paper"
    "direction", "entry", "stop", "target", "partial_exit", "rr",
    "confidence", "long_pts", "short_pts",
    "htf_bias", "htf_strength", "nearest_zone_score",
    "delta_dir", "session_name", "session_quality",
    "final_score", "ml_prediction",
    "f_htf", "f_zone", "f_delta", "f_session", "f_stacked", "f_sentiment", "f_ml",
    "outcome", "exit_price", "actual_rr", "exit_time", "notes",
]

DEFAULT_SETTINGS = {
    "auto_log":        True,
    "auto_evaluate":   True,
    "min_confidence":  60,
    "min_zone_score":  5,
    "min_rr":          1.5,
    "dedup_minutes":   30,
    "eval_interval_h": 2,
    "risk_per_trade":  1.0,
}

# ─── Settings ─────────────────────────────────────────────────────────────────

def load_trade_settings() -> dict:
    if TRADE_SETTINGS_PATH.exists():
        try:
            with open(TRADE_SETTINGS_PATH, encoding="utf-8") as f:
                saved = json.load(f)
            return {**DEFAULT_SETTINGS, **saved}
        except Exception:
            pass
    return DEFAULT_SETTINGS.copy()


def save_trade_settings(settings: dict) -> None:
    TRADE_SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(TRADE_SETTINGS_PATH, "w", encoding="utf-8") as f:
        json.dump(settings, f, indent=2)


# ─── Trade Log I/O ────────────────────────────────────────────────────────────

def _ensure_log() -> None:
    TRADE_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not TRADE_LOG_PATH.exists():
        pd.DataFrame(columns=COLUMNS).to_csv(TRADE_LOG_PATH, index=False)


def load_trade_log() -> pd.DataFrame:
    _ensure_log()
    try:
        df = pd.read_csv(TRADE_LOG_PATH)
    except Exception:
        return pd.DataFrame(columns=COLUMNS)
    if df.empty:
        return pd.DataFrame(columns=COLUMNS)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    for col in ("entry", "stop", "target", "partial_exit", "rr",
                "exit_price", "actual_rr", "confidence", "long_pts", "short_pts",
                "htf_strength", "nearest_zone_score", "session_quality",
                "final_score", "f_htf", "f_zone", "f_delta",
                "f_session", "f_stacked", "f_sentiment", "f_ml"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _extract_factor_pts(factors: list) -> dict:
    mapping = {
        "HTF Bias (Daily + 1H)":    "f_htf",
        "MTF Zone (15m OB/FVG/KL)": "f_zone",
        "Order Flow Delta (5m)":    "f_delta",
        "Session Quality":          "f_session",
        "Stacked Imbalance (5m)":   "f_stacked",
        "Sentiment Score":          "f_sentiment",
        "ML Prediction":            "f_ml",
    }
    result = {v: 0 for v in mapping.values()}
    for f in factors:
        key = mapping.get(f.get("factor", ""))
        if key:
            result[key] = int(f.get("pts", 0))
    return result


# ─── Logging ──────────────────────────────────────────────────────────────────

def log_trade(sig: dict) -> bool:
    """
    Write a signal to trade_log.csv.
    Returns True if written, False if duplicate (same direction in last dedup_minutes).
    """
    settings = load_trade_settings()
    if not settings.get("auto_log", True):
        return False
    if (sig.get("confidence") or 0) < settings.get("min_confidence", 0):
        return False
    if (sig.get("nearest_zone_score") or 0) < settings.get("min_zone_score", 0):
        return False
    if (sig.get("rr") or 0) < settings.get("min_rr", 0):
        return False

    with _csv_lock:
        _ensure_log()
        df = load_trade_log()

        now_il = datetime.now(IL_TZ)
        dedup  = settings.get("dedup_minutes", 30)

        # Deduplication: same direction already logged within dedup window
        if not df.empty and "timestamp" in df.columns and "direction" in df.columns:
            cutoff = pd.Timestamp(now_il).tz_convert("UTC") - pd.Timedelta(minutes=dedup)
            recent = df[df["timestamp"] >= cutoff]
            if not recent.empty and (recent["direction"] == sig.get("direction")).any():
                return False

        fpts = _extract_factor_pts(sig.get("factors", []))
        row = _build_row(sig, fpts, trade_type="real")
        updated = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        updated.to_csv(TRADE_LOG_PATH, index=False)
        return True


def _build_row(sig: dict, fpts: dict, trade_type: str = "real") -> dict:
    """Build a single CSV row dict from a signal dict."""
    now_il = datetime.now(IL_TZ)
    return {
        "id":                 str(uuid.uuid4())[:8],
        "timestamp":          now_il.isoformat(),
        "date":               now_il.strftime("%Y-%m-%d"),
        "time_il":            now_il.strftime("%H:%M"),
        "trade_type":         trade_type,
        "direction":          sig.get("direction"),
        "entry":              sig.get("entry"),
        "stop":               sig.get("stop"),
        "target":             sig.get("target"),
        "partial_exit":       sig.get("partial_exit"),
        "rr":                 sig.get("rr"),
        "confidence":         sig.get("confidence"),
        "long_pts":           sig.get("long_pts"),
        "short_pts":          sig.get("short_pts"),
        "htf_bias":           sig.get("htf_bias"),
        "htf_strength":       sig.get("htf_strength"),
        "nearest_zone_score": sig.get("nearest_zone_score"),
        "delta_dir":          sig.get("delta_dir"),
        "session_name":       sig.get("session", {}).get("name", ""),
        "session_quality":    sig.get("session_quality"),
        "final_score":        sig.get("final_score"),
        "ml_prediction":      sig.get("ml_prediction"),
        **fpts,
        "outcome":            "OPEN",
        "exit_price":         None,
        "actual_rr":          None,
        "exit_time":          None,
        "notes":              "",
    }


def log_paper_trade(sig: dict) -> bool:
    """
    Log a directional signal as a paper trade regardless of alert threshold.

    Called every time the signal engine returns LONG or SHORT — even when
    confidence, zone score, or R:R don't meet the alert criteria.  Paper
    trades are evaluated automatically by evaluate_open_trades() just like
    real trades, giving the ML model training data on quiet days.

    Deduplication: one paper trade per direction per 30-minute window.
    """
    if sig.get("direction") not in ("LONG", "SHORT"):
        return False
    if sig.get("blackout"):
        return False

    settings = load_trade_settings()
    dedup = settings.get("dedup_minutes", 30)

    with _csv_lock:
        _ensure_log()
        df = load_trade_log()
        now_il = datetime.now(IL_TZ)

        if not df.empty and "timestamp" in df.columns:
            cutoff = pd.Timestamp(now_il).tz_convert("UTC") - pd.Timedelta(minutes=dedup)
            recent = df[
                (df["timestamp"] >= cutoff) &
                (df.get("trade_type", pd.Series(dtype=str)) == "paper")
            ] if "trade_type" in df.columns else df[df["timestamp"] >= cutoff]
            if not recent.empty and (recent["direction"] == sig.get("direction")).any():
                return False

        fpts = _extract_factor_pts(sig.get("factors", []))
        row  = _build_row(sig, fpts, trade_type="paper")
        updated = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        updated.to_csv(TRADE_LOG_PATH, index=False)
        return True


def update_trade_manual(
    trade_id: str,
    outcome: str,
    exit_price: float | None = None,
    notes: str = "",
) -> bool:
    """Manually set outcome for a trade by its id field."""
    with _csv_lock:
        df = load_trade_log()
        mask = df["id"] == trade_id
        if not mask.any():
            return False
        idx = df[mask].index[0]
        entry = float(df.at[idx, "entry"]) if pd.notna(df.at[idx, "entry"]) else None
        stop  = float(df.at[idx, "stop"])  if pd.notna(df.at[idx, "stop"])  else None

        df.at[idx, "outcome"]   = outcome
        df.at[idx, "exit_time"] = datetime.now(IL_TZ).strftime("%Y-%m-%d %H:%M")
        if notes:
            df.at[idx, "notes"] = notes
        if exit_price is not None:
            df.at[idx, "exit_price"] = exit_price
            if entry is not None and stop is not None and abs(entry - stop) > 0:
                df.at[idx, "actual_rr"] = round(abs(exit_price - entry) / abs(entry - stop), 2)
        df.to_csv(TRADE_LOG_PATH, index=False)
        return True


# ─── Auto-Evaluation ──────────────────────────────────────────────────────────

def evaluate_open_trades(df_price: pd.DataFrame) -> int:
    """
    Check OPEN trades against recent price bars.
    Updates outcome: WIN / LOSS / PARTIAL (hit 1R but still open).
    Returns number of trades resolved.
    """
    if df_price.empty:
        return 0
    with _csv_lock:
        _ensure_log()
        df = load_trade_log()
        open_mask = df["outcome"] == "OPEN"
        if not open_mask.any():
            return 0

        # pandas 2.x: columns loaded from an all-NaN CSV are inferred as float64.
        # df.at assignment then raises TypeError when the value is a string.
        # Cast string-typed columns to object before we write into them.
        for _col in ("exit_time", "outcome", "notes"):
            if _col in df.columns:
                df[_col] = df[_col].astype(object)

        time_col = "datetime_il" if "datetime_il" in df_price.columns else "datetime"
        updated_count = 0

        for idx in df[open_mask].index:
            row = df.loc[idx]
            direction = row.get("direction")
            entry     = float(row["entry"])        if pd.notna(row.get("entry"))        else None
            stop      = float(row["stop"])         if pd.notna(row.get("stop"))         else None
            target    = float(row["target"])       if pd.notna(row.get("target"))       else None
            partial   = float(row["partial_exit"]) if pd.notna(row.get("partial_exit")) else None

            if not all([entry, stop, target, direction]):
                continue

            # Filter price bars after trade entry
            entry_ts = row.get("timestamp")
            bars = df_price.copy()
            if pd.notna(entry_ts):
                try:
                    ets = pd.to_datetime(entry_ts, utc=True).tz_convert(None)
                    bar_ts = pd.to_datetime(bars[time_col], utc=True, errors="coerce").dt.tz_convert(None)
                    bars = bars[bar_ts >= ets]
                except Exception:
                    pass

            if bars.empty:
                continue

            outcome    = None
            exit_price = None
            exit_time  = None
            hit_partial = False

            for _, bar in bars.iterrows():
                high = float(bar["high"])
                low  = float(bar["low"])
                bt   = str(bar.get(time_col, "")).split("+")[0].split(".")[0]

                if direction == "LONG":
                    if low <= stop:
                        outcome, exit_price, exit_time = "LOSS", stop, bt
                        break
                    if high >= target:
                        outcome, exit_price, exit_time = "WIN", target, bt
                        break
                    if partial and high >= partial:
                        hit_partial = True
                else:  # SHORT
                    if high >= stop:
                        outcome, exit_price, exit_time = "LOSS", stop, bt
                        break
                    if low <= target:
                        outcome, exit_price, exit_time = "WIN", target, bt
                        break
                    if partial and low <= partial:
                        hit_partial = True

            if outcome is None and hit_partial:
                outcome = "PARTIAL"

            if outcome is not None:
                actual_rr = None
                if exit_price and entry and stop and abs(entry - stop) > 0:
                    actual_rr = round(abs(exit_price - entry) / abs(entry - stop), 2)
                df.at[idx, "outcome"]    = outcome
                df.at[idx, "exit_price"] = exit_price
                df.at[idx, "actual_rr"]  = actual_rr
                df.at[idx, "exit_time"]  = exit_time
                updated_count += 1

        if updated_count:
            df.to_csv(TRADE_LOG_PATH, index=False)
        return updated_count


# ─── Analytics ────────────────────────────────────────────────────────────────

def factor_win_rates(df: pd.DataFrame) -> pd.DataFrame:
    """Win rate and average R:R broken down by factor buckets."""
    closed = df[df["outcome"].isin(["WIN", "LOSS", "PARTIAL"])].copy()
    if closed.empty:
        return pd.DataFrame()
    closed["win"] = (closed["outcome"] == "WIN").astype(int)

    def _group(sub, factor, value):
        if len(sub) < 2:
            return None
        avg_rr = round(sub["actual_rr"].dropna().mean(), 2) if sub["actual_rr"].notna().any() else None
        return {
            "Factor": factor, "Value": value,
            "Trades": len(sub), "Wins": int(sub["win"].sum()),
            "Win %": round(sub["win"].mean() * 100, 1),
            "Avg R:R": avg_rr,
        }

    rows = []

    # HTF Bias
    for val in ["bullish", "bearish", "neutral"]:
        r = _group(closed[closed["htf_bias"] == val], "HTF Bias", val)
        if r: rows.append(r)

    # Zone Score buckets
    for lo, hi, label in [(0, 4, "0-4 weak"), (5, 7, "5-7 medium"), (8, 10, "8-10 strong")]:
        r = _group(closed[closed["nearest_zone_score"].between(lo, hi)], "Zone Score", label)
        if r: rows.append(r)

    # Delta direction
    for val in ["Bullish", "Bearish", "Neutral"]:
        r = _group(closed[closed["delta_dir"] == val], "Delta", val)
        if r: rows.append(r)

    # Session
    for val in closed["session_name"].dropna().unique():
        r = _group(closed[closed["session_name"] == val], "Session", str(val))
        if r: rows.append(r)

    # Confidence buckets
    for lo, hi, label in [(55, 69, "55–69%"), (70, 84, "70–84%"), (85, 100, "85–100%")]:
        r = _group(closed[closed["confidence"].between(lo, hi)], "Confidence", label)
        if r: rows.append(r)

    # ML Prediction vs outcome
    for val in ["LONG", "SHORT", "NEUTRAL"]:
        r = _group(closed[closed["ml_prediction"] == val], "ML Prediction", val)
        if r: rows.append(r)

    # HTF matches direction
    def _htf_match(row):
        d = row.get("direction", "")
        b = row.get("htf_bias", "")
        if (d == "LONG" and b == "bullish") or (d == "SHORT" and b == "bearish"):
            return "HTF Agrees"
        return "HTF Disagrees"
    closed["htf_match"] = closed.apply(_htf_match, axis=1)
    for val in ["HTF Agrees", "HTF Disagrees"]:
        r = _group(closed[closed["htf_match"] == val], "HTF Alignment", val)
        if r: rows.append(r)

    return pd.DataFrame(rows).sort_values("Win %", ascending=False).reset_index(drop=True)


def equity_curve(df: pd.DataFrame, risk_per_trade: float = 1.0) -> pd.DataFrame:
    """Cumulative R series for equity curve. Returns DataFrame with [timestamp, cumR]."""
    closed = df[df["outcome"].isin(["WIN", "LOSS", "PARTIAL"])].copy()
    if closed.empty:
        return pd.DataFrame(columns=["timestamp", "cumR"])
    closed = closed.sort_values("timestamp").reset_index(drop=True)
    closed["r"] = closed.apply(
        lambda row: (
            float(row["actual_rr"]) * risk_per_trade if row["outcome"] == "WIN"
            else (-risk_per_trade if row["outcome"] == "LOSS"
                  else 0.5 * risk_per_trade)
        ),
        axis=1,
    )
    closed["cumR"] = closed["r"].cumsum()
    return closed[["timestamp", "date", "time_il", "direction", "r", "cumR"]]


def summary_stats(df: pd.DataFrame) -> dict:
    """High-level stats dict for dashboard display."""
    closed = df[df["outcome"].isin(["WIN", "LOSS", "PARTIAL"])]
    total  = len(closed)
    if total == 0:
        return {"total": 0, "wins": 0, "losses": 0, "win_rate": 0,
                "avg_rr_wins": None, "total_r": 0, "open_trades": len(df[df["outcome"] == "OPEN"])}
    wins   = len(closed[closed["outcome"] == "WIN"])
    losses = len(closed[closed["outcome"] == "LOSS"])
    rr_wins = closed[closed["outcome"] == "WIN"]["actual_rr"].dropna()
    total_r = equity_curve(df)["cumR"].iloc[-1] if not equity_curve(df).empty else 0
    return {
        "total":       total,
        "wins":        wins,
        "losses":      losses,
        "partials":    total - wins - losses,
        "win_rate":    round(wins / total * 100, 1) if total else 0,
        "avg_rr_wins": round(float(rr_wins.mean()), 2) if not rr_wins.empty else None,
        "total_r":     round(float(total_r), 2),
        "open_trades": len(df[df["outcome"] == "OPEN"]),
    }

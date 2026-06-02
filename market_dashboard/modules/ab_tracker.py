"""
ab_tracker.py — A/B Performance Tracking engine.

Tracks signal accuracy before/after config changes over rolling windows.
Each experiment records signals automatically from the trade_signals page;
outcomes are resolved automatically from NQ 5m price data (~1h after signal).

Storage: logs/ab_experiments.json (rolling, no size limit)
"""
import json
import logging
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytz

logger = logging.getLogger(__name__)

_ROOT      = Path(__file__).resolve().parents[2]
_FILE      = _ROOT / "logs" / "ab_experiments.json"
_IL_TZ     = pytz.timezone("Asia/Jerusalem")
_WIN_PCT   = 0.12   # % price move required to classify WIN/LOSS
_MIN_AGE   = 75     # minutes before a PENDING signal is eligible for resolution


# ─── Persistence ─────────────────────────────────────────────────────────────

def _load() -> dict:
    if not _FILE.exists():
        return {"experiments": []}
    try:
        return json.loads(_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {"experiments": []}


def _save(data: dict) -> None:
    _FILE.parent.mkdir(parents=True, exist_ok=True)
    _FILE.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


# ─── Experiment management ────────────────────────────────────────────────────

def create_experiment(
    name: str,
    description: str = "",
    config_snapshot: dict | None = None,
) -> str:
    """
    Create a new experiment and deactivate the previous one.
    Returns the new experiment_id.
    """
    data = _load()
    for exp in data["experiments"]:
        exp["active"] = False

    exp_id = f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    data["experiments"].append({
        "exp_id":          exp_id,
        "name":            name,
        "description":     description,
        "created_at":      datetime.now(_IL_TZ).isoformat(),
        "config_snapshot": config_snapshot or {},
        "signals":         [],
        "active":          True,
    })
    _save(data)
    logger.info(f"New experiment: {exp_id} — {name}")
    return exp_id


def get_active_experiment() -> dict | None:
    data = _load()
    for exp in reversed(data["experiments"]):
        if exp.get("active"):
            return exp
    return None


def list_experiments(limit: int = 20) -> list:
    return list(reversed(_load()["experiments"]))[:limit]


# ─── Signal recording ─────────────────────────────────────────────────────────

def _has_signal_this_hour(exp_id: str, direction: str, date_hour: str) -> bool:
    for exp in _load()["experiments"]:
        if exp["exp_id"] == exp_id:
            return any(
                s.get("date_hour") == date_hour and s.get("direction") == direction
                for s in exp["signals"]
            )
    return False


def record_signal(signal_data: dict) -> str:
    """
    Record a signal into the active experiment.
    Deduplicates by direction + hour to prevent refresh spam.
    Returns signal_id, or "" if no active experiment / duplicate.
    """
    active = get_active_experiment()
    if not active:
        return ""

    direction = signal_data.get("direction", "NEUTRAL")
    now_il    = datetime.now(_IL_TZ)
    date_hour = now_il.strftime("%Y-%m-%d_%H")

    if _has_signal_this_hour(active["exp_id"], direction, date_hour):
        return ""

    signal_id = str(uuid.uuid4())[:8]
    entry_price = (
        signal_data.get("entry")
        or signal_data.get("current_price")
        or 0
    )
    record = {
        "signal_id":    signal_id,
        "date":         now_il.strftime("%Y-%m-%d"),
        "time_il":      now_il.strftime("%H:%M"),
        "date_hour":    date_hour,
        "recorded_at":  now_il.isoformat(),
        "direction":    direction,
        "confidence":   float(signal_data.get("confidence", 0)),
        "session":      signal_data.get("session", ""),
        "regime":       str(signal_data.get("regime", "")),
        "score":        float(signal_data.get("score", signal_data.get("final_score", 50) or 50)),
        "entry_price":  float(entry_price or 0),
        "outcome":      "PENDING",
        "outcome_price": 0.0,
        "pnl_points":   0.0,
        "resolved_at":  "",
    }

    data = _load()
    for exp in data["experiments"]:
        if exp["exp_id"] == active["exp_id"]:
            exp["signals"].append(record)
            break
    _save(data)
    return signal_id


# ─── Auto-resolution ──────────────────────────────────────────────────────────

def resolve_pending_outcomes(price_df) -> int:
    """
    Auto-resolve PENDING signals using NQ 5m price data.

    Rules:
      WIN:     price moves > _WIN_PCT % in signal direction within 1 hour
      LOSS:    price moves > _WIN_PCT % against signal within 1 hour
      NEUTRAL: price didn't move enough (or signal was NEUTRAL)

    Only resolves signals older than _MIN_AGE minutes.
    Returns number of signals resolved.
    """
    if price_df is None or price_df.empty:
        return 0

    import pandas as pd

    data          = _load()
    resolved      = 0
    now_utc       = datetime.now(timezone.utc)
    close_col     = next(
        (c for c in ("Close", "close") if c in price_df.columns), None
    )
    if not close_col:
        return 0

    for exp in data["experiments"]:
        for sig in exp["signals"]:
            if sig["outcome"] != "PENDING":
                continue

            # Parse recorded time and check minimum age
            try:
                rec = datetime.fromisoformat(sig["recorded_at"])
                if rec.tzinfo is None:
                    rec = _IL_TZ.localize(rec)
                rec_utc = rec.astimezone(timezone.utc)
            except Exception:
                continue

            if (now_utc - rec_utc).total_seconds() / 60 < _MIN_AGE:
                continue

            # Slice price bars: [signal_time, signal_time + 1h]
            sig_ts = pd.Timestamp(rec_utc)
            end_ts = sig_ts + pd.Timedelta(hours=1)

            try:
                idx = price_df.index
                if hasattr(idx, "tz") and idx.tz is not None:
                    bars = price_df[(idx >= sig_ts) & (idx <= end_ts)]
                else:
                    bars = price_df[
                        (idx >= sig_ts.tz_localize(None))
                        & (idx <= end_ts.tz_localize(None))
                    ]
            except Exception:
                continue

            if bars.empty:
                sig["outcome"]     = "NEUTRAL"
                sig["resolved_at"] = now_utc.isoformat()
                resolved += 1
                continue

            entry = float(bars.iloc[0][close_col])
            exit_ = float(bars.iloc[-1][close_col])
            move  = (exit_ - entry) / entry * 100 if entry else 0

            direction = sig.get("direction", "NEUTRAL")
            if direction == "NEUTRAL":
                outcome = "NEUTRAL"
                pnl     = 0.0
            elif direction == "LONG":
                if move >= _WIN_PCT:
                    outcome, pnl = "WIN",  round(exit_ - entry, 1)
                elif move <= -_WIN_PCT:
                    outcome, pnl = "LOSS", round(exit_ - entry, 1)
                else:
                    outcome, pnl = "NEUTRAL", 0.0
            else:  # SHORT
                if move <= -_WIN_PCT:
                    outcome, pnl = "WIN",  round(entry - exit_, 1)
                elif move >= _WIN_PCT:
                    outcome, pnl = "LOSS", round(entry - exit_, 1)
                else:
                    outcome, pnl = "NEUTRAL", 0.0

            sig.update(
                outcome=outcome,
                pnl_points=pnl,
                entry_price=round(entry, 2),
                outcome_price=round(exit_, 2),
                resolved_at=now_utc.isoformat(),
            )
            resolved += 1

    if resolved:
        _save(data)
    return resolved


# ─── Metrics ──────────────────────────────────────────────────────────────────

def get_experiment_metrics(exp_id: str, days: int = 14) -> dict:
    data   = _load()
    exp    = next((e for e in data["experiments"] if e["exp_id"] == exp_id), None)
    empty  = {"total": 0, "resolved": 0, "pending": 0, "wins": 0, "losses": 0,
              "neutrals": 0, "win_rate": 0.0, "loss_rate": 0.0, "accuracy": 0.0,
              "avg_pnl": 0.0, "avg_conf_wins": 0.0, "avg_conf_losses": 0.0,
              "exp_name": "", "created_at": ""}
    if not exp:
        return empty

    cutoff  = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
    signals = [s for s in exp["signals"] if s.get("date", "") >= cutoff]
    if not signals:
        return {**empty, "exp_name": exp["name"], "created_at": exp.get("created_at", "")[:10]}

    resolved = [s for s in signals if s["outcome"] != "PENDING"]
    wins     = [s for s in resolved if s["outcome"] == "WIN"]
    losses   = [s for s in resolved if s["outcome"] == "LOSS"]
    neutrals = [s for s in resolved if s["outcome"] == "NEUTRAL"]
    dir_res  = [s for s in resolved if s.get("direction") in ("LONG", "SHORT")]
    n_res    = len(resolved)

    return {
        "total":           len(signals),
        "resolved":        n_res,
        "pending":         len(signals) - n_res,
        "wins":            len(wins),
        "losses":          len(losses),
        "neutrals":        len(neutrals),
        "win_rate":        round(len(wins) / n_res * 100, 1)        if n_res   else 0.0,
        "loss_rate":       round(len(losses) / n_res * 100, 1)      if n_res   else 0.0,
        "accuracy":        round(len(wins) / len(dir_res) * 100, 1) if dir_res else 0.0,
        "avg_pnl":         round(sum(s["pnl_points"] for s in resolved) / n_res, 1) if n_res else 0.0,
        "avg_conf_wins":   round(sum(s["confidence"] for s in wins) / len(wins), 1)   if wins  else 0.0,
        "avg_conf_losses": round(sum(s["confidence"] for s in losses) / len(losses), 1) if losses else 0.0,
        "exp_name":        exp["name"],
        "created_at":      exp.get("created_at", "")[:10],
    }


def compare_experiments(exp_id_a: str, exp_id_b: str, days: int = 14) -> dict:
    m_a = get_experiment_metrics(exp_id_a, days)
    m_b = get_experiment_metrics(exp_id_b, days)
    keys  = ["win_rate", "accuracy", "avg_pnl", "avg_conf_wins"]
    delta = {k: round(m_b.get(k, 0) - m_a.get(k, 0), 1) for k in keys}
    pos   = sum(1 for v in delta.values() if v > 0)
    verdict = (
        "B is better ✅" if pos >= 3
        else "A is better ⚠️" if pos <= 1
        else "Mixed results 🟡"
    )
    return {"a": m_a, "b": m_b, "delta": delta, "verdict": verdict}

"""
Agent Safety Layer — validates all config changes before applying.
Enforces parameter limits, cooldowns, minimum sample requirements,
change history logging, dry-run, and revert capability.
"""
from __future__ import annotations

import json
import math
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pytz

_MD   = Path(__file__).resolve().parents[1]
_ROOT = _MD.parent

CHANGE_LOG_PATH    = _MD / "logs" / "config_changes.log"
CHANGE_HIST_PATH   = _MD / "logs" / "config_changes.json"
TRADE_LOG_PATH     = _MD / "logs" / "trade_log.csv"

IL_TZ = pytz.timezone("Asia/Jerusalem")

# ─── Parameter limits ─────────────────────────────────────────────────────────

PARAMETER_LIMITS: dict[str, tuple] = {
    # Fundamental weights (each, not sum)
    "weights.sentiment":        (0.10, 0.45),
    "weights.macro":            (0.05, 0.30),
    "weights.bonds":            (0.05, 0.20),
    "weights.vix":              (0.05, 0.25),
    "weights.sectors":          (0.02, 0.15),
    "weights.mes":              (0.10, 0.40),
    # Signal / entry thresholds
    "signal.min_confidence":    (55,   85),
    "signal.min_zone_score":    (4,     9),
    "signal.min_rr":            (1.2,  3.0),
    "signal.min_pts":           (45,   70),
    "signal.min_margin":        (10,   25),
    # Trade tracking
    "trade.dedup_minutes":      (15,  120),
    "trade.eval_interval_h":    (1,    24),
    "trade.min_confidence":     (55,   85),
    "trade.min_zone_score":     (3,    9),
    "trade.min_rr":             (1.2,  3.0),
    # ICT config
    "ict.ote_fib_lo":           (0.50, 0.70),
    "ict.ote_fib_hi":           (0.70, 0.90),
    "ict.eqh_eql_tolerance_pts": (1.0, 10.0),
    "ict.macro_window_multiplier": (1.0, 1.5),
    # VWAP config
    "vwap.band1_weight_pts":    (4,   15),
    "vwap.band2_weight_pts":    (8,   20),
    "vwap.slope_pts":           (2,   10),
    # Session multipliers
    "session.rth_open":         (0.8,  1.5),
    "session.ny_morning":       (0.8,  1.5),
    "session.pm_session":       (0.5,  1.3),
    "session.power_hour":       (0.5,  1.3),
    "session.lunch_lull":       (0.0,  0.5),
    # ATR stop multiplier
    "risk.atr_mult":            (1.0,  2.5),
}

# Safety floors — cannot go below these regardless of limits
HARD_FLOORS: dict[str, float] = {
    "signal.min_rr":         1.2,
    "signal.min_confidence": 55,
    "trade.min_rr":          1.2,
    "risk.atr_mult":         1.0,
}

CHANGE_RULES = {
    "max_change_pct":              0.15,   # 15% max per param per session
    "cooldown_hours":              24,     # hours between changes to same param
    "min_trades_for_weight":       20,     # minimum closed trades before touching weights
    "min_trades_for_threshold":    15,
    "min_trades_for_session":      10,
}

# ─── Validation ───────────────────────────────────────────────────────────────

def validate_change(param: str, new_value: Any, current_value: Any = None) -> tuple[bool, str]:
    """Returns (is_valid, reason). Checks limits, floors, and change magnitude."""

    # Hard floor check
    if param in HARD_FLOORS and float(new_value) < HARD_FLOORS[param]:
        return False, f"{param} cannot go below hard floor {HARD_FLOORS[param]}"

    # Range check
    limits = PARAMETER_LIMITS.get(param)
    if limits is None:
        return False, f"Unknown parameter '{param}' — not in allowed list"
    lo, hi = limits
    try:
        v = float(new_value)
    except (TypeError, ValueError):
        return False, f"Value must be numeric, got {type(new_value).__name__}"
    if not (lo <= v <= hi):
        return False, f"{param} must be in [{lo}, {hi}], got {v}"

    # Max change magnitude
    if current_value is not None:
        try:
            cur = float(current_value)
            if cur != 0:
                pct_change = abs(v - cur) / abs(cur)
                if pct_change > CHANGE_RULES["max_change_pct"]:
                    return False, (f"Change of {pct_change*100:.1f}% exceeds max "
                                   f"{CHANGE_RULES['max_change_pct']*100:.0f}% per session")
        except Exception:
            pass

    return True, "OK"


def validate_weights(new_weights: dict) -> tuple[bool, str]:
    """Validate that all fundamental weights are valid and sum to ~1.0."""
    for key, val in new_weights.items():
        ok, reason = validate_change(f"weights.{key}", val)
        if not ok:
            return False, reason
    total = sum(new_weights.values())
    if abs(total - 1.0) > 0.02:
        return False, f"Weights must sum to 1.0, got {total:.3f}"
    return True, "OK"


def check_cooldown(param: str) -> tuple[bool, str]:
    """Returns (can_change, reason). Prevents changes to same param within cooldown_hours."""
    hist = _load_change_history()
    cutoff = datetime.now(IL_TZ) - timedelta(hours=CHANGE_RULES["cooldown_hours"])
    recent_params = [
        e["param"] for e in hist
        if _parse_ts(e.get("timestamp")) and _parse_ts(e["timestamp"]) > cutoff
    ]
    if param in recent_params:
        return False, f"{param} was already changed within last {CHANGE_RULES['cooldown_hours']}h — cooldown active"
    return True, "OK"


def check_min_trades(change_type: str) -> tuple[bool, str]:
    """Check if there are enough closed trades to justify a change."""
    try:
        import pandas as pd
        df = pd.read_csv(TRADE_LOG_PATH)
        closed = df[df["outcome"].isin(["WIN", "LOSS", "PARTIAL"])]
        n = len(closed)
    except Exception:
        n = 0

    min_map = {
        "weight":    CHANGE_RULES["min_trades_for_weight"],
        "threshold": CHANGE_RULES["min_trades_for_threshold"],
        "session":   CHANGE_RULES["min_trades_for_session"],
    }
    required = min_map.get(change_type, CHANGE_RULES["min_trades_for_weight"])
    if n < required:
        return False, f"Need {required} closed trades for {change_type} changes, have {n}"
    return True, "OK"


# ─── Apply / Dry-run / Revert ─────────────────────────────────────────────────

def dry_run(proposed_changes: list[dict]) -> dict:
    """
    Simulate applying proposed changes without writing to disk.
    Returns {valid: [...], invalid: [...], warnings: [...]}
    """
    valid, invalid, warnings = [], [], []
    seen_params = set()

    for change in proposed_changes:
        param    = change.get("param", "")
        new_val  = change.get("new_value")
        cur_val  = change.get("current_value")
        reason_ok = change.get("reasoning", "no reason given")

        if param in seen_params:
            invalid.append({"param": param, "reason": "Duplicate param in same batch"})
            continue
        seen_params.add(param)

        ok, reason = validate_change(param, new_val, cur_val)
        if not ok:
            invalid.append({"param": param, "new_value": new_val, "reason": reason})
            continue

        cd_ok, cd_reason = check_cooldown(param)
        if not cd_ok:
            warnings.append({"param": param, "warning": cd_reason})

        valid.append({
            "param":    param,
            "current":  cur_val,
            "proposed": new_val,
            "change":   round((float(new_val) - float(cur_val)) / float(cur_val) * 100, 1)
                        if cur_val and float(cur_val) != 0 else "N/A",
            "reasoning": reason_ok,
        })

    return {"valid": valid, "invalid": invalid, "warnings": warnings,
            "apply_count": len(valid), "reject_count": len(invalid)}


def apply_change(
    param: str,
    new_value: Any,
    current_value: Any,
    reasoning: str,
    agent_name: str = "unknown",
    skip_cooldown: bool = False,
) -> tuple[bool, str]:
    """
    Validate, apply to the correct config file, and log.
    Returns (success, message).
    """
    ok, reason = validate_change(param, new_value, current_value)
    if not ok:
        return False, reason

    if not skip_cooldown:
        cd_ok, cd_reason = check_cooldown(param)
        if not cd_ok:
            return False, cd_reason

    # Route to correct file
    success, msg = _write_param(param, new_value)
    if not success:
        return False, msg

    # Log the change
    _log_change(param, current_value, new_value, reasoning, agent_name)
    return True, f"Applied {param}: {current_value} → {new_value}"


def revert_last_change() -> tuple[bool, str]:
    """Undo the most recent config change."""
    hist = _load_change_history()
    if not hist:
        return False, "No changes to revert"
    last = hist[-1]
    success, msg = _write_param(last["param"], last["old_value"])
    if success:
        hist.pop()
        _save_change_history(hist)
        _append_log(f"REVERTED | {last['param']}: {last['new_value']} → {last['old_value']} | "
                    f"original reason: {last['reasoning']}")
        return True, f"Reverted {last['param']} to {last['old_value']}"
    return False, msg


def revert_to_defaults() -> tuple[bool, str]:
    """Reset all config files to factory defaults."""
    _ROOT_CONF = _ROOT / "scores_news" / "config"
    defaults = {
        "weights": {"sentiment": 0.30, "macro": 0.15, "bonds": 0.10,
                    "vix": 0.15, "sectors": 0.05, "mes": 0.25},
        "thresholds": {"yield_inversion": 2.0, "vix_spike": 12.0},
    }
    try:
        (_ROOT_CONF / "weights.json").write_text(
            json.dumps(defaults["weights"], indent=2))
        (_ROOT_CONF / "thresholds.json").write_text(
            json.dumps(defaults["thresholds"], indent=2))
        _append_log("FULL RESET to factory defaults")
        return True, "Reset to defaults"
    except Exception as e:
        return False, str(e)


def get_change_history(n: int = 50) -> list:
    return _load_change_history()[-n:]


# ─── Internal helpers ─────────────────────────────────────────────────────────

def _write_param(param: str, value: Any) -> tuple[bool, str]:
    """Write a single parameter to the appropriate config file."""
    _ROOT_CONF = _ROOT / "scores_news" / "config"
    try:
        section, key = param.split(".", 1) if "." in param else (param, param)

        if section == "weights":
            path = _ROOT_CONF / "weights.json"
            data = json.loads(path.read_text()) if path.exists() else {}
            data[key] = float(value)
            path.write_text(json.dumps(data, indent=2))

        elif section == "thresholds" or param.startswith("signal."):
            path = _ROOT_CONF / "thresholds.json"
            data = json.loads(path.read_text()) if path.exists() else {}
            data[key] = float(value)
            path.write_text(json.dumps(data, indent=2))

        elif section == "trade":
            path = _MD / "logs" / "trade_settings.json"
            data = json.loads(path.read_text()) if path.exists() else {}
            remap = {"dedup_minutes": "dedup_minutes", "eval_interval_h": "eval_interval_h",
                     "min_confidence": "min_confidence", "min_zone_score": "min_zone_score",
                     "min_rr": "min_rr"}
            data[remap.get(key, key)] = float(value)
            path.write_text(json.dumps(data, indent=2))

        elif section in ("ict", "vwap", "session", "risk"):
            path = _MD / "logs" / "agent_config.json"
            data = json.loads(path.read_text()) if path.exists() else {}
            data.setdefault(section, {})[key] = float(value)
            path.write_text(json.dumps(data, indent=2))

        else:
            return False, f"No file mapping for section '{section}'"

        return True, "OK"

    except Exception as e:
        return False, str(e)


def _load_change_history() -> list:
    if CHANGE_HIST_PATH.exists():
        try:
            return json.loads(CHANGE_HIST_PATH.read_text())
        except Exception:
            pass
    return []


def _save_change_history(hist: list) -> None:
    CHANGE_HIST_PATH.parent.mkdir(parents=True, exist_ok=True)
    CHANGE_HIST_PATH.write_text(json.dumps(hist, indent=2))


def _log_change(param, old, new, reasoning, agent_name) -> None:
    ts  = datetime.now(IL_TZ).strftime("%Y-%m-%d %H:%M")
    entry = {
        "timestamp": datetime.now(IL_TZ).isoformat(),
        "param":     param, "old_value": old,
        "new_value": new,   "reasoning": reasoning,
        "agent":     agent_name,
    }
    hist = _load_change_history()
    hist.append(entry)
    _save_change_history(hist)
    _append_log(
        f"{ts} | {agent_name.upper()} | {param}: {old} → {new}\n"
        f"  Reasoning: {reasoning}"
    )


def _append_log(text: str) -> None:
    CHANGE_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CHANGE_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(text + "\n")


def _parse_ts(ts_str: str):
    try:
        return datetime.fromisoformat(str(ts_str)).astimezone(IL_TZ)
    except Exception:
        return None

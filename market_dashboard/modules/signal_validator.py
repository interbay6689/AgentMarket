"""
Signal Cross-Validator — compares rule-based signal_engine output vs
Claude Signal Agent proposal and produces a consensus result.

Agreement levels:
  FULL      — both engines agree on same direction (LONG/SHORT) → higher conviction
  PARTIAL   — one engine is NEUTRAL/NO_SIGNAL; other has a direction → wait
  CONFLICT  — one LONG, one SHORT → DO NOT TRADE
  NEUTRAL   — both NEUTRAL/NO_SIGNAL → sit on hands

Results are logged to logs/validation_log.json for historical tracking.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytz

_MD   = Path(__file__).resolve().parents[1]
_ROOT = _MD.parent
for p in (str(_MD), str(_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

IL_TZ = pytz.timezone("Asia/Jerusalem")
_LOGS = _MD / "logs"
_PROPOSALS_PATH  = _LOGS / "agent_proposals.json"
_VALIDATION_PATH = _LOGS / "validation_log.json"

# A proposal older than this is considered stale
FRESH_MAX_MINUTES = 90


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _load_latest_proposal() -> dict | None:
    if not _PROPOSALS_PATH.exists():
        return None
    try:
        data = json.loads(_PROPOSALS_PATH.read_text())
        if not isinstance(data, list) or not data:
            return None
        return data[-1]
    except Exception:
        return None


def _proposal_age_minutes(proposal: dict) -> float | None:
    logged_at = proposal.get("logged_at")
    if not logged_at:
        return None
    try:
        ts = datetime.fromisoformat(logged_at)
        if ts.tzinfo is None:
            ts = IL_TZ.localize(ts)
        now = datetime.now(IL_TZ)
        return (now - ts).total_seconds() / 60
    except Exception:
        return None


def _normalise_agent_direction(proposal: dict) -> str:
    """Extract direction from either a wrapped {'proposal': {...}} or flat dict."""
    p = proposal.get("proposal", proposal)
    d = p.get("direction", "NO_SIGNAL")
    return d if d in ("LONG", "SHORT") else "NEUTRAL"


# ─── Main validation ──────────────────────────────────────────────────────────

def validate(rule_sig: dict | None = None) -> dict:
    """
    Compare rule engine signal vs latest agent proposal.

    Parameters
    ----------
    rule_sig : dict, optional
        Pre-computed rule signal (avoids a second call when already computed).
        If None, calls generate_signal() internally.

    Returns
    -------
    dict with keys:
        rule, agent, agreement, consensus_direction, consensus_confidence,
        consensus_label, note, validated_at
    """
    # ── Rule-based signal ─────────────────────────────────────────────────────
    if rule_sig is None:
        try:
            from modules.signal_engine import generate_signal
            rule_sig = generate_signal()
        except Exception as e:
            rule_sig = {"direction": "NEUTRAL", "confidence": 0, "error": str(e)}

    rule_dir  = rule_sig.get("direction", "NEUTRAL")
    rule_conf = int(rule_sig.get("confidence", 0))
    rule_blackout = rule_sig.get("blackout", False)

    rule_info = {
        "direction":  rule_dir,
        "confidence": rule_conf,
        "session":    rule_sig.get("session", "?"),
        "htf_bias":   rule_sig.get("htf_bias", "?"),
        "score":      rule_sig.get("score", rule_conf),
        "blackout":   rule_blackout,
        "alert":      rule_sig.get("alert", False),
    }

    # ── Agent proposal ────────────────────────────────────────────────────────
    proposal  = _load_latest_proposal()
    agent_dir = "PENDING"
    agent_conf = 0
    age_min    = None
    is_fresh   = False
    agent_info: dict = {}

    if proposal is None:
        agent_info = {"status": "no_proposals", "direction": "PENDING"}
    else:
        age_min  = _proposal_age_minutes(proposal)
        is_fresh = age_min is not None and age_min <= FRESH_MAX_MINUTES
        p        = proposal.get("proposal", proposal)
        raw_dir  = p.get("direction", "NO_SIGNAL")
        agent_dir  = raw_dir if raw_dir in ("LONG", "SHORT") else "NEUTRAL"
        agent_conf = int(p.get("confidence", 0))
        agent_info = {
            "direction":  agent_dir,
            "confidence": agent_conf,
            "rr":         p.get("rr"),
            "logged_at":  proposal.get("logged_at", ""),
            "age_min":    round(age_min, 1) if age_min is not None else None,
            "is_fresh":   is_fresh,
            "session":    p.get("session", "?"),
            "key_factors": p.get("key_factors", [])[:3],
        }

    # ── Agreement logic ───────────────────────────────────────────────────────
    if rule_blackout:
        agreement          = "BLACKOUT"
        consensus_dir      = "NEUTRAL"
        consensus_conf     = 0
        consensus_label    = "🚫 Blackout — No Trading"
        note = "Economic event blackout in effect. Both engines suppressed."

    elif not is_fresh:
        agreement          = "PENDING"
        consensus_dir      = rule_dir
        consensus_conf     = rule_conf
        consensus_label    = "⏳ Waiting for AI Agent"
        age_str = f"{age_min:.0f}m ago" if age_min is not None else "never"
        note = (
            f"No fresh AI Agent proposal (last: {age_str}). "
            "Using rule engine only. Run Signal Agent to get consensus."
        )

    elif rule_dir == agent_dir and rule_dir in ("LONG", "SHORT"):
        agreement          = "FULL"
        consensus_dir      = rule_dir
        consensus_conf     = (rule_conf + agent_conf) // 2
        consensus_label    = f"✅ Full Agreement — {rule_dir}"
        note = (
            f"Both engines agree: {rule_dir}. "
            f"Rule: {rule_conf}% · Agent: {agent_conf}% → consensus {consensus_conf}%."
        )

    elif rule_dir in ("LONG", "SHORT") and agent_dir == "NEUTRAL":
        agreement          = "PARTIAL"
        consensus_dir      = "NEUTRAL"
        consensus_conf     = rule_conf // 2
        consensus_label    = f"⚠️ Partial — Rule says {rule_dir}, Agent NEUTRAL"
        note = (
            f"Rule engine sees {rule_dir} ({rule_conf}%) but AI Agent returned NEUTRAL. "
            "Wait for stronger confirmation before trading."
        )

    elif agent_dir in ("LONG", "SHORT") and rule_dir == "NEUTRAL":
        agreement          = "PARTIAL"
        consensus_dir      = "NEUTRAL"
        consensus_conf     = agent_conf // 2
        consensus_label    = f"⚠️ Partial — Agent says {agent_dir}, Rule NEUTRAL"
        note = (
            f"AI Agent sees {agent_dir} ({agent_conf}%) but rule engine returned NEUTRAL. "
            "Wait for stronger confirmation before trading."
        )

    elif rule_dir in ("LONG", "SHORT") and agent_dir in ("LONG", "SHORT") and rule_dir != agent_dir:
        agreement          = "CONFLICT"
        consensus_dir      = "NEUTRAL"
        consensus_conf     = 0
        consensus_label    = f"❌ Conflict — Rule {rule_dir} vs Agent {agent_dir}"
        note = (
            f"DIRECTIONAL CONFLICT: rule engine says {rule_dir}, AI Agent says {agent_dir}. "
            "Do NOT trade — wait for both engines to align."
        )

    else:
        agreement          = "NEUTRAL"
        consensus_dir      = "NEUTRAL"
        consensus_conf     = 0
        consensus_label    = "⚪ Both Neutral — No Setup"
        note = "Both engines agree: no tradeable setup at this time."

    result = {
        "validated_at":        datetime.now(IL_TZ).isoformat(),
        "rule":                rule_info,
        "agent":               agent_info,
        "agreement":           agreement,
        "consensus_direction": consensus_dir,
        "consensus_confidence": consensus_conf,
        "consensus_label":     consensus_label,
        "note":                note,
    }

    _log_validation(result)
    return result


# ─── Persistence ──────────────────────────────────────────────────────────────

def _log_validation(result: dict) -> None:
    _LOGS.mkdir(parents=True, exist_ok=True)
    entries: list = []
    if _VALIDATION_PATH.exists():
        try:
            entries = json.loads(_VALIDATION_PATH.read_text())
        except Exception:
            pass
    # Keep a compact summary for the log (strip verbose rule/agent sub-dicts)
    log_entry = {
        "ts":               result["validated_at"],
        "rule_dir":         result["rule"]["direction"],
        "rule_conf":        result["rule"]["confidence"],
        "agent_dir":        result["agent"].get("direction", "PENDING"),
        "agent_conf":       result["agent"].get("confidence", 0),
        "agreement":        result["agreement"],
        "consensus_dir":    result["consensus_direction"],
        "consensus_conf":   result["consensus_confidence"],
    }
    entries.append(log_entry)
    _VALIDATION_PATH.write_text(json.dumps(entries[-500:], indent=2))


def load_validation_log(n: int = 50) -> list[dict]:
    if not _VALIDATION_PATH.exists():
        return []
    try:
        data = json.loads(_VALIDATION_PATH.read_text())
        return data[-n:] if isinstance(data, list) else []
    except Exception:
        return []


def agreement_stats(n: int = 100) -> dict:
    """Aggregate statistics over the last N validation entries."""
    entries = load_validation_log(n)
    if not entries:
        return {"total": 0}

    total    = len(entries)
    by_type  = {}
    for e in entries:
        a = e.get("agreement", "UNKNOWN")
        by_type[a] = by_type.get(a, 0) + 1

    full_count     = by_type.get("FULL", 0)
    conflict_count = by_type.get("CONFLICT", 0)

    return {
        "total":            total,
        "full_pct":         round(full_count / total * 100, 1),
        "conflict_pct":     round(conflict_count / total * 100, 1),
        "by_type":          by_type,
        "full_long":  sum(1 for e in entries if e.get("agreement") == "FULL" and e.get("consensus_dir") == "LONG"),
        "full_short": sum(1 for e in entries if e.get("agreement") == "FULL" and e.get("consensus_dir") == "SHORT"),
    }

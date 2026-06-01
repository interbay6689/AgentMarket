"""
Config Optimizer — runs daily at 07:00 Israel time.
Uses Claude claude-sonnet-4-6 with adaptive thinking to review
cumulative trade performance and propose parameter adjustments.

Safety guarantees (enforced by agent_safety layer):
  - Dry-run first, ALWAYS
  - max 15% change per param per session
  - 24h cooldown per param
  - Minimum 20 closed trades before touching weights
  - Hard floors: min_rr ≥ 1.2, min_confidence ≥ 55

The optimizer is evidence-driven: it reasons about the last N trades
and changes only params with clear statistical support.
"""
from __future__ import annotations

import json
import os
import sys
import traceback
from datetime import datetime
from pathlib import Path

import pytz

_MD   = Path(__file__).resolve().parents[1]
_ROOT = _MD.parent
for p in (str(_MD), str(_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from agents.agent_tools import TOOL_SCHEMAS, execute_tool
from agents.agent_safety import check_min_trades

IL_TZ  = pytz.timezone("Asia/Jerusalem")
_LOGS  = _MD / "logs"
_STATUS_PATH = _LOGS / "agent_status.json"

AGENT_NAME = "config_optimizer"
MODEL      = "claude-sonnet-4-6"
MAX_TOOL_ROUNDS = 8

_ALLOWED_TOOLS = {
    "read_trade_log_summary",
    "read_all_configs",
    "read_insights_log",
    "read_fundamental_state",
    "read_statistical_edge",
    "run_dry_run",
    "update_fundamental_weights",
    "update_entry_filter",
    "update_agent_config",
    "get_config_history",
}

_SYSTEM_PROMPT = """You are the Config Optimizer for an NQ futures trading system.
You run once per day (07:00 Israel time) and adjust system parameters based on hard evidence.

## Decision Framework
1. Read current configs (read_all_configs)
2. Read trade performance (read_trade_log_summary) — need ≥ 20 closed trades to touch weights
3. Read insights from Analysis Agent (read_insights_log) — these are distilled learnings
4. Read recent config change history (get_config_history) — avoid reverting recent changes
5. Identify 1-3 specific, evidence-backed changes
6. ALWAYS dry-run first (run_dry_run) — never skip this step
7. If dry-run is valid → apply changes with clear reasoning
8. If dry-run rejects → do NOT apply, explain why

## What you can change
- weights.sentiment / macro / bonds / vix / sectors / mes  (sum must = 1.0)
- signal.min_confidence (55-85), signal.min_zone_score (4-9), signal.min_rr (1.2-3.0)
- session.rth_open, session.ny_morning, session.pm_session, session.power_hour, session.lunch_lull
- risk.atr_mult (1.0-2.5)
- ict.ote_fib_lo / ote_fib_hi, ict.eqh_eql_tolerance_pts

## Evidence requirements
- Weight changes: need ≥ 20 closed trades + clear correlation between a category and wins
- Threshold changes: need ≥ 15 closed trades + pattern showing the current threshold is wrong
- Session changes: need ≥ 10 closed trades + session-specific data

## Change limits (enforced by safety layer — you can't override these)
- Max 15% change per param in one session
- 24h cooldown between changes to the same param
- Hard floors: min_rr ≥ 1.2, min_confidence ≥ 55

## Output format
After applying (or deciding not to apply), give a summary:
- Changes applied: list with old→new and reasoning
- Changes rejected: why
- Observations: patterns worth monitoring next week
- Estimated impact: directional effect on win rate / R:R

Be conservative: 1 good change beats 3 marginal ones. Cite specific trade data."""


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


def _log_run(result: dict) -> None:
    path = _LOGS / "config_optimizer_log.json"
    entries = []
    if path.exists():
        try:
            entries = json.loads(path.read_text())
        except Exception:
            pass
    entries.append(result)
    path.write_text(json.dumps(entries[-100:], indent=2))


def _agent_tools() -> list[dict]:
    return [s for s in TOOL_SCHEMAS if s["name"] in _ALLOWED_TOOLS]


# ─── Main run ──────────────────────────────────────────────────────────────────

def run(force: bool = False) -> dict:
    """
    Run the Config Optimizer.
    - force=True: bypass minimum-trade check (for testing).
    Returns result dict.
    """
    _update_status("running")
    run_result: dict = {
        "started_at": datetime.now(IL_TZ).isoformat(),
        "agent":      AGENT_NAME,
        "model":      MODEL,
    }

    try:
        # Quick pre-check: enough trades?
        if not force:
            ok, reason = check_min_trades("weight")
            if not ok:
                run_result.update({"outcome": "skipped", "reason": reason})
                _update_status("done")
                _log_run(run_result)
                return run_result

        # Lazy import
        try:
            import anthropic as _anthropic
        except ImportError:
            run_result.update({"outcome": "error",
                                "reason": "anthropic package not installed — run: pip install anthropic"})
            _update_status("error", error="anthropic not installed")
            _log_run(run_result)
            return run_result

        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            run_result.update({"outcome": "error", "reason": "ANTHROPIC_API_KEY not set"})
            _update_status("error", error="ANTHROPIC_API_KEY not set")
            _log_run(run_result)
            return run_result

        client = _anthropic.Anthropic(api_key=api_key)

        today_str  = datetime.now(IL_TZ).strftime("%Y-%m-%d %H:%M")
        user_msg = (
            f"Daily optimization run — {today_str} Israel time.\n\n"
            "Please review the system's performance and apply any evidence-backed parameter changes.\n"
            "Remember: dry-run FIRST, then apply only if valid.\n"
            "Start by reading current configs and trade performance."
        )

        messages: list[dict] = [
            {"role": "user", "content": user_msg},
        ]

        tools          = _agent_tools()
        changes_applied = 0
        dry_runs_run    = 0
        tool_calls      = 0
        final_text      = ""

        for _ in range(MAX_TOOL_ROUNDS + 1):
            response = client.messages.create(
                model=MODEL,
                max_tokens=2000,
                thinking={"type": "adaptive"},
                system=[{
                    "type": "text",
                    "text": _SYSTEM_PROMPT,
                    "cache_control": {"type": "ephemeral"},
                }],
                tools=tools,
                messages=messages,
            )

            messages.append({"role": "assistant", "content": response.content})

            if response.stop_reason == "end_turn":
                final_text = next(
                    (b.text for b in response.content
                     if hasattr(b, "text") and b.type == "text"), ""
                )
                break

            if response.stop_reason != "tool_use":
                break

            tool_results = []
            for block in response.content:
                if block.type != "tool_use":
                    continue
                tool_calls += 1
                result = execute_tool(block.name, block.input if block.input else {})

                if block.name == "run_dry_run":
                    dry_runs_run += 1
                elif block.name in (
                    "update_fundamental_weights",
                    "update_entry_filter",
                    "update_agent_config",
                ):
                    if isinstance(result, dict):
                        applied = result.get("applied", 0) or (1 if result.get("success") else 0)
                        changes_applied += applied

                tool_results.append({
                    "type":        "tool_result",
                    "tool_use_id": block.id,
                    "content":     json.dumps(result, default=str)[:4000],
                })

            messages.append({"role": "user", "content": tool_results})

        run_result.update({
            "outcome":         "done",
            "changes_applied": changes_applied,
            "dry_runs_run":    dry_runs_run,
            "tool_calls":      tool_calls,
            "summary":         final_text[:800],
            "finished_at":     datetime.now(IL_TZ).isoformat(),
        })
        _update_status("done")

    except Exception as exc:
        err = traceback.format_exc()
        run_result.update({"outcome": "error", "error": str(exc)})
        _update_status("error", error=err[:500])

    _log_run(run_result)
    return run_result

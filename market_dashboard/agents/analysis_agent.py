"""
Analysis Agent — runs every 60 minutes.
1. Calls evaluate_open_trades() to auto-close any expired trades.
2. Uses Claude claude-sonnet-4-6 to analyze recently closed trades and
   write actionable insights to logs/insights_log.json.

The insights feed directly into Config Optimizer's daily decisions.
"""
from __future__ import annotations

import json
import os
import sys
import traceback
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import pytz

_MD   = Path(__file__).resolve().parents[1]
_ROOT = _MD.parent
for p in (str(_MD), str(_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

import anthropic

from agents.agent_tools import TOOL_SCHEMAS, execute_tool
from modules.trade_tracker import evaluate_open_trades, load_trade_log

IL_TZ  = pytz.timezone("Asia/Jerusalem")
_LOGS  = _MD / "logs"
_STATUS_PATH = _LOGS / "agent_status.json"

AGENT_NAME = "analysis_agent"
MODEL      = "claude-sonnet-4-6"
MAX_TOOL_ROUNDS = 5

_ALLOWED_TOOLS = {
    "read_trade_log_summary",
    "read_fundamental_state",
    "read_statistical_edge",
    "read_insights_log",
    "write_insight",
}

_SYSTEM_PROMPT = """You are the Analysis Agent for an NQ futures trading system.
Your role: evaluate recently closed trades and extract actionable learning patterns.

## Your job each hour
1. Read the trade log summary — focus on trades closed in the last 2-4 hours
2. For each recently closed trade (WIN/LOSS/PARTIAL), diagnose:
   - Root cause of the outcome (which factor was decisive?)
   - Factor quality assessment: were the signals actually aligned?
   - What should change in the future?
3. Write 1-3 high-quality insights (not noise) using write_insight

## Insight format (required fields)
- trade_id:        string — trade UUID
- outcome:         "WIN" | "LOSS" | "PARTIAL"
- root_cause:      string — primary reason for outcome (1 sentence)
- factor_quality:  dict — {"htf": "aligned|conflicting|neutral", "zone": ..., "delta": ..., "session": ...}
- recommendation:  string — specific actionable change (e.g. "Raise min_zone_score to 7 on Fridays")
- confidence:      int 1-10 — how confident in this analysis
- pattern:         string (optional) — repeating pattern across trades

## Rules
- Only write insights for trades with REAL data (not hypothetical)
- If there are no recently closed trades → write one insight summarizing overall patterns
- Be specific: cite actual values (e.g. "zone_score was 5 but trade failed — floor is 6")
- Do not repeat insights already in insights_log for the same trade_id
- Maximum 3 insights per run to stay focused"""


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
    path = _LOGS / "analysis_agent_log.json"
    entries = []
    if path.exists():
        try:
            entries = json.loads(path.read_text())
        except Exception:
            pass
    entries.append(result)
    path.write_text(json.dumps(entries[-200:], indent=2))


def _get_recently_closed(hours: int = 4) -> pd.DataFrame:
    """Return trades closed within the last N hours."""
    df = load_trade_log()
    if df.empty:
        return df
    closed = df[df["outcome"].isin(["WIN", "LOSS", "PARTIAL"])].copy()
    if closed.empty:
        return closed
    # Parse exit_time if available, else timestamp
    for col in ("exit_time", "timestamp"):
        if col in closed.columns:
            try:
                closed["_dt"] = pd.to_datetime(closed[col], errors="coerce", utc=True)
                closed = closed.dropna(subset=["_dt"])
                if not closed.empty:
                    cutoff = datetime.now(pytz.utc) - timedelta(hours=hours)
                    return closed[closed["_dt"] >= cutoff].drop(columns=["_dt"])
            except Exception:
                pass
    return closed.tail(5)  # fallback: last 5


def _agent_tools() -> list[dict]:
    return [s for s in TOOL_SCHEMAS if s["name"] in _ALLOWED_TOOLS]


# ─── Main run ──────────────────────────────────────────────────────────────────

def run() -> dict:
    """
    Run the Analysis Agent.
    1. Auto-evaluate open trades (no Claude cost).
    2. Call Claude to analyze recent closed trades and write insights.
    """
    _update_status("running")
    run_result: dict = {
        "started_at": datetime.now(IL_TZ).isoformat(),
        "agent":      AGENT_NAME,
        "model":      MODEL,
    }

    try:
        # Step 1: auto-evaluate (free, no API)
        eval_count = 0
        try:
            eval_count = evaluate_open_trades()
            if not isinstance(eval_count, int):
                eval_count = 0
        except Exception as e:
            run_result["eval_warning"] = str(e)

        run_result["trades_evaluated"] = eval_count

        # Step 2: check how many recently closed trades exist
        recent = _get_recently_closed(hours=4)
        run_result["recent_closed"] = len(recent)

        # Check API key
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            run_result.update({"outcome": "error", "reason": "ANTHROPIC_API_KEY not set"})
            _update_status("error", error="ANTHROPIC_API_KEY not set")
            _log_run(run_result)
            return run_result

        client = anthropic.Anthropic(api_key=api_key)

        # Build context message
        ctx_lines = [
            f"## Analysis Task — {datetime.now(IL_TZ).strftime('%Y-%m-%d %H:%M')} Israel Time",
            f"- Auto-evaluated {eval_count} previously open trade(s)",
            f"- Recently closed trades (last 4h): {len(recent)}",
        ]
        if not recent.empty:
            ctx_lines.append("\n## Recent closed trades:")
            for _, row in recent.iterrows():
                ctx_lines.append(
                    f"  • {row.get('id','?')[:8]} | {row.get('direction','?')} "
                    f"| outcome={row.get('outcome','?')} "
                    f"| zone_score={row.get('nearest_zone_score','?')} "
                    f"| htf={row.get('htf_bias','?')} "
                    f"| session={row.get('session_name','?')} "
                    f"| actual_rr={row.get('actual_rr','?')}"
                )
        ctx_lines.append(
            "\nUse read_trade_log_summary to get full stats, then write insights for recent trades."
        )

        messages: list[dict] = [
            {"role": "user", "content": "\n".join(ctx_lines)},
        ]

        tools            = _agent_tools()
        insights_written = 0
        tool_calls       = 0
        final_text       = ""

        for _ in range(MAX_TOOL_ROUNDS + 1):
            response = client.messages.create(
                model=MODEL,
                max_tokens=1500,
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
                if block.name == "write_insight" and result.get("success"):
                    insights_written += 1
                tool_results.append({
                    "type":        "tool_result",
                    "tool_use_id": block.id,
                    "content":     json.dumps(result, default=str)[:4000],
                })

            messages.append({"role": "user", "content": tool_results})

        run_result.update({
            "outcome":         "done",
            "insights_written": insights_written,
            "tool_calls":       tool_calls,
            "final_text":       final_text[:400],
            "finished_at":      datetime.now(IL_TZ).isoformat(),
        })
        _update_status("done")

    except Exception as exc:
        err = traceback.format_exc()
        run_result.update({"outcome": "error", "error": str(exc)})
        _update_status("error", error=err[:500])

    _log_run(run_result)
    return run_result

"""
Signal Agent — uses Claude (claude-haiku-4-5) with tool_use to generate
trade proposals by cross-referencing technical + fundamental state.

Flow:
  1. Load indicator_state.json (pre-computed by indicators_agent)
  2. Call Claude with all TOOL_SCHEMAS, providing indicator snapshot in context
  3. Claude calls read_* tools to gather data it needs
  4. Claude calls write_trade_proposal when it finds a valid setup
  5. Result written to agent_proposals.json
"""
from __future__ import annotations

import json
import os
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

import pytz

_MD   = Path(__file__).resolve().parents[1]
_ROOT = _MD.parent
for p in (str(_MD), str(_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from agents.agent_tools import TOOL_SCHEMAS, execute_tool
from agents.indicators_agent import load_state as load_indicator_state

IL_TZ  = pytz.timezone("Asia/Jerusalem")
_LOGS  = _MD / "logs"
_STATUS_PATH = _LOGS / "agent_status.json"

AGENT_NAME = "signal_agent"
MODEL      = "claude-haiku-4-5-20251001"
MAX_TOOL_ROUNDS = 6

# Only the tools Signal Agent is allowed to use
_ALLOWED_TOOLS = {
    "read_current_price",
    "read_technical_state",
    "read_fundamental_state",
    "read_trade_log_summary",
    "read_statistical_edge",
    "write_trade_proposal",
}

_SYSTEM_PROMPT = """You are the Signal Agent for an NQ futures day trading system.
Your role: analyze the current market state and generate high-probability trade proposals.

## Scoring System (100 points total)
- HTF Bias confirmation:  20 pts
- MTF Zone quality:       20 pts  (zone score ≥ 7 = full pts)
- Delta confirmation:     15 pts
- Session timing:         10 pts  (NY Morning / RTH Open = max)
- Stacked imbalances:     10 pts
- Fundamental sentiment:  15 pts
- Statistical edge:       10 pts

## Entry criteria (all required)
- Total score ≥ 65 pts
- Zone score ≥ 6 (price near OB/FVG/Key Level cluster)
- R:R ≥ 1.5
- No active Lunch Lull session
- HTF bias not opposing entry

## Proposal format (write_trade_proposal fields)
- direction:   "LONG" or "SHORT"
- entry:       float (specific price)
- stop:        float (ATR-based or below/above key level)
- target:      float (nearest opposing zone or +ATR×2.5)
- rr:          float (calculated)
- confidence:  int (0-100, your scoring result)
- reasoning:   string (≤200 words, cite specific indicators)
- key_factors: list of 3-5 strings (most important factors)
- session:     string
- htf_bias:    string

## Rules
- If score < 65: respond with NO_SIGNAL and reasoning — do NOT write a proposal
- If session = "lunch_lull": always NO_SIGNAL
- Never fabricate prices — always read_current_price first
- Check both technical AND fundamental alignment
- One proposal per run maximum

Respond concisely. When done (proposal written or NO_SIGNAL decided), output your final summary."""


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
    path = _LOGS / "signal_agent_log.json"
    entries = []
    if path.exists():
        try:
            entries = json.loads(path.read_text())
        except Exception:
            pass
    entries.append(result)
    path.write_text(json.dumps(entries[-200:], indent=2))


# ─── Tool filter ───────────────────────────────────────────────────────────────

def _agent_tools() -> list[dict]:
    return [s for s in TOOL_SCHEMAS if s["name"] in _ALLOWED_TOOLS]


# ─── Main run ──────────────────────────────────────────────────────────────────

def run(force: bool = False) -> dict:
    """
    Run the Signal Agent.
    - Loads indicator state (pre-computed).
    - If composite_score < 65 and not forced, skips (saves API cost).
    - Otherwise calls Claude with tool_use loop.
    Returns result dict with signal or no_signal.
    """
    _update_status("running")
    run_result: dict = {
        "started_at": datetime.now(IL_TZ).isoformat(),
        "agent":      AGENT_NAME,
        "model":      MODEL,
    }

    try:
        # Load pre-computed indicator state
        indicator_state = load_indicator_state()
        composite       = indicator_state.get("composite_score", {})
        should_escalate = composite.get("escalate", False) or force

        if not should_escalate:
            run_result.update({
                "outcome":  "skipped",
                "reason":   f"Composite score {composite.get('total', 0)} < 65 — no setup",
                "score":    composite.get("total", 0),
            })
            _update_status("done")
            _log_run(run_result)
            return run_result

        # Lazy import — keeps the module loadable even without the package installed
        try:
            import anthropic as _anthropic
        except ImportError:
            run_result.update({"outcome": "error",
                                "reason": "anthropic package not installed — run: pip install anthropic"})
            _update_status("error", error="anthropic not installed")
            _log_run(run_result)
            return run_result

        # Check API key
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            run_result.update({"outcome": "error", "reason": "ANTHROPIC_API_KEY not set"})
            _update_status("error", error="ANTHROPIC_API_KEY not set")
            _log_run(run_result)
            return run_result

        client = _anthropic.Anthropic(api_key=api_key)

        # Build context snapshot for the user message
        ctx_lines = [
            "## Current Market Snapshot (pre-computed, may be up to 5 minutes old)",
            f"- Session: {indicator_state.get('session', '?')}",
            f"- Price: {indicator_state.get('price', '?')}",
            f"- HTF Bias: {indicator_state.get('htf_bias', '?')} "
              f"(strength: {indicator_state.get('htf_strength', '?')})",
            f"- Nearest zone score: {indicator_state.get('nearest_zone_score', 0)}/10",
            f"- Delta signal: {indicator_state.get('delta', {}).get('delta_signal', '?')}",
            f"- VWAP zone: {indicator_state.get('vwap', {}).get('zone', '?')}",
            f"- Stacked imbalances: {indicator_state.get('stacked', {}).get('count', 0)}",
            f"- Quick score: {composite.get('total', '?')}/100 "
              f"(components: {composite.get('components', {})})",
            "",
            "Use your tools to gather live data and generate a trade proposal if conditions are met.",
        ]

        messages: list[dict] = [
            {"role": "user", "content": "\n".join(ctx_lines)},
        ]

        tools      = _agent_tools()
        tool_calls = 0
        proposal_written = False
        final_text = ""

        # Tool-use loop
        for round_n in range(MAX_TOOL_ROUNDS + 1):
            response = client.messages.create(
                model=MODEL,
                max_tokens=1024,
                system=[{
                    "type": "text",
                    "text": _SYSTEM_PROMPT,
                    "cache_control": {"type": "ephemeral"},
                }],
                tools=tools,
                messages=messages,
            )

            # Append assistant turn
            messages.append({"role": "assistant", "content": response.content})

            if response.stop_reason == "end_turn":
                final_text = next(
                    (b.text for b in response.content
                     if hasattr(b, "text") and b.type == "text"),
                    "",
                )
                break

            if response.stop_reason != "tool_use":
                break

            # Execute each tool call
            tool_results = []
            for block in response.content:
                if block.type != "tool_use":
                    continue
                tool_calls += 1
                result = execute_tool(block.name, block.input if block.input else {})
                if block.name == "write_trade_proposal" and result.get("success"):
                    proposal_written = True
                tool_results.append({
                    "type":        "tool_result",
                    "tool_use_id": block.id,
                    "content":     json.dumps(result, default=str)[:4000],
                })

            messages.append({"role": "user", "content": tool_results})

        no_signal = (
            not proposal_written and (
                "NO_SIGNAL" in final_text.upper() or
                "no setup" in final_text.lower() or
                "no signal" in final_text.lower()
            )
        )

        run_result.update({
            "outcome":          "proposal" if proposal_written else "no_signal",
            "proposal_written": proposal_written,
            "tool_calls":       tool_calls,
            "final_text":       final_text[:500],
            "composite_score":  composite.get("total", 0),
            "finished_at":      datetime.now(IL_TZ).isoformat(),
        })

        _update_status("done")

    except Exception as exc:
        err = traceback.format_exc()
        run_result.update({"outcome": "error", "error": str(exc)})
        _update_status("error", error=err[:500])

    _log_run(run_result)
    return run_result

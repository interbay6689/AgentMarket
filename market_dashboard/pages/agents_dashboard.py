"""
Agents Dashboard — monitor and control the multi-agent AI system.
Sprint 2: Orchestrator controls + live run buttons.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytz
import streamlit as st

_MD   = Path(__file__).resolve().parents[1]
_ROOT = _MD.parent
for p in (str(_MD), str(_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from agents.agent_safety import (
    get_change_history,
    dry_run,
    revert_last_change,
    PARAMETER_LIMITS,
    CHANGE_RULES,
)
from agents import orchestrator

IL_TZ      = pytz.timezone("Asia/Jerusalem")
_LOGS      = _MD / "logs"
_AGENT_CFG = _LOGS / "agent_config.json"
_AGENT_ST  = _LOGS / "agent_status.json"


# ─── helpers ──────────────────────────────────────────────────────────────────

def _load_agent_status() -> dict:
    """Load persisted agent status. Returns dict keyed by agent name."""
    if _AGENT_ST.exists():
        try:
            return json.loads(_AGENT_ST.read_text())
        except Exception:
            pass
    defaults = {
        "orchestrator":    {"enabled": False, "last_run": None, "status": "idle", "runs": 0},
        "signal_agent":    {"enabled": False, "last_run": None, "status": "idle", "runs": 0},
        "indicators":      {"enabled": False, "last_run": None, "status": "idle", "runs": 0},
        "analysis_agent":  {"enabled": False, "last_run": None, "status": "idle", "runs": 0},
        "config_optimizer":{"enabled": False, "last_run": None, "status": "idle", "runs": 0},
    }
    return defaults


def _save_agent_status(data: dict) -> None:
    _LOGS.mkdir(parents=True, exist_ok=True)
    _AGENT_ST.write_text(json.dumps(data, indent=2))


def _load_agent_config() -> dict:
    if _AGENT_CFG.exists():
        try:
            return json.loads(_AGENT_CFG.read_text())
        except Exception:
            pass
    return {}


def _status_chip(status: str) -> str:
    colors = {
        "idle":    ("⚪", "#888"),
        "running": ("🟡", "#f0a500"),
        "done":    ("🟢", "#00c851"),
        "error":   ("🔴", "#ff4444"),
    }
    icon, color = colors.get(status, ("⚪", "#888"))
    return f'<span style="color:{color};font-weight:600">{icon} {status.upper()}</span>'


# ─── tabs ─────────────────────────────────────────────────────────────────────

def _tab_agent_status() -> None:
    st.subheader("Agent Status")

    # ── Orchestrator controls ──────────────────────────────────────────────────
    orch_status = orchestrator.get_status()
    is_running  = orch_status.get("_scheduler_running", False)
    has_apsched = orch_status.get("_apscheduler_available", False)

    with st.container(border=True):
        c1, c2, c3, c4 = st.columns([2, 1, 1, 1])
        c1.markdown("**🧠 Orchestrator Scheduler**")
        c2.markdown(
            _status_chip("running" if is_running else "idle"),
            unsafe_allow_html=True,
        )
        if not has_apsched:
            c3.caption("apscheduler not installed")
        elif is_running:
            if c3.button("⏹ Stop", key="btn_stop"):
                orchestrator.stop()
                st.rerun()
        else:
            if c3.button("▶ Start", key="btn_start", type="primary"):
                orchestrator.start(indicators_interval_minutes=5)
                st.rerun()

        if c4.button("🔄 Run Now", key="btn_run_now", help="Run indicators + signal once"):
            with st.spinner("Running indicators + signal agent…"):
                result = orchestrator.run_once("indicators_signal")
            outcome = result.get("outcome", result.get("status", "?"))
            if "error" in result:
                st.error(f"Error: {result['error']}")
            else:
                st.success(f"Done — {outcome}")
            st.rerun()

    st.divider()

    # ── Per-agent status table ─────────────────────────────────────────────────
    agent_info = {
        "orchestrator":    ("🧠 Orchestrator",    "Schedules and coordinates all agents"),
        "signal_agent":    ("🎯 Signal Agent",     "Claude claude-haiku-4-5 — MTF confluence, 100pt scoring"),
        "indicators":      ("📊 Indicators Agent", "VWAP, EQH/EQL, OTE, delta, order flow (no API cost)"),
        "analysis_agent":  ("🔍 Analysis Agent",   "Per-trade evaluation — Sprint 3"),
        "config_optimizer":("⚙️ Config Optimizer", "Parameter optimization — Sprint 4"),
    }

    status_data = _load_agent_status()
    # Merge live scheduler data
    for key in agent_info:
        if key in orch_status and isinstance(orch_status[key], dict):
            status_data[key] = {**status_data.get(key, {}), **orch_status[key]}

    cols = st.columns([2, 3, 1, 1, 2])
    cols[0].markdown("**Agent**")
    cols[1].markdown("**Role**")
    cols[2].markdown("**Status**")
    cols[3].markdown("**Runs**")
    cols[4].markdown("**Last Run**")
    st.divider()

    for agent_key, (display_name, role) in agent_info.items():
        s = status_data.get(agent_key, {})
        c0, c1, c2, c3, c4 = st.columns([2, 3, 1, 1, 2])
        c0.markdown(f"**{display_name}**")
        c1.caption(role)
        c2.markdown(_status_chip(s.get("status", "idle")), unsafe_allow_html=True)
        c3.write(s.get("runs", 0))
        last = s.get("last_run")
        c4.caption(last[:16] if last else "—")

    st.divider()

    # ── Recent orchestrator log ────────────────────────────────────────────────
    with st.expander("📋 Orchestrator Activity Log (last 10)"):
        log_entries = orchestrator.get_orch_log(10)
        if log_entries:
            for entry in reversed(log_entries):
                ts   = str(entry.get("timestamp", ""))[:16]
                job  = entry.get("job", entry.get("event", "?"))
                stat = entry.get("status", entry.get("event", "?"))
                score = entry.get("score", "")
                score_str = f" | score={score}" if score != "" else ""
                err  = entry.get("error", "")
                err_str = f" ⚠ {err[:60]}" if err else ""
                st.caption(f"{ts} — **{job}** → {stat}{score_str}{err_str}")
        else:
            st.caption("No activity yet — start the scheduler or run manually.")


def _tab_config_history() -> None:
    st.subheader("Configuration Change History")

    hist = get_change_history(50)
    if not hist:
        st.info("No configuration changes recorded yet.")
        return

    rows = []
    for e in reversed(hist):
        ts = e.get("timestamp", "")[:16]
        rows.append({
            "Time":      ts,
            "Agent":     e.get("agent", "—"),
            "Parameter": e.get("param", "—"),
            "Old":       e.get("old_value", "—"),
            "New":       e.get("new_value", "—"),
            "Reason":    e.get("reasoning", "—"),
        })

    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)

    if st.button("⏪ Revert Last Change", type="secondary"):
        ok, msg = revert_last_change()
        if ok:
            st.success(msg)
            st.rerun()
        else:
            st.error(msg)


def _tab_dry_run() -> None:
    st.subheader("Dry-Run Proposed Changes")
    st.caption("Test parameter changes without applying them.")

    with st.form("dry_run_form"):
        raw = st.text_area(
            "Proposed changes (JSON list)",
            value=json.dumps([
                {"param": "weights.sentiment", "new_value": 0.30,
                 "current_value": 0.28, "reasoning": "Sentiment has been accurate this week"},
            ], indent=2),
            height=200,
        )
        submitted = st.form_submit_button("Run Simulation")

    if submitted:
        try:
            changes = json.loads(raw)
        except json.JSONDecodeError as e:
            st.error(f"Invalid JSON: {e}")
            return

        result = dry_run(changes)

        col1, col2, col3 = st.columns(3)
        col1.metric("Valid", result["apply_count"], delta=None)
        col2.metric("Rejected", result["reject_count"], delta=None)
        col3.metric("Warnings", len(result["warnings"]))

        if result["valid"]:
            st.markdown("**✅ Valid changes:**")
            st.dataframe(pd.DataFrame(result["valid"]), use_container_width=True, hide_index=True)

        if result["invalid"]:
            st.markdown("**❌ Invalid changes:**")
            st.dataframe(pd.DataFrame(result["invalid"]), use_container_width=True, hide_index=True)

        if result["warnings"]:
            st.markdown("**⚠️ Warnings (cooldown):**")
            for w in result["warnings"]:
                st.warning(f"{w['param']}: {w['warning']}")


def _tab_param_limits() -> None:
    st.subheader("Parameter Limits Reference")

    rows = [
        {"Parameter": p, "Min": lo, "Max": hi}
        for p, (lo, hi) in PARAMETER_LIMITS.items()
    ]
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)

    st.divider()
    st.markdown("**Change Rules**")
    rules_df = pd.DataFrame([
        {"Rule": k, "Value": v}
        for k, v in CHANGE_RULES.items()
    ])
    st.dataframe(rules_df, use_container_width=True, hide_index=True)


# ─── main ─────────────────────────────────────────────────────────────────────

def app() -> None:
    st.title("🤖 AI Agents")

    tab1, tab2, tab3, tab4 = st.tabs([
        "🟢 Agent Status",
        "📋 Change History",
        "🧪 Dry Run",
        "📏 Parameter Limits",
    ])

    with tab1:
        _tab_agent_status()
    with tab2:
        _tab_config_history()
    with tab3:
        _tab_dry_run()
    with tab4:
        _tab_param_limits()

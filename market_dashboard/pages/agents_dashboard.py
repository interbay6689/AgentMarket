"""
Agents Dashboard — monitor and control the multi-agent AI system.
Sprint 1: Agent Status tab only.
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

    agent_info = {
        "orchestrator":    ("🧠 Orchestrator",    "Coordinates all agents, makes final trade decisions"),
        "signal_agent":    ("🎯 Signal Agent",     "Generates MTF confluence signals (100pt scoring)"),
        "indicators":      ("📊 Indicators Agent", "VWAP, EQH/EQL, OTE, order flow metrics"),
        "analysis_agent":  ("🔍 Analysis Agent",   "Per-trade evaluation and insight generation"),
        "config_optimizer":("⚙️ Config Optimizer", "Optimizes parameters based on trade outcomes"),
    }

    status_data = _load_agent_status()

    cols = st.columns([2, 2, 1, 1, 1, 1])
    cols[0].markdown("**Agent**")
    cols[1].markdown("**Role**")
    cols[2].markdown("**Status**")
    cols[3].markdown("**Runs**")
    cols[4].markdown("**Last Run**")
    cols[5].markdown("**Enable**")
    st.divider()

    changed = False
    for agent_key, (display_name, role) in agent_info.items():
        s = status_data.get(agent_key, {})
        c0, c1, c2, c3, c4, c5 = st.columns([2, 2, 1, 1, 1, 1])
        c0.markdown(f"**{display_name}**")
        c1.caption(role)
        c2.markdown(_status_chip(s.get("status", "idle")), unsafe_allow_html=True)
        c3.write(s.get("runs", 0))

        last = s.get("last_run")
        c4.caption(last[:16] if last else "—")

        new_enabled = c5.toggle(
            "on", value=s.get("enabled", False),
            key=f"toggle_{agent_key}",
            label_visibility="collapsed",
        )
        if new_enabled != s.get("enabled", False):
            status_data[agent_key]["enabled"] = new_enabled
            changed = True

    if changed:
        _save_agent_status(status_data)
        st.toast("Agent settings saved", icon="✅")

    st.divider()
    st.caption(
        "⚠️ Agents are currently in **Sprint 1 (passive mode)** — "
        "they read data and log analysis but do not auto-execute trades. "
        "Full agent scheduling arrives in Sprint 2."
    )


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

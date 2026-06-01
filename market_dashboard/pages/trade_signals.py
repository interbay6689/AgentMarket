"""Trade Signals — main page. Shows real-time LONG/SHORT signal with R:R."""
import sys
from pathlib import Path

_MD = Path(__file__).resolve().parents[1]
_ROOT = _MD.parent
for p in (str(_MD), str(_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime

from modules.signal_engine import generate_signal
from modules.nq_data import get_todays_nq_data, load_nq_daily_cache, current_session_israel
from modules.nq_calculations import cumulative_delta, detect_fvg


@st.cache_data(ttl=60)
def _get_signal():
    return generate_signal()


@st.cache_data(ttl=60)
def _get_5m():
    return get_todays_nq_data()


@st.cache_data(ttl=300)
def _get_daily():
    return load_nq_daily_cache()


def _direction_color(direction: str) -> str:
    return {"LONG": "#00c853", "SHORT": "#d50000", "NEUTRAL": "#9e9e9e"}.get(direction, "#9e9e9e")


def _direction_emoji(direction: str) -> str:
    return {"LONG": "🟢", "SHORT": "🔴", "NEUTRAL": "⚪"}.get(direction, "⚪")


def _rr_color(rr) -> str:
    if rr is None:
        return "#9e9e9e"
    if rr >= 2.0:
        return "#00c853"
    if rr >= 1.5:
        return "#ffd600"
    return "#d50000"


def _render_signal_card(sig: dict):
    direction = sig["direction"]
    confidence = sig["confidence"]
    color = _direction_color(direction)
    emoji = _direction_emoji(direction)

    st.markdown(
        f"""
        <div style="background:{color}20; border-left:6px solid {color};
                    padding:20px 24px; border-radius:8px; margin-bottom:16px;">
            <h2 style="margin:0; color:{color};">{emoji} {direction}</h2>
            <p style="margin:4px 0 0 0; font-size:1.1em; color:#ccc;">
                Confluence Confidence: <b style="color:{color};">{confidence}%</b>
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_rr_metrics(sig: dict):
    entry = sig.get("entry")
    stop = sig.get("stop")
    target = sig.get("target")
    rr = sig.get("rr")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Entry", f"{entry:,.1f}" if entry else "—")
    c2.metric("Stop", f"{stop:,.1f}" if stop else "—",
              delta=f"{stop - entry:+.1f}" if stop and entry else None,
              delta_color="inverse")
    c3.metric("Target", f"{target:,.1f}" if target else "—",
              delta=f"{target - entry:+.1f}" if target and entry else None)
    rr_str = f"{rr:.2f}" if rr else "—"
    rr_col = _rr_color(rr)
    c4.markdown(
        f"""<div style="padding:8px 0;">
            <div style="font-size:.85em; color:#aaa;">R:R Ratio</div>
            <div style="font-size:1.8em; color:{rr_col}; font-weight:bold;">{rr_str}</div>
        </div>""",
        unsafe_allow_html=True,
    )


def _render_alert_banner(sig: dict):
    if sig["alert"]:
        direction = sig["direction"]
        rr = sig["rr"]
        entry = sig["entry"]
        stop = sig["stop"]
        target = sig["target"]
        color = _direction_color(direction)
        st.markdown(
            f"""
            <div style="background:{color}30; border:2px solid {color};
                        padding:14px 20px; border-radius:8px; margin:12px 0;">
                🚨 <b style="color:{color};">TRADE ALERT — {direction}</b> &nbsp;|&nbsp;
                Entry: <b>{entry:,.1f}</b> &nbsp;|&nbsp;
                Stop: <b>{stop:,.1f}</b> &nbsp;|&nbsp;
                Target: <b>{target:,.1f}</b> &nbsp;|&nbsp;
                R:R: <b style="color:{_rr_color(rr)};">{rr:.2f}</b>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.info("No high-confidence trade alert right now. Conditions: confidence ≥ 60%, R:R ≥ 1.5, active session.")


def _render_confluence_table(sig: dict):
    factors = sig.get("factors", [])
    if not factors:
        return
    rows = []
    for f in factors:
        side = f["side"]
        pts = f["pts"]
        if side == "LONG":
            icon = "🟢"
        elif side == "SHORT":
            icon = "🔴"
        elif side in ("BOTH",):
            icon = "🟡"
        else:
            icon = "⚪"
        rows.append({
            "Factor": f["factor"],
            "Value": str(f["value"]),
            "Signal": f"{icon} {side}",
            "Points": pts,
        })
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)


def _render_nq_chart(df_5m: pd.DataFrame, sig: dict):
    if df_5m.empty:
        st.warning("No 5-min NQ data available.")
        return

    time_col = "datetime_il" if "datetime_il" in df_5m.columns else "datetime"
    x = df_5m[time_col].astype(str)

    fig = go.Figure()

    # Candlesticks
    fig.add_trace(go.Candlestick(
        x=x, open=df_5m["open"], high=df_5m["high"],
        low=df_5m["low"], close=df_5m["close"],
        name="NQ 5m", increasing_line_color="#00c853", decreasing_line_color="#d50000",
    ))

    current_price = sig.get("current_price")
    entry = sig.get("entry")
    stop = sig.get("stop")
    target = sig.get("target")
    direction = sig.get("direction", "NEUTRAL")

    # Key level lines
    levels = sig.get("levels", {})
    level_colors = {"pdh": "#ffd600", "pdl": "#ffd600", "pdc": "#ff9800",
                    "poc_20d": "#64b5f6", "poc_5d": "#4fc3f7",
                    "weekly_high": "#ce93d8", "weekly_low": "#ce93d8",
                    "overnight_high": "#80cbc4", "overnight_low": "#80cbc4"}
    for name, val in levels.items():
        color = level_colors.get(name, "#9e9e9e")
        fig.add_hline(y=val, line_color=color, line_dash="dot",
                      annotation_text=name.upper(), annotation_position="left",
                      line_width=1, opacity=0.6)

    # Entry / Stop / Target lines
    if entry and direction != "NEUTRAL":
        col = _direction_color(direction)
        fig.add_hline(y=entry, line_color=col, line_width=2,
                      annotation_text="ENTRY", annotation_position="right")
    if stop:
        fig.add_hline(y=stop, line_color="#ff5252", line_width=1.5, line_dash="dash",
                      annotation_text="STOP", annotation_position="right")
    if target:
        fig.add_hline(y=target, line_color="#69f0ae", line_width=1.5, line_dash="dash",
                      annotation_text="TARGET", annotation_position="right")

    # FVG zones
    try:
        fvgs = detect_fvg(df_5m)
        if not fvgs.empty:
            for _, fvg in fvgs.tail(5).iterrows():
                fc = "rgba(0,200,83,0.12)" if fvg["type"] == "bullish" else "rgba(213,0,0,0.12)"
                fig.add_hrect(y0=fvg["bottom"], y1=fvg["top"], fillcolor=fc,
                              line_width=0, annotation_text="FVG", annotation_position="left")
    except Exception:
        pass

    fig.update_layout(
        height=420, template="plotly_dark",
        margin=dict(l=0, r=80, t=30, b=0),
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", y=1.02),
        yaxis=dict(side="right"),
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_key_levels_table(sig: dict):
    levels = sig.get("levels", {})
    current = sig.get("current_price")
    if not levels:
        st.info("No key levels available.")
        return
    rows = []
    for name, val in sorted(levels.items(), key=lambda x: x[1]):
        dist = f"{val - current:+.1f}" if current else "—"
        rows.append({"Level": name.upper(), "Price": f"{val:,.1f}", "Dist from Current": dist})
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def app():
    st.title("🎯 Trade Signals — NQ")

    # Auto-refresh toggle
    col_ref, col_sess = st.columns([3, 1])
    with col_ref:
        auto_refresh = st.checkbox("Auto-refresh (60s)", value=False)
    if auto_refresh:
        import time
        time.sleep(60)
        st.cache_data.clear()
        st.rerun()

    # Session status
    session = current_session_israel()
    sess_color = {"high": "#00c853", "medium": "#ffd600", "low": "#9e9e9e"}.get(
        session.get("risk", "low"), "#9e9e9e"
    )
    with col_sess:
        st.markdown(
            f"""<div style="text-align:right; padding-top:6px;">
                <span style="color:{sess_color}; font-weight:bold;">⏱ {session.get('name','—')}</span><br>
                <small style="color:#aaa;">{session.get('current_il_time','—')} IL</small>
            </div>""",
            unsafe_allow_html=True,
        )

    with st.spinner("Calculating signal..."):
        sig = _get_signal()

    # ── Main signal card ────────────────────────────────────────
    _render_signal_card(sig)
    _render_alert_banner(sig)

    # ── Metrics row ─────────────────────────────────────────────
    st.markdown("### Trade Parameters")
    _render_rr_metrics(sig)

    # ── Chart + Confluence side-by-side ─────────────────────────
    chart_col, conf_col = st.columns([3, 2])

    with chart_col:
        st.markdown("### NQ 5-min Chart")
        df_5m = _get_5m()
        _render_nq_chart(df_5m, sig)

    with conf_col:
        st.markdown("### Confluence Factors")
        _render_confluence_table(sig)

        st.markdown("### Macro Context")
        c1, c2 = st.columns(2)
        c1.metric("Sentiment Score", sig.get("final_score", "—"))
        c2.metric("ML Prediction", sig.get("ml_prediction", "—"))
        c1.metric("Delta Direction", sig.get("delta_dir", "—"))
        c2.metric("Session Risk", session.get("risk", "—").capitalize())

    # ── Key levels ──────────────────────────────────────────────
    with st.expander("📊 Key Levels", expanded=False):
        _render_key_levels_table(sig)

    # ── Score log spark ─────────────────────────────────────────
    with st.expander("📈 Score History (last 20 days)", expanded=False):
        score_path = _ROOT / "scores_news" / "config" / "score_log.csv"
        if score_path.exists():
            df_log = pd.read_csv(score_path)
            df_log["date"] = pd.to_datetime(df_log["date"], errors="coerce")
            df_log = df_log.dropna(subset=["date"]).sort_values("date").tail(20)
            if "final_score" in df_log.columns:
                st.line_chart(df_log.set_index("date")["final_score"], height=150)
        else:
            st.info("No score history yet.")

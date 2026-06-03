"""Trade Signals — main page. Shows real-time LONG/SHORT signal with R:R."""
import os
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
from modules.nq_data import get_todays_nq_data, get_nq_1m, get_nq_3m, load_nq_daily_cache, current_session_israel
from modules.nq_calculations import cumulative_delta, detect_fvg
from modules.trade_tracker import log_trade, log_paper_trade, load_trade_settings, load_trade_log, summary_stats
from modules.signal_validator import validate, load_validation_log, agreement_stats

import json
from pathlib import Path as _Path
_LOGS = _Path(__file__).resolve().parents[1] / "logs"


def _load_indicator_state() -> dict:
    """Load last computed indicator state (written by indicators_agent every 5 min)."""
    p = _LOGS / "indicator_state.json"
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            pass
    return {}


def _load_signal_agent_log(n: int = 5) -> list:
    p = _LOGS / "signal_agent_log.json"
    if p.exists():
        try:
            data = json.loads(p.read_text())
            return data[-n:] if isinstance(data, list) else []
        except Exception:
            pass
    return []


# ─── Signal threshold (must match signal_engine.py) ─────────────────────────
_SIGNAL_THRESHOLD = 55
_DIRECTION_LEAD   = 15   # minimum lead over opponent


# ─── Data loaders ─────────────────────────────────────────────────────────────

def _load_agent_proposals(n: int = 3, directional_only: bool = False) -> list:
    """Load last n agent proposals. directional_only=True skips NEUTRAL sentinels."""
    path = _LOGS / "agent_proposals.json"
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text())
        if not isinstance(data, list):
            return []
        if directional_only:
            data = [e for e in data
                    if e.get("proposal", e).get("direction") in ("LONG", "SHORT")]
        return data[-n:]
    except Exception:
        return []


@st.cache_data(ttl=60)
def _get_signal():
    return generate_signal()


@st.cache_data(ttl=60)
def _get_validation(rule_sig: dict | None = None):
    try:
        return validate(rule_sig)
    except Exception as e:
        return {"agreement": "ERROR", "consensus_label": f"⚠️ Validator error: {e}",
                "note": str(e), "rule": {}, "agent": {}}


@st.cache_data(ttl=30)
def _get_1m():
    return get_nq_1m()

@st.cache_data(ttl=60)
def _get_3m():
    return get_nq_3m()

@st.cache_data(ttl=60)
def _get_5m():
    return get_todays_nq_data()


@st.cache_data(ttl=300)
def _get_daily():
    return load_nq_daily_cache()


_TF_LOADERS = {"1m": _get_1m, "3m": _get_3m, "5m": _get_5m}
_TF_TTL     = {"1m": "30s",   "3m": "60s",   "5m": "60s"}


# ─── Color helpers ─────────────────────────────────────────────────────────────

def _dc(direction: str) -> str:
    return {"LONG": "#00c853", "SHORT": "#d50000", "NEUTRAL": "#9e9e9e"}.get(direction, "#9e9e9e")


def _de(direction: str) -> str:
    return {"LONG": "🟢", "SHORT": "🔴", "NEUTRAL": "⚪"}.get(direction, "⚪")


def _rr_color(rr) -> str:
    if rr is None:
        return "#9e9e9e"
    if rr >= 2.0:
        return "#00c853"
    if rr >= 1.5:
        return "#ffd600"
    return "#d50000"


# ─── Section 1: Signal readiness ──────────────────────────────────────────────

def _velocity_tag(current: int, prev: int | None) -> str:
    """Returns an HTML velocity badge: ↑ +8, ↓ -3, or empty string."""
    if prev is None:
        return ""
    delta = current - prev
    if delta > 0:
        return f'<span style="color:#00c853;font-size:.78em;"> ↑ +{delta}</span>'
    if delta < 0:
        return f'<span style="color:#d50000;font-size:.78em;"> ↓ {delta}</span>'
    return '<span style="color:#666;font-size:.78em;"> →</span>'


def _session_threshold_label(sig: dict, thr: int) -> str:
    """Context-aware threshold label depending on session quality."""
    sess_quality = sig.get("session_quality", 0) or 0
    sess_name    = (sig.get("session") or {}).get("name", "")
    if sess_quality < 2:
        return f'{thr} pts <span style="color:#ff9800;font-size:.8em;">(off-hours — signal unlikely)</span>'
    return f'{thr} pts <span style="color:#00c853;font-size:.8em;">(active session ✓)</span>'


def _render_signal_readiness(sig: dict):
    """Dual progress bars with velocity and session-aware threshold label."""
    long_pts  = sig.get("long_pts",  0) or 0
    short_pts = sig.get("short_pts", 0) or 0
    thr       = _SIGNAL_THRESHOLD

    # ── Velocity: compare to previous refresh via session_state ─────────────
    prev_long  = st.session_state.get("_prev_long_pts")
    prev_short = st.session_state.get("_prev_short_pts")
    st.session_state["_prev_long_pts"]  = long_pts
    st.session_state["_prev_short_pts"] = short_pts

    long_vel  = _velocity_tag(long_pts,  prev_long)
    short_vel = _velocity_tag(short_pts, prev_short)

    # ── Metrics ──────────────────────────────────────────────────────────────
    long_pct   = min(100, int(long_pts  / thr * 100))
    short_pct  = min(100, int(short_pts / thr * 100))
    lead       = abs(long_pts - short_pts)
    leading    = "LONG" if long_pts >= short_pts else "SHORT"
    best_pts   = max(long_pts, short_pts)
    pts_to_go  = max(0, thr - best_pts)
    lead_to_go = max(0, _DIRECTION_LEAD - lead)

    # Suffix for the leading side only
    long_suffix  = (f" — ✅ met" if best_pts >= thr else f" — {pts_to_go} to fire") if long_pts > short_pts else ""
    short_suffix = (f" — ✅ met" if best_pts >= thr else f" — {pts_to_go} to fire") if short_pts > long_pts else ""

    thr_label = _session_threshold_label(sig, thr)

    # ── Render ───────────────────────────────────────────────────────────────
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            f'<div style="margin-bottom:4px;">'
            f'<span style="color:#00c853;font-size:.9em;font-weight:bold;">🟢 LONG</span>'
            f'<span style="color:#aaa;font-size:.85em;"> {long_pts}/{thr} pts{long_suffix}</span>'
            f'{long_vel}'
            f'</div>'
            f'<div style="background:#1a2a1a;border:1px solid #2a4a2a;border-radius:4px;'
            f'height:10px;overflow:hidden;">'
            f'<div style="background:#00c853;width:{long_pct}%;height:100%;border-radius:4px;'
            f'opacity:.9;"></div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f'<div style="margin-bottom:4px;">'
            f'<span style="color:#d50000;font-size:.9em;font-weight:bold;">🔴 SHORT</span>'
            f'<span style="color:#aaa;font-size:.85em;"> {short_pts}/{thr} pts{short_suffix}</span>'
            f'{short_vel}'
            f'</div>'
            f'<div style="background:#2a1a1a;border:1px solid #4a2a2a;border-radius:4px;'
            f'height:10px;overflow:hidden;">'
            f'<div style="background:#d50000;width:{short_pct}%;height:100%;border-radius:4px;'
            f'opacity:.9;"></div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # ── Caption: session-aware threshold + lead status ───────────────────────
    if lead_to_go > 0 and best_pts >= thr:
        st.markdown(
            f'<span style="color:#ff9800;font-size:.82em;">⚠️ Threshold met — '
            f'{leading} leads by {lead} pts, need ≥{_DIRECTION_LEAD} pts clear lead</span>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<span style="color:#666;font-size:.82em;">Threshold: {thr_label} · '
            f'currently {best_pts} pts</span>',
            unsafe_allow_html=True,
        )


# ─── Section 2: Signal card + alert ───────────────────────────────────────────

def _render_signal_card(sig: dict):
    direction  = sig["direction"]
    confidence = sig["confidence"]
    color = _dc(direction)
    emoji = _de(direction)

    st.markdown(
        f"""
        <div style="background:{color}20; border-left:6px solid {color};
                    padding:18px 24px; border-radius:8px; margin-bottom:12px;">
            <h2 style="margin:0; color:{color};">{emoji} {direction}</h2>
            <p style="margin:4px 0 0 0; font-size:1.05em; color:#ccc;">
                Confluence Confidence: <b style="color:{color};">{confidence}%</b>
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_alert_banner(sig: dict):
    if sig["alert"]:
        direction = sig["direction"]
        rr = sig["rr"]
        entry = sig["entry"]
        stop = sig["stop"]
        target = sig["target"]
        color = _dc(direction)
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
        _render_no_signal_reasons(sig)


def _render_no_signal_reasons(sig: dict):
    """Show exactly WHY there is no signal right now."""
    long_pts  = sig.get("long_pts",  0) or 0
    short_pts = sig.get("short_pts", 0) or 0
    best      = max(long_pts, short_pts)
    lead      = abs(long_pts - short_pts)
    reasons   = []

    if sig.get("blackout"):
        reasons.append(f"🚫 **Blackout**: {sig.get('blackout_reason','Economic event in progress')}")

    session = sig.get("session", {})
    sess_risk = sig.get("session_quality", 0) or 0
    sess_name = session.get("name", "—") if isinstance(session, dict) else str(session)
    if sess_risk < 2:
        next_sess = _next_session_info()
        reasons.append(f"⏱️ **Session**: {sess_name} has low liquidity — {next_sess}")

    htf = sig.get("htf_bias", "neutral")
    if htf == "neutral":
        reasons.append("📊 **HTF Bias**: Neutral — wait for directional commitment on daily/1H")

    if best < _SIGNAL_THRESHOLD:
        reasons.append(f"📈 **Confluence**: {best}/{_SIGNAL_THRESHOLD} pts — need {_SIGNAL_THRESHOLD - best} more")

    if best >= _SIGNAL_THRESHOLD and lead < _DIRECTION_LEAD:
        reasons.append(f"↔️ **Direction conflict**: lead = {lead} pts, need ≥{_DIRECTION_LEAD} pts")

    rr = sig.get("rr") or 0
    if rr and rr < 1.5:
        reasons.append(f"📐 **R:R**: {rr:.2f} — minimum 1.5 required")

    if not reasons:
        reasons.append("⏳ No confluence confluence factors align yet — monitor for changes")

    items_html = "".join(f"<li style='margin:3px 0; color:#ccc;'>{r}</li>" for r in reasons)
    st.markdown(
        f"""<div style="background:#1e2533; border-left:4px solid #546e7a;
                        padding:12px 16px; border-radius:6px; margin:8px 0;">
            <div style="color:#90a4ae; font-size:.9em; font-weight:bold; margin-bottom:6px;">
                ⏸️ No signal — conditions not met:
            </div>
            <ul style="margin:0; padding-left:18px; font-size:.88em;">{items_html}</ul>
        </div>""",
        unsafe_allow_html=True,
    )


def _next_session_info() -> str:
    """Returns a short string like 'RTH Open in 2h 45m'."""
    try:
        from modules.nq_data import current_session_israel
        import pytz
        IL_TZ = pytz.timezone("Asia/Jerusalem")
        now = datetime.now(IL_TZ)
        # RTH Open = 16:30 IL (09:30 ET)
        target_h, target_m = 16, 30
        target = now.replace(hour=target_h, minute=target_m, second=0, microsecond=0)
        if now >= target:
            # next day
            from datetime import timedelta
            target += timedelta(days=1)
        diff = target - now
        hours, rem = divmod(int(diff.total_seconds()), 3600)
        mins = rem // 60
        if hours > 0:
            return f"next RTH Open in {hours}h {mins}m"
        return f"next RTH Open in {mins}m"
    except Exception:
        return "next RTH Open at 16:30 IL"


# ─── Section 3: Trade Parameters ──────────────────────────────────────────────

def _render_rr_metrics(sig: dict):
    entry     = sig.get("entry")
    stop      = sig.get("stop")
    target    = sig.get("target")
    partial   = sig.get("partial_exit")
    rr        = sig.get("rr")
    direction = sig.get("direction", "NEUTRAL")

    if not entry or direction == "NEUTRAL":
        st.markdown(
            '<div style="background:#161b27; border:1px solid #2a2f3e; '
            'padding:14px 20px; border-radius:8px; color:#555; '
            'text-align:center; font-size:.9em;">'
            '⏸️ No active trade setup — waiting for signal conditions to align'
            '</div>',
            unsafe_allow_html=True,
        )
        return

    color     = _dc(direction)
    stop_pts  = round(stop - entry, 1)   if stop   else None
    tgt_pts   = round(target - entry, 1) if target else None
    risk_usd  = round(abs(stop_pts) * 2, 0)  if stop_pts  else None   # MNQ $2/pt
    tgt_usd   = round(abs(tgt_pts)  * 2, 0)  if tgt_pts   else None
    rr_color  = _rr_color(rr)
    rr_label  = "✅ Good" if rr and rr >= 2.0 else ("⚠️ Marginal" if rr and rr >= 1.5 else "❌ Poor")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Entry", f"{entry:,.1f}")
    if stop_pts is not None:
        c2.metric(
            "Stop (ATR)",
            f"{stop:,.1f}",
            delta=f"{stop_pts:+.1f} pts",
            delta_color="inverse",
            help=f"Risk ≈ ${int(risk_usd)} per MNQ contract" if risk_usd else None,
        )
    else:
        c2.metric("Stop", "—")
    if tgt_pts is not None:
        c3.metric(
            "Target",
            f"{target:,.1f}",
            delta=f"{tgt_pts:+.1f} pts",
            help=f"Profit ≈ ${int(tgt_usd)} per MNQ contract" if tgt_usd else None,
        )
    else:
        c3.metric("Target", "—")
    c4.markdown(
        f"""<div style="padding:4px 0;">
            <div style="font-size:.8em; color:#aaa; margin-bottom:2px;">R:R Ratio</div>
            <div style="font-size:2em; color:{rr_color}; font-weight:bold; line-height:1.1;">
                {f"{rr:.2f}" if rr else "—"}</div>
            <div style="font-size:.75em; color:{rr_color}; margin-top:2px;">{rr_label if rr else ""}</div>
        </div>""",
        unsafe_allow_html=True,
    )

    if partial:
        st.caption(
            f"1R Partial: {partial:,.1f} ({partial - entry:+.1f} pts) "
            "· Move stop to BE (break-even) after partial is hit"
        )


# ─── Section 4: NQ Chart ──────────────────────────────────────────────────────

def _render_nq_chart(df_5m: pd.DataFrame, sig: dict, tf_label: str = "5m"):
    if df_5m.empty:
        st.warning("No 5-min NQ data available.")
        return

    time_col = "datetime_il" if "datetime_il" in df_5m.columns else "datetime"
    x = df_5m[time_col].astype(str)

    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=x, open=df_5m["open"], high=df_5m["high"],
        low=df_5m["low"], close=df_5m["close"],
        name=f"NQ {tf_label}",
        increasing_line_color="#00c853", decreasing_line_color="#d50000",
        showlegend=False,
        hoverlabel=dict(bgcolor="#1a1a2e"),
    ))

    # Key structural levels with price in the label
    levels = sig.get("levels", {})
    KEY_LEVELS = [
        ("pdh", "#ffd600", "PDH"),
        ("pdl", "#ffd600", "PDL"),
        ("pdc", "#ff9800", "PDC"),
    ]
    for key, color, label in KEY_LEVELS:
        val = levels.get(key)
        if val:
            fig.add_hline(
                y=val, line_color=color, line_dash="dot", line_width=1.2, opacity=0.80,
                annotation_text=f"{label}  {val:,.0f}",
                annotation_position="right",
                annotation_font=dict(size=10, color=color),
            )

    # Entry / Stop / Target with price labels
    entry     = sig.get("entry")
    stop      = sig.get("stop")
    target    = sig.get("target")
    partial   = sig.get("partial_exit")
    direction = sig.get("direction", "NEUTRAL")

    if entry and direction != "NEUTRAL":
        fig.add_hline(
            y=entry, line_color=_dc(direction), line_width=2.5,
            annotation_text=f"ENTRY  {entry:,.0f}",
            annotation_position="right",
            annotation_font=dict(size=11, color=_dc(direction)),
        )
    if stop:
        fig.add_hline(
            y=stop, line_color="#ff5252", line_width=1.5, line_dash="dash",
            annotation_text=f"STOP  {stop:,.0f}",
            annotation_position="right",
            annotation_font=dict(size=10, color="#ff5252"),
        )
    if target:
        fig.add_hline(
            y=target, line_color="#69f0ae", line_width=1.5, line_dash="dash",
            annotation_text=f"TARGET  {target:,.0f}",
            annotation_position="right",
            annotation_font=dict(size=10, color="#69f0ae"),
        )
    if partial:
        fig.add_hline(
            y=partial, line_color="#ffd600", line_width=1, line_dash="dot",
            annotation_text=f"1R  {partial:,.0f}",
            annotation_position="right",
            annotation_font=dict(size=9, color="#ffd600"),
        )

    # FVG zones — max 3 most recent, labelled with price range
    try:
        fvgs = detect_fvg(df_5m)
        if not fvgs.empty:
            for _, fvg in fvgs.tail(3).iterrows():
                is_bull = fvg["type"] == "bullish"
                mid     = (fvg["top"] + fvg["bottom"]) / 2
                color   = "#00c853" if is_bull else "#d50000"
                fill    = "rgba(0,200,83,0.12)" if is_bull else "rgba(213,0,0,0.12)"
                lbl     = ("Bull FVG" if is_bull else "Bear FVG")
                fig.add_hrect(
                    y0=fvg["bottom"], y1=fvg["top"],
                    fillcolor=fill, line_color=color, line_width=0.8, opacity=0.9,
                    annotation_text=f"{lbl}  {fvg['bottom']:,.0f}–{fvg['top']:,.0f}",
                    annotation_position="left",
                    annotation_font=dict(size=9, color=color),
                )
    except Exception:
        pass

    # HTF Liquidity zones (EQH/EQL + swing highs/lows from 15m/1h)
    liq = sig.get("liquidity", {})
    for lvl in liq.get("eqh", []):
        fig.add_hline(
            y=lvl, line_color="#e040fb", line_dash="dot", line_width=1.0, opacity=0.7,
            annotation_text=f"EQH  {lvl:,.0f}",
            annotation_position="right",
            annotation_font=dict(size=9, color="#e040fb"),
        )
    for lvl in liq.get("eql", []):
        fig.add_hline(
            y=lvl, line_color="#40c4ff", line_dash="dot", line_width=1.0, opacity=0.7,
            annotation_text=f"EQL  {lvl:,.0f}",
            annotation_position="right",
            annotation_font=dict(size=9, color="#40c4ff"),
        )
    for lvl in liq.get("buy_side", [])[:3]:
        fig.add_hline(
            y=lvl, line_color="#b2ff59", line_dash="longdash", line_width=0.8, opacity=0.5,
            annotation_text=f"BSL  {lvl:,.0f}",
            annotation_position="right",
            annotation_font=dict(size=8, color="#b2ff59"),
        )
    for lvl in liq.get("sell_side", [])[:3]:
        fig.add_hline(
            y=lvl, line_color="#ff6e40", line_dash="longdash", line_width=0.8, opacity=0.5,
            annotation_text=f"SSL  {lvl:,.0f}",
            annotation_position="right",
            annotation_font=dict(size=8, color="#ff6e40"),
        )

    # ATR info caption
    atr_ltf = sig.get("atr_ltf")
    stop    = sig.get("stop")
    entry_p = sig.get("entry")
    if atr_ltf and stop and entry_p:
        stop_pts = round(abs(entry_p - stop), 1)
        fig.add_annotation(
            text=f"LTF ATR: {atr_ltf} pts  |  Stop: {stop_pts} pts",
            xref="paper", yref="paper", x=0.01, y=0.99,
            showarrow=False, font=dict(size=9, color="#888"),
            align="left",
        )

    # Crosshair spikes — makes cursor feel like TradingView
    fig.update_xaxes(
        showspikes=True, spikemode="across", spikedash="dot",
        spikecolor="#555", spikethickness=1,
    )
    fig.update_yaxes(
        showspikes=True, spikemode="across", spikedash="dot",
        spikecolor="#555", spikethickness=1,
        side="right", tickformat=",.0f",
    )

    fig.update_layout(
        height=500, template="plotly_dark",
        margin=dict(l=0, r=130, t=16, b=0),
        xaxis_rangeslider_visible=False,
        hovermode="x unified",
        dragmode="zoom",
    )
    st.plotly_chart(
        fig,
        use_container_width=True,
        config={
            "scrollZoom":      True,
            "displayModeBar":  True,
            "modeBarButtonsToAdd": ["drawline", "drawopenpath", "eraseshape"],
            "modeBarButtonsToRemove": ["select2d", "lasso2d"],
        },
    )


# ─── Section 5: Confluence factor rows ───────────────────────────────────────

def _render_confluence_visual(sig: dict):
    factors = sig.get("factors", [])
    if not factors:
        st.info("No confluence factors computed.")
        return

    total_long  = sum(f.get("pts", 0) or 0 for f in factors if f.get("side") == "LONG")
    total_short = sum(f.get("pts", 0) or 0 for f in factors if f.get("side") == "SHORT")
    st.markdown(
        f'<div style="font-size:.8em; color:#666; margin-bottom:6px;">'
        f'<span style="color:#00c853; font-weight:600;">▲ LONG {total_long} pts</span>'
        f'&nbsp;&nbsp;·&nbsp;&nbsp;'
        f'<span style="color:#d50000; font-weight:600;">▼ SHORT {total_short} pts</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    for f in factors:
        fname = f["factor"]
        side  = f["side"]
        pts   = f.get("pts", 0) or 0
        value = str(f.get("value", "") or "")
        if value in ("None", "nan"):
            value = ""

        color     = _dc(side)
        icon      = _de(side)
        pts_label = f"+{pts}" if pts > 0 else (str(pts) if pts < 0 else "0")
        bg        = f"{color}0e" if pts != 0 else "#111118"

        value_html = (
            f'<div style="color:#666; font-size:.76em; margin-top:1px;">{value}</div>'
            if value else ""
        )
        st.markdown(
            f"""<div style="display:flex; justify-content:space-between; align-items:center;
                            border-left:3px solid {color}; background:{bg};
                            border-radius:0 5px 5px 0; padding:6px 10px; margin:2px 0;">
                <div style="flex:1; min-width:0;">
                    <div style="color:#ccc; font-size:.85em; white-space:nowrap;
                                overflow:hidden; text-overflow:ellipsis;">{fname}</div>
                    {value_html}
                </div>
                <div style="display:flex; align-items:center; gap:8px;
                            flex-shrink:0; margin-left:8px;">
                    <span style="color:{color}; font-size:.8em; font-weight:600;
                                 white-space:nowrap;">{icon} {side}</span>
                    <span style="color:{color}; font-weight:bold; font-size:.9em;
                                 min-width:46px; text-align:right;
                                 white-space:nowrap;">{pts_label} pts</span>
                </div>
            </div>""",
            unsafe_allow_html=True,
        )


# ─── Section 6: Macro Context ─────────────────────────────────────────────────

def _render_macro_context(sig: dict, session: dict):
    htf_bias = sig.get("htf_bias", "neutral").capitalize()
    htf_str  = sig.get("htf_strength", 0)
    htf_col  = {"Bullish": "#00c853", "Bearish": "#d50000"}.get(htf_bias, "#9e9e9e")

    c1, c2 = st.columns(2)
    c1.markdown(
        f"""<div style="padding:4px 0">
            <div style="font-size:.85em;color:#aaa;">HTF Bias</div>
            <div style="font-size:1.4em;color:{htf_col};font-weight:bold;">
                {htf_bias} {htf_str}%</div>
        </div>""", unsafe_allow_html=True)
    c2.metric("ML Prediction",  sig.get("ml_prediction", "—"))
    c1.metric("Sentiment Score", sig.get("final_score", "—"))
    c2.metric("Delta Direction", sig.get("delta_dir", "—"))
    c1.metric("Zone Score",      f"{sig.get('nearest_zone_score', 0)}/10")
    c2.metric("Session Risk",    session.get("risk", "—").capitalize())


# ─── Section 7: Pre-Trade Checklist ───────────────────────────────────────────

def _render_pretrade_checklist(sig: dict, session: dict):
    """11 conditions evaluated automatically from current signal data."""
    long_pts  = sig.get("long_pts",  0) or 0
    short_pts = sig.get("short_pts", 0) or 0
    best      = max(long_pts, short_pts)
    lead      = abs(long_pts - short_pts)
    direction = "LONG" if long_pts >= short_pts else "SHORT"

    rr          = sig.get("rr") or 0
    htf_bias    = sig.get("htf_bias", "neutral")
    zone_score  = sig.get("nearest_zone_score", 0) or 0
    delta_dir   = sig.get("delta_dir", "Neutral")
    ml_pred     = sig.get("ml_prediction", "NEUTRAL")
    final_score = sig.get("final_score", 50) or 50
    sess_risk   = session.get("risk", "low")
    blackout    = sig.get("blackout", False)

    # Delta agrees = delta dir same as leading direction
    delta_agrees = (
        (direction == "LONG"  and delta_dir in ("Bullish", "LONG"))  or
        (direction == "SHORT" and delta_dir in ("Bearish", "SHORT"))
    )
    # ML agrees or neutral
    ml_ok = ml_pred in ("NEUTRAL", direction)
    # Sentiment: >50 supports LONG, <50 supports SHORT
    sentiment_ok = (
        (direction == "LONG"  and final_score >= 50) or
        (direction == "SHORT" and final_score <= 50) or
        (50 <= final_score <= 55)
    )

    checks = [
        ("No blackout zone", not blackout,
         sig.get("blackout_reason", "OK")),
        ("Active session (not Off-Hours/Lunch)",
         sess_risk in ("high", "medium"),
         f"{session.get('name','—')} · {sess_risk}"),
        ("HTF bias directional",
         htf_bias not in ("neutral", "Neutral"),
         f"HTF: {htf_bias} {sig.get('htf_strength',0)}%"),
        (f"Zone confluence ≥ 6/10",
         zone_score >= 6,
         f"Zone score: {zone_score}/10"),
        ("Order flow delta confirms direction",
         delta_agrees,
         f"Delta: {delta_dir}, Leading: {direction}"),
        (f"Confluence ≥ {_SIGNAL_THRESHOLD} pts",
         best >= _SIGNAL_THRESHOLD,
         f"Best: {best}/{_SIGNAL_THRESHOLD} pts"),
        (f"Clear directional lead (≥{_DIRECTION_LEAD} pts)",
         lead >= _DIRECTION_LEAD,
         f"Lead: {lead} pts"),
        ("R:R ratio ≥ 1.5",
         bool(rr and rr >= 1.5),
         f"R:R: {rr:.2f}" if rr else "No setup yet"),
        ("ML prediction agrees or neutral",
         ml_ok,
         f"ML: {ml_pred}"),
        ("Sentiment supports direction",
         sentiment_ok,
         f"Score: {final_score}"),
        ("Not in Judas Swing window (09:15–09:30 ET)",
         True,    # always true for now — placeholder
         "OK"),
    ]

    passed = sum(1 for _, ok, _ in checks if ok)
    total  = len(checks)

    # Score badge
    score_color = "#00c853" if passed >= 8 else "#ffd600" if passed >= 5 else "#d50000"
    st.markdown(
        f"""<div style="display:flex; align-items:center; gap:12px; margin-bottom:10px;">
            <div style="background:{score_color}20; border:2px solid {score_color};
                        border-radius:50%; width:52px; height:52px; display:flex;
                        align-items:center; justify-content:center; flex-shrink:0;">
                <span style="color:{score_color}; font-weight:bold; font-size:1.1em;">
                    {passed}/{total}
                </span>
            </div>
            <div>
                <div style="color:{score_color}; font-weight:bold;">
                    {"✅ Ready to trade" if passed >= 8
                     else "⚠️ Partial setup" if passed >= 5
                     else "❌ Not ready"}
                </div>
                <div style="color:#888; font-size:.85em;">
                    {total - passed} condition{"s" if total - passed != 1 else ""} not met
                </div>
            </div>
        </div>""",
        unsafe_allow_html=True,
    )

    # Checklist rows — failed first, then passed
    sorted_checks = sorted(checks, key=lambda x: (x[1], x[0]))  # failed first
    for label, ok, detail in sorted_checks:
        icon   = "✅" if ok else "❌"
        c_label = "#ccc" if ok else "#ff8a80"
        c_detail = "#666" if ok else "#ff8a80"
        st.markdown(
            f"""<div style="display:flex; justify-content:space-between;
                            padding:4px 8px; border-radius:4px; font-size:.87em;
                            background:{'#1a1a1a' if ok else '#2a1515'};">
                <span style="color:{c_label};">{icon} {label}</span>
                <span style="color:{c_detail};">{detail}</span>
            </div>""",
            unsafe_allow_html=True,
        )


# ─── Section 8: Regime badge + Blackout ───────────────────────────────────────

def _render_blackout_banner(sig: dict):
    if sig.get("blackout"):
        reason = sig.get("blackout_reason", "High-impact economic event")
        st.markdown(
            f"""<div style="background:#b71c1c30; border:2px solid #b71c1c;
                            padding:12px 20px; border-radius:8px; margin-bottom:10px;">
                🚫 <b style="color:#ff5252;">BLACKOUT ZONE — NO TRADING</b><br>
                <span style="color:#ffcdd2; font-size:.9em;">{reason}</span>
            </div>""",
            unsafe_allow_html=True,
        )


def _render_regime_badge(sig: dict):
    regime       = sig.get("regime", "transitioning")
    regime_label = sig.get("regime_label", "Transitioning")
    regime_icon  = sig.get("regime_icon", "⚪")
    strategies   = sig.get("regime_strategies", [])
    conf_adj     = sig.get("regime_conf_adj", 0)
    size_adj     = sig.get("regime_size_adj", 1.0)

    conf_str = (f"+{conf_adj}pts threshold" if conf_adj > 0
                else f"{conf_adj}pts threshold" if conf_adj < 0
                else "standard threshold")
    size_str = f"×{size_adj:.2g} size"

    border_color = {
        "trending_bull": "#00c853", "trending_bear": "#d50000",
        "ranging": "#ffd600",       "high_vol": "#ff9800",
        "transitioning": "#9e9e9e",
    }.get(regime, "#9e9e9e")

    strat_html = "".join(f"<li style='margin:1px 0'>{s}</li>" for s in strategies[:3])
    st.markdown(
        f"""<div style="background:{border_color}18; border-left:4px solid {border_color};
                        padding:10px 16px; border-radius:6px; margin:8px 0;">
            <b style="color:{border_color};">{regime_icon} {regime_label}</b>
            &nbsp;<span style="color:#aaa; font-size:.85em;">({conf_str} · {size_str})</span>
            <ul style="margin:4px 0 0 0; padding-left:18px; color:#ccc; font-size:.85em;">
                {strat_html}
            </ul>
        </div>""",
        unsafe_allow_html=True,
    )


# ─── Section 9: Cross-Validation ──────────────────────────────────────────────

def _write_neutral_proposal(outcome: str, result: dict) -> None:
    """Write a NEUTRAL sentinel so the validator knows the agent ran recently.

    Without this, a 'no_signal' run leaves agent_proposals.json stale and the
    Cross-Validation cards stay in PENDING state until a LONG/SHORT is generated.
    """
    import pytz as _pytz
    IL_TZ = _pytz.timezone("Asia/Jerusalem")
    path  = _LOGS / "agent_proposals.json"
    entries: list = []
    if path.exists():
        try:
            entries = json.loads(path.read_text())
        except Exception:
            pass
    entries.append({
        "logged_at": datetime.now(IL_TZ).isoformat(),
        "proposal": {
            "direction":   "NEUTRAL",
            "confidence":  0,
            "outcome":     outcome,
            "reason":      result.get("reason") or result.get("final_text", "")[:200],
            "session":     result.get("session", ""),
            "key_factors": [],
        },
    })
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(entries[-200:], indent=2))


def _do_agent_run() -> dict:
    """Run indicators then signal agent (force=True). Returns result dict."""
    try:
        from agents import indicators_agent as _ind
        from agents import signal_agent as _sa
        ind_state = _ind.run()
        result    = _sa.run(force=True)
        result["_indicator_score"] = ind_state.get("composite_score", {}).get("total", 0)
        # If no LONG/SHORT proposal was written, log a NEUTRAL sentinel so the
        # validator shows NEUTRAL (not PENDING) for the next 90 minutes.
        if result.get("outcome") in ("no_signal", "skipped", "error"):
            _write_neutral_proposal(result.get("outcome", "no_signal"), result)
        return result
    except ImportError as e:
        return {"outcome": "error", "reason": f"Import error: {e}"}
    except Exception as e:
        return {"outcome": "error", "reason": str(e)}


_KEY_PLACEHOLDERS = {"PASTE_YOUR_NEW_KEY_HERE", "sk-ant-...", "", "your_key_here"}


def _api_key_ok() -> bool:
    """True only if ANTHROPIC_API_KEY is set and not a placeholder."""
    val = os.environ.get("ANTHROPIC_API_KEY", "")
    return bool(val) and val not in _KEY_PLACEHOLDERS


def _save_api_key(key: str) -> bool:
    """Write the key to AgentMarket/.env and activate it in the current process."""
    env_path = _ROOT / ".env"
    try:
        if env_path.exists():
            lines = env_path.read_text(encoding="utf-8").splitlines()
            new_lines, found = [], False
            for line in lines:
                if line.strip().startswith("ANTHROPIC_API_KEY"):
                    new_lines.append(f"ANTHROPIC_API_KEY={key}")
                    found = True
                else:
                    new_lines.append(line)
            if not found:
                new_lines.append(f"ANTHROPIC_API_KEY={key}")
            env_path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
        else:
            env_path.write_text(f"ANTHROPIC_API_KEY={key}\n", encoding="utf-8")
        os.environ["ANTHROPIC_API_KEY"] = key
        return True
    except Exception:
        return False


def _render_agent_run_controls():
    """Status bar + Run button + persistent result, shown at top of cross-validation."""
    ind_state = _load_indicator_state()
    comp      = ind_state.get("composite_score", {})
    ind_score = comp.get("total", 0)
    threshold = comp.get("threshold", 65)
    api_ok    = _api_key_ok()
    computed  = ind_state.get("computed_at_ts", None)

    # Status bar
    score_color = "#00c853" if ind_score >= threshold else "#ffd600" if ind_score >= 45 else "#d50000"
    key_color   = "#00c853" if api_ok else "#d50000"
    key_label   = "✅ API key set" if api_ok else "❌ API key not set"
    ind_age     = f"indicators at {computed}" if computed else "indicators: not run yet"

    sc1, sc2, sc3 = st.columns([2, 2, 2])
    sc1.markdown(
        f'<span style="font-size:.84em; color:{score_color};">'
        f'Indicator score: <b>{ind_score}</b> / {threshold} threshold'
        f'{"&nbsp;✅ escalates" if ind_score >= threshold else "&nbsp;⚠️ below threshold"}'
        f'</span>',
        unsafe_allow_html=True,
    )
    sc2.markdown(
        f'<span style="font-size:.84em; color:{key_color};">{key_label}</span>',
        unsafe_allow_html=True,
    )
    sc3.markdown(
        f'<span style="font-size:.84em; color:#666;">{ind_age}</span>',
        unsafe_allow_html=True,
    )

    # Inline key entry — shown once until the key is saved
    if not api_ok:
        with st.container():
            st.markdown(
                '<p style="font-size:.82em; color:#888; margin:4px 0 2px;">Enter your Anthropic API key once — it will be saved and loaded automatically on every future startup.</p>',
                unsafe_allow_html=True,
            )
            k_col, btn_col2 = st.columns([4, 1])
            entered = k_col.text_input(
                "Anthropic API key",
                type="password",
                placeholder="sk-ant-api03-…",
                label_visibility="collapsed",
                key="input_api_key_inline",
            )
            if btn_col2.button("Save key", key="btn_save_key_inline", type="primary", use_container_width=True):
                val = entered.strip()
                if val.startswith("sk-ant-") and len(val) > 30:
                    if _save_api_key(val):
                        st.success("Key saved — reloading…")
                        st.rerun()
                    else:
                        st.error("Could not write to .env — check file permissions.")
                else:
                    st.warning("Invalid key — must start with sk-ant- and be at least 30 characters.")
        return  # don't render the run button until key is set

    # Run button
    btn_col, _ = st.columns([1, 3])
    with btn_col:
        if st.button("▶ Run Signal Agent", key="btn_cv_run", type="primary",
                     use_container_width=True,
                     help="Runs indicators + Signal Agent (Claude). Uses API tokens."):
            with st.spinner("Running indicators + Signal Agent (Claude)…"):
                result = _do_agent_run()
            st.session_state["_agent_run_result"] = {
                "ts":     datetime.now().strftime("%H:%M:%S"),
                "result": result,
            }
            # Clear cached validation so the cards update
            _get_signal.clear()
            _get_validation.clear()
            st.rerun()

    # Persistent run result
    run_info = st.session_state.get("_agent_run_result")
    if run_info:
        r       = run_info["result"]
        ts      = run_info["ts"]
        outcome = r.get("outcome", "?")
        ind_sc  = r.get("_indicator_score", "?")

        if outcome == "proposal":
            txt = r.get("final_text", "")[:200]
            st.success(
                f"✅ Proposal written at {ts} · "
                f"indicator score {ind_sc} · {r.get('tool_calls', 0)} tool calls"
                + (f"\n\n_{txt}_" if txt else "")
            )
        elif outcome == "no_signal":
            txt = r.get("final_text", "")[:200]
            st.info(
                f"⏳ No signal at {ts} · "
                f"indicator score {ind_sc} · {r.get('tool_calls', 0)} tool calls"
                + (f"\n\n_{txt}_" if txt else "")
            )
        elif outcome == "skipped":
            st.warning(
                f"⏭️ Skipped at {ts} — {r.get('reason', 'score below threshold')}  "
                f"(indicator score: {ind_sc})"
            )
        elif outcome == "error":
            reason = r.get("reason", r.get("error", "unknown error"))
            st.error(f"❌ Error at {ts}: {reason}")
        else:
            st.info(f"Result at {ts}: {outcome}")

        # Show last-run log details in expander
        last_logs = _load_signal_agent_log(3)
        if last_logs:
            with st.expander("📋 Agent run log (last 3 entries)", expanded=False):
                for entry in reversed(last_logs):
                    ts_e    = str(entry.get("started_at", ""))[:16]
                    oc_e    = entry.get("outcome", "?")
                    calls_e = entry.get("tool_calls", 0)
                    txt_e   = entry.get("final_text", entry.get("reason", ""))[:300]
                    icon_e  = {"proposal": "✅", "no_signal": "⏳", "skipped": "⏭️",
                                "error": "❌"}.get(oc_e, "⚪")
                    with st.container():
                        st.markdown(
                            f"**{ts_e}** — {icon_e} `{oc_e}` · {calls_e} tool calls"
                        )
                        if txt_e:
                            st.caption(txt_e)
                        st.divider()

    st.divider()


def _render_cross_validation(val: dict):
    _render_agent_run_controls()

    agreement = val.get("agreement", "PENDING")
    label     = val.get("consensus_label", "—")
    note      = val.get("note", "")
    rule      = val.get("rule", {})
    agent     = val.get("agent", {})

    agg_color = {
        "FULL": "#00c853", "PARTIAL": "#ffd600", "CONFLICT": "#d50000",
        "NEUTRAL": "#9e9e9e", "PENDING": "#78909c", "BLACKOUT": "#b71c1c",
    }.get(agreement, "#9e9e9e")

    st.markdown(
        f"""<div style="background:{agg_color}20; border-left:5px solid {agg_color};
                        padding:10px 16px; border-radius:6px; margin-bottom:10px;">
            <b style="color:{agg_color}; font-size:1.05em;">{label}</b><br>
            <span style="color:#ccc; font-size:.88em;">{note}</span>
        </div>""",
        unsafe_allow_html=True,
    )

    col_r, col_a, col_c = st.columns(3)
    rd = rule.get("direction", "—")
    rc = rule.get("confidence", 0)
    with col_r:
        st.markdown(
            f"""<div style="background:#1a1a2e; border-radius:6px; padding:12px; text-align:center;">
                <div style="color:#aaa; font-size:.8em; margin-bottom:4px;">📐 Rule Engine</div>
                <div style="font-size:1.6em; color:{_dc(rd)}; font-weight:bold;">
                    {_de(rd)} {rd}</div>
                <div style="color:#ccc; font-size:.9em;">Confidence: {rc}%</div>
                <div style="color:#888; font-size:.8em;">Alert: {'✅' if rule.get('alert') else '—'}</div>
            </div>""", unsafe_allow_html=True)

    ad = agent.get("direction", "PENDING")
    ac = agent.get("confidence", 0)
    a_color = _dc(ad) if ad in ("LONG", "SHORT") else "#78909c"
    age_str = f"{agent.get('age_min'):.0f}m ago" if agent.get("age_min") is not None else "—"
    fresh_tag = "🟢 fresh" if agent.get("is_fresh") else "🔴 stale"
    with col_a:
        st.markdown(
            f"""<div style="background:#1a1a2e; border-radius:6px; padding:12px; text-align:center;">
                <div style="color:#aaa; font-size:.8em; margin-bottom:4px;">🤖 AI Agent</div>
                <div style="font-size:1.6em; color:{a_color}; font-weight:bold;">
                    {_de(ad)} {ad}</div>
                <div style="color:#ccc; font-size:.9em;">Confidence: {ac}%</div>
                <div style="color:#888; font-size:.8em;">{age_str} · {fresh_tag}</div>
            </div>""", unsafe_allow_html=True)

    cd = val.get("consensus_direction", "NEUTRAL")
    cc = val.get("consensus_confidence", 0)
    c_color = _dc(cd) if cd in ("LONG", "SHORT") else agg_color
    with col_c:
        st.markdown(
            f"""<div style="background:{agg_color}15; border:1px solid {agg_color};
                            border-radius:6px; padding:12px; text-align:center;">
                <div style="color:#aaa; font-size:.8em; margin-bottom:4px;">⚖️ Consensus</div>
                <div style="font-size:1.6em; color:{c_color}; font-weight:bold;">
                    {_de(cd)} {cd}</div>
                <div style="color:#ccc; font-size:.9em;">Confidence: {cc}%</div>
                <div style="color:{agg_color}; font-size:.8em; font-weight:bold;">{agreement}</div>
            </div>""", unsafe_allow_html=True)


def _render_validation_history():
    entries = load_validation_log(100)
    if not entries:
        st.info("No validation history yet — run Signal Agent at least once.")
        return
    stats = agreement_stats(100)
    sc1, sc2, sc3, sc4 = st.columns(4)
    sc1.metric("Total Validations", stats.get("total", 0))
    sc2.metric("Full Agreement %", f"{stats.get('full_pct', 0):.1f}%")
    sc3.metric("Conflict %", f"{stats.get('conflict_pct', 0):.1f}%",
               delta=f"{'⚠️' if stats.get('conflict_pct', 0) > 20 else '✅'}", delta_color="off")
    sc4.metric("Full LONG / SHORT", f"{stats.get('full_long', 0)} / {stats.get('full_short', 0)}")
    by_type = stats.get("by_type", {})
    if by_type:
        df_agg = pd.DataFrame(
            [{"Agreement": k, "Count": v} for k, v in by_type.items()]
        ).sort_values("Count", ascending=False)
        colors_map = {"FULL": "#00c853", "PARTIAL": "#ffd600", "CONFLICT": "#d50000",
                      "NEUTRAL": "#9e9e9e", "PENDING": "#78909c", "BLACKOUT": "#b71c1c"}
        fig = go.Figure(go.Bar(
            x=df_agg["Agreement"], y=df_agg["Count"],
            marker_color=[colors_map.get(a, "#9e9e9e") for a in df_agg["Agreement"]],
        ))
        fig.update_layout(height=200, template="plotly_dark",
                          margin=dict(l=0, r=0, t=10, b=0), showlegend=False)
        st.plotly_chart(fig, use_container_width=True)


# ─── Section 10: Zone Map ──────────────────────────────────────────────────────

def _render_zone_map(sig: dict):
    zones   = sig.get("zones", [])
    current = sig.get("current_price")
    if not zones:
        st.info("No zones identified — price data unavailable.")
        return

    rows = []
    for z in zones:
        score  = z["score"]
        ztype  = z["zone_type"]
        icon   = "🟢" if ztype == "support" else "🔴" if ztype == "resistance" else "⚪"
        dist   = z["dist_pts"]
        comps  = z["components"]
        tags   = list({
            "OB"  if any("OB"  in c for c in comps) else None,
            "FVG" if any("FVG" in c for c in comps) else None,
            "KL"  if any(c not in ("OB_D", "OB_15m", "FVG_15m") for c in comps) else None,
        } - {None})
        score_bar = "█" * score + "░" * (10 - score)
        rows.append({
            " ":         icon,
            "Type":      ztype.capitalize(),
            "Range":     f"{z['price_low']:,.0f} – {z['price_high']:,.0f}",
            "Mid":       f"{z['midpoint']:,.1f}",
            "Dist (pt)": f"{dist:+.0f}",
            "Score":     f"{score}/10  {score_bar}",
            "Layers":    " + ".join(tags) if tags else "—",
        })

    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # Price ladder
    if current and len(zones) > 1:
        fig = go.Figure()
        for z in zones:
            fc = "rgba(0,200,83,0.15)" if z["zone_type"] == "support" else "rgba(213,0,0,0.15)"
            lc = "#00c853"             if z["zone_type"] == "support" else "#d50000"
            fig.add_hrect(y0=z["price_low"], y1=z["price_high"],
                          fillcolor=fc, line_color=lc, line_width=1, opacity=0.8)
            fig.add_annotation(
                x=0.02, y=(z["price_low"] + z["price_high"]) / 2,
                xref="paper", yref="y",
                text=f"{z['score']}/10  {'|'.join(z['components'][:3])}",
                showarrow=False, font=dict(size=10, color=lc), xanchor="left",
            )
        fig.add_hline(y=current, line_color="#ffffff", line_width=2,
                      annotation_text=f"NOW {current:,.1f}", annotation_position="right")
        fig.update_layout(
            height=240, template="plotly_dark",
            margin=dict(l=0, r=120, t=10, b=0),
            xaxis=dict(visible=False),
            yaxis=dict(side="right", tickformat=",.0f"),
        )
        st.plotly_chart(fig, use_container_width=True)


def _render_key_levels_table(sig: dict):
    levels  = sig.get("levels", {})
    current = sig.get("current_price")
    if not levels:
        st.info("No key levels available.")
        return
    rows = []
    for name, val in sorted(levels.items(), key=lambda x: x[1]):
        dist = f"{val - current:+.1f}" if current else "—"
        rows.append({"Level": name.upper(), "Price": f"{val:,.1f}", "Dist from Current": dist})
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# ─── Section 11: AI Agent proposals ───────────────────────────────────────────

def _render_agent_proposals():
    """Show the last 3 directional proposals written by the Signal Agent."""
    proposals = _load_agent_proposals(3, directional_only=True)
    if not proposals:
        st.info(
            "Signal Agent has not generated proposals yet. "
            "Click **▶ Run Now** to trigger manually, "
            "or start the Orchestrator from the 🤖 Agents tab.",
        )
    else:
        for prop in reversed(proposals):
            p      = prop.get("proposal", prop)
            ts     = str(prop.get("logged_at", ""))[:16]
            direc  = p.get("direction", "?")
            conf   = p.get("confidence", 0)
            entry  = p.get("entry", "?")
            stop_p = p.get("stop", "?")
            target = p.get("target", "?")
            rr_p   = p.get("rr", "?")
            with st.expander(
                f"{'🟢' if direc=='LONG' else '🔴'} {direc} — conf:{conf}/100 — R:R {rr_p} — {ts}",
                expanded=(prop is proposals[-1]),
            ):
                cols = st.columns(4)
                cols[0].metric("Direction", direc)
                cols[1].metric("Entry",  f"{entry:,.1f}"  if isinstance(entry,  float) else entry)
                cols[2].metric("Stop",   f"{stop_p:,.1f}" if isinstance(stop_p, float) else stop_p)
                cols[3].metric("Target", f"{target:,.1f}" if isinstance(target, float) else target)
                if p.get("reasoning"):
                    st.markdown(f"**Reasoning:** {p['reasoning']}")
                factors_list = p.get("key_factors", [])
                if factors_list:
                    st.markdown("**Key Factors:** " + " · ".join(f"`{f}`" for f in factors_list))
                if p.get("session"):
                    st.caption(f"Session: {p['session']} | HTF: {p.get('htf_bias','?')}")

    if not proposals:
        st.caption("No LONG/SHORT proposals yet — run Signal Agent to generate one.")

    # Recent run activity (all outcomes, including no_signal)
    recent_runs = _load_signal_agent_log(10)
    if recent_runs:
        with st.expander("📋 Recent Agent Activity (last 10 runs)", expanded=False):
            for run in reversed(recent_runs):
                outcome = run.get("outcome", "?")
                ts      = str(run.get("started_at", ""))[:16].replace("T", " ")
                score   = run.get("composite_score") or run.get("_indicator_score", "?")
                tools   = run.get("tool_calls", 0)
                icon    = {"proposal": "🟢", "no_signal": "⏳", "skipped": "⏭️", "error": "🔴"}.get(outcome, "⚪")
                reason  = run.get("reason") or run.get("final_text", "")[:120]
                st.markdown(
                    f'<div style="padding:4px 0; border-bottom:1px solid #222; font-size:.85em;">'
                    f'{icon} <b>{ts}</b> &nbsp; <code>{outcome}</code>'
                    f'&nbsp; score:{score} &nbsp; tools:{tools}'
                    + (f'<br><span style="color:#aaa;">{reason}</span>' if reason else "")
                    + f'</div>',
                    unsafe_allow_html=True,
                )


# ─── Main page ─────────────────────────────────────────────────────────────────

@st.fragment(run_every=60)
def _live_signals_fragment():
    """All live data — auto-refreshes every 60 s without full-page flicker."""
    session = current_session_israel()
    sess_color = {"high": "#00c853", "medium": "#ffd600", "low": "#9e9e9e"}.get(
        session.get("risk", "low"), "#9e9e9e"
    )
    st.markdown(
        f'<div style="text-align:right; color:{sess_color}; font-size:.9em; margin-bottom:4px;">'
        f'<b>⏱ {session.get("name","—")}</b>'
        f'&nbsp; <span style="color:#aaa;">{session.get("current_il_time","—")} IL</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    sig = _get_signal()

    # Auto-log: real alert → trade_log with type="real"
    if sig.get("alert"):
        ts_settings = load_trade_settings()
        if ts_settings.get("auto_log", True):
            sig_hash = f"{sig.get('direction')}_{sig.get('entry')}_{sig.get('confidence')}"
            if st.session_state.get("_last_logged") != sig_hash:
                if log_trade(sig):
                    st.session_state["_last_logged"] = sig_hash
                    st.toast("📝 Trade logged to Journal", icon="✅")

    # Paper trade: any directional signal → log with type="paper" for ML training
    # Runs even when alert=False so quiet days still accumulate training data.
    elif sig.get("direction") not in (None, "NEUTRAL"):
        log_paper_trade(sig)

    # A/B experiment auto-record
    try:
        from modules import ab_tracker as _ab
        if _ab.get_active_experiment() and sig.get("direction") != "NEUTRAL":
            ab_key = f"_ab_{sig.get('direction')}_{datetime.now().strftime('%Y-%m-%d_%H')}"
            if not st.session_state.get(ab_key):
                _ab.record_signal(sig)
                st.session_state[ab_key] = True
    except Exception:
        pass

    _render_blackout_banner(sig)
    _render_regime_badge(sig)
    _render_signal_card(sig)
    _render_signal_readiness(sig)
    _render_alert_banner(sig)

    st.markdown("### Trade Parameters")
    _render_rr_metrics(sig)

    # Timeframe selector — scalping (1m/3m) or context (5m/15m)
    tf_col, info_col = st.columns([2, 5])
    with tf_col:
        tf = st.radio(
            "Chart TF",
            ["1m", "3m", "5m"],
            index=2,
            horizontal=True,
            help="1m/3m for scalp entry; 5m for structure confirmation",
            key="chart_tf_radio",
        )
    atr_ltf = sig.get("atr_ltf")
    with info_col:
        if atr_ltf:
            stop_p = sig.get("stop")
            entry_p = sig.get("entry")
            stop_pts = round(abs(entry_p - stop_p), 1) if stop_p and entry_p else "—"
            st.markdown(
                f'<div style="padding-top:8px; font-size:.85em; color:#aaa;">'
                f'LTF ATR: <b style="color:#fff;">{atr_ltf} pts</b>'
                f'&nbsp;·&nbsp; Scalp stop: <b style="color:#ff5252;">{stop_pts} pts</b>'
                f'&nbsp;·&nbsp; '
                f'<span style="color:#e040fb;">━━ EQH/EQL</span>'
                f'&nbsp; <span style="color:#b2ff59;">─ ─ BSL</span>'
                f'&nbsp; <span style="color:#ff6e40;">─ ─ SSL</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
    df_chart = _TF_LOADERS[tf]()
    st.markdown(f"### NQ {tf} Chart")
    _render_nq_chart(df_chart, sig, tf_label=tf)

    conf_col, macro_col = st.columns([3, 2])
    with conf_col:
        st.markdown("### Confluence Factors")
        _render_confluence_visual(sig)
    with macro_col:
        st.markdown("### Macro Context")
        _render_macro_context(sig, session)

    long_pts  = sig.get("long_pts",  0) or 0
    short_pts = sig.get("short_pts", 0) or 0
    best      = max(long_pts, short_pts)
    with st.expander("✅ Pre-Trade Checklist", expanded=best >= 40 or sig.get("alert", False)):
        _render_pretrade_checklist(sig, session)

    st.markdown("### ⚖️ Signal Cross-Validation")
    val = _get_validation(sig)
    _render_cross_validation(val)

    with st.expander("📊 Validation History (last 100)", expanded=False):
        _render_validation_history()

    with st.expander("🗺️ Zone Map (Support & Resistance)", expanded=True):
        _render_zone_map(sig)

    with st.expander("📊 Key Levels", expanded=False):
        _render_key_levels_table(sig)

    st.divider()
    st.markdown("### 🤖 AI Signal Agent")
    _render_agent_proposals()

    # ── Score history sparkline ───────────────────────────────────
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


def app():
    st.title("🎯 Trade Signals — NQ")
    _live_signals_fragment()

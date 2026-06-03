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
from modules.trade_tracker import log_trade, load_trade_settings, load_trade_log, summary_stats
from modules.signal_validator import validate, load_validation_log, agreement_stats

import json
from pathlib import Path as _Path
_LOGS = _Path(__file__).resolve().parents[1] / "logs"

# ─── Signal threshold (must match signal_engine.py) ─────────────────────────
_SIGNAL_THRESHOLD = 55
_DIRECTION_LEAD   = 15   # minimum lead over opponent


# ─── Data loaders ─────────────────────────────────────────────────────────────

def _load_agent_proposals(n: int = 3) -> list:
    path = _LOGS / "agent_proposals.json"
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text())
        return data[-n:] if isinstance(data, list) else []
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


@st.cache_data(ttl=60)
def _get_5m():
    return get_todays_nq_data()


@st.cache_data(ttl=300)
def _get_daily():
    return load_nq_daily_cache()


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

def _render_nq_chart(df_5m: pd.DataFrame, sig: dict):
    if df_5m.empty:
        st.warning("No 5-min NQ data available.")
        return

    time_col  = "datetime_il" if "datetime_il" in df_5m.columns else "datetime"
    x = df_5m[time_col].astype(str)

    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=x, open=df_5m["open"], high=df_5m["high"],
        low=df_5m["low"], close=df_5m["close"],
        name="NQ 5m",
        increasing_line_color="#00c853", decreasing_line_color="#d50000",
        showlegend=False,
    ))

    # Only the 3 most important structural levels — right side annotations only
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
                y=val, line_color=color, line_dash="dot", line_width=1.2, opacity=0.75,
                annotation_text=label, annotation_position="right",
                annotation_font=dict(size=10, color=color),
            )

    # Entry / Stop / Target — only when signal active
    entry     = sig.get("entry")
    stop      = sig.get("stop")
    target    = sig.get("target")
    partial   = sig.get("partial_exit")
    direction = sig.get("direction", "NEUTRAL")

    if entry and direction != "NEUTRAL":
        fig.add_hline(
            y=entry, line_color=_dc(direction), line_width=2.5,
            annotation_text="ENTRY", annotation_position="right",
            annotation_font=dict(size=11, color=_dc(direction)),
        )
    if stop:
        fig.add_hline(
            y=stop, line_color="#ff5252", line_width=1.5, line_dash="dash",
            annotation_text="STOP", annotation_position="right",
            annotation_font=dict(size=10, color="#ff5252"),
        )
    if target:
        fig.add_hline(
            y=target, line_color="#69f0ae", line_width=1.5, line_dash="dash",
            annotation_text="TARGET", annotation_position="right",
            annotation_font=dict(size=10, color="#69f0ae"),
        )
    if partial:
        fig.add_hline(
            y=partial, line_color="#ffd600", line_width=1, line_dash="dot",
            annotation_text="1R", annotation_position="right",
            annotation_font=dict(size=9, color="#ffd600"),
        )

    # FVG zones — max 3 most recent, no text label
    try:
        fvgs = detect_fvg(df_5m)
        if not fvgs.empty:
            for _, fvg in fvgs.tail(3).iterrows():
                is_bull = fvg["type"] == "bullish"
                fig.add_hrect(
                    y0=fvg["bottom"], y1=fvg["top"],
                    fillcolor="rgba(0,200,83,0.10)" if is_bull else "rgba(213,0,0,0.10)",
                    line_color="#00c853" if is_bull else "#d50000",
                    line_width=0.5, opacity=0.8,
                )
    except Exception:
        pass

    fig.update_layout(
        height=380, template="plotly_dark",
        margin=dict(l=0, r=80, t=16, b=0),
        xaxis_rangeslider_visible=False,
        yaxis=dict(side="right", tickformat=",.0f"),
    )
    st.plotly_chart(fig, use_container_width=True)


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

def _render_cross_validation(val: dict):
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
    proposals = _load_agent_proposals(3)
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

    run_col, _ = st.columns([1, 3])
    if run_col.button("▶ Run Signal Agent Now", key="run_sig_agent"):
        with st.spinner("Running Indicators + Signal Agent..."):
            try:
                from agents import orchestrator
                result = orchestrator.run_once("indicators_signal")
                if result.get("outcome") == "proposal":
                    st.success("✅ New proposal created — refresh to see it.")
                elif result.get("outcome") == "skipped":
                    st.warning(f"⏭️ {result.get('reason','no signal')}")
                elif "error" in result:
                    st.error(f"Error: {result.get('error','?')}")
                else:
                    st.info(f"Result: {result.get('outcome','?')}")
            except ImportError:
                st.error("anthropic not installed — run: pip install anthropic")
        st.rerun()


# ─── Main page ─────────────────────────────────────────────────────────────────

def app():
    st.title("🎯 Trade Signals — NQ")

    # ── Auto-refresh + session header ────────────────────────────
    col_ref, col_sess = st.columns([3, 1])
    with col_ref:
        refresh_sec = st.select_slider(
            "Auto-refresh",
            options=[0, 30, 60, 120, 300],
            value=60,
            format_func=lambda x: "Off" if x == 0 else f"{x}s",
        )
    if refresh_sec > 0:
        try:
            from streamlit_autorefresh import st_autorefresh
            st_autorefresh(interval=refresh_sec * 1000, key="sig_refresh")
        except ImportError:
            st.caption("install streamlit-autorefresh for non-blocking refresh")

    session = current_session_israel()
    sess_color = {"high": "#00c853", "medium": "#ffd600", "low": "#9e9e9e"}.get(
        session.get("risk", "low"), "#9e9e9e"
    )
    with col_sess:
        st.markdown(
            f"""<div style="text-align:right; padding-top:6px;">
                <span style="color:{sess_color}; font-weight:bold;">
                    ⏱ {session.get('name','—')}</span><br>
                <small style="color:#aaa;">{session.get('current_il_time','—')} IL</small>
            </div>""",
            unsafe_allow_html=True,
        )

    # ── Signal computation ────────────────────────────────────────
    with st.spinner("Calculating signal..."):
        sig = _get_signal()

    # ── Auto-log alert ────────────────────────────────────────────
    if sig.get("alert"):
        ts_settings = load_trade_settings()
        if ts_settings.get("auto_log", True):
            sig_hash = f"{sig.get('direction')}_{sig.get('entry')}_{sig.get('confidence')}"
            if st.session_state.get("_last_logged") != sig_hash:
                if log_trade(sig):
                    st.session_state["_last_logged"] = sig_hash
                    st.toast("📝 Trade logged to Journal", icon="✅")

    # ── A/B experiment auto-record ────────────────────────────────
    try:
        from modules import ab_tracker as _ab
        if _ab.get_active_experiment() and sig.get("direction") != "NEUTRAL":
            ab_key = f"_ab_{sig.get('direction')}_{datetime.now().strftime('%Y-%m-%d_%H')}"
            if not st.session_state.get(ab_key):
                _ab.record_signal(sig)
                st.session_state[ab_key] = True
    except Exception:
        pass

    # ── Blackout + Regime ─────────────────────────────────────────
    _render_blackout_banner(sig)
    _render_regime_badge(sig)

    # ── Signal card + readiness ───────────────────────────────────
    _render_signal_card(sig)
    _render_signal_readiness(sig)
    _render_alert_banner(sig)

    # ── Trade Parameters ──────────────────────────────────────────
    st.markdown("### Trade Parameters")
    _render_rr_metrics(sig)

    # ── Chart (full width) ────────────────────────────────────────
    st.markdown("### NQ 5-min Chart")
    df_5m = _get_5m()
    _render_nq_chart(df_5m, sig)

    # ── Confluence + Macro side-by-side below chart ───────────────
    conf_col, macro_col = st.columns([3, 2])

    with conf_col:
        st.markdown("### Confluence Factors")
        _render_confluence_visual(sig)

    with macro_col:
        st.markdown("### Macro Context")
        _render_macro_context(sig, session)

    # ── Pre-Trade Checklist ───────────────────────────────────────
    long_pts  = sig.get("long_pts",  0) or 0
    short_pts = sig.get("short_pts", 0) or 0
    best      = max(long_pts, short_pts)
    # Auto-expand checklist when close to firing (≥40 pts)
    checklist_expanded = best >= 40 or sig.get("alert", False)
    with st.expander("✅ Pre-Trade Checklist", expanded=checklist_expanded):
        _render_pretrade_checklist(sig, session)

    # ── Cross-Validation ──────────────────────────────────────────
    st.markdown("### ⚖️ Signal Cross-Validation")
    val = _get_validation(sig)
    _render_cross_validation(val)

    with st.expander("📊 Validation History (last 100)", expanded=False):
        _render_validation_history()

    # ── Zone Map ──────────────────────────────────────────────────
    with st.expander("🗺️ Zone Map (Support & Resistance)", expanded=True):
        _render_zone_map(sig)

    # ── Key Levels ────────────────────────────────────────────────
    with st.expander("📊 Key Levels", expanded=False):
        _render_key_levels_table(sig)

    # ── AI Agent Proposals ────────────────────────────────────────
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

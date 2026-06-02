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
    entry        = sig.get("entry")
    stop         = sig.get("stop")
    target       = sig.get("target")
    partial      = sig.get("partial_exit")
    rr           = sig.get("rr")

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Entry",      f"{entry:,.1f}"   if entry   else "—")
    c2.metric("Stop (ATR)", f"{stop:,.1f}"    if stop    else "—",
              delta=f"{stop - entry:+.1f}"    if stop and entry else None,
              delta_color="inverse")
    c3.metric("Target",     f"{target:,.1f}"  if target  else "—",
              delta=f"{target - entry:+.1f}"  if target and entry else None)
    c4.metric("1R Partial", f"{partial:,.1f}" if partial else "—",
              delta=f"{partial - entry:+.1f}" if partial and entry else None)
    c5.metric("BE Level",   f"{entry:,.1f}"   if entry   else "—",
              help="Move stop to entry after partial exit is hit")
    rr_str = f"{rr:.2f}" if rr else "—"
    rr_col = _rr_color(rr)
    c6.markdown(
        f"""<div style="padding:8px 0;">
            <div style="font-size:.85em; color:#aaa;">R:R Ratio</div>
            <div style="font-size:1.8em; color:{rr_col}; font-weight:bold;">{rr_str}</div>
        </div>""",
        unsafe_allow_html=True,
    )


def _render_blackout_banner(sig: dict):
    if sig.get("blackout"):
        reason = sig.get("blackout_reason", "High-impact economic event")
        st.markdown(
            f"""
            <div style="background:#b71c1c30; border:2px solid #b71c1c;
                        padding:12px 20px; border-radius:8px; margin-bottom:10px;">
                🚫 <b style="color:#ff5252;">BLACKOUT ZONE — NO TRADING</b><br>
                <span style="color:#ffcdd2; font-size:.9em;">{reason}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )


def _render_regime_badge(sig: dict):
    regime       = sig.get("regime", "transitioning")
    regime_label = sig.get("regime_label", "Transitioning")
    regime_icon  = sig.get("regime_icon", "⚪")
    strategies   = sig.get("regime_strategies", [])
    conf_adj     = sig.get("regime_conf_adj", 0)
    size_adj     = sig.get("regime_size_adj", 1.0)

    conf_str = f"+{conf_adj}pts threshold" if conf_adj > 0 else f"{conf_adj}pts threshold" if conf_adj < 0 else "standard threshold"
    size_str = f"×{size_adj:.2g} size"

    border_color = {
        "trending_bull":  "#00c853",
        "trending_bear":  "#d50000",
        "ranging":        "#ffd600",
        "high_vol":       "#ff9800",
        "transitioning":  "#9e9e9e",
    }.get(regime, "#9e9e9e")

    strat_html = "".join(f"<li style='margin:1px 0'>{s}</li>" for s in strategies[:3])
    st.markdown(
        f"""
        <div style="background:{border_color}18; border-left:4px solid {border_color};
                    padding:10px 16px; border-radius:6px; margin:8px 0;">
            <b style="color:{border_color};">{regime_icon} {regime_label}</b>
            &nbsp;<span style="color:#aaa; font-size:.85em;">({conf_str} · {size_str})</span>
            <ul style="margin:4px 0 0 0; padding-left:18px; color:#ccc; font-size:.85em;">
                {strat_html}
            </ul>
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
                      annotation_text="STOP (ATR)", annotation_position="right")
    if target:
        fig.add_hline(y=target, line_color="#69f0ae", line_width=1.5, line_dash="dash",
                      annotation_text="TARGET", annotation_position="right")
    partial = sig.get("partial_exit")
    if partial:
        fig.add_hline(y=partial, line_color="#ffd600", line_width=1, line_dash="dot",
                      annotation_text="1R Partial", annotation_position="right")

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


def _render_cross_validation(val: dict):
    """Side-by-side comparison: rule engine vs AI agent vs consensus."""
    agreement = val.get("agreement", "PENDING")
    label     = val.get("consensus_label", "—")
    note      = val.get("note", "")
    rule      = val.get("rule", {})
    agent     = val.get("agent", {})

    # Agreement colour
    agg_color = {
        "FULL":     "#00c853",
        "PARTIAL":  "#ffd600",
        "CONFLICT": "#d50000",
        "NEUTRAL":  "#9e9e9e",
        "PENDING":  "#78909c",
        "BLACKOUT": "#b71c1c",
    }.get(agreement, "#9e9e9e")

    # Consensus banner
    st.markdown(
        f"""
        <div style="background:{agg_color}20; border-left:5px solid {agg_color};
                    padding:10px 16px; border-radius:6px; margin-bottom:10px;">
            <b style="color:{agg_color}; font-size:1.05em;">{label}</b><br>
            <span style="color:#ccc; font-size:.88em;">{note}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Three-column breakdown
    col_r, col_a, col_c = st.columns(3)

    # Rule engine column
    rd = rule.get("direction", "—")
    rc = rule.get("confidence", 0)
    r_color = _direction_color(rd)
    with col_r:
        st.markdown(
            f"""<div style="background:#1a1a2e; border-radius:6px; padding:12px; text-align:center;">
                <div style="color:#aaa; font-size:.8em; margin-bottom:4px;">📐 Rule Engine</div>
                <div style="font-size:1.6em; color:{r_color}; font-weight:bold;">
                    {_direction_emoji(rd)} {rd}
                </div>
                <div style="color:#ccc; font-size:.9em;">Confidence: {rc}%</div>
                <div style="color:#888; font-size:.8em;">Alert: {'✅' if rule.get('alert') else '—'}</div>
            </div>""",
            unsafe_allow_html=True,
        )

    # AI Agent column
    ad = agent.get("direction", "PENDING")
    ac = agent.get("confidence", 0)
    a_color = _direction_color(ad) if ad in ("LONG", "SHORT") else "#78909c"
    age_str = (
        f"{agent.get('age_min'):.0f}m ago" if agent.get("age_min") is not None else "—"
    )
    fresh_tag = "🟢 fresh" if agent.get("is_fresh") else "🔴 stale"
    with col_a:
        st.markdown(
            f"""<div style="background:#1a1a2e; border-radius:6px; padding:12px; text-align:center;">
                <div style="color:#aaa; font-size:.8em; margin-bottom:4px;">🤖 AI Agent</div>
                <div style="font-size:1.6em; color:{a_color}; font-weight:bold;">
                    {_direction_emoji(ad)} {ad}
                </div>
                <div style="color:#ccc; font-size:.9em;">Confidence: {ac}%</div>
                <div style="color:#888; font-size:.8em;">{age_str} · {fresh_tag}</div>
            </div>""",
            unsafe_allow_html=True,
        )

    # Consensus column
    cd = val.get("consensus_direction", "NEUTRAL")
    cc = val.get("consensus_confidence", 0)
    c_color = _direction_color(cd) if cd in ("LONG", "SHORT") else agg_color
    with col_c:
        st.markdown(
            f"""<div style="background:{agg_color}15; border:1px solid {agg_color};
                            border-radius:6px; padding:12px; text-align:center;">
                <div style="color:#aaa; font-size:.8em; margin-bottom:4px;">⚖️ Consensus</div>
                <div style="font-size:1.6em; color:{c_color}; font-weight:bold;">
                    {_direction_emoji(cd)} {cd}
                </div>
                <div style="color:#ccc; font-size:.9em;">Confidence: {cc}%</div>
                <div style="color:{agg_color}; font-size:.8em; font-weight:bold;">{agreement}</div>
            </div>""",
            unsafe_allow_html=True,
        )


def _render_validation_history():
    """Compact agreement rate chart from validation_log.json."""
    entries = load_validation_log(100)
    if not entries:
        st.info("No validation history yet — run Signal Agent at least once.")
        return

    stats = agreement_stats(100)
    sc1, sc2, sc3, sc4 = st.columns(4)
    sc1.metric("Total Validations", stats.get("total", 0))
    sc2.metric("Full Agreement %", f"{stats.get('full_pct', 0):.1f}%",
               help="Both engines said same direction")
    sc3.metric("Conflict %", f"{stats.get('conflict_pct', 0):.1f}%",
               delta=f"{'⚠️' if stats.get('conflict_pct', 0) > 20 else '✅'}",
               delta_color="off")
    sc4.metric("Full LONG / SHORT",
               f"{stats.get('full_long', 0)} / {stats.get('full_short', 0)}")

    # Agreement type bar chart
    by_type = stats.get("by_type", {})
    if by_type:
        df_agg = pd.DataFrame(
            [{"Agreement": k, "Count": v} for k, v in by_type.items()]
        ).sort_values("Count", ascending=False)
        colors_map = {
            "FULL": "#00c853", "PARTIAL": "#ffd600", "CONFLICT": "#d50000",
            "NEUTRAL": "#9e9e9e", "PENDING": "#78909c", "BLACKOUT": "#b71c1c",
        }
        fig = go.Figure(go.Bar(
            x=df_agg["Agreement"], y=df_agg["Count"],
            marker_color=[colors_map.get(a, "#9e9e9e") for a in df_agg["Agreement"]],
        ))
        fig.update_layout(height=200, template="plotly_dark",
                          margin=dict(l=0, r=0, t=10, b=0), showlegend=False)
        st.plotly_chart(fig, use_container_width=True)


def _render_zone_map(sig: dict):
    """Visual zone map: all identified S/R zones with confluence scores."""
    zones   = sig.get("zones", [])
    current = sig.get("current_price")
    if not zones:
        st.info("No zones identified — price data unavailable.")
        return

    rows = []
    for z in zones:
        score    = z["score"]
        ztype    = z["zone_type"]
        icon     = "🟢" if ztype == "support" else "🔴" if ztype == "resistance" else "⚪"
        dist     = z["dist_pts"]
        comps    = z["components"]
        has_ob   = any("OB" in c for c in comps)
        has_fvg  = any("FVG" in c for c in comps)
        has_kl   = any(c not in ("OB_D", "OB_15m", "FVG_15m") for c in comps)
        tags     = []
        if has_ob:  tags.append("OB")
        if has_fvg: tags.append("FVG")
        if has_kl:  tags.append("KL")
        score_bar = "█" * score + "░" * (10 - score)
        rows.append({
            " ":         icon,
            "Type":      ztype.capitalize(),
            "Range":     f"{z['price_low']:,.0f} – {z['price_high']:,.0f}",
            "Mid":       f"{z['midpoint']:,.1f}",
            "Dist (pt)": f"{dist:+.0f}" if ztype == "support" else f"+{dist:.0f}",
            "Score":     f"{score}/10  {score_bar}",
            "Layers":    " + ".join(tags) if tags else "—",
        })

    df_zones = pd.DataFrame(rows)

    # Color-code score column via Streamlit styling
    st.dataframe(df_zones, use_container_width=True, hide_index=True)

    # Mini price ladder chart
    if current and len(zones) > 1:
        import plotly.graph_objects as go
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
            height=250, template="plotly_dark",
            margin=dict(l=0, r=120, t=10, b=0),
            xaxis=dict(visible=False),
            yaxis=dict(side="right", tickformat=",.0f"),
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

    # Auto-refresh — non-blocking JS timer (no sleep)
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

    # ── Auto-log alert if enabled ────────────────────────────────
    if sig.get("alert"):
        ts_settings = load_trade_settings()
        if ts_settings.get("auto_log", True):
            sig_hash = f"{sig.get('direction')}_{sig.get('entry')}_{sig.get('confidence')}"
            if st.session_state.get("_last_logged") != sig_hash:
                if log_trade(sig):
                    st.session_state["_last_logged"] = sig_hash
                    st.toast("📝 Trade logged to Journal", icon="✅")

    # ── A/B Experiment auto-record ───────────────────────────────
    # Records one signal per hour per direction to avoid refresh duplicates.
    try:
        from modules import ab_tracker as _ab
        if _ab.get_active_experiment() and sig.get("direction") != "NEUTRAL":
            ab_key = f"_ab_{sig.get('direction')}_{datetime.now().strftime('%Y-%m-%d_%H')}"
            if not st.session_state.get(ab_key):
                _ab.record_signal(sig)
                st.session_state[ab_key] = True
    except Exception:
        pass

    # ── Blackout / Regime status ────────────────────────────────
    _render_blackout_banner(sig)
    _render_regime_badge(sig)

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
        htf_bias = sig.get("htf_bias", "neutral").capitalize()
        htf_str  = sig.get("htf_strength", 0)
        htf_col  = {"Bullish": "#00c853", "Bearish": "#d50000"}.get(htf_bias, "#9e9e9e")
        c1.markdown(
            f"""<div style="padding:4px 0">
                <div style="font-size:.85em;color:#aaa;">HTF Bias</div>
                <div style="font-size:1.4em;color:{htf_col};font-weight:bold;">{htf_bias} {htf_str}%</div>
            </div>""", unsafe_allow_html=True)
        c2.metric("ML Prediction", sig.get("ml_prediction", "—"))
        c1.metric("Sentiment Score", sig.get("final_score", "—"))
        c2.metric("Delta Direction", sig.get("delta_dir", "—"))
        nzs = sig.get("nearest_zone_score", 0)
        c1.metric("Zone Score", f"{nzs}/10")
        c2.metric("Session Risk", session.get("risk", "—").capitalize())

    # ── Cross-Validation ────────────────────────────────────────
    st.markdown("### ⚖️ Signal Cross-Validation")
    val = _get_validation(sig)
    _render_cross_validation(val)

    with st.expander("📊 Validation History (last 100)", expanded=False):
        _render_validation_history()

    # ── Zone Map ────────────────────────────────────────────────
    with st.expander("🗺️ Zone Map (Support & Resistance)", expanded=True):
        _render_zone_map(sig)

    # ── Key levels ──────────────────────────────────────────────
    with st.expander("📊 Key Levels", expanded=False):
        _render_key_levels_table(sig)

    # ── AI Agent Proposals ──────────────────────────────────────
    st.divider()
    st.markdown("### 🤖 AI Signal Agent — הצעות אחרונות")

    proposals = _load_agent_proposals(3)
    if not proposals:
        st.info(
            "Signal Agent טרם יצר הצעות. "
            "לחץ **▶ הפעל עכשיו** כדי להריץ אותו ידנית, "
            "או הפעל את ה-Orchestrator מהטאב 🤖 Agents.",
        )
    else:
        for prop in reversed(proposals):
            p      = prop.get("proposal", prop)  # support both wrapped and flat
            ts     = str(prop.get("logged_at", ""))[:16]
            direc  = p.get("direction", "?")
            conf   = p.get("confidence", 0)
            entry  = p.get("entry", "?")
            stop   = p.get("stop", "?")
            target = p.get("target", "?")
            rr     = p.get("rr", "?")
            color  = {"LONG": "#00c853", "SHORT": "#d50000"}.get(direc, "#9e9e9e")

            with st.expander(
                f"{'🟢' if direc=='LONG' else '🔴'} {direc} — "
                f"conf: {conf}/100 — R:R {rr} — {ts}",
                expanded=(prop is proposals[-1]),
            ):
                cols = st.columns(4)
                cols[0].metric("כיוון", direc)
                cols[1].metric("כניסה", f"{entry:,.1f}" if isinstance(entry, float) else entry)
                cols[2].metric("סטופ", f"{stop:,.1f}" if isinstance(stop, float) else stop)
                cols[3].metric("יעד", f"{target:,.1f}" if isinstance(target, float) else target)

                if p.get("reasoning"):
                    st.markdown(f"**נימוק:** {p['reasoning']}")
                factors = p.get("key_factors", [])
                if factors:
                    st.markdown("**גורמים מרכזיים:** " + " · ".join(f"`{f}`" for f in factors))
                if p.get("session"):
                    st.caption(f"Session: {p['session']} | HTF: {p.get('htf_bias','?')}")

    # Run Signal Agent button
    run_col, _ = st.columns([1, 3])
    if run_col.button("▶ הפעל Signal Agent עכשיו", key="run_sig_agent"):
        with st.spinner("מריץ Indicators + Signal Agent..."):
            try:
                from agents import orchestrator
                result = orchestrator.run_once("indicators_signal")
                outcome = result.get("outcome", "?")
                if result.get("outcome") == "proposal":
                    st.success("✅ הצעה חדשה נוצרה! רענן את הדף.")
                elif result.get("outcome") == "skipped":
                    st.warning(f"⏭️ Signal Agent: {result.get('reason','no signal')}")
                elif "error" in result:
                    st.error(f"שגיאה: {result.get('error', result.get('reason','?'))}")
                else:
                    st.info(f"תוצאה: {outcome}")
            except ImportError:
                st.error("anthropic לא מותקן — הרץ: pip install anthropic")
        st.rerun()

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

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import pytz
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modules import nq_data, nq_calculations

IL_TZ = pytz.timezone("Asia/Jerusalem")
ET_TZ = pytz.timezone("America/New_York")

SESSIONS = [
    {"name": "Overnight",   "il_start": "02:00", "il_end": "11:00", "et_start": "19:00", "et_end": "04:00",
     "color": "#2c3e50", "participants": "Overnight desks, Asian prop firms, HFT"},
    {"name": "Pre-Market",  "il_start": "14:30", "il_end": "16:30", "et_start": "07:30", "et_end": "09:30",
     "color": "#1a5276", "participants": "JPM/GS/MS program desks, Index Arb, Options MMs"},
    {"name": "RTH Open",    "il_start": "16:30", "il_end": "17:00", "et_start": "09:30", "et_end": "10:00",
     "color": "#1f618d", "participants": "Index Arb, CTAs, HFT — Peak Volume #1"},
    {"name": "NY Morning",  "il_start": "17:00", "il_end": "20:00", "et_start": "10:00", "et_end": "13:00",
     "color": "#6c3483", "participants": "Macro Funds, Systematic CTAs, Equity L/S HFs"},
    {"name": "Lunch Lull",  "il_start": "20:00", "il_end": "21:30", "et_start": "13:00", "et_end": "14:30",
     "color": "#922b21", "participants": "⚠️ Low liquidity — HFT dominates — avoid trading"},
    {"name": "PM Session",  "il_start": "21:30", "il_end": "22:30", "et_start": "14:30", "et_end": "15:30",
     "color": "#117a65", "participants": "Portfolio Managers, London Close, MOC accumulation"},
    {"name": "Power Hour",  "il_start": "22:30", "il_end": "23:00", "et_start": "15:30", "et_end": "16:00",
     "color": "#b7950b", "participants": "MOC orders, Day trader exits — Peak Volume #2"},
    {"name": "Cash Close",  "il_start": "23:00", "il_end": "23:10", "et_start": "16:00", "et_end": "16:10",
     "color": "#784212", "participants": "MOC execution, Index rebalancing"},
]

KEY_EVENTS_ET = [
    {"time": "08:30", "label": "CPI/PPI/NFP/Retail Sales (if scheduled)", "color": "red"},
    {"time": "09:15", "label": "⚡ Judas Swing Window starts", "color": "orange"},
    {"time": "09:30", "label": "RTH Open — Peak Volume", "color": "yellow"},
    {"time": "10:00", "label": "ISM / Consumer Confidence (if scheduled)", "color": "cyan"},
    {"time": "14:00", "label": "Fed Minutes / FOMC (if scheduled)", "color": "magenta"},
    {"time": "15:00", "label": "MOC orders begin accumulating", "color": "lime"},
    {"time": "15:30", "label": "⚡ Power Hour begins", "color": "gold"},
    {"time": "16:00", "label": "Cash Close — MOC execution", "color": "white"},
]

PARTICIPANTS_BY_SESSION = {
    "Overnight":   ["Overnight desks (skeleton crew)", "Asian prop firms", "Macro funds (passive)", "HFT (spread only)"],
    "Pre-Market":  ["Program trading desks (JPM, GS, MS)", "Index Arbitrage desks", "Options Market Makers (delta hedge)", "Retail futures traders"],
    "RTH Open":    ["Index Arb desks (NQ vs QQQ ms speed)", "Options MM (gamma hedge)", "Institutional Programs (MOO orders)", "Systematic CTAs", "HFT"],
    "NY Morning":  ["Discretionary Macro Funds", "Systematic CTAs (momentum)", "Equity Long/Short HFs", "Program trading (ongoing rebalancing)"],
    "Lunch Lull":  ["HFT (dominates thin market)", "Algorithms (spread maintenance)", "Quiet accumulation (watch for blocks)"],
    "PM Session":  ["Portfolio Managers (must finish before close)", "MOC order accumulation", "London close desks exiting"],
    "Power Hour":  ["MOC orders (index funds, mandatory)", "Day trader exits", "Window dressing (end of month/quarter)"],
    "Cash Close":  ["MOC execution", "Index rebalancing algorithms", "Settlement desks"],
}

DOW_NAMES = {0: "Monday", 1: "Tuesday", 2: "Wednesday", 3: "Thursday", 4: "Friday"}


# ─── Cached data fetchers ─────────────────────────────────────────────────────

@st.cache_data(ttl=300)
def _get_5m():
    return nq_data.get_todays_nq_data()

@st.cache_data(ttl=300)
def _get_corr():
    return nq_data.fetch_correlated_assets()

@st.cache_data(ttl=3600)
def _get_daily():
    return nq_data.load_nq_daily_cache()

@st.cache_data(ttl=3600)
def _get_hourly():
    return nq_data.load_nq_hourly_cache()

@st.cache_data(ttl=300)
def _get_15m():
    return nq_data.fetch_nq_15m(days_back=5)

@st.cache_data(ttl=120)
def _get_signal():
    """Live AI signal — 2-minute cache so it stays fresh without hammering yfinance."""
    try:
        from modules import signal_engine
        return signal_engine.generate_signal()
    except Exception as e:
        return {"direction": "NEUTRAL", "confidence": 0, "alert": False,
                "long_pts": 0, "short_pts": 0, "factors": [], "error": str(e)}


def _price_badge(val: float, label: str, delta: float = None):
    color = "normal" if delta is None else ("inverse" if delta < 0 else "normal")
    st.metric(label, f"{val:,.1f}", delta=f"{delta:+.2f}%" if delta is not None else None,
              delta_color=color)


def _sig_color(direction: str) -> str:
    return {"LONG": "#00c851", "SHORT": "#ff4444"}.get(direction, "#9e9e9e")


# ─── Main app ─────────────────────────────────────────────────────────────────

def app():
    st.title("📊 NQ Order Flow Dashboard — Daily Trading Plan")
    st.caption("Nasdaq 100 Futures | Israel Time | Order Flow · ICT/SMC · AI Signal")

    tabs = st.tabs([
        "🎯 War Plan",
        "🕐 Timeline",
        "📏 Key Levels",
        "🌊 Order Flow",
        "⚔️ ICT/SMC",
        "📜 Historical",
        "✅ Checklist",
    ])

    # ── data load ──────────────────────────────────────────────────────────────
    with st.spinner("Loading NQ data…"):
        df_5m    = _get_5m()
        df_daily = _get_daily()
        df_hourly = _get_hourly()
        df_15m   = _get_15m()
        corr     = _get_corr()
        sig      = _get_signal()

    current_session = nq_data.current_session_israel()
    levels = nq_calculations.calculate_key_levels(df_daily, df_hourly)

    current_price = float(df_5m["close"].iloc[-1]) if not df_5m.empty else None
    prev_close    = levels.get("pdc")
    pct_change    = ((current_price - prev_close) / prev_close * 100) if (current_price and prev_close) else None

    # Sprint-1 pre-computations (shared across tabs)
    opening_range = nq_calculations.calculate_opening_range(df_5m, minutes=15)
    atr_data      = nq_calculations.calculate_atr_range_consumed(df_daily, df_5m)
    ob_df         = nq_calculations.detect_order_blocks(df_5m) if not df_5m.empty else pd.DataFrame()
    ms_df         = nq_calculations.detect_market_structure(df_5m) if not df_5m.empty else pd.DataFrame()
    pd_daily      = nq_calculations.classify_premium_discount(
        current_price,
        levels.get("weekly_high", current_price or 0),
        levels.get("weekly_low",  current_price or 0),
    ) if current_price else {"pct": 50, "zone": "—", "bias": "neutral"}

    # Sprint-2 pre-computations
    vwap_data = nq_calculations.vwap_position(df_5m) if not df_5m.empty else {}
    gap_data  = nq_calculations.gap_fill_stats(df_daily)
    dow_data  = nq_calculations.day_of_week_bias(df_daily)
    vol_anom  = nq_calculations.volume_anomaly_score(df_5m) if not df_5m.empty else 1.0

    # ──────────────────────────────────────────────────────────────────────────
    # TAB 1 — War Plan
    # ──────────────────────────────────────────────────────────────────────────
    with tabs[0]:
        st.subheader("🎯 Daily War Plan")
        now_il = datetime.now(IL_TZ)
        st.markdown(f"**IL Time:** `{now_il.strftime('%H:%M:%S')}` | **Date:** `{now_il.strftime('%d/%m/%Y')}`")

        # ── Row 1: Price + session + VIX + VWAP ─────────────────────────────
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            if current_price:
                _price_badge(current_price, "💰 NQ Price", pct_change)
            else:
                st.metric("💰 NQ Price", "N/A")
        with col2:
            risk_emoji = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(current_session.get("risk", "low"), "⚪")
            st.metric("📍 Current Session", current_session.get("name", "—"))
            st.caption(f"{risk_emoji} Activity: {current_session.get('risk', '—')}")
        with col3:
            if vwap_data:
                sigma = vwap_data.get("sigma", 0)
                zone  = vwap_data.get("zone", "at")
                zone_label = {
                    "extreme_high": "🔴 +2σ Extreme",
                    "above":        "🟡 Above VWAP",
                    "at":           "🟢 At VWAP",
                    "below":        "🟡 Below VWAP",
                    "extreme_low":  "🔴 −2σ Extreme",
                }.get(zone, "—")
                st.metric("📊 VWAP", f"{vwap_data.get('vwap', 0):,.1f}")
                st.caption(f"{zone_label} ({sigma:+.2f}σ)")
            else:
                st.metric("VWAP", "N/A")
        with col4:
            vix_df = corr.get("VIX", pd.DataFrame())
            if not vix_df.empty and "close" in vix_df.columns:
                vix_val = float(vix_df["close"].dropna().iloc[-1])
                vix_color = "🔴" if vix_val > 20 else "🟡" if vix_val > 15 else "🟢"
                st.metric(f"{vix_color} VIX", f"{vix_val:.2f}")
            else:
                st.metric("VIX", "N/A")

        # ── Row 2: ATR / OR / Premium-Discount / OB ──────────────────────────
        st.markdown("---")
        s1c1, s1c2, s1c3, s1c4 = st.columns(4)
        with s1c1:
            if atr_data:
                consumed = atr_data["consumed_pct"]
                color = "🔴" if consumed > 80 else "🟡" if consumed > 55 else "🟢"
                st.metric(f"{color} ATR Range Consumed",
                          f"{consumed:.0f}%",
                          delta=f"{atr_data['remaining_pts']:.0f} pts left")
                st.caption(f"ATR: {atr_data['atr']:.0f} | Today: {atr_data['today_range']:.0f}")
            else:
                st.metric("ATR Range", "N/A")
        with s1c2:
            if opening_range:
                or_status = "✅ Complete" if opening_range.get("or_complete") else f"⏳ {opening_range.get('or_bars', 0)} bars"
                st.metric("📐 Opening Range (15m)", or_status)
                st.caption(f"H: {opening_range['or_high']:.0f} | L: {opening_range['or_low']:.0f} | Range: {opening_range['or_range']:.0f}")
            else:
                st.metric("📐 Opening Range", "⏳ Pre-Open")
                st.caption("09:30–09:45 ET | 16:30–16:45 IL")
        with s1c3:
            bias_icons = {"bullish": "🟢 Discount", "neutral": "🟡 Equilibrium", "bearish": "🔴 Premium"}
            pd_label = bias_icons.get(pd_daily["bias"], "—")
            st.metric("📊 Premium / Discount", pd_label)
            st.caption(f"{pd_daily['zone']} ({pd_daily['pct']:.1f}% of weekly range)")
        with s1c4:
            valid_obs = ob_df[ob_df["valid"] == True] if not ob_df.empty else pd.DataFrame()
            if not valid_obs.empty:
                last_ob = valid_obs.iloc[-1]
                ob_emoji = "🟢" if last_ob["type"] == "bullish" else "🔴"
                st.metric(f"{ob_emoji} Active Order Block", f"{last_ob['low']:.0f}–{last_ob['high']:.0f}")
                st.caption(f"{'Bullish' if last_ob['type'] == 'bullish' else 'Bearish'} OB | {len(valid_obs)} active")
            else:
                st.metric("Order Block", "None detected")

        # ── AI Signal Card ───────────────────────────────────────────────────
        st.divider()
        direction   = sig.get("direction", "NEUTRAL")
        confidence  = sig.get("confidence", 0)
        sig_color   = _sig_color(direction)
        long_pts    = sig.get("long_pts", 0)
        short_pts   = sig.get("short_pts", 0)
        alert       = sig.get("alert", False)
        blackout    = sig.get("blackout", False)

        st.markdown("### 🤖 AI Signal — Live Multi-Timeframe Confluence")

        ai_c1, ai_c2, ai_c3, ai_c4, ai_c5 = st.columns([2, 1, 1, 1, 2])
        with ai_c1:
            dir_emoji = {"LONG": "▲ LONG", "SHORT": "▼ SHORT", "NEUTRAL": "— NEUTRAL"}.get(direction, "—")
            st.markdown(
                f"<div style='background:{sig_color}22;border-left:4px solid {sig_color};"
                f"padding:12px 16px;border-radius:6px;'>"
                f"<span style='font-size:1.4rem;font-weight:700;color:{sig_color}'>{dir_emoji}</span>"
                f"<br><span style='font-size:0.85rem;color:#ccc'>Confidence: {confidence}%</span>"
                f"{'<br><span style=\"color:#ffd600;font-size:0.8rem\">🔔 ALERT — Trade Signal!</span>' if alert else ''}"
                f"{'<br><span style=\"color:#ff6e40;font-size:0.8rem\">⛔ BLACKOUT</span>' if blackout else ''}"
                f"</div>",
                unsafe_allow_html=True,
            )
        with ai_c2:
            st.metric("🟢 LONG pts", long_pts)
        with ai_c3:
            st.metric("🔴 SHORT pts", short_pts)
        with ai_c4:
            rr = sig.get("rr")
            st.metric("R:R", f"{rr:.1f}" if rr else "—")
        with ai_c5:
            entry  = sig.get("entry")
            stop   = sig.get("stop")
            target = sig.get("target")
            if entry and stop and target:
                risk_pts   = abs(entry - stop)
                reward_pts = abs(target - entry)
                st.caption(f"Entry: **{entry:,.0f}**")
                st.caption(f"Stop: {stop:,.0f}  (−{risk_pts:.0f} pts)")
                st.caption(f"Target: {target:,.0f}  (+{reward_pts:.0f} pts)")
            else:
                st.caption("No active trade parameters")

        # Factor mini-table
        factors = sig.get("factors", [])
        if factors:
            with st.expander("📊 Confluence Factor Breakdown", expanded=False):
                rows = []
                for f in factors:
                    side_emoji = {"LONG": "🟢", "SHORT": "🔴", "NEUTRAL": "⚪", "BOTH": "🔵", "LOW": "⬛"}.get(f.get("side", ""), "")
                    rows.append({
                        "Factor": f.get("factor", ""),
                        "Value":  str(f.get("value", "")),
                        "Side":   f"{side_emoji} {f.get('side', '')}",
                        "Pts":    f.get("pts", 0),
                    })
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        if sig.get("error"):
            st.caption(f"⚠️ Signal engine error: {sig['error']}")

        # ── Market Context: Gap + DOW + Volume Anomaly ───────────────────────
        st.divider()
        ctx_cols = st.columns(3)

        with ctx_cols[0]:
            st.markdown("**📅 Day-of-Week Edge**")
            dow_today = datetime.now(ET_TZ).weekday()
            wr = dow_data.get(dow_today)
            day_name = DOW_NAMES.get(dow_today, "Today")
            if wr is not None:
                wr_pct = round(wr * 100, 1)
                edge_color = "🟢" if wr_pct >= 55 else "🔴" if wr_pct <= 45 else "🟡"
                st.markdown(f"{edge_color} **{day_name}**: NQ closed UP **{wr_pct}%** of sessions")
                st.progress(wr)
            else:
                st.caption("Not enough history for day-of-week analysis")

        with ctx_cols[1]:
            st.markdown("**📈 Opening Gap**")
            if gap_data:
                gap_pts = gap_data.get("today_gap_pts", 0)
                gap_dir = gap_data.get("today_gap_dir", "none")
                if gap_dir == "up":
                    fill_pct = gap_data.get("gap_up_fill_pct")
                    st.markdown(f"🔼 **Gap UP** {gap_pts:+.0f} pts")
                    if fill_pct:
                        st.caption(f"Historical fill rate: **{fill_pct}%**")
                elif gap_dir == "down":
                    fill_pct = gap_data.get("gap_down_fill_pct")
                    st.markdown(f"🔽 **Gap DOWN** {gap_pts:+.0f} pts")
                    if fill_pct:
                        st.caption(f"Historical fill rate: **{fill_pct}%**")
                else:
                    st.markdown("↔️ **No significant gap** today")
            else:
                st.caption("Gap data unavailable")

        with ctx_cols[2]:
            st.markdown("**📊 Volume Anomaly**")
            vanom_color = "🔴" if vol_anom > 2.0 else "🟢" if vol_anom > 1.2 else "🟡" if vol_anom > 0.7 else "⬛"
            label = "Institutional activity" if vol_anom > 2.0 else \
                    "Above average" if vol_anom > 1.2 else \
                    "Normal" if vol_anom > 0.7 else "Low conviction"
            st.markdown(f"{vanom_color} **{vol_anom:.2f}×** average")
            st.caption(label)
            if vol_anom > 2.0:
                st.warning("⚠️ Unusual volume — institutions active")

        st.divider()

        # ── Smart Banners ─────────────────────────────────────────────────────
        banners = []
        if not df_5m.empty:
            fvg_df = nq_calculations.detect_fvg(df_5m)
            if not fvg_df.empty:
                last_fvg = fvg_df.iloc[-1]
                banners.append(("info", f"🟦 **FVG {last_fvg['type'].upper()}** at {last_fvg['bottom']:.0f}–{last_fvg['top']:.0f} — potential price magnet"))
            div_series = nq_calculations.detect_delta_divergence(df_5m)
            last_div = div_series.iloc[-4:].sum()
            if last_div > 0:
                banners.append(("warning", "⚠️ **Bearish Delta Divergence** — price rising but selling pressure dominates"))
            elif last_div < 0:
                banners.append(("success", "📈 **Bullish Delta Divergence** — price falling but buying pressure dominates"))
            stacks = nq_calculations.detect_stacked_imbalances(df_5m)
            if stacks:
                last_stack = stacks[-1]
                banners.append(("info", f"📚 **Stacked Imbalance {last_stack['direction'].upper()}** — {last_stack['candle_count']} consecutive candles | {last_stack['price_start']:.0f}→{last_stack['price_end']:.0f}"))
            if opening_range and opening_range.get("or_complete") and current_price:
                if current_price > opening_range["or_high"]:
                    banners.append(("success", f"📐 **OR Breakout BULLISH** — broke above OR High ({opening_range['or_high']:.0f}) — bullish session bias"))
                elif current_price < opening_range["or_low"]:
                    banners.append(("error", f"📐 **OR Breakdown BEARISH** — broke below OR Low ({opening_range['or_low']:.0f}) — bearish session bias"))
            if not ms_df.empty:
                last_ms = ms_df.iloc[-1]
                ms_is_mss = "MSS" in last_ms["label"]
                ms_type = "warning" if ms_is_mss else "info"
                ms_text = "**MSS — Trend Reversal!**" if ms_is_mss else "**BOS — Trend Continuation**"
                ms_emoji = "🚀" if "bullish" in last_ms["type"] else "📉"
                banners.append((ms_type, f"{ms_emoji} {ms_text} | {last_ms['label']} @ {last_ms['price']:.0f}"))
            if not ob_df.empty and current_price:
                valid_obs_nearby = ob_df[ob_df["valid"] == True]
                for _, ob in valid_obs_nearby.tail(3).iterrows():
                    dist = abs(current_price - (ob["high"] + ob["low"]) / 2)
                    if dist / current_price < 0.003:
                        ob_emoji = "🟢" if ob["type"] == "bullish" else "🔴"
                        banners.append(("info", f"{ob_emoji} **{ob['type'].upper()} Order Block** — price within {dist:.0f} pts of OB at {ob['low']:.0f}–{ob['high']:.0f}"))
            if atr_data and atr_data.get("consumed_pct", 0) > 85:
                banners.append(("error", f"⚠️ **ATR {atr_data['consumed_pct']:.0f}% consumed** — daily range nearly exhausted, poor R:R"))
            # VWAP extreme
            if vwap_data and abs(vwap_data.get("sigma", 0)) > 2.0:
                sigma = vwap_data.get("sigma", 0)
                side  = "above" if sigma > 0 else "below"
                banners.append(("warning", f"📊 **VWAP Extreme** — price is {abs(sigma):.1f}σ {side} VWAP — mean reversion risk"))

        if current_session.get("name") == "Lunch Lull":
            banners.insert(0, ("error", "⛔ **Lunch Lull active** — low liquidity, false breakouts common, avoid trading"))

        if banners:
            for btype, btext in banners:
                getattr(st, btype)(btext)
        else:
            st.success("✅ No special signals at this time — normal market conditions")

        # ── Correlated Assets ─────────────────────────────────────────────────
        st.divider()
        st.subheader("📈 Correlated Assets")
        asset_cols = st.columns(5)
        asset_map = [("VIX", "Volatility"), ("DXY", "Dollar"), ("TNX", "10Y Yield"), ("ES", "S&P500 Futs"), ("Gold", "Gold")]
        for i, (name, label) in enumerate(asset_map):
            df_a = corr.get(name, pd.DataFrame())
            with asset_cols[i]:
                if not df_a.empty and "close" in df_a.columns:
                    vals = df_a["close"].dropna()
                    if len(vals) >= 2:
                        chg = (vals.iloc[-1] / vals.iloc[-2] - 1) * 100
                        st.metric(f"{name} ({label})", f"{vals.iloc[-1]:.2f}", delta=f"{chg:+.2f}%",
                                  delta_color="inverse" if name in ("VIX", "TNX") else "normal")
                    else:
                        st.metric(name, "—")
                else:
                    st.metric(name, "N/A")

        # ── Session Participants ───────────────────────────────────────────────
        participants = PARTICIPANTS_BY_SESSION.get(current_session.get("name", ""), [])
        if participants:
            st.divider()
            st.subheader(f"👥 Session Participants: {current_session.get('name', '')}")
            for p in participants:
                st.markdown(f"• {p}")

    # ──────────────────────────────────────────────────────────────────────────
    # TAB 2 — Session Timeline
    # ──────────────────────────────────────────────────────────────────────────
    with tabs[1]:
        st.subheader("🕐 Session Timeline — Israel Time")

        now_il = datetime.now(IL_TZ)
        now_minutes = now_il.hour * 60 + now_il.minute

        def hm_to_min(s):
            h, m = map(int, s.split(":"))
            return h * 60 + m

        fig_tl = go.Figure()
        for i, s in enumerate(SESSIONS):
            start_m = hm_to_min(s["il_start"])
            end_m   = hm_to_min(s["il_end"])
            if end_m < start_m:
                end_m += 24 * 60
            is_active = start_m <= now_minutes < end_m
            opacity = 1.0 if is_active else 0.5
            fig_tl.add_trace(go.Bar(
                x=[end_m - start_m],
                y=[s["name"]],
                base=[start_m],
                orientation="h",
                marker=dict(color=s["color"], opacity=opacity,
                            line=dict(color="white" if is_active else "rgba(255,255,255,0.2)", width=2 if is_active else 0.5)),
                text=f"{s['il_start']}–{s['il_end']} IL / {s['et_start']}–{s['et_end']} ET",
                textposition="inside",
                hovertemplate=f"<b>{s['name']}</b><br>{s['il_start']}–{s['il_end']} IL<br>{s['et_start']}–{s['et_end']} ET<br>{s['participants']}<extra></extra>",
                name=s["name"],
                showlegend=False,
            ))

        fig_tl.add_vline(x=now_minutes, line_dash="dash", line_color="red", line_width=2,
                         annotation_text=f"⏰ {now_il.strftime('%H:%M')} IL",
                         annotation_position="top")
        for ev in KEY_EVENTS_ET:
            try:
                ev_il = nq_data.et_to_israel(ev["time"])
                ev_il_min = hm_to_min(ev_il)
                fig_tl.add_vline(x=ev_il_min, line_dash="dot", line_color=ev["color"], line_width=1,
                                 annotation_text=f"{ev_il} ✦", annotation_position="bottom right",
                                 annotation_font_size=9)
            except Exception:
                pass

        tick_vals = list(range(0, 1441, 60))
        tick_text = [f"{v // 60:02d}:00" for v in tick_vals]
        fig_tl.update_layout(
            title="Session Timeline — Israel Time (UTC +2/+3)",
            xaxis=dict(title="Hour (IL)", range=[0, 1440], tickvals=tick_vals, ticktext=tick_text),
            yaxis=dict(title=""),
            height=420,
            paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
            font=dict(color="white"), bargap=0.3,
        )
        st.plotly_chart(fig_tl, use_container_width=True)

        st.divider()
        st.subheader("👥 Participants by Session")
        for s in SESSIONS:
            parts = PARTICIPANTS_BY_SESSION.get(s["name"], [])
            with st.expander(f"{s['name']} | {s['il_start']}–{s['il_end']} IL"):
                cols = st.columns(2)
                with cols[0]:
                    for p in parts:
                        st.markdown(f"• {p}")
                with cols[1]:
                    st.caption(s["participants"])

    # ──────────────────────────────────────────────────────────────────────────
    # TAB 3 — Key Levels & Volume Profile
    # ──────────────────────────────────────────────────────────────────────────
    with tabs[2]:
        st.subheader("📏 Key Levels & Volume Profile")

        # level_colors defined here so it's always in scope for the levels table below
        level_colors = {
            "pdh":           ("Prior Day High",  "red"),
            "pdl":           ("Prior Day Low",   "green"),
            "pdc":           ("Prior Day Close", "gray"),
            "weekly_high":   ("Weekly High",     "darkred"),
            "weekly_low":    ("Weekly Low",      "darkgreen"),
            "poc_20d":       ("POC 20d",         "gold"),
            "poc_5d":        ("POC 5d",          "orange"),
            "val_20d":       ("VAL 20d",         "#8888ff"),
            "vah_20d":       ("VAH 20d",         "#ff88ff"),
            "overnight_high":("Overnight High",  "lightblue"),
            "overnight_low": ("Overnight Low",   "lightblue"),
        }

        if df_5m.empty:
            st.warning("5-minute data unavailable — market may be closed")
        else:
            fig_kl = make_subplots(rows=1, cols=2, column_widths=[0.75, 0.25],
                                   shared_yaxes=True,
                                   subplot_titles=["NQ 5m — Today + VWAP", "Volume Profile"])

            dt_col = "datetime_il" if "datetime_il" in df_5m.columns else "datetime"
            fig_kl.add_trace(go.Candlestick(
                x=df_5m[dt_col],
                open=df_5m["open"], high=df_5m["high"],
                low=df_5m["low"],  close=df_5m["close"],
                name="NQ 5m",
                increasing_line_color="#26a69a",
                decreasing_line_color="#ef5350",
            ), row=1, col=1)

            # ── VWAP + ±1σ / ±2σ bands ───────────────────────────────────────
            if len(df_5m) >= 5:
                try:
                    vwap_line = nq_calculations.calculate_vwap(df_5m)
                    b1u, b1l  = nq_calculations.calculate_vwap_bands(df_5m, n_std=1.0)
                    b2u, b2l  = nq_calculations.calculate_vwap_bands(df_5m, n_std=2.0)

                    fig_kl.add_trace(go.Scatter(
                        x=df_5m[dt_col], y=vwap_line,
                        mode="lines", name="VWAP",
                        line=dict(color="#ffd600", width=2, dash="solid"),
                        opacity=0.9,
                    ), row=1, col=1)
                    fig_kl.add_trace(go.Scatter(
                        x=df_5m[dt_col], y=b1u,
                        mode="lines", name="+1σ",
                        line=dict(color="#ffd600", width=1, dash="dot"), opacity=0.5,
                        showlegend=True,
                    ), row=1, col=1)
                    fig_kl.add_trace(go.Scatter(
                        x=df_5m[dt_col], y=b1l,
                        mode="lines", name="−1σ",
                        line=dict(color="#ffd600", width=1, dash="dot"), opacity=0.5,
                        showlegend=True,
                    ), row=1, col=1)
                    fig_kl.add_trace(go.Scatter(
                        x=df_5m[dt_col], y=b2u,
                        mode="lines", name="+2σ",
                        line=dict(color="orange", width=1, dash="dot"), opacity=0.4,
                        showlegend=True,
                    ), row=1, col=1)
                    fig_kl.add_trace(go.Scatter(
                        x=df_5m[dt_col], y=b2l,
                        mode="lines", name="−2σ",
                        line=dict(color="orange", width=1, dash="dot"), opacity=0.4,
                        showlegend=True,
                    ), row=1, col=1)
                except Exception:
                    pass

            # ── Key level lines ───────────────────────────────────────────────
            for key, (label, color) in level_colors.items():
                val = levels.get(key)
                if val and not np.isnan(val):
                    fig_kl.add_hline(y=val, line_dash="dash" if "poc" not in key else "dot",
                                     line_color=color, line_width=1,
                                     annotation_text=f"{label}: {val:.0f}",
                                     annotation_position="left",
                                     row=1, col=1)

            # ── FVG zones ─────────────────────────────────────────────────────
            fvg_df = nq_calculations.detect_fvg(df_5m)
            if not fvg_df.empty:
                x_end = df_5m[dt_col].iloc[-1]
                for _, fvg in fvg_df.iterrows():
                    fig_kl.add_shape(
                        type="rect",
                        x0=fvg["datetime"], x1=x_end,
                        y0=fvg["bottom"], y1=fvg["top"],
                        fillcolor="rgba(0,255,100,0.12)" if fvg["type"] == "bullish" else "rgba(255,50,50,0.12)",
                        line=dict(width=0), row=1, col=1,
                    )

            # ── Opening Range overlay ─────────────────────────────────────────
            if opening_range and opening_range.get("or_complete"):
                x_start = df_5m[dt_col].iloc[0]
                x_end   = df_5m[dt_col].iloc[-1]
                fig_kl.add_shape(type="rect",
                    x0=x_start, x1=x_end,
                    y0=opening_range["or_low"], y1=opening_range["or_high"],
                    fillcolor="rgba(255,215,0,0.07)", line=dict(color="gold", width=1, dash="dot"),
                    row=1, col=1)
                for or_key, or_label in [("or_high", f"OR H {opening_range['or_high']:.0f}"),
                                          ("or_low",  f"OR L {opening_range['or_low']:.0f}"),
                                          ("or_mid",  f"OR Mid {opening_range['or_mid']:.0f}")]:
                    val = opening_range.get(or_key)
                    if val:
                        fig_kl.add_hline(y=val, line_dash="dot", line_color="gold", line_width=1,
                                         annotation_text=or_label, annotation_position="right",
                                         row=1, col=1)

            # ── Order Block overlays ──────────────────────────────────────────
            if not ob_df.empty:
                x_end = df_5m[dt_col].iloc[-1]
                for _, ob in ob_df[ob_df["valid"] == True].tail(5).iterrows():
                    fill   = "rgba(0,200,100,0.15)" if ob["type"] == "bullish" else "rgba(220,50,50,0.15)"
                    border = "rgba(0,200,100,0.6)"  if ob["type"] == "bullish" else "rgba(220,50,50,0.6)"
                    fig_kl.add_shape(type="rect",
                        x0=ob["datetime"], x1=x_end,
                        y0=ob["low"], y1=ob["high"],
                        fillcolor=fill, line=dict(color=border, width=1), row=1, col=1)
                    fig_kl.add_annotation(
                        x=x_end, y=(ob["high"] + ob["low"]) / 2,
                        text=f"{'B' if ob['type'] == 'bullish' else 'S'}-OB",
                        font=dict(size=9, color=border), showarrow=False,
                        xanchor="right", row=1, col=1)

            # ── EQH / EQL liquidity lines from signal ─────────────────────────
            liquidity = sig.get("liquidity", {})
            x_start = df_5m[dt_col].iloc[0]
            x_end   = df_5m[dt_col].iloc[-1]
            for lvl in liquidity.get("eqh", []):
                fig_kl.add_shape(type="line",
                    x0=x_start, x1=x_end, y0=lvl, y1=lvl,
                    line=dict(color="#e040fb", width=1, dash="dot"), row=1, col=1)
                fig_kl.add_annotation(x=x_end, y=lvl, text=f"EQH {lvl:.0f}",
                    font=dict(size=8, color="#e040fb"), showarrow=False, xanchor="right", row=1, col=1)
            for lvl in liquidity.get("eql", []):
                fig_kl.add_shape(type="line",
                    x0=x_start, x1=x_end, y0=lvl, y1=lvl,
                    line=dict(color="#40c4ff", width=1, dash="dot"), row=1, col=1)
                fig_kl.add_annotation(x=x_end, y=lvl, text=f"EQL {lvl:.0f}",
                    font=dict(size=8, color="#40c4ff"), showarrow=False, xanchor="right", row=1, col=1)
            for lvl in liquidity.get("buy_side", []):
                fig_kl.add_shape(type="line",
                    x0=x_start, x1=x_end, y0=lvl, y1=lvl,
                    line=dict(color="#b2ff59", width=1, dash="dash"), row=1, col=1)
            for lvl in liquidity.get("sell_side", []):
                fig_kl.add_shape(type="line",
                    x0=x_start, x1=x_end, y0=lvl, y1=lvl,
                    line=dict(color="#ff6e40", width=1, dash="dash"), row=1, col=1)

            # ── Volume profile ────────────────────────────────────────────────
            vp = nq_calculations.calculate_volume_profile(df_5m, price_bins=40)
            if not vp.empty:
                fig_kl.add_trace(go.Bar(
                    x=vp["volume"], y=vp["price"], orientation="h",
                    marker_color="rgba(100,180,255,0.5)", name="Volume Profile",
                    hovertemplate="Price: %{y:.0f}<br>Vol: %{x:,.0f}<extra></extra>",
                ), row=1, col=2)
                poc = nq_calculations.get_poc(vp)
                if poc:
                    fig_kl.add_hline(y=poc, line_dash="dash", line_color="gold",
                                     annotation_text=f"POC: {poc:.0f}", row=1, col=2)

            fig_kl.update_layout(
                height=580, paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
                font=dict(color="white"), xaxis_rangeslider_visible=False,
                legend=dict(orientation="h", y=1.05),
            )
            st.plotly_chart(fig_kl, use_container_width=True)

        # ── Levels table ──────────────────────────────────────────────────────
        st.subheader("📋 Levels Table")
        if current_price and levels:
            rows = []
            for key, (label, _) in level_colors.items():
                val = levels.get(key)
                if val and not np.isnan(val):
                    dist = val - current_price
                    rows.append({"Level": label, "Price": f"{val:.0f}",
                                 "Distance (pts)": f"{dist:+.0f}",
                                 "Type": "Support 🟢" if dist < 0 else "Resistance 🔴"})
            if opening_range:
                for or_key, or_label in [("or_high", "OR High (15m)"), ("or_low", "OR Low (15m)"), ("or_mid", "OR Mid")]:
                    val = opening_range.get(or_key)
                    if val:
                        dist = val - current_price
                        rows.append({"Level": f"📐 {or_label}", "Price": f"{val:.0f}",
                                     "Distance (pts)": f"{dist:+.0f}",
                                     "Type": "Support 🟢" if dist < 0 else "Resistance 🔴"})
            if not ob_df.empty:
                for _, ob in ob_df[ob_df["valid"] == True].tail(3).iterrows():
                    mid  = (ob["high"] + ob["low"]) / 2
                    dist = mid - current_price
                    emoji = "🟢" if ob["type"] == "bullish" else "🔴"
                    rows.append({"Level": f"{emoji} {ob['type'].capitalize()} OB",
                                 "Price": f"{ob['low']:.0f}–{ob['high']:.0f}",
                                 "Distance (pts)": f"{dist:+.0f}",
                                 "Type": "Support 🟢" if ob["type"] == "bullish" else "Resistance 🔴"})
            # VWAP levels
            if vwap_data:
                for key, label in [("vwap", "VWAP"), ("band1_upper", "+1σ VWAP"), ("band1_lower", "−1σ VWAP"),
                                   ("band2_upper", "+2σ VWAP"), ("band2_lower", "−2σ VWAP")]:
                    val = vwap_data.get(key)
                    if val:
                        dist = val - current_price
                        rows.append({"Level": f"📊 {label}", "Price": f"{val:.0f}",
                                     "Distance (pts)": f"{dist:+.0f}",
                                     "Type": "Support 🟢" if dist < 0 else "Resistance 🔴"})
            if rows:
                df_lev = pd.DataFrame(rows)
                df_lev["Distance (pts)"] = pd.to_numeric(df_lev["Distance (pts)"].str.replace("+", ""), errors="coerce")
                df_lev = df_lev.sort_values("Distance (pts)").reset_index(drop=True)
                df_lev["Distance (pts)"] = df_lev["Distance (pts)"].apply(lambda x: f"{x:+.0f}" if pd.notna(x) else "—")
                st.dataframe(df_lev, use_container_width=True, hide_index=True)

    # ──────────────────────────────────────────────────────────────────────────
    # TAB 4 — Order Flow Indicators
    # ──────────────────────────────────────────────────────────────────────────
    with tabs[3]:
        st.subheader("🌊 Order Flow Indicators")

        if df_5m.empty:
            st.warning("5-minute data unavailable")
        else:
            cd         = nq_calculations.cumulative_delta(df_5m)
            div        = nq_calculations.detect_delta_divergence(df_5m)
            absorption = nq_calculations.detect_absorption(df_5m)
            dz         = nq_calculations.delta_zscore(df_5m)

            bearish_div = int((div.iloc[-8:] > 0).sum())
            bullish_div = int((div.iloc[-8:] < 0).sum())
            if bearish_div > 0:
                st.error(f"⚠️ Bearish Delta Divergence ({bearish_div} signals in last 8 bars) — Distribution: price rising but selling dominates")
            elif bullish_div > 0:
                st.success(f"📈 Bullish Delta Divergence ({bullish_div} signals in last 8 bars) — Accumulation: price falling but buying dominates")

            # Volume anomaly alert
            vol_c1, vol_c2 = st.columns(2)
            with vol_c1:
                vanom_color = "🔴" if vol_anom > 2.0 else "🟢" if vol_anom > 1.2 else "🟡"
                st.metric(f"{vanom_color} Volume Anomaly", f"{vol_anom:.2f}×",
                          delta="Institutional" if vol_anom > 2.0 else "Normal")
            with vol_c2:
                dz_now = float(dz.iloc[-1]) if not dz.empty else 0
                dz_color = "🔴" if dz_now > 2 else "🟢" if dz_now < -2 else "🟡"
                st.metric(f"{dz_color} Delta Z-Score (last bar)", f"{dz_now:+.2f}σ",
                          delta="Extreme selling" if dz_now > 2 else "Extreme buying" if dz_now < -2 else "Normal")

            dt_col = "datetime_il" if "datetime_il" in df_5m.columns else "datetime"

            # 3-panel chart: Price / Cumulative Delta / Delta Z-score
            fig_cd = make_subplots(rows=3, cols=1, shared_xaxes=True,
                                   row_heights=[0.5, 0.3, 0.2],
                                   subplot_titles=["NQ Price + VWAP", "Cumulative Delta", "Delta Z-Score (σ)"])

            fig_cd.add_trace(go.Candlestick(
                x=df_5m[dt_col], open=df_5m["open"], high=df_5m["high"],
                low=df_5m["low"], close=df_5m["close"],
                increasing_line_color="#26a69a", decreasing_line_color="#ef5350",
                name="NQ", showlegend=False,
            ), row=1, col=1)

            # VWAP on price chart
            try:
                vwap_line = nq_calculations.calculate_vwap(df_5m)
                fig_cd.add_trace(go.Scatter(
                    x=df_5m[dt_col], y=vwap_line,
                    mode="lines", name="VWAP",
                    line=dict(color="#ffd600", width=1.5),
                    showlegend=True,
                ), row=1, col=1)
            except Exception:
                pass

            # Divergence markers
            div_up = df_5m[div < 0]
            div_dn = df_5m[div > 0]
            if not div_up.empty:
                fig_cd.add_trace(go.Scatter(x=div_up[dt_col], y=div_up["low"] * 0.9995,
                                            mode="markers", marker=dict(symbol="triangle-up", size=10, color="lime"),
                                            name="Bullish Div"), row=1, col=1)
            if not div_dn.empty:
                fig_cd.add_trace(go.Scatter(x=div_dn[dt_col], y=div_dn["high"] * 1.0005,
                                            mode="markers", marker=dict(symbol="triangle-down", size=10, color="red"),
                                            name="Bearish Div"), row=1, col=1)

            # Absorption markers
            abs_bull = df_5m[absorption == 1]
            abs_bear = df_5m[absorption == -1]
            if not abs_bull.empty:
                fig_cd.add_trace(go.Scatter(x=abs_bull[dt_col], y=abs_bull["close"],
                                            mode="markers", marker=dict(symbol="square", size=8, color="cyan", opacity=0.7),
                                            name="Bullish Absorption"), row=1, col=1)
            if not abs_bear.empty:
                fig_cd.add_trace(go.Scatter(x=abs_bear[dt_col], y=abs_bear["close"],
                                            mode="markers", marker=dict(symbol="square", size=8, color="magenta", opacity=0.7),
                                            name="Bearish Absorption"), row=1, col=1)

            # BOS / MSS markers
            if not ms_df.empty:
                for _, ev in ms_df.iterrows():
                    is_mss = "MSS" in ev["label"]
                    color  = "cyan" if "bullish" in ev["type"] else "orange"
                    symbol = "triangle-up" if "bullish" in ev["type"] else "triangle-down"
                    y_pos  = ev["price"] * 0.9998 if "bullish" in ev["type"] else ev["price"] * 1.0002
                    fig_cd.add_trace(go.Scatter(
                        x=[ev["datetime"]], y=[y_pos],
                        mode="markers+text",
                        marker=dict(symbol=symbol, size=12 if is_mss else 9,
                                    color=color, line=dict(color="white", width=1)),
                        text=[ev["label"]], textposition="bottom center" if "bullish" in ev["type"] else "top center",
                        textfont=dict(size=8, color=color),
                        name=ev["label"], showlegend=False,
                    ), row=1, col=1)

            # Order Block zones
            if not ob_df.empty:
                x_end = df_5m[dt_col].iloc[-1]
                for _, ob in ob_df[ob_df["valid"] == True].tail(4).iterrows():
                    fill   = "rgba(0,200,100,0.12)" if ob["type"] == "bullish" else "rgba(220,50,50,0.12)"
                    border = "rgba(0,200,100,0.5)"  if ob["type"] == "bullish" else "rgba(220,50,50,0.5)"
                    fig_cd.add_shape(type="rect",
                        x0=ob["datetime"], x1=x_end, y0=ob["low"], y1=ob["high"],
                        fillcolor=fill, line=dict(color=border, width=1), row=1, col=1)

            # Cumulative delta bars
            cd_colors = ["#26a69a" if v >= 0 else "#ef5350" for v in cd]
            fig_cd.add_trace(go.Bar(x=df_5m[dt_col], y=cd, marker_color=cd_colors,
                                    name="Cumulative Delta", showlegend=False), row=2, col=1)
            fig_cd.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=1)

            # Delta Z-score line
            dz_colors = ["#ef5350" if v > 2 else "#26a69a" if v < -2 else "#9e9e9e" for v in dz]
            fig_cd.add_trace(go.Bar(x=df_5m[dt_col], y=dz, marker_color=dz_colors,
                                    name="Delta Z-Score", showlegend=False), row=3, col=1)
            fig_cd.add_hline(y=2,  line_dash="dot", line_color="red",  opacity=0.5, row=3, col=1)
            fig_cd.add_hline(y=-2, line_dash="dot", line_color="lime", opacity=0.5, row=3, col=1)
            fig_cd.add_hline(y=0,  line_dash="dash", line_color="gray", row=3, col=1)

            fig_cd.update_layout(height=700, paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
                                 font=dict(color="white"), xaxis_rangeslider_visible=False)
            st.plotly_chart(fig_cd, use_container_width=True)

            # Stacked Imbalances
            stacks = nq_calculations.detect_stacked_imbalances(df_5m, min_stack=3)
            if stacks:
                st.subheader("📚 Stacked Imbalances Detected")
                for s in stacks[-5:]:
                    direction_emoji = "🟢" if s["direction"] == "bullish" else "🔴"
                    st.markdown(f"{direction_emoji} **{s['direction'].upper()}** — {s['candle_count']} candles | {s['price_start']:.0f} → {s['price_end']:.0f}")

            # Absorption table
            if absorption.abs().sum() > 0:
                st.subheader("🧲 Absorption Events")
                abs_rows = []
                for i, val in enumerate(absorption):
                    if val != 0:
                        row = df_5m.iloc[i]
                        abs_rows.append({
                            "Time":      str(row[dt_col]).split("+")[0],
                            "Direction": "🟢 Bullish" if val > 0 else "🔴 Bearish",
                            "Volume":    f"{int(row['volume']):,}",
                            "Price":     f"{row['close']:.0f}",
                            "Range":     f"{(row['high'] - row['low']):.1f}",
                        })
                if abs_rows:
                    st.dataframe(pd.DataFrame(abs_rows), use_container_width=True, hide_index=True)

    # ──────────────────────────────────────────────────────────────────────────
    # TAB 5 — ICT/SMC Analysis
    # ──────────────────────────────────────────────────────────────────────────
    with tabs[4]:
        st.subheader("⚔️ ICT/SMC Analysis")

        # ── Killzones ─────────────────────────────────────────────────────────
        st.markdown("### 🎯 Killzones")
        kz_col1, kz_col2, kz_col3 = st.columns(3)
        now_il = datetime.now(IL_TZ)
        now_et = now_il.astimezone(ET_TZ)
        now_et_hm = now_et.hour * 60 + now_et.minute

        def in_kz(start_hm, end_hm):
            s_min = int(start_hm[:2]) * 60 + int(start_hm[3:])
            e_min = int(end_hm[:2]) * 60 + int(end_hm[3:])
            return s_min <= now_et_hm < e_min

        kz_configs = [
            ("🏙️ London KZ", "02:00", "05:00", "07:00–11:00 IL"),
            ("🗽 NY AM KZ",  "07:00", "10:00", "14:00–17:00 IL"),
            ("🌆 NY PM KZ",  "13:30", "16:00", "20:30–23:00 IL"),
        ]
        for col, (title, start_et, end_et, il_label) in zip([kz_col1, kz_col2, kz_col3], kz_configs):
            with col:
                active = in_kz(start_et, end_et)
                status = "🟢 Active now" if active else "⚫ Not active"
                st.metric(title, status)
                st.caption(f"ET: {start_et}–{end_et} | {il_label}")

        st.divider()

        # ── AMD Cycle ─────────────────────────────────────────────────────────
        st.markdown("### 🔄 AMD Cycle — Accumulation / Manipulation / Distribution")
        amd_col1, amd_col2, amd_col3 = st.columns(3)

        def amd_phase():
            if now_et_hm < 7 * 60 + 30:
                return "Accumulation", "💡 Silent position building — overnight discovering directional bias"
            elif now_et_hm < 10 * 60:
                return "Manipulation", "⚡ Judas Swing window — false move before true direction"
            else:
                return "Distribution", "📤 Distribution phase — institutions selling/buying to retail"

        current_phase, phase_desc = amd_phase()
        with amd_col1:
            color = "🔵" if current_phase == "Accumulation" else "⚫"
            st.metric("💡 Accumulation", color + " Active" if current_phase == "Accumulation" else "⚫ Complete")
            st.caption("Overnight + Pre-Market")
        with amd_col2:
            color = "🟡" if current_phase == "Manipulation" else "⚫"
            st.metric("⚡ Manipulation", color + " Active" if current_phase == "Manipulation" else "⚫ Complete")
            st.caption("09:15–09:45 ET (Judas Swing)")
        with amd_col3:
            color = "🔴" if current_phase == "Distribution" else "⚫"
            st.metric("📤 Distribution", color + " Active" if current_phase == "Distribution" else "⚫ Not yet")
            st.caption("RTH 10:00+ ET")
        st.info(f"**Current phase: {current_phase}** — {phase_desc}")

        # ── Judas Swing ───────────────────────────────────────────────────────
        st.divider()
        st.markdown("### ⚡ Judas Swing Detector")
        overnight_high = levels.get("overnight_high")
        overnight_low  = levels.get("overnight_low")
        if not df_5m.empty and overnight_high and overnight_low:
            judas = nq_calculations.detect_judas_swing(df_5m, overnight_high, overnight_low)
            if judas.get("detected"):
                if judas.get("direction") == "bearish":
                    st.error(f"🚨 **Bearish Judas Swing!** — swept above Overnight High ({overnight_high:.0f}), "
                             f"reached {judas.get('sweep_high', 0):.0f}, then reversed → true direction: DOWN")
                else:
                    st.success(f"🚀 **Bullish Judas Swing!** — swept below Overnight Low ({overnight_low:.0f}), "
                               f"reached {judas.get('sweep_low', 0):.0f}, then reversed → true direction: UP")
            else:
                st.info(f"📊 Overnight Range: {overnight_low:.0f} – {overnight_high:.0f} | No Judas Swing detected yet")
        else:
            st.info("📊 Overnight High/Low not available — check after 09:30 ET (16:30 IL)")

        # ── Opening Range ─────────────────────────────────────────────────────
        st.divider()
        st.markdown("### 📐 Opening Range (OR 15 min)")
        if opening_range:
            or_col1, or_col2, or_col3, or_col4 = st.columns(4)
            with or_col1: st.metric("OR High", f"{opening_range.get('or_high', 0):.0f}")
            with or_col2: st.metric("OR Low",  f"{opening_range.get('or_low', 0):.0f}")
            with or_col3: st.metric("OR Range (pts)", f"{opening_range.get('or_range', 0):.1f}")
            with or_col4:
                status = "✅ Complete" if opening_range.get("or_complete") else f"⏳ {opening_range.get('or_bars', 0)}/3 bars"
                st.metric("Status", status)
            if current_price and opening_range.get("or_complete"):
                if current_price > opening_range["or_high"]:
                    st.success(f"🚀 **OR Breakout Bullish** — price ({current_price:.0f}) above OR High ({opening_range['or_high']:.0f}). OR High → support on retest.")
                elif current_price < opening_range["or_low"]:
                    st.error(f"📉 **OR Breakdown Bearish** — price ({current_price:.0f}) below OR Low ({opening_range['or_low']:.0f}). OR Low → resistance on retest.")
                else:
                    st.info("📊 Price **inside** Opening Range — wait for a decisive breakout.")
        else:
            st.info("⏳ Opening Range will be calculated after 09:45 ET (16:45 IL)")

        # ── OTE Zone (Optimal Trade Entry) ────────────────────────────────────
        st.divider()
        st.markdown("### 🎯 OTE — Optimal Trade Entry Zone (ICT Fibonacci)")
        st.caption("OTE = 61.8%–78.6% retracement of the last impulse wave — the institutional entry window")

        if not df_5m.empty and len(df_5m) >= 10:
            try:
                sh_df, sl_df = nq_calculations.detect_swings(df_5m, n=3)
                if not sh_df.empty and not sl_df.empty:
                    htf_bias_val = sig.get("htf_bias", "neutral")
                    # Use most recent swing for OTE
                    last_sh = float(sh_df.iloc[-1]["price"])
                    last_sl = float(sl_df.iloc[-1]["price"])
                    ote_direction = "bullish" if htf_bias_val == "bullish" else "bearish"
                    ote = nq_calculations.calculate_ote_zone(last_sl, last_sh, direction=ote_direction)

                    if ote:
                        ote_c1, ote_c2, ote_c3, ote_c4 = st.columns(4)
                        with ote_c1:
                            st.metric("OTE Direction", "🟢 Bullish" if ote_direction == "bullish" else "🔴 Bearish")
                        with ote_c2:
                            st.metric("OTE Zone", f"{ote['ote_low']:.0f}–{ote['ote_high']:.0f}")
                        with ote_c3:
                            st.metric("61.8% Fib", f"{ote['fib_618']:.0f}")
                        with ote_c4:
                            st.metric("78.6% Fib", f"{ote['fib_786']:.0f}")

                        if current_price:
                            in_ote = ote["ote_low"] <= current_price <= ote["ote_high"]
                            if in_ote:
                                st.success(f"✅ **Price is IN the OTE zone** ({ote['ote_low']:.0f}–{ote['ote_high']:.0f}) — optimal institutional entry area!")
                            else:
                                dist_to_ote = min(abs(current_price - ote["ote_low"]), abs(current_price - ote["ote_high"]))
                                st.info(f"📊 Price is {dist_to_ote:.0f} pts from OTE zone ({ote['ote_low']:.0f}–{ote['ote_high']:.0f})")
                    else:
                        st.info("OTE calculation requires a clear impulse swing")
                else:
                    st.info("Not enough swing data for OTE — need more intraday bars")
            except Exception as e:
                st.caption(f"OTE: {e}")
        else:
            st.info("OTE requires intraday 5m data")

        # ── Order Blocks ──────────────────────────────────────────────────────
        st.divider()
        st.markdown("### 🟥🟩 Order Blocks")
        if not ob_df.empty:
            valid_obs   = ob_df[ob_df["valid"] == True]
            invalid_obs = ob_df[ob_df["valid"] == False]
            ob_display_cols = st.columns(2)
            with ob_display_cols[0]:
                st.markdown(f"**✅ Active OBs: {len(valid_obs)}**")
                if not valid_obs.empty:
                    for _, ob in valid_obs.tail(5).iterrows():
                        emoji = "🟢" if ob["type"] == "bullish" else "🔴"
                        dist  = abs(current_price - (ob["high"] + ob["low"]) / 2) if current_price else 0
                        near  = " 🎯 **Close!**" if dist < 20 else ""
                        st.markdown(f"{emoji} **{ob['type'].upper()} OB** | {ob['low']:.0f}–{ob['high']:.0f} | dist: {dist:.0f} pts{near}")
            with ob_display_cols[1]:
                st.markdown(f"**❌ Invalidated OBs: {len(invalid_obs)}**")
                st.caption("OB is invalidated when price closes beyond the OB zone")

            with st.expander("📖 What is an Order Block?"):
                st.markdown("""
**Order Block** = the last candle in the opposite direction before a strong institutional move.

- **Bullish OB**: last bearish candle before a strong bullish impulse → institution accumulated long.
  When price returns → buy environment.
- **Bearish OB**: last bullish candle before a strong bearish impulse → institution distributed short.
  When price returns → sell environment.

**Invalidation**: price closing beyond the OB = institution was bypassed = OB no longer valid.
                """)
        else:
            st.info("No Order Blocks detected in current data")

        # ── BOS / MSS ─────────────────────────────────────────────────────────
        st.divider()
        st.markdown("### 🔀 Market Structure — BOS / MSS")
        if not ms_df.empty:
            last_ms    = ms_df.iloc[-1]
            is_mss     = "MSS" in last_ms["label"]
            if is_mss:
                if "bullish" in last_ms["type"]:
                    st.success(f"🚀 **Bullish MSS** @ {last_ms['price']:.0f} — Trend Reversal: bearish → bullish")
                else:
                    st.error(f"📉 **Bearish MSS** @ {last_ms['price']:.0f} — Trend Reversal: bullish → bearish")
            else:
                if "bullish" in last_ms["type"]:
                    st.info(f"↑ **Bullish BOS** @ {last_ms['price']:.0f} — Continuation: broke prior swing high")
                else:
                    st.info(f"↓ **Bearish BOS** @ {last_ms['price']:.0f} — Continuation: broke prior swing low")
            rows_ms = []
            for _, r in ms_df.tail(8).iterrows():
                is_r_mss = "MSS" in r["label"]
                rows_ms.append({
                    "Time":        str(r["datetime"]).split("+")[0].split(".")[0],
                    "Signal":      r["label"],
                    "Price":       f"{r['price']:.0f}",
                    "Importance":  "🔴 Reversal" if is_r_mss else "🟡 Continuation",
                })
            st.dataframe(pd.DataFrame(rows_ms), use_container_width=True, hide_index=True)
        else:
            st.info("No Market Structure events detected — need more intraday data")

        # ── Premium / Discount + ATR ──────────────────────────────────────────
        st.divider()
        st.markdown("### 📊 Premium / Discount & ATR Consumed")
        pd_col, atr_col = st.columns(2)
        with pd_col:
            zone_colors = {"Deep Discount": "🟢🟢", "Discount": "🟢", "Equilibrium": "🟡",
                           "Premium": "🔴", "Deep Premium": "🔴🔴"}
            zone_icon = zone_colors.get(pd_daily["zone"], "⚪")
            st.metric(f"{zone_icon} {pd_daily['zone']}", f"{pd_daily['pct']:.1f}%",
                      delta="Buy environment" if pd_daily["bias"] == "bullish" else
                            "Sell environment" if pd_daily["bias"] == "bearish" else "Equilibrium")
            st.caption("Price position within weekly range (0%=Deep Discount, 100%=Deep Premium)")
            if pd_daily["pct"] < 35:
                st.success("📉→📈 Price in Discount — institutions prefer to buy here")
            elif pd_daily["pct"] > 65:
                st.error("📈→📉 Price in Premium — institutions prefer to sell here")
            else:
                st.info("Price at Equilibrium — wait for extension from 50%")
        with atr_col:
            if atr_data:
                consumed = atr_data["consumed_pct"]
                st.metric("📏 ATR Consumed Today", f"{consumed:.0f}%",
                          delta=f"~{atr_data['remaining_pts']:.0f} pts remaining")
                st.progress(min(consumed / 100, 1.0))
                st.caption(f"ATR(14d): {atr_data['atr']:.0f} pts | Today range: {atr_data['today_range']:.0f}")
                if consumed > 80:
                    st.error("⚠️ Range exhausted — very poor R:R for new entries")
                elif consumed > 60:
                    st.warning("⚠️ Range advanced — select entries carefully")
                else:
                    st.success("✅ Range open — good movement potential")
            else:
                st.metric("ATR Consumed", "N/A")

        # ── ICT Macro Windows ─────────────────────────────────────────────────
        st.divider()
        st.markdown("### ⏱️ ICT Macro Windows — Algorithmic Time Windows")
        st.caption("~20-min windows where institutions manipulate price before true displacement")
        try:
            macros = nq_calculations.detect_ict_macros(df_5m)
            if macros:
                for macro in macros:
                    status  = macro.get("status", "pending")
                    active  = macro.get("active", False)
                    label   = macro.get("pattern_label", "—")
                    pattern = macro.get("pattern", "")
                    with st.container():
                        cols = st.columns([2, 1, 1, 3, 1])
                        name_str = f"{'🔴 LIVE — ' if active else ''}{macro['name']}"
                        cols[0].markdown(f"**{name_str}**")
                        cols[1].caption(f"🇺🇸 {macro['et_start']}–{macro['et_end']}")
                        cols[2].caption(f"🇮🇱 {macro['il_start']}–{macro['il_end']}")
                        cols[3].markdown(label)
                        if status == "complete" and macro.get("range_pts"):
                            cols[4].caption(f"±{macro['range_pts']:.0f}p / {macro.get('move_pts', 0):+.0f}p")
                        elif status == "active":
                            cols[4].markdown("🟡")
                        else:
                            cols[4].caption("⏰")
            else:
                st.info("ICT Macro data unavailable — requires 5m data with datetime_et")
        except Exception as e:
            st.caption(f"ICT Macros: {e}")

        with st.expander("📖 ICT/SMC Quick Reference"):
            st.markdown("""
| Concept | Definition | Trade Signal |
|---------|-----------|-------------|
| **Order Block (OB)** | Last opposing candle before institutional impulse | Buy/sell on return to block |
| **BOS** | Swing break in trend direction | Continuation move |
| **MSS** | Swing break against trend | Potential reversal |
| **OR (Opening Range)** | First 15-min High/Low | Breakout = session bias |
| **Premium/Discount** | >65%=premium (sell), <35%=discount (buy) | Enter on correct side |
| **OTE** | 61.8–78.6% Fibonacci retracement | Optimal institutional entry |
| **FVG** | Gap between candle[i-2].high and candle[i].low | Price returns to fill |
| **Judas Swing** | Overnight range breakout + reversal | Enter in reversal direction |
| **Killzone** | High-probability time window | Only trade inside KZ |
| **AMD** | Accumulation→Manipulation→Distribution | Identify phase |
| **Absorption** | High vol + narrow range at extreme | Holds direction |
| **VWAP** | Volume Weighted Average Price | Institutional benchmark |
            """)

    # ──────────────────────────────────────────────────────────────────────────
    # TAB 6 — Historical Patterns
    # ──────────────────────────────────────────────────────────────────────────
    with tabs[5]:
        st.subheader("📜 Historical Patterns & Edge")

        if df_daily.empty:
            st.warning("Daily data unavailable")
        else:
            # ── Row 1: Similar Days + Win Rate by Hour ──────────────────────
            col_sim, col_wr = st.columns(2)
            with col_sim:
                st.markdown("### 🔍 Similar Days to Today")
                similar = nq_calculations.find_similar_setups(df_daily)
                if not similar.empty:
                    similar["similarity"]       = (similar["similarity"] * 100).round(1)
                    similar["next_day_change"]  = similar["next_day_change"].round(2)
                    similar["Result"]           = similar["next_day_change"].apply(
                        lambda x: f"🟢 +{x:.1f}%" if x > 0 else f"🔴 {x:.1f}%")
                    display_sim = similar[["date", "similarity", "Result"]].copy()
                    display_sim.columns = ["Date", "Similarity %", "Next Day Result"]
                    st.dataframe(display_sim, use_container_width=True, hide_index=True)
                    up_days = (similar["next_day_change"] > 0).sum()
                    total   = len(similar)
                    win_pct = up_days / total * 100
                    if win_pct >= 60:
                        st.success(f"📈 {up_days}/{total} similar days went UP ({win_pct:.0f}% bullish bias)")
                    elif win_pct <= 40:
                        st.error(f"📉 {total - up_days}/{total} similar days fell ({100 - win_pct:.0f}% bearish bias)")
                    else:
                        st.info(f"📊 {win_pct:.0f}% of similar days went up — neutral bias")
                else:
                    st.info("Not enough history for similarity analysis (need ≥30 days)")

            with col_wr:
                st.markdown("### ⏰ Win Rate by Hour (ET)")
                if not df_hourly.empty:
                    wr_df = nq_calculations.session_win_rates(df_hourly)
                    if not wr_df.empty:
                        fig_wr = px.bar(wr_df, x="hour_et", y="win_rate_pct",
                                        title="% of bars NQ closed UP per ET hour",
                                        color="win_rate_pct",
                                        color_continuous_scale=["red", "yellow", "green"],
                                        range_color=[30, 70])
                        fig_wr.add_hline(y=50, line_dash="dash", line_color="white", line_width=1)
                        fig_wr.update_layout(paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
                                             font=dict(color="white"), showlegend=False,
                                             coloraxis_showscale=False, height=320)
                        st.plotly_chart(fig_wr, use_container_width=True)
                else:
                    st.info("Hourly data unavailable")

            st.divider()

            # ── Row 2: Day-of-Week Bias + Gap Fill ─────────────────────────
            col_dow, col_gap = st.columns(2)

            with col_dow:
                st.markdown("### 📅 Day-of-Week Bias")
                if dow_data:
                    dow_df = pd.DataFrame([
                        {"Day": DOW_NAMES.get(k, str(k)), "Win %": round(v * 100, 1), "Bars": k}
                        for k, v in sorted(dow_data.items())
                    ])
                    today_dow = datetime.now(ET_TZ).weekday()
                    fig_dow = px.bar(dow_df, x="Day", y="Win %",
                                     color="Win %",
                                     color_continuous_scale=["red", "yellow", "green"],
                                     range_color=[40, 65],
                                     title="NQ historical win % by day of week")
                    fig_dow.add_hline(y=50, line_dash="dash", line_color="white", line_width=1)
                    # Highlight today
                    today_name = DOW_NAMES.get(today_dow, "")
                    today_row  = dow_df[dow_df["Day"] == today_name]
                    if not today_row.empty:
                        fig_dow.add_vline(x=today_name, line_color="yellow", line_width=2, opacity=0.5)
                    fig_dow.update_layout(paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
                                          font=dict(color="white"), showlegend=False,
                                          coloraxis_showscale=False, height=280)
                    st.plotly_chart(fig_dow, use_container_width=True)
                    # Highlight today's edge
                    today_wr = dow_data.get(today_dow)
                    if today_wr is not None:
                        wr_pct = round(today_wr * 100, 1)
                        if wr_pct >= 55:
                            st.success(f"Today ({DOW_NAMES[today_dow]}): **{wr_pct}%** historical win rate — bullish edge")
                        elif wr_pct <= 45:
                            st.error(f"Today ({DOW_NAMES[today_dow]}): **{wr_pct}%** historical win rate — bearish edge")
                        else:
                            st.info(f"Today ({DOW_NAMES[today_dow]}): **{wr_pct}%** — no clear day-of-week edge")
                else:
                    st.info("Need ≥20 days of history for DOW analysis")

            with col_gap:
                st.markdown("### 📈 Opening Gap Analysis")
                if gap_data:
                    gap_pts = gap_data.get("today_gap_pts", 0)
                    gap_dir = gap_data.get("today_gap_dir", "none")

                    gm1, gm2 = st.columns(2)
                    with gm1:
                        fill_up = gap_data.get("gap_up_fill_pct")
                        st.metric("Gap-Up Fill Rate", f"{fill_up}%" if fill_up else "N/A")
                        st.caption("Historical: % of up-gaps that fill same day")
                    with gm2:
                        fill_dn = gap_data.get("gap_down_fill_pct")
                        st.metric("Gap-Down Fill Rate", f"{fill_dn}%" if fill_dn else "N/A")
                        st.caption("Historical: % of down-gaps that fill same day")

                    st.markdown("---")
                    if gap_dir == "up":
                        fill_pct = gap_data.get("gap_up_fill_pct")
                        st.markdown(f"**Today's gap:** 🔼 UP **{gap_pts:+.0f} pts**")
                        if fill_pct:
                            if fill_pct >= 70:
                                st.success(f"📉 High fill probability: **{fill_pct}%** of similar up-gaps filled same day → watch for early reversal")
                            elif fill_pct >= 50:
                                st.warning(f"📊 Moderate fill probability: **{fill_pct}%**")
                            else:
                                st.info(f"📈 Low fill probability: **{fill_pct}%** — momentum likely continues")
                    elif gap_dir == "down":
                        fill_pct = gap_data.get("gap_down_fill_pct")
                        st.markdown(f"**Today's gap:** 🔽 DOWN **{gap_pts:+.0f} pts**")
                        if fill_pct:
                            if fill_pct >= 70:
                                st.success(f"📈 High fill probability: **{fill_pct}%** → watch for bounce/reversal")
                            else:
                                st.info(f"📉 Fill probability: **{fill_pct}%**")
                    else:
                        st.info("↔️ No significant opening gap today (< 2 pts)")
                else:
                    st.info("Gap analysis requires ≥10 days of daily data")

            st.divider()

            # ── Session Reference Table ────────────────────────────────────────
            st.markdown("### 📊 Session Reference Table")
            stats_rows = []
            for sess in SESSIONS:
                name = sess["name"]
                if name in ("Overnight", "Lunch Lull", "Cash Close"):
                    continue
                stats_rows.append({
                    "Session":      name,
                    "IL Hours":     f"{sess['il_start']}–{sess['il_end']}",
                    "ET Hours":     f"{sess['et_start']}–{sess['et_end']}",
                    "Participants": sess["participants"][:60] + "…" if len(sess["participants"]) > 60 else sess["participants"],
                })
            st.dataframe(pd.DataFrame(stats_rows), use_container_width=True, hide_index=True)

    # ──────────────────────────────────────────────────────────────────────────
    # TAB 7 — Checklist
    # ──────────────────────────────────────────────────────────────────────────
    with tabs[6]:
        st.subheader("✅ Pre-Trade Checklist")

        conditions = {}

        # 1. Session check
        bad_sessions = ("Lunch Lull", "Overnight", "Globex / Off-Hours")
        conditions["good_session"] = current_session.get("name") not in bad_sessions

        # 2. Near key level
        near_level = False
        if current_price and levels:
            for key in ["pdh", "pdl", "poc_5d", "poc_20d", "overnight_high", "overnight_low"]:
                val = levels.get(key)
                if val and not np.isnan(val):
                    if abs(current_price - val) / current_price < 0.0025:
                        near_level = True
                        break
        conditions["near_level"] = near_level

        # 3. No divergence in last 4 bars
        no_divergence = True
        if not df_5m.empty:
            div_c = nq_calculations.detect_delta_divergence(df_5m)
            if abs(div_c.iloc[-4:].sum()) > 0:
                no_divergence = False
        conditions["no_divergence"] = no_divergence

        # 4. Not in Judas window
        not_judas_window = not (9 * 60 + 15 <= now_et_hm <= 9 * 60 + 30)
        conditions["not_judas"] = not_judas_window

        # 5. Stacked imbalance present
        has_stack = False
        if not df_5m.empty:
            stacks_c = nq_calculations.detect_stacked_imbalances(df_5m)
            has_stack = len(stacks_c) > 0
        conditions["has_stack"] = has_stack

        # 6. FVG present
        has_fvg = False
        if not df_5m.empty:
            fvg_df_c = nq_calculations.detect_fvg(df_5m)
            has_fvg = not fvg_df_c.empty
        conditions["has_fvg"] = has_fvg

        # 7. Volume above average
        vol_ok = False
        if not df_5m.empty and len(df_5m) > 10:
            recent_vol = df_5m["volume"].iloc[-1]
            avg_vol    = df_5m["volume"].iloc[-20:].mean()
            vol_ok     = recent_vol >= avg_vol * 0.8
        conditions["vol_ok"] = vol_ok

        # 8. Near valid Order Block
        near_ob = False
        if current_price and not ob_df.empty:
            for _, ob in ob_df[ob_df["valid"] == True].iterrows():
                mid = (ob["high"] + ob["low"]) / 2
                if abs(current_price - mid) / current_price < 0.003:
                    near_ob = True
                    break
        conditions["near_ob"] = near_ob

        # 9. BOS/MSS confirmation
        conditions["has_ms"] = not ms_df.empty

        # 10. OR breakout in clear direction
        or_breakout = False
        if opening_range and opening_range.get("or_complete") and current_price:
            or_breakout = current_price > opening_range["or_high"] or current_price < opening_range["or_low"]
        conditions["or_breakout"] = or_breakout

        # 11. ATR range not exhausted
        atr_ok = True
        if atr_data:
            atr_ok = atr_data.get("consumed_pct", 0) < 80
        conditions["atr_ok"] = atr_ok

        # 12. VWAP position not extreme (avoid entries at ±2σ — mean-reversion risk)
        vwap_ok = True
        if vwap_data:
            sigma_now = abs(vwap_data.get("sigma", 0))
            vwap_ok   = sigma_now < 2.0
        conditions["vwap_ok"] = vwap_ok

        # 13. AI Signal is directional (not NEUTRAL)
        conditions["ai_directional"] = sig.get("direction", "NEUTRAL") != "NEUTRAL"

        checklist = [
            ("good_session",    "Current session is suitable for trading (not Lunch Lull / Overnight)"),
            ("near_level",      "Price is near a key level (PDH/PDL/POC/FVG ±0.25%)"),
            ("no_divergence",   "No Delta Divergence in last 4 bars"),
            ("not_judas",       "Not in Judas Swing window (09:15–09:30 ET)"),
            ("has_stack",       "Stacked Imbalance present in potential direction"),
            ("has_fvg",         "Active Fair Value Gap exists"),
            ("vol_ok",          "Volume is at least 80% of session average"),
            ("near_ob",         "Price is near an active Order Block (±0.3%)"),
            ("has_ms",          "Clear BOS / MSS signal in today's data"),
            ("or_breakout",     "Opening Range has broken in a clear direction"),
            ("atr_ok",          "ATR range not exhausted (more than 20% of daily range remaining)"),
            ("vwap_ok",         "Price is not at VWAP extremes (within ±2σ — avoids mean-reversion trap)"),
            ("ai_directional",  "AI Signal Engine has a directional bias (not NEUTRAL)"),
        ]

        score    = sum(1 for key, _ in checklist if conditions.get(key, False))
        total    = len(checklist)
        score_pct = score / total * 100

        col_score, col_session = st.columns([1, 2])
        with col_score:
            st.metric("Checklist Score", f"{score}/{total}")
            st.progress(score / total)
        with col_session:
            if score_pct >= 70:
                st.success(f"✅ {score}/{total} conditions met ({score_pct:.0f}%) — **Good conditions for a trade**")
            elif score_pct >= 45:
                st.warning(f"⚠️ {score}/{total} conditions met ({score_pct:.0f}%) — **Caution: not all conditions present**")
            else:
                st.error(f"❌ {score}/{total} conditions met ({score_pct:.0f}%) — **Not recommended to trade now**")

        # AI Signal alignment note
        if sig.get("alert"):
            st.success("🔔 AI Signal Alert is ACTIVE — signal engine confirms trade conditions")
        elif sig.get("direction", "NEUTRAL") != "NEUTRAL":
            st.info(f"🤖 AI Signal: {sig['direction']} ({sig.get('confidence', 0)}%) — below alert threshold")
        else:
            st.caption("🤖 AI Signal: NEUTRAL — no directional bias from signal engine")

        st.divider()
        st.markdown("### Condition Details")
        failed      = [(k, l) for k, l in checklist if not conditions.get(k, False)]
        passed_list = [(k, l) for k, l in checklist if conditions.get(k, False)]
        for key, label in failed + passed_list:
            p = conditions.get(key, False)
            st.markdown(f"{'✅' if p else '❌'} {label}")

        st.divider()
        st.markdown("### 📝 Manual Notes")
        st.text_area("Enter trade plan notes for today:", height=100,
                     placeholder="e.g., No economic data today, VIX low, bullish trend from yesterday…")

        st.markdown("#### 📋 ICT Pre-Trade Framework")
        st.markdown("""
1. **Daily bias?** (Daily Chart — Higher High/Low? SMA20 position?)
2. **Session bias?** (Asian range: above/below? OR direction?)
3. **Nearest PD Array?** (FVG? OB? Breaker? VWAP?)
4. **Nearest liquidity pool?** (EQH/EQL? Buy-side/Sell-side stops?)
5. **Killzone?** (Only expect setups inside the killzone)
6. **OTE Zone?** (Is price at 61.8–78.6% retracement of the impulse?)
7. **Confirmation?** (MSS? BOS? Delta? Volume anomaly?)
        """)

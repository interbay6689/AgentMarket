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


def _price_badge(val: float, label: str, delta: float = None):
    color = "normal" if delta is None else ("inverse" if delta < 0 else "normal")
    st.metric(label, f"{val:,.1f}", delta=f"{delta:+.2f}%" if delta is not None else None,
              delta_color=color)


def app():
    st.title("📊 NQ Order Flow Dashboard — Daily Trading Plan")
    st.caption("Nasdaq 100 Futures | Israel Time | Order Flow & ICT/SMC Analysis")

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
        df_5m = _get_5m()
        df_daily = _get_daily()
        df_hourly = _get_hourly()
        corr = _get_corr()

    current_session = nq_data.current_session_israel()
    levels = nq_calculations.calculate_key_levels(df_daily, df_hourly)

    # current price
    current_price = float(df_5m["close"].iloc[-1]) if not df_5m.empty else None
    prev_close = levels.get("pdc")
    pct_change = ((current_price - prev_close) / prev_close * 100) if (current_price and prev_close) else None

    # Sprint-1 pre-computations (shared across tabs)
    opening_range = nq_calculations.calculate_opening_range(df_5m, minutes=15)
    atr_data = nq_calculations.calculate_atr_range_consumed(df_daily, df_5m)
    ob_df = nq_calculations.detect_order_blocks(df_5m) if not df_5m.empty else pd.DataFrame()
    ms_df = nq_calculations.detect_market_structure(df_5m) if not df_5m.empty else pd.DataFrame()
    pd_daily = nq_calculations.classify_premium_discount(
        current_price,
        levels.get("weekly_high", current_price or 0),
        levels.get("weekly_low",  current_price or 0),
    ) if current_price else {"pct": 50, "zone": "—", "bias": "neutral"}

    # ──────────────────────────────────────────────────────────────────────────
    # TAB 1 — War Plan
    # ──────────────────────────────────────────────────────────────────────────
    with tabs[0]:
        st.subheader("🎯 Daily War Plan")
        now_il = datetime.now(IL_TZ)
        st.markdown(f"**IL Time:** `{now_il.strftime('%H:%M:%S')}` | **Date:** `{now_il.strftime('%d/%m/%Y')}`")

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
            participants = PARTICIPANTS_BY_SESSION.get(current_session.get("name", ""), [])
            st.metric("👥 Key Participants", f"{len(participants)} types")
            if participants:
                st.caption(participants[0])
        with col4:
            vix_df = corr.get("VIX", pd.DataFrame())
            if not vix_df.empty and "close" in vix_df.columns:
                vix_val = float(vix_df["close"].dropna().iloc[-1])
                vix_color = "🔴" if vix_val > 20 else "🟡" if vix_val > 15 else "🟢"
                st.metric(f"{vix_color} VIX", f"{vix_val:.2f}")
            else:
                st.metric("VIX", "N/A")

        # ── Sprint-1 row: ATR / OR / Premium-Discount / OB ──────────────────
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

        st.divider()

        # Smart banner
        banners = []
        if not df_5m.empty:
            fvg_df = nq_calculations.detect_fvg(df_5m)
            if not fvg_df.empty:
                last_fvg = fvg_df.iloc[-1]
                banners.append(f"🟦 **FVG {last_fvg['type'].upper()}** at {last_fvg['bottom']:.0f}–{last_fvg['top']:.0f} — potential price magnet")
            div_series = nq_calculations.detect_delta_divergence(df_5m)
            last_div = div_series.iloc[-4:].sum()
            if last_div > 0:
                banners.append("⚠️ **Bullish Delta Divergence** detected — price may not reflect buying pressure")
            elif last_div < 0:
                banners.append("⚠️ **Bearish Delta Divergence** detected — possible distribution")
            stacks = nq_calculations.detect_stacked_imbalances(df_5m)
            if stacks:
                last_stack = stacks[-1]
                banners.append(f"📚 **Stacked Imbalance {last_stack['direction'].upper()}** — {last_stack['candle_count']} consecutive candles | {last_stack['price_start']:.0f} → {last_stack['price_end']:.0f}")

            # Sprint-1 banners
            if opening_range and opening_range.get("or_complete") and current_price:
                if current_price > opening_range["or_high"]:
                    banners.append(f"📐 **OR Breakout BULLISH** — price broke above OR High ({opening_range['or_high']:.0f}) — bullish session bias")
                elif current_price < opening_range["or_low"]:
                    banners.append(f"📐 **OR Breakdown BEARISH** — price broke below OR Low ({opening_range['or_low']:.0f}) — bearish session bias")

            if not ms_df.empty:
                last_ms = ms_df.iloc[-1]
                ms_emoji = "🚀" if "bullish" in last_ms["type"] else "📉"
                ms_is_mss = "MSS" in last_ms["label"]
                ms_text = "**MSS — Trend Reversal!**" if ms_is_mss else "**BOS — Trend Continuation**"
                banners.append(f"{ms_emoji} {ms_text} | {last_ms['label']} @ {last_ms['price']:.0f}")

            if not ob_df.empty and current_price:
                valid_obs = ob_df[ob_df["valid"] == True]
                for _, ob in valid_obs.tail(3).iterrows():
                    dist = abs(current_price - (ob["high"] + ob["low"]) / 2)
                    if dist / current_price < 0.003:
                        ob_emoji = "🟢" if ob["type"] == "bullish" else "🔴"
                        banners.append(f"{ob_emoji} **{ob['type'].upper()} Order Block** — price within {dist:.0f} pts of OB at {ob['low']:.0f}–{ob['high']:.0f}")

            if atr_data and atr_data.get("consumed_pct", 0) > 85:
                banners.append(f"⚠️ **ATR {atr_data['consumed_pct']:.0f}% consumed** — daily range nearly exhausted, poor R:R for new entries")

        if current_session.get("name") == "Lunch Lull":
            banners.insert(0, "⛔ **Lunch Lull active** — low liquidity, false breakouts common, avoid trading")

        if banners:
            for b in banners:
                st.info(b)
        else:
            st.success("✅ No special signals at this time — normal market conditions")

        st.divider()
        st.subheader("📈 Correlated Assets")
        asset_cols = st.columns(4)
        asset_map = [("VIX", "^VIX"), ("DXY", "DX-Y.NYB"), ("TNX", "10Y Yield"), ("Gold", "GC=F")]
        for i, (name, _) in enumerate(asset_map):
            df_a = corr.get(name, pd.DataFrame())
            with asset_cols[i]:
                if not df_a.empty and "close" in df_a.columns:
                    vals = df_a["close"].dropna()
                    if len(vals) >= 2:
                        chg = (vals.iloc[-1] / vals.iloc[-2] - 1) * 100
                        st.metric(name, f"{vals.iloc[-1]:.2f}", delta=f"{chg:+.2f}%",
                                  delta_color="inverse" if name in ("VIX", "TNX") else "normal")
                    else:
                        st.metric(name, "—")
                else:
                    st.metric(name, "N/A")

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
            end_m = hm_to_min(s["il_end"])
            # handle sessions crossing midnight
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

        # current time marker
        fig_tl.add_vline(x=now_minutes, line_dash="dash", line_color="red", line_width=2,
                         annotation_text=f"⏰ {now_il.strftime('%H:%M')} IL",
                         annotation_position="top")

        # key event markers
        for ev in KEY_EVENTS_ET:
            try:
                ev_et_min = hm_to_min(ev["time"])
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
            xaxis=dict(title="Hour (IL)", range=[0, 1440],
                       tickvals=tick_vals, ticktext=tick_text),
            yaxis=dict(title=""),
            height=420,
            paper_bgcolor="#0e1117",
            plot_bgcolor="#0e1117",
            font=dict(color="white"),
            bargap=0.3,
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

        if df_5m.empty:
            st.warning("5-minute data unavailable — market may be closed")
        else:
            fig_kl = make_subplots(rows=1, cols=2, column_widths=[0.75, 0.25],
                                   shared_yaxes=True,
                                   subplot_titles=["NQ 5m — Today", "Volume Profile"])

            dt_col = "datetime_il" if "datetime_il" in df_5m.columns else "datetime"
            fig_kl.add_trace(go.Candlestick(
                x=df_5m[dt_col],
                open=df_5m["open"], high=df_5m["high"],
                low=df_5m["low"], close=df_5m["close"],
                name="NQ 5m",
                increasing_line_color="#26a69a",
                decreasing_line_color="#ef5350",
            ), row=1, col=1)

            level_colors = {
                "pdh": ("Prior Day High", "red"),
                "pdl": ("Prior Day Low", "green"),
                "pdc": ("Prior Day Close", "gray"),
                "weekly_high": ("Weekly High", "darkred"),
                "weekly_low": ("Weekly Low", "darkgreen"),
                "poc_20d": ("POC 20d", "gold"),
                "poc_5d": ("POC 5d", "orange"),
                "overnight_high": ("Overnight High", "lightblue"),
                "overnight_low": ("Overnight Low", "lightblue"),
            }
            for key, (label, color) in level_colors.items():
                val = levels.get(key)
                if val and not np.isnan(val):
                    fig_kl.add_hline(y=val, line_dash="dash" if "poc" not in key else "dot",
                                     line_color=color, line_width=1,
                                     annotation_text=f"{label}: {val:.0f}",
                                     annotation_position="left",
                                     row=1, col=1)

            fvg_df = nq_calculations.detect_fvg(df_5m)
            if not fvg_df.empty:
                x_end = df_5m[dt_col].iloc[-1]
                for _, fvg in fvg_df.iterrows():
                    fig_kl.add_shape(
                        type="rect",
                        x0=fvg["datetime"], x1=x_end,
                        y0=fvg["bottom"], y1=fvg["top"],
                        fillcolor="rgba(0,255,100,0.12)" if fvg["type"] == "bullish" else "rgba(255,50,50,0.12)",
                        line=dict(width=0),
                        row=1, col=1,
                    )

            # Opening Range overlay
            if opening_range and opening_range.get("or_complete"):
                x_start = df_5m[dt_col].iloc[0]
                x_end = df_5m[dt_col].iloc[-1]
                fig_kl.add_shape(type="rect",
                    x0=x_start, x1=x_end,
                    y0=opening_range["or_low"], y1=opening_range["or_high"],
                    fillcolor="rgba(255,215,0,0.07)", line=dict(color="gold", width=1, dash="dot"),
                    row=1, col=1)
                fig_kl.add_hline(y=opening_range["or_high"], line_dash="dot", line_color="gold",
                                 line_width=1, annotation_text=f"OR High: {opening_range['or_high']:.0f}",
                                 annotation_position="right", row=1, col=1)
                fig_kl.add_hline(y=opening_range["or_low"], line_dash="dot", line_color="gold",
                                 line_width=1, annotation_text=f"OR Low: {opening_range['or_low']:.0f}",
                                 annotation_position="right", row=1, col=1)
                fig_kl.add_hline(y=opening_range["or_mid"], line_dash="dash", line_color="rgba(255,215,0,0.4)",
                                 line_width=1, annotation_text=f"OR Mid: {opening_range['or_mid']:.0f}",
                                 annotation_position="right", row=1, col=1)

            # Order Block overlays
            if not ob_df.empty:
                x_end = df_5m[dt_col].iloc[-1]
                valid_obs = ob_df[ob_df["valid"] == True].tail(5)
                for _, ob in valid_obs.iterrows():
                    fill = "rgba(0,200,100,0.15)" if ob["type"] == "bullish" else "rgba(220,50,50,0.15)"
                    border = "rgba(0,200,100,0.6)"  if ob["type"] == "bullish" else "rgba(220,50,50,0.6)"
                    fig_kl.add_shape(type="rect",
                        x0=ob["datetime"], x1=x_end,
                        y0=ob["low"], y1=ob["high"],
                        fillcolor=fill, line=dict(color=border, width=1),
                        row=1, col=1)
                    fig_kl.add_annotation(
                        x=x_end, y=(ob["high"] + ob["low"]) / 2,
                        text=f"{'B' if ob['type'] == 'bullish' else 'S'}-OB",
                        font=dict(size=9, color=border), showarrow=False,
                        xanchor="right", row=1, col=1)

            # volume profile on right
            vp = nq_calculations.calculate_volume_profile(df_5m, price_bins=40)
            if not vp.empty:
                fig_kl.add_trace(go.Bar(
                    x=vp["volume"], y=vp["price"],
                    orientation="h",
                    marker_color="rgba(100,180,255,0.5)",
                    name="Volume Profile",
                    hovertemplate="Price: %{y:.0f}<br>Vol: %{x:,.0f}<extra></extra>",
                ), row=1, col=2)

                poc = nq_calculations.get_poc(vp)
                if poc:
                    fig_kl.add_hline(y=poc, line_dash="dash", line_color="gold",
                                     annotation_text=f"POC: {poc:.0f}", row=1, col=2)

            fig_kl.update_layout(height=550, paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
                                 font=dict(color="white"), xaxis_rangeslider_visible=False,
                                 showlegend=False)
            st.plotly_chart(fig_kl, use_container_width=True)

        # levels table
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

            # Add Opening Range levels
            if opening_range:
                for or_key, or_label in [("or_high", "OR High (15m)"), ("or_low", "OR Low (15m)"), ("or_mid", "OR Mid")]:
                    val = opening_range.get(or_key)
                    if val:
                        dist = val - current_price
                        rows.append({"Level": f"📐 {or_label}", "Price": f"{val:.0f}",
                                     "Distance (pts)": f"{dist:+.0f}",
                                     "Type": "Support 🟢" if dist < 0 else "Resistance 🔴"})

            # Add valid Order Blocks
            if not ob_df.empty:
                valid_obs = ob_df[ob_df["valid"] == True].tail(3)
                for _, ob in valid_obs.iterrows():
                    mid = (ob["high"] + ob["low"]) / 2
                    dist = mid - current_price
                    emoji = "🟢" if ob["type"] == "bullish" else "🔴"
                    rows.append({"Level": f"{emoji} {ob['type'].capitalize()} OB", "Price": f"{ob['low']:.0f}–{ob['high']:.0f}",
                                 "Distance (pts)": f"{dist:+.0f}",
                                 "Type": "Support 🟢" if ob["type"] == "bullish" else "Resistance 🔴"})

            if rows:
                df_lev = pd.DataFrame(rows).sort_values("Distance (pts)")
                st.dataframe(df_lev, use_container_width=True, hide_index=True)

    # ──────────────────────────────────────────────────────────────────────────
    # TAB 4 — Order Flow Indicators
    # ──────────────────────────────────────────────────────────────────────────
    with tabs[3]:
        st.subheader("🌊 Order Flow Indicators")

        if df_5m.empty:
            st.warning("5-minute data unavailable")
        else:
            cd = nq_calculations.cumulative_delta(df_5m)
            div = nq_calculations.detect_delta_divergence(df_5m)
            absorption = nq_calculations.detect_absorption(df_5m)

            bearish_div = int((div.iloc[-8:] > 0).sum())
            bullish_div = int((div.iloc[-8:] < 0).sum())
            if bearish_div > 0:
                st.error(f"⚠️ Bearish Delta Divergence detected ({bearish_div} signals) — Distribution: price rising but selling dominates")
            elif bullish_div > 0:
                st.success(f"📈 Bullish Delta Divergence detected ({bullish_div} signals) — Accumulation: price falling but buying dominates")

            dt_col = "datetime_il" if "datetime_il" in df_5m.columns else "datetime"
            fig_cd = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                   row_heights=[0.6, 0.4],
                                   subplot_titles=["NQ Price", "Cumulative Delta"])

            fig_cd.add_trace(go.Candlestick(
                x=df_5m[dt_col], open=df_5m["open"], high=df_5m["high"],
                low=df_5m["low"], close=df_5m["close"],
                increasing_line_color="#26a69a", decreasing_line_color="#ef5350",
                name="NQ", showlegend=False
            ), row=1, col=1)

            # mark divergences on price
            div_up = df_5m[div < 0]
            div_dn = df_5m[div > 0]
            if not div_up.empty:
                fig_cd.add_trace(go.Scatter(x=div_up[dt_col], y=div_up["low"] * 0.9995,
                                            mode="markers", marker=dict(symbol="triangle-up", size=10, color="lime"),
                                            name="Bullish Div", showlegend=True), row=1, col=1)
            if not div_dn.empty:
                fig_cd.add_trace(go.Scatter(x=div_dn[dt_col], y=div_dn["high"] * 1.0005,
                                            mode="markers", marker=dict(symbol="triangle-down", size=10, color="red"),
                                            name="Bearish Div", showlegend=True), row=1, col=1)

            # absorption markers
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

            # BOS / MSS markers on price chart
            if not ms_df.empty:
                for _, ev in ms_df.iterrows():
                    is_mss = "MSS" in ev["label"]
                    color = "cyan" if "bullish" in ev["type"] else "orange"
                    symbol = "triangle-up" if "bullish" in ev["type"] else "triangle-down"
                    y_pos = ev["price"] * 0.9998 if "bullish" in ev["type"] else ev["price"] * 1.0002
                    fig_cd.add_trace(go.Scatter(
                        x=[ev["datetime"]], y=[y_pos],
                        mode="markers+text",
                        marker=dict(symbol=symbol, size=12 if is_mss else 9,
                                    color=color, line=dict(color="white", width=1)),
                        text=[ev["label"]],
                        textposition="bottom center" if "bullish" in ev["type"] else "top center",
                        textfont=dict(size=8, color=color),
                        name=ev["label"], showlegend=False,
                    ), row=1, col=1)

            # Order Block zones on price chart
            if not ob_df.empty:
                x_end = df_5m[dt_col].iloc[-1]
                for _, ob in ob_df[ob_df["valid"] == True].tail(4).iterrows():
                    fill = "rgba(0,200,100,0.12)" if ob["type"] == "bullish" else "rgba(220,50,50,0.12)"
                    border = "rgba(0,200,100,0.5)"  if ob["type"] == "bullish" else "rgba(220,50,50,0.5)"
                    fig_cd.add_shape(type="rect",
                        x0=ob["datetime"], x1=x_end,
                        y0=ob["low"], y1=ob["high"],
                        fillcolor=fill, line=dict(color=border, width=1),
                        row=1, col=1)

            cd_colors = ["#26a69a" if v >= 0 else "#ef5350" for v in cd]
            fig_cd.add_trace(go.Bar(x=df_5m[dt_col], y=cd, marker_color=cd_colors,
                                    name="Cumulative Delta", showlegend=False), row=2, col=1)
            fig_cd.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=1)

            fig_cd.update_layout(height=600, paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
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
            if not df_5m.empty and absorption.abs().sum() > 0:
                st.subheader("🧲 Absorption Events")
                abs_rows = []
                for i, val in enumerate(absorption):
                    if val != 0:
                        row = df_5m.iloc[i]
                        abs_rows.append({
                            "Time": str(row[dt_col]).split("+")[0],
                            "Direction": "🟢 Bullish" if val > 0 else "🔴 Bearish",
                            "Volume": f"{int(row['volume']):,}",
                            "Price": f"{row['close']:.0f}",
                            "Range": f"{(row['high'] - row['low']):.1f}",
                        })
                if abs_rows:
                    st.dataframe(pd.DataFrame(abs_rows), use_container_width=True, hide_index=True)

    # ──────────────────────────────────────────────────────────────────────────
    # TAB 5 — ICT/SMC Analysis
    # ──────────────────────────────────────────────────────────────────────────
    with tabs[4]:
        st.subheader("⚔️ ICT/SMC Analysis")

        # Killzones
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
            ("🗽 NY AM KZ", "07:00", "10:00", "14:00–17:00 IL"),
            ("🌆 NY PM KZ", "13:30", "16:00", "20:30–23:00 IL"),
        ]
        for col, (title, start_et, end_et, il_label) in zip([kz_col1, kz_col2, kz_col3], kz_configs):
            with col:
                active = in_kz(start_et, end_et)
                status = "🟢 Active now" if active else "⚫ Not active"
                st.metric(title, status)
                st.caption(f"ET: {start_et}–{end_et} | {il_label}")

        st.divider()

        # AMD Cycle
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
        phase_colors = {"Accumulation": "🔵", "Manipulation": "🟡", "Distribution": "🔴"}
        with amd_col1:
            color = "🔵" if current_phase == "Accumulation" else "⚫"
            st.metric("💡 Accumulation", color + " Active" if current_phase == "Accumulation" else color + " Complete")
            st.caption("Overnight + Pre-Market")
        with amd_col2:
            color = "🟡" if current_phase == "Manipulation" else "⚫"
            st.metric("⚡ Manipulation", color + " Active" if current_phase == "Manipulation" else color + " Complete")
            st.caption("09:15–09:45 ET (Judas Swing)")
        with amd_col3:
            color = "🔴" if current_phase == "Distribution" else "⚫"
            st.metric("📤 Distribution", color + " Active" if current_phase == "Distribution" else color + " Not yet")
            st.caption("RTH 10:00+ ET")

        st.info(f"**Current phase: {current_phase}** — {phase_desc}")

        # Judas Swing
        st.divider()
        st.markdown("### ⚡ Judas Swing Detector")
        overnight_high = levels.get("overnight_high")
        overnight_low = levels.get("overnight_low")
        if not df_5m.empty and overnight_high and overnight_low:
            judas = nq_calculations.detect_judas_swing(df_5m, overnight_high, overnight_low)
            if judas.get("detected"):
                direction = judas.get("direction", "")
                level = judas.get("level", 0)
                if direction == "bearish":
                    st.error(f"🚨 **Bearish Judas Swing detected!** — price swept above Overnight High ({overnight_high:.0f}), "
                             f"reached {judas.get('sweep_high', 0):.0f}, then reversed — true direction: DOWN")
                else:
                    st.success(f"🚀 **Bullish Judas Swing detected!** — price swept below Overnight Low ({overnight_low:.0f}), "
                               f"reached {judas.get('sweep_low', 0):.0f}, then reversed — true direction: UP")
            else:
                st.info(f"📊 Overnight Range: {overnight_low:.0f} – {overnight_high:.0f} | No Judas Swing detected yet")
        else:
            st.info("📊 Overnight High/Low not available — check after 09:30 ET (16:30 IL)")

        # ── Opening Range ────────────────────────────────────────────────────
        st.divider()
        st.markdown("### 📐 Opening Range (OR 15 min)")
        if opening_range:
            or_col1, or_col2, or_col3, or_col4 = st.columns(4)
            with or_col1:
                st.metric("OR High", f"{opening_range.get('or_high', 0):.0f}")
            with or_col2:
                st.metric("OR Low", f"{opening_range.get('or_low', 0):.0f}")
            with or_col3:
                st.metric("OR Range (pts)", f"{opening_range.get('or_range', 0):.1f}")
            with or_col4:
                status = "✅ Complete" if opening_range.get("or_complete") else f"⏳ {opening_range.get('or_bars', 0)}/3 bars"
                st.metric("Status", status)

            if current_price and opening_range.get("or_complete"):
                if current_price > opening_range["or_high"]:
                    st.success(f"🚀 **OR Breakout Bullish** — price ({current_price:.0f}) above OR High ({opening_range['or_high']:.0f}). Expect bullish continuation, OR High becomes support.")
                elif current_price < opening_range["or_low"]:
                    st.error(f"📉 **OR Breakdown Bearish** — price ({current_price:.0f}) below OR Low ({opening_range['or_low']:.0f}). Expect bearish continuation, OR Low becomes resistance.")
                else:
                    st.info(f"📊 Price **inside** Opening Range — no clear bias. Wait for a decisive breakout.")

            with st.expander("📖 How to use the Opening Range"):
                st.markdown("""
**Opening Range (15 min) — 09:30–09:45 ET | 16:30–16:45 IL**

- **OR High breakout**: Bullish session bias. OR High becomes support on a retest.
- **OR Low breakdown**: Bearish session bias. OR Low becomes resistance.
- **Price inside OR**: Indecisive market — wait for direction.
- **OR Mid**: Balance zone. Breaking above/below gives half-session bias.
- **Statistics**: ~70% of the time price does not return inside the OR after a clean breakout.
                """)
        else:
            st.info("⏳ Opening Range will be calculated after 09:45 ET (16:45 IL)")

        # ── Order Blocks ──────────────────────────────────────────────────────
        st.divider()
        st.markdown("### 🟥🟩 Order Blocks")
        if not ob_df.empty:
            valid_obs = ob_df[ob_df["valid"] == True]
            invalid_obs = ob_df[ob_df["valid"] == False]

            ob_display_cols = st.columns(2)
            with ob_display_cols[0]:
                st.markdown(f"**✅ Active OBs: {len(valid_obs)}**")
                if not valid_obs.empty:
                    for _, ob in valid_obs.tail(5).iterrows():
                        emoji = "🟢" if ob["type"] == "bullish" else "🔴"
                        dist = abs(current_price - (ob["high"] + ob["low"]) / 2) if current_price else 0
                        near = " 🎯 **Close!**" if dist < 20 else ""
                        st.markdown(f"{emoji} **{ob['type'].upper()} OB** | {ob['low']:.0f}–{ob['high']:.0f} | dist: {dist:.0f} pts{near}")
            with ob_display_cols[1]:
                st.markdown(f"**❌ Invalidated OBs: {len(invalid_obs)}**")
                st.caption("OB is invalidated when price closes beyond the OB zone")

            with st.expander("📖 What is an Order Block?"):
                st.markdown("""
**Order Block** = the last candle in the opposite direction before a strong institutional move.

- **Bullish OB**: last bearish candle before a strong bullish impulse → that was institutional buying.
  - When price returns to this zone — institution "defends" the position → buy environment.
- **Bearish OB**: last bullish candle before a strong bearish impulse → that was institutional selling.
  - When price returns — sell environment.

**What invalidates an OB?** Price closing beyond the OB boundary = institution was bypassed = OB no longer valid.
                """)
        else:
            st.info("No Order Blocks detected in current data")

        # ── BOS / MSS ─────────────────────────────────────────────────────────
        st.divider()
        st.markdown("### 🔀 Market Structure — BOS / MSS")
        if not ms_df.empty:
            last_ms = ms_df.iloc[-1]
            is_mss = "MSS" in last_ms["label"]

            if is_mss:
                if "bullish" in last_ms["type"]:
                    st.success(f"🚀 **Bullish MSS detected** @ {last_ms['price']:.0f} — **Trend Reversal**: market shifted from bearish to bullish structure")
                else:
                    st.error(f"📉 **Bearish MSS detected** @ {last_ms['price']:.0f} — **Trend Reversal**: market shifted from bullish to bearish structure")
            else:
                if "bullish" in last_ms["type"]:
                    st.info(f"↑ **Bullish BOS** @ {last_ms['price']:.0f} — **Continuation**: broke prior swing high")
                else:
                    st.info(f"↓ **Bearish BOS** @ {last_ms['price']:.0f} — **Continuation**: broke prior swing low")

            st.markdown("**Recent structure events:**")
            rows_ms = []
            for _, r in ms_df.tail(8).iterrows():
                is_r_mss = "MSS" in r["label"]
                rows_ms.append({
                    "Time": str(r["datetime"]).split("+")[0].split(".")[0],
                    "Signal": r["label"],
                    "Price": f"{r['price']:.0f}",
                    "Importance": "🔴 Reversal" if is_r_mss else "🟡 Continuation",
                })
            st.dataframe(pd.DataFrame(rows_ms), use_container_width=True, hide_index=True)

            with st.expander("📖 BOS vs MSS — The Difference"):
                st.markdown("""
| | BOS (Break of Structure) | MSS (Market Structure Shift) |
|--|--|--|
| **Meaning** | Break **with** current trend | Break **against** current trend |
| **Signal** | Trend continuation | Potential reversal |
| **Example** | In uptrend — price breaks HH = BOS ↑ | In uptrend — price breaks HL = MSS ↓ |
| **Entry** | Continuation trade | Reversal trade |
| **Confirmation** | FVG / OB in direction | FVG / OB against prior trend |
                """)
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
            st.caption("Price position within weekly range (0%=low, 100%=high)")
            if pd_daily["pct"] < 35:
                st.success("📉→📈 Price in Discount — institutions prefer to buy in this zone")
            elif pd_daily["pct"] > 65:
                st.error("📈→📉 Price in Premium — institutions prefer to sell in this zone")
            else:
                st.info("Price at Equilibrium — avoid entries, wait for extension from 50%")
        with atr_col:
            if atr_data:
                consumed = atr_data["consumed_pct"]
                st.metric("📏 ATR Consumed Today",
                          f"{consumed:.0f}%",
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
        st.caption("~20-minute windows where institutional algorithms manipulate price before the true displacement")

        try:
            macros = nq_calculations.detect_ict_macros(df_5m)
            if macros:
                now_et_str = datetime.now(ET_TZ).strftime("%H:%M")
                for macro in macros:
                    status  = macro.get("status", "pending")
                    active  = macro.get("active", False)
                    label   = macro.get("pattern_label", "—")
                    pattern = macro.get("pattern", "")

                    # Color by pattern
                    if active:
                        bg = "#1a3a5c"
                        border = "2px solid #3498db"
                    elif "sweep_low" in pattern or "displacement_bullish" in pattern:
                        bg = "#0d3b27"; border = "1px solid #00c851"
                    elif "sweep_high" in pattern or "displacement_bearish" in pattern:
                        bg = "#3b0d0d"; border = "1px solid #ff4444"
                    elif status == "pending":
                        bg = "#1a1a2e"; border = "1px solid #444"
                    else:
                        bg = "#1c1c1c"; border = "1px solid #333"

                    with st.container():
                        cols = st.columns([2, 1, 1, 3, 1])
                        name_str = f"{'🔴 LIVE — ' if active else ''}{macro['name']}"
                        cols[0].markdown(f"**{name_str}**")
                        cols[1].caption(f"🇺🇸 {macro['et_start']}–{macro['et_end']} ET")
                        cols[2].caption(f"🇮🇱 {macro['il_start']}–{macro['il_end']}")
                        cols[3].markdown(label)
                        if status == "complete" and macro.get("range_pts"):
                            rng = macro["range_pts"]
                            move = macro.get("move_pts", 0)
                            cols[4].caption(f"±{rng:.0f}p / {move:+.0f}p")
                        elif status == "active":
                            cols[4].markdown("🟡")
                        else:
                            cols[4].caption("⏰")
            else:
                st.info("ICT Macro data unavailable — requires 5m data with datetime_et")
        except Exception as e:
            st.caption(f"ICT Macros: {e}")

        with st.expander("📖 ICT Macros Guide"):
            st.markdown("""
**ICT Macro Windows** are specific time windows where ICT (Inner Circle Trader) algorithms
create liquidity sweeps before the true displacement:

| Window | ET Time | IL Time | Characteristic |
|--------|---------|---------|----------------|
| London Open Macro | 02:33–02:55 | 09:33–09:55 | Stop hunt on overnight range |
| London Continuation | 04:03–04:20 | 11:03–11:20 | Departure from London range |
| Pre-Market Macro | 08:50–09:10 | 15:50–16:10 | RTH prep — sweep of Pre-market H/L |
| **NY AM Macro 1** | **09:50–10:10** | **16:50–17:10** | **Most important — Stop hunt + reversal** |
| **NY AM Macro 2** | **10:50–11:10** | **17:50–18:10** | **Distribution/Accumulation continuation** |
| Noon Macro | 11:50–12:10 | 18:50–19:10 | Midday reversal setup |
| PM Macro 1 | 13:10–13:30 | 20:10–20:30 | PM session manipulation start |
| PM Macro 2 | 14:50–15:10 | 21:50–22:10 | Power Hour preparation |
| PM Close Setup | 15:15–15:45 | 22:15–22:45 | MOC accumulation + stop hunt |

**Patterns:**
- 🔻 **Sweep High + Reversal** = Bearish Judas swing — price broke above high, then reversed → short entry
- 🔺 **Sweep Low + Reversal** = Bullish Judas swing — price broke below low, then reversed → long entry
- 🚀 **Displacement** = sharp clear move in one direction — continuation trade
            """)

        # ICT Concepts Reference
        st.divider()
        st.markdown("### 📖 ICT/SMC Quick Reference")
        with st.expander("Show concept glossary"):
            st.markdown("""
| Concept | Definition | Trade Signal |
|---------|-----------|-------------|
| **Order Block (OB)** | Last candle in opposite direction before institutional impulse | Buy/sell on return to block |
| **BOS (Break of Structure)** | Swing break in the direction of existing trend | Continuation move |
| **MSS (Market Structure Shift)** | Swing break against existing trend | Potential reversal |
| **Opening Range (OR)** | First 15-min High/Low | Breakout = session bias |
| **Premium/Discount** | Price position in range: >65%=sell, <35%=buy | Enter on correct side |
| **ATR Consumed** | % of daily range already consumed | >80% = avoid entries |
| **FVG (Fair Value Gap)** | Gap between candle N-2 high and candle N low | Price returns to fill the gap |
| **Judas Swing** | Overnight range breakout + reversal within 30 min | Enter in reversal direction |
| **Killzone** | Time window of high institutional activity | Only expect setups inside KZ |
| **AMD** | Accumulation → Manipulation → Distribution | Identify phase + trade accordingly |
| **Stacked Imbalance** | 3+ consecutive one-directional candles | Strong institutional direction |
| **Absorption** | High volume + narrow range + close at extreme | Institution absorbing, holding direction |
            """)

    # ──────────────────────────────────────────────────────────────────────────
    # TAB 6 — Historical Patterns
    # ──────────────────────────────────────────────────────────────────────────
    with tabs[5]:
        st.subheader("📜 Historical Patterns")

        if df_daily.empty:
            st.warning("Daily data unavailable")
        else:
            col_sim, col_wr = st.columns(2)

            with col_sim:
                st.markdown("### 🔍 Similar Days to Today")
                similar = nq_calculations.find_similar_setups(df_daily)
                if not similar.empty:
                    similar["similarity"] = (similar["similarity"] * 100).round(1)
                    similar["next_day_change"] = similar["next_day_change"].round(2)
                    similar["Result"] = similar["next_day_change"].apply(
                        lambda x: f"🟢 +{x:.1f}%" if x > 0 else f"🔴 {x:.1f}%")
                    display_sim = similar[["date", "similarity", "Result"]].copy()
                    display_sim.columns = ["Date", "Similarity %", "Next Day Result"]
                    st.dataframe(display_sim, use_container_width=True, hide_index=True)

                    up_days = (similar["next_day_change"] > 0).sum()
                    total = len(similar)
                    win_pct = up_days / total * 100
                    if win_pct >= 60:
                        st.success(f"📈 {up_days}/{total} similar days went up next day ({win_pct:.0f}% bullish bias)")
                    elif win_pct <= 40:
                        st.error(f"📉 {total - up_days}/{total} similar days fell next day ({100 - win_pct:.0f}% bearish bias)")
                    else:
                        st.info(f"📊 {win_pct:.0f}% of similar days went up — neutral bias")
                else:
                    st.info("Not enough history for similarity analysis")

            with col_wr:
                st.markdown("### ⏰ Win Rate by Hour (ET)")
                if not df_hourly.empty:
                    wr_df = nq_calculations.session_win_rates(df_hourly)
                    if not wr_df.empty:
                        fig_wr = px.bar(wr_df, x="hour_et", y="win_rate_pct",
                                        title="% of days NQ closed up in this ET hour",
                                        color="win_rate_pct",
                                        color_continuous_scale=["red", "yellow", "green"],
                                        range_color=[30, 70])
                        fig_wr.add_hline(y=50, line_dash="dash", line_color="white", line_width=1)
                        fig_wr.update_layout(paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
                                             font=dict(color="white"), showlegend=False,
                                             coloraxis_showscale=False, height=350)
                        st.plotly_chart(fig_wr, use_container_width=True)
                else:
                    st.info("Hourly data unavailable")

            # Historical stats table
            st.markdown("### 📊 Session Reference Table")
            if len(df_daily) >= 5:
                stats_rows = []
                for sess in SESSIONS:
                    name = sess["name"]
                    if name in ("Overnight", "Lunch Lull", "Cash Close"):
                        continue
                    stats_rows.append({
                        "Session": name,
                        "IL Hours": f"{sess['il_start']}–{sess['il_end']}",
                        "ET Hours": f"{sess['et_start']}–{sess['et_end']}",
                        "Participants": sess["participants"][:50] + "..." if len(sess["participants"]) > 50 else sess["participants"],
                    })
                st.dataframe(pd.DataFrame(stats_rows), use_container_width=True, hide_index=True)

    # ──────────────────────────────────────────────────────────────────────────
    # TAB 7 — Checklist
    # ──────────────────────────────────────────────────────────────────────────
    with tabs[6]:
        st.subheader("✅ Pre-Trade Checklist")

        # compute conditions automatically
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
            div = nq_calculations.detect_delta_divergence(df_5m)
            if abs(div.iloc[-4:].sum()) > 0:
                no_divergence = False
        conditions["no_divergence"] = no_divergence

        # 4. Not in Judas window (09:15-09:30 ET)
        not_judas_window = not (9 * 60 + 15 <= now_et_hm <= 9 * 60 + 30)
        conditions["not_judas"] = not_judas_window

        # 5. Stacked imbalance present
        has_stack = False
        if not df_5m.empty:
            stacks = nq_calculations.detect_stacked_imbalances(df_5m)
            has_stack = len(stacks) > 0
        conditions["has_stack"] = has_stack

        # 6. FVG present
        has_fvg = False
        if not df_5m.empty:
            fvg_df = nq_calculations.detect_fvg(df_5m)
            has_fvg = not fvg_df.empty
        conditions["has_fvg"] = has_fvg

        # 7. Volume above average (approximation)
        vol_ok = False
        if not df_5m.empty and len(df_5m) > 10:
            recent_vol = df_5m["volume"].iloc[-1]
            avg_vol = df_5m["volume"].iloc[-20:].mean()
            vol_ok = recent_vol >= avg_vol * 0.8
        conditions["vol_ok"] = vol_ok

        # Sprint-1 conditions
        # 8. Near valid Order Block
        near_ob = False
        if current_price and not ob_df.empty:
            valid_obs = ob_df[ob_df["valid"] == True]
            for _, ob in valid_obs.iterrows():
                mid = (ob["high"] + ob["low"]) / 2
                if abs(current_price - mid) / current_price < 0.003:
                    near_ob = True
                    break
        conditions["near_ob"] = near_ob

        # 9. BOS/MSS confirmation (last structure event aligns with any bias)
        has_ms_signal = not ms_df.empty
        conditions["has_ms"] = has_ms_signal

        # 10. OR breakout in clear direction
        or_breakout = False
        if opening_range and opening_range.get("or_complete") and current_price:
            or_breakout = current_price > opening_range["or_high"] or current_price < opening_range["or_low"]
        conditions["or_breakout"] = or_breakout

        # 11. ATR range not exhausted (< 80%)
        atr_ok = True
        if atr_data:
            atr_ok = atr_data.get("consumed_pct", 0) < 80
        conditions["atr_ok"] = atr_ok

        checklist = [
            ("good_session",  "Current session is suitable for trading (not Lunch Lull / Overnight)"),
            ("near_level",    "Price is near a key level (PDH/PDL/POC/FVG ±0.25%)"),
            ("no_divergence", "No Delta Divergence in last 4 bars"),
            ("not_judas",     "Not in Judas Swing window (09:15–09:30 ET)"),
            ("has_stack",     "Stacked Imbalance present in potential direction"),
            ("has_fvg",       "Active Fair Value Gap exists"),
            ("vol_ok",        "Volume is at least 80% of session average"),
            ("near_ob",       "Price is near an active Order Block (±0.3%)"),
            ("has_ms",        "Clear BOS / MSS signal in today's data"),
            ("or_breakout",   "Opening Range has broken in a clear direction"),
            ("atr_ok",        "ATR range not exhausted — more than 20% of daily range remaining"),
        ]

        score = sum(1 for key, _ in checklist if conditions.get(key, False))
        total = len(checklist)
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

        st.divider()
        st.markdown("### Condition Details")
        # Show failed conditions first
        failed = [(k, l) for k, l in checklist if not conditions.get(k, False)]
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
1. **What is the daily bias?** (Daily Chart — Higher High/Low?)
2. **What is the session bias?** (Asian range: above/below?)
3. **What is the nearest PD Array?** (FVG? OB? Breaker?)
4. **What is the nearest liquidity pool?** (Stops above high / below low?)
5. **When is the killzone?** (Only expect signals inside the killzone)
6. **Is there confirmation?** (MSS? BOS? Delta?)
        """)

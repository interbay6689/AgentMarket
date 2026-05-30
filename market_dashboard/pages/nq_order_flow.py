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

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from AgentMarket.market_dashboard.modules import nq_data, nq_calculations

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
    st.title("📊 NQ Order Flow Dashboard — תכנון יום מסחר")
    st.caption("נאסדק 100 חוזים עתידיים | שעון ישראל | Order Flow & ICT/SMC Analysis")

    tabs = st.tabs([
        "🎯 תוכנית המסחר",
        "🕐 ציר הזמן",
        "📏 רמות מפתח",
        "🌊 Order Flow",
        "⚔️ ICT/SMC",
        "📜 דפוסים היסטוריים",
        "✅ צ'קליסט",
    ])

    # ── data load ──────────────────────────────────────────────────────────────
    with st.spinner("טוען נתוני NQ..."):
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
        st.subheader("🎯 תוכנית המסחר היומית")
        now_il = datetime.now(IL_TZ)
        st.markdown(f"**שעון ישראל:** `{now_il.strftime('%H:%M:%S')}` | **תאריך:** `{now_il.strftime('%d/%m/%Y')}`")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            if current_price:
                _price_badge(current_price, "💰 מחיר NQ נוכחי", pct_change)
            else:
                st.metric("💰 מחיר NQ", "לא זמין")
        with col2:
            risk_emoji = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(current_session.get("risk", "low"), "⚪")
            st.metric("📍 סשן נוכחי", current_session.get("name", "—"))
            st.caption(f"{risk_emoji} רמת פעילות: {current_session.get('risk', '—')}")
        with col3:
            participants = PARTICIPANTS_BY_SESSION.get(current_session.get("name", ""), [])
            st.metric("👥 משתתפים עיקריים", f"{len(participants)} סוגים")
            if participants:
                st.caption(participants[0])
        with col4:
            vix_df = corr.get("VIX", pd.DataFrame())
            if not vix_df.empty and "close" in vix_df.columns:
                vix_val = float(vix_df["close"].dropna().iloc[-1])
                vix_color = "🔴" if vix_val > 20 else "🟡" if vix_val > 15 else "🟢"
                st.metric(f"{vix_color} VIX", f"{vix_val:.2f}")
            else:
                st.metric("VIX", "לא זמין")

        # ── Sprint-1 row: ATR / OR / Premium-Discount / OB ──────────────────
        st.markdown("---")
        s1c1, s1c2, s1c3, s1c4 = st.columns(4)
        with s1c1:
            if atr_data:
                consumed = atr_data["consumed_pct"]
                color = "🔴" if consumed > 80 else "🟡" if consumed > 55 else "🟢"
                st.metric(f"{color} ATR Range Consumed",
                          f"{consumed:.0f}%",
                          delta=f"{atr_data['remaining_pts']:.0f} נקודות נותרו")
                st.caption(f"ATR: {atr_data['atr']:.0f} | היום: {atr_data['today_range']:.0f}")
            else:
                st.metric("ATR Range", "לא זמין")
        with s1c2:
            if opening_range:
                or_status = "✅ הושלם" if opening_range.get("or_complete") else f"⏳ {opening_range.get('or_bars', 0)} נרות"
                st.metric("📐 Opening Range (15m)", or_status)
                st.caption(f"H: {opening_range['or_high']:.0f} | L: {opening_range['or_low']:.0f} | Range: {opening_range['or_range']:.0f}")
            else:
                st.metric("📐 Opening Range", "⏳ לפני Open")
                st.caption("09:30–09:45 ET | 16:30–16:45 ישראל")
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
                st.metric(f"{ob_emoji} Order Block פעיל", f"{last_ob['low']:.0f}–{last_ob['high']:.0f}")
                st.caption(f"{'Bullish' if last_ob['type'] == 'bullish' else 'Bearish'} OB | {len(valid_obs)} פעילים")
            else:
                st.metric("Order Block", "לא זוהה")

        st.divider()

        # Smart banner
        banners = []
        if not df_5m.empty:
            fvg_df = nq_calculations.detect_fvg(df_5m)
            if not fvg_df.empty:
                last_fvg = fvg_df.iloc[-1]
                banners.append(f"🟦 **FVG {last_fvg['type'].upper()}** קיים ב-{last_fvg['bottom']:.0f}–{last_fvg['top']:.0f} — מגנט מחיר פוטנציאלי")
            div_series = nq_calculations.detect_delta_divergence(df_5m)
            last_div = div_series.iloc[-4:].sum()
            if last_div > 0:
                banners.append("⚠️ **Delta Divergence שורי** זוהה — מחיר עשוי לא לשקף לחץ מכירה")
            elif last_div < 0:
                banners.append("⚠️ **Delta Divergence דובי** זוהה — Distribution אפשרי")
            stacks = nq_calculations.detect_stacked_imbalances(df_5m)
            if stacks:
                last_stack = stacks[-1]
                banners.append(f"📚 **Stacked Imbalance {last_stack['direction'].upper()}** — {last_stack['candle_count']} נרות רצופים | {last_stack['price_start']:.0f} → {last_stack['price_end']:.0f}")

            # Sprint-1 banners
            if opening_range and opening_range.get("or_complete") and current_price:
                if current_price > opening_range["or_high"]:
                    banners.append(f"📐 **OR Breakout BULLISH** — מחיר פרץ מעל Opening Range High ({opening_range['or_high']:.0f}) — bias שורי לסשן")
                elif current_price < opening_range["or_low"]:
                    banners.append(f"📐 **OR Breakdown BEARISH** — מחיר פרץ מתחת Opening Range Low ({opening_range['or_low']:.0f}) — bias דובי לסשן")

            if not ms_df.empty:
                last_ms = ms_df.iloc[-1]
                ms_emoji = "🚀" if "bullish" in last_ms["type"] else "📉"
                ms_is_mss = "MSS" in last_ms["label"]
                ms_text = "**MSS — היפוך מגמה!**" if ms_is_mss else "**BOS — המשך מגמה**"
                banners.append(f"{ms_emoji} {ms_text} זוהה | {last_ms['label']} @ {last_ms['price']:.0f}")

            if not ob_df.empty and current_price:
                valid_obs = ob_df[ob_df["valid"] == True]
                for _, ob in valid_obs.tail(3).iterrows():
                    dist = abs(current_price - (ob["high"] + ob["low"]) / 2)
                    if dist / current_price < 0.003:
                        ob_emoji = "🟢" if ob["type"] == "bullish" else "🔴"
                        banners.append(f"{ob_emoji} **Order Block {ob['type'].upper()}** — מחיר קרוב מאוד ל-OB ב-{ob['low']:.0f}–{ob['high']:.0f} ({dist:.0f} נקודות)")

            if atr_data and atr_data.get("consumed_pct", 0) > 85:
                banners.append(f"⚠️ **ATR {atr_data['consumed_pct']:.0f}% נוצל** — range יומי כמעט מוצה, כניסות חדשות בעלות R:R ירוד")

        if current_session.get("name") == "Lunch Lull":
            banners.insert(0, "⛔ **Lunch Lull פעיל** — נפח נמוך, false breakouts שכיחים, מומלץ להימנע ממסחר")

        if banners:
            for b in banners:
                st.info(b)
        else:
            st.success("✅ אין אותות מיוחדים כרגע — שוק רגיל")

        st.divider()
        st.subheader("📈 נכסים מתואמים")
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
                    st.metric(name, "לא זמין")

        if participants:
            st.divider()
            st.subheader(f"👥 משתתפים בסשן: {current_session.get('name', '')}")
            for p in participants:
                st.markdown(f"• {p}")

    # ──────────────────────────────────────────────────────────────────────────
    # TAB 2 — Session Timeline
    # ──────────────────────────────────────────────────────────────────────────
    with tabs[1]:
        st.subheader("🕐 ציר זמן הסשנים — שעון ישראל")

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
                hovertemplate=f"<b>{s['name']}</b><br>{s['il_start']}–{s['il_end']} ישראל<br>{s['et_start']}–{s['et_end']} ET<br>{s['participants']}<extra></extra>",
                name=s["name"],
                showlegend=False,
            ))

        # current time marker
        fig_tl.add_vline(x=now_minutes, line_dash="dash", line_color="red", line_width=2,
                         annotation_text=f"⏰ {now_il.strftime('%H:%M')} ישראל",
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
            title="ציר זמן סשנים — שעון ישראל (גרינוויץ +2/+3)",
            xaxis=dict(title="שעה (ישראל)", range=[0, 1440],
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
        st.subheader("👥 פירוט משתתפים לפי סשן")
        for s in SESSIONS:
            parts = PARTICIPANTS_BY_SESSION.get(s["name"], [])
            with st.expander(f"{s['name']} | {s['il_start']}–{s['il_end']} ישראל"):
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
        st.subheader("📏 רמות מפתח ו-Volume Profile")

        if df_5m.empty:
            st.warning("נתוני 5 דקות לא זמינים — ייתכן ששוק סגור")
        else:
            fig_kl = make_subplots(rows=1, cols=2, column_widths=[0.75, 0.25],
                                   shared_yaxes=True,
                                   subplot_titles=["NQ 5-דקות היום", "Volume Profile"])

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
        st.subheader("📋 טבלת רמות")
        if current_price and levels:
            rows = []
            for key, (label, _) in level_colors.items():
                val = levels.get(key)
                if val and not np.isnan(val):
                    dist = val - current_price
                    rows.append({"רמה": label, "מחיר": f"{val:.0f}",
                                 "מרחק (נקודות)": f"{dist:+.0f}",
                                 "סוג": "תמיכה 🟢" if dist < 0 else "התנגדות 🔴"})

            # Add Opening Range levels
            if opening_range:
                for or_key, or_label in [("or_high", "OR High (15m)"), ("or_low", "OR Low (15m)"), ("or_mid", "OR Mid")]:
                    val = opening_range.get(or_key)
                    if val:
                        dist = val - current_price
                        rows.append({"רמה": f"📐 {or_label}", "מחיר": f"{val:.0f}",
                                     "מרחק (נקודות)": f"{dist:+.0f}",
                                     "סוג": "תמיכה 🟢" if dist < 0 else "התנגדות 🔴"})

            # Add valid Order Blocks
            if not ob_df.empty:
                valid_obs = ob_df[ob_df["valid"] == True].tail(3)
                for _, ob in valid_obs.iterrows():
                    mid = (ob["high"] + ob["low"]) / 2
                    dist = mid - current_price
                    emoji = "🟢" if ob["type"] == "bullish" else "🔴"
                    rows.append({"רמה": f"{emoji} {ob['type'].capitalize()} OB", "מחיר": f"{ob['low']:.0f}–{ob['high']:.0f}",
                                 "מרחק (נקודות)": f"{dist:+.0f}",
                                 "סוג": "תמיכה 🟢" if ob["type"] == "bullish" else "התנגדות 🔴"})

            if rows:
                df_lev = pd.DataFrame(rows).sort_values("מרחק (נקודות)")
                st.dataframe(df_lev, use_container_width=True, hide_index=True)

    # ──────────────────────────────────────────────────────────────────────────
    # TAB 4 — Order Flow Indicators
    # ──────────────────────────────────────────────────────────────────────────
    with tabs[3]:
        st.subheader("🌊 Order Flow Indicators")

        if df_5m.empty:
            st.warning("נתוני 5 דקות לא זמינים")
        else:
            cd = nq_calculations.cumulative_delta(df_5m)
            div = nq_calculations.detect_delta_divergence(df_5m)
            absorption = nq_calculations.detect_absorption(df_5m)

            bearish_div = int((div.iloc[-8:] > 0).sum())
            bullish_div = int((div.iloc[-8:] < 0).sum())
            if bearish_div > 0:
                st.error(f"⚠️ Delta Divergence דובי זוהה ({bearish_div} איתות) — Distribution: מחיר עולה אך selling dominates")
            elif bullish_div > 0:
                st.success(f"📈 Delta Divergence שורי זוהה ({bullish_div} איתות) — Accumulation: מחיר יורד אך buying dominates")

            dt_col = "datetime_il" if "datetime_il" in df_5m.columns else "datetime"
            fig_cd = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                   row_heights=[0.6, 0.4],
                                   subplot_titles=["NQ מחיר", "Cumulative Delta"])

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
                st.subheader("📚 Stacked Imbalances זוהו")
                for s in stacks[-5:]:
                    direction_emoji = "🟢" if s["direction"] == "bullish" else "🔴"
                    st.markdown(f"{direction_emoji} **{s['direction'].upper()}** — {s['candle_count']} נרות | {s['price_start']:.0f} → {s['price_end']:.0f}")

            # Absorption table
            if not df_5m.empty and absorption.abs().sum() > 0:
                st.subheader("🧲 Absorption Events")
                abs_rows = []
                for i, val in enumerate(absorption):
                    if val != 0:
                        row = df_5m.iloc[i]
                        abs_rows.append({
                            "זמן": str(row[dt_col]).split("+")[0],
                            "כיוון": "🟢 Bullish" if val > 0 else "🔴 Bearish",
                            "Volume": f"{int(row['volume']):,}",
                            "מחיר": f"{row['close']:.0f}",
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
            ("🏙️ London KZ", "02:00", "05:00", "07:00–11:00 ישראל"),
            ("🗽 NY AM KZ", "07:00", "10:00", "14:00–17:00 ישראל"),
            ("🌆 NY PM KZ", "13:30", "16:00", "20:30–23:00 ישראל"),
        ]
        for col, (title, start_et, end_et, il_label) in zip([kz_col1, kz_col2, kz_col3], kz_configs):
            with col:
                active = in_kz(start_et, end_et)
                status = "🟢 פעיל עכשיו" if active else "⚫ לא פעיל"
                st.metric(title, status)
                st.caption(f"ET: {start_et}–{end_et} | {il_label}")

        st.divider()

        # AMD Cycle
        st.markdown("### 🔄 AMD Cycle — Accumulation / Manipulation / Distribution")
        amd_col1, amd_col2, amd_col3 = st.columns(3)

        def amd_phase():
            if now_et_hm < 7 * 60 + 30:
                return "Accumulation", "💡 בניית פוזיציות בשקט — overnight לומד את הכיוון"
            elif now_et_hm < 10 * 60:
                return "Manipulation", "⚡ חלון Judas Swing — תנועה מזויפת לפני הכיוון האמיתי"
            else:
                return "Distribution", "📤 חלוקה והוצאת הרווח — המוסדי מוכר/קונה לציבור"

        current_phase, phase_desc = amd_phase()
        phase_colors = {"Accumulation": "🔵", "Manipulation": "🟡", "Distribution": "🔴"}
        with amd_col1:
            color = "🔵" if current_phase == "Accumulation" else "⚫"
            st.metric("💡 Accumulation", color + " פעיל" if current_phase == "Accumulation" else color + " הסתיים")
            st.caption("Overnight + Pre-Market")
        with amd_col2:
            color = "🟡" if current_phase == "Manipulation" else "⚫"
            st.metric("⚡ Manipulation", color + " פעיל" if current_phase == "Manipulation" else color + " הסתיים")
            st.caption("09:15–09:45 ET (Judas Swing)")
        with amd_col3:
            color = "🔴" if current_phase == "Distribution" else "⚫"
            st.metric("📤 Distribution", color + " פעיל" if current_phase == "Distribution" else color + " עדיין לא")
            st.caption("RTH 10:00+ ET")

        st.info(f"**שלב נוכחי: {current_phase}** — {phase_desc}")

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
                    st.error(f"🚨 **Judas Swing דובי זוהה!** — מחיר פרץ מעל Overnight High ({overnight_high:.0f}), "
                             f"הגיע ל-{judas.get('sweep_high', 0):.0f}, ואז הפך — כיוון אמיתי: ירידה")
                else:
                    st.success(f"🚀 **Judas Swing שורי זוהה!** — מחיר פרץ מתחת Overnight Low ({overnight_low:.0f}), "
                               f"הגיע ל-{judas.get('sweep_low', 0):.0f}, ואז הפך — כיוון אמיתי: עלייה")
            else:
                st.info(f"📊 Overnight Range: {overnight_low:.0f} – {overnight_high:.0f} | Judas Swing לא זוהה עדיין")
        else:
            st.info("📊 Overnight High/Low לא זמין — בדוק לאחר שעה 09:30 ET (16:30 ישראל)")

        # ── Opening Range ────────────────────────────────────────────────────
        st.divider()
        st.markdown("### 📐 Opening Range (OR 15 דקות)")
        if opening_range:
            or_col1, or_col2, or_col3, or_col4 = st.columns(4)
            with or_col1:
                st.metric("OR High", f"{opening_range.get('or_high', 0):.0f}")
            with or_col2:
                st.metric("OR Low", f"{opening_range.get('or_low', 0):.0f}")
            with or_col3:
                st.metric("OR Range (נקודות)", f"{opening_range.get('or_range', 0):.1f}")
            with or_col4:
                status = "✅ הושלם" if opening_range.get("or_complete") else f"⏳ {opening_range.get('or_bars', 0)}/3 נרות"
                st.metric("סטטוס", status)

            if current_price and opening_range.get("or_complete"):
                if current_price > opening_range["or_high"]:
                    st.success(f"🚀 **OR Breakout Bullish** — מחיר ({current_price:.0f}) מעל OR High ({opening_range['or_high']:.0f}). צפה לרציפות שורית, OR High הופך לתמיכה.")
                elif current_price < opening_range["or_low"]:
                    st.error(f"📉 **OR Breakdown Bearish** — מחיר ({current_price:.0f}) מתחת OR Low ({opening_range['or_low']:.0f}). צפה לרציפות דובית, OR Low הופך להתנגדות.")
                else:
                    st.info(f"📊 מחיר **בתוך** ה-Opening Range — אין bias ברור. המתן לפריצה ברורה.")

            with st.expander("📖 כיצד להשתמש ב-Opening Range"):
                st.markdown("""
**Opening Range (15 דקות) — 09:30–09:45 ET | 16:30–16:45 ישראל**

- **OR High פריצה**: Bias שורי לאורך כל הסשן. OR High הופך לתמיכה בחזרה אליו.
- **OR Low פריצה**: Bias דובי לאורך כל הסשן. OR Low הופך להתנגדות.
- **מחיר בתוך ה-OR**: שוק מתחשב — המתן לכיוון.
- **OR Mid**: אזור איזון. פריצה מעל/מתחת נותנת bias לחצי.
- **סטטיסטיקה**: ~70% מהפעמים מחיר לא חוזר לתוך ה-OR לאחר פריצה ברורה.
                """)
        else:
            st.info("⏳ Opening Range יחושב לאחר 09:45 ET (16:45 ישראל)")

        # ── Order Blocks ──────────────────────────────────────────────────────
        st.divider()
        st.markdown("### 🟥🟩 Order Blocks")
        if not ob_df.empty:
            valid_obs = ob_df[ob_df["valid"] == True]
            invalid_obs = ob_df[ob_df["valid"] == False]

            ob_display_cols = st.columns(2)
            with ob_display_cols[0]:
                st.markdown(f"**✅ OBs פעילים: {len(valid_obs)}**")
                if not valid_obs.empty:
                    for _, ob in valid_obs.tail(5).iterrows():
                        emoji = "🟢" if ob["type"] == "bullish" else "🔴"
                        dist = abs(current_price - (ob["high"] + ob["low"]) / 2) if current_price else 0
                        near = " 🎯 **קרוב!**" if dist < 20 else ""
                        st.markdown(f"{emoji} **{ob['type'].upper()} OB** | {ob['low']:.0f}–{ob['high']:.0f} | מרחק: {dist:.0f} נקודות{near}")
            with ob_display_cols[1]:
                st.markdown(f"**❌ OBs שבוטלו: {len(invalid_obs)}**")
                st.caption("OB מבוטל כאשר מחיר סגר מחוץ לתחום ה-OB")

            with st.expander("📖 מהו Order Block?"):
                st.markdown("""
**Order Block** = הנר האחרון בכיוון ההפוך לפני תנועה מוסדית חזקה.

- **Bullish OB**: הנר הדובי האחרון לפני impulse שורי חזק → זו הייתה הקנייה המוסדית.
  - כשמחיר חוזר לאזור זה — מוסד "מגן" על הפוזיציה → סביבת קנייה.
- **Bearish OB**: הנר השורי האחרון לפני impulse דובי חזק → זו הייתה המכירה המוסדית.
  - כשמחיר חוזר — סביבת מכירה.

**מה מבטל OB?** מחיר שסוגר מעבר לגבול ה-OB = המוסד "נעקף" = OB לא תקף.
                """)
        else:
            st.info("Order Blocks לא זוהו בנתונים הנוכחיים")

        # ── BOS / MSS ─────────────────────────────────────────────────────────
        st.divider()
        st.markdown("### 🔀 Market Structure — BOS / MSS")
        if not ms_df.empty:
            last_ms = ms_df.iloc[-1]
            is_mss = "MSS" in last_ms["label"]

            if is_mss:
                if "bullish" in last_ms["type"]:
                    st.success(f"🚀 **MSS Bullish זוהה** @ {last_ms['price']:.0f} — **היפוך מגמה**: שוק עבר ממבנה דובי לשורי")
                else:
                    st.error(f"📉 **MSS Bearish זוהה** @ {last_ms['price']:.0f} — **היפוך מגמה**: שוק עבר ממבנה שורי לדובי")
            else:
                if "bullish" in last_ms["type"]:
                    st.info(f"↑ **BOS Bullish** @ {last_ms['price']:.0f} — **המשך עלייה**: פריצת swing high קודם")
                else:
                    st.info(f"↓ **BOS Bearish** @ {last_ms['price']:.0f} — **המשך ירידה**: שבירת swing low קודם")

            # Show last 5 events
            st.markdown("**אירועי מבנה אחרונים:**")
            display_ms = ms_df.tail(8).copy()
            display_ms["סוג"] = display_ms["label"]
            display_ms["מחיר"] = display_ms["price"].apply(lambda x: f"{x:.0f}")
            display_ms["זמן"] = display_ms["datetime"].astype(str).str.split("+").str[0]
            rows_ms = []
            for _, r in ms_df.tail(8).iterrows():
                is_r_mss = "MSS" in r["label"]
                rows_ms.append({
                    "זמן": str(r["datetime"]).split("+")[0].split(".")[0],
                    "אות": r["label"],
                    "מחיר": f"{r['price']:.0f}",
                    "חשיבות": "🔴 היפוך" if is_r_mss else "🟡 המשך",
                })
            st.dataframe(pd.DataFrame(rows_ms), use_container_width=True, hide_index=True)

            with st.expander("📖 BOS vs MSS — ההבדל"):
                st.markdown("""
| | BOS (Break of Structure) | MSS (Market Structure Shift) |
|--|--|--|
| **משמעות** | שבירה **בכיוון** המגמה הקיימת | שבירה **נגד** המגמה הקיימת |
| **אות** | המשך תנועה | היפוך פוטנציאלי |
| **דוגמה** | בuptrend — מחיר שובר HH = BOS ↑ | בuptrend — מחיר שובר HL = MSS ↓ |
| **כניסה** | continuation trade | reversal trade |
| **אימות** | FVG / OB בכיוון | FVG / OB נגד המגמה הקודמת |
                """)
        else:
            st.info("Market Structure events לא זוהו — נדרש יותר נתונים intraday")

        # ── Premium / Discount + ATR ──────────────────────────────────────────
        st.divider()
        st.markdown("### 📊 Premium / Discount & ATR Consumed")
        pd_col, atr_col = st.columns(2)
        with pd_col:
            zone_colors = {"Deep Discount": "🟢🟢", "Discount": "🟢", "Equilibrium": "🟡",
                           "Premium": "🔴", "Deep Premium": "🔴🔴"}
            zone_icon = zone_colors.get(pd_daily["zone"], "⚪")
            st.metric(f"{zone_icon} {pd_daily['zone']}", f"{pd_daily['pct']:.1f}%",
                      delta="סביבת קנייה" if pd_daily["bias"] == "bullish" else
                            "סביבת מכירה" if pd_daily["bias"] == "bearish" else "איזון")
            st.caption("מיקום מחיר ביחס ל-weekly range (0%=low, 100%=high)")
            if pd_daily["pct"] < 35:
                st.success("📉→📈 מחיר ב-Discount — מוסד מעדיף לקנות באזור זה")
            elif pd_daily["pct"] > 65:
                st.error("📈→📉 מחיר ב-Premium — מוסד מעדיף למכור באזור זה")
            else:
                st.info("מחיר ב-Equilibrium — נמנע מכניסות, המתן להתרחקות מ-50%")
        with atr_col:
            if atr_data:
                consumed = atr_data["consumed_pct"]
                st.metric("📏 ATR Consumed היום",
                          f"{consumed:.0f}%",
                          delta=f"נותרו ~{atr_data['remaining_pts']:.0f} נקודות")
                st.progress(min(consumed / 100, 1.0))
                st.caption(f"ATR({14}d): {atr_data['atr']:.0f} נקודות | Range היום: {atr_data['today_range']:.0f}")
                if consumed > 80:
                    st.error("⚠️ Range מוצה — R:R לכניסות חדשות ירוד מאוד")
                elif consumed > 60:
                    st.warning("⚠️ Range מתקדם — בחר כניסות בזהירות")
                else:
                    st.success("✅ Range פתוח — פוטנציאל תנועה טוב")
            else:
                st.metric("ATR Consumed", "לא זמין")

        # ICT Concepts Reference
        st.divider()
        st.markdown("### 📖 מדריך ICT/SMC מהיר")
        with st.expander("הצג מדריך מושגים"):
            st.markdown("""
| מושג | הגדרה | אות מסחר |
|------|--------|-----------|
| **Order Block (OB)** | הנר האחרון בכיוון ההפוך לפני impulse מוסדי | קנייה/מכירה בחזרה לblock |
| **BOS (Break of Structure)** | שבירת swing בכיוון המגמה הקיימת | המשך תנועה |
| **MSS (Market Structure Shift)** | שבירת swing נגד המגמה הקיימת | היפוך פוטנציאלי |
| **Opening Range (OR)** | High/Low של 15 דקות ראשונות | פריצה = bias לסשן |
| **Premium/Discount** | מיקום מחיר ב-range: >65%=מכור, <35%=קנה | כניסה בצד הנכון |
| **ATR Consumed** | כמה % מה-range היומי כבר "נאכל" | >80% = הימנע מכניסות |
| **FVG (Fair Value Gap)** | פער בין high נר N-2 לlow נר N | מחיר חוזר למלא את הפער |
| **Judas Swing** | פריצת overnight range + היפוך תוך 30 דקות | כניסה בכיוון ההיפוך |
| **Killzone** | חלון זמן של פעילות מוסדית גבוהה | סטאפים בתוך ה-killzone |
| **AMD** | Accumulation → Manipulation → Distribution | זיהוי שלב + כניסה בהתאם |
| **Stacked Imbalance** | 3+ נרות חד-כיווניים ברצף | כיוון מוסדי חזק |
| **Absorption** | volume גדול + range צר + close בקצה | מוסדי בולע ומחזיק כיוון |
            """)

    # ──────────────────────────────────────────────────────────────────────────
    # TAB 6 — Historical Patterns
    # ──────────────────────────────────────────────────────────────────────────
    with tabs[5]:
        st.subheader("📜 דפוסים היסטוריים")

        if df_daily.empty:
            st.warning("נתוני daily לא זמינים")
        else:
            col_sim, col_wr = st.columns(2)

            with col_sim:
                st.markdown("### 🔍 ימים דומים להיום")
                similar = nq_calculations.find_similar_setups(df_daily)
                if not similar.empty:
                    similar["similarity"] = (similar["similarity"] * 100).round(1)
                    similar["next_day_change"] = similar["next_day_change"].round(2)
                    similar["תוצאה"] = similar["next_day_change"].apply(
                        lambda x: f"🟢 +{x:.1f}%" if x > 0 else f"🔴 {x:.1f}%")
                    display_sim = similar[["date", "similarity", "תוצאה"]].copy()
                    display_sim.columns = ["תאריך", "דמיון %", "תוצאת יום הבא"]
                    st.dataframe(display_sim, use_container_width=True, hide_index=True)

                    up_days = (similar["next_day_change"] > 0).sum()
                    total = len(similar)
                    win_pct = up_days / total * 100
                    if win_pct >= 60:
                        st.success(f"📈 מתוך {total} ימים דומים: {up_days} ירדו לאחר מכן ({win_pct:.0f}% bias שורי)")
                    elif win_pct <= 40:
                        st.error(f"📉 מתוך {total} ימים דומים: {total - up_days} ירדו ({100 - win_pct:.0f}% bias דובי)")
                    else:
                        st.info(f"📊 מתוך {total} ימים דומים: {win_pct:.0f}% עלו — bias ניטרלי")
                else:
                    st.info("אין מספיק היסטוריה לניתוח דמיון")

            with col_wr:
                st.markdown("### ⏰ Win Rate לפי שעה (ET)")
                if not df_hourly.empty:
                    wr_df = nq_calculations.session_win_rates(df_hourly)
                    if not wr_df.empty:
                        fig_wr = px.bar(wr_df, x="hour_et", y="win_rate_pct",
                                        title="% ימים שNQ עלה בשעה זו (ET)",
                                        color="win_rate_pct",
                                        color_continuous_scale=["red", "yellow", "green"],
                                        range_color=[30, 70])
                        fig_wr.add_hline(y=50, line_dash="dash", line_color="white", line_width=1)
                        fig_wr.update_layout(paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
                                             font=dict(color="white"), showlegend=False,
                                             coloraxis_showscale=False, height=350)
                        st.plotly_chart(fig_wr, use_container_width=True)
                else:
                    st.info("נתוני שעתיים לא זמינים")

            # Historical stats table
            st.markdown("### 📊 סטטיסטיקות לפי סשן (Daily)")
            if len(df_daily) >= 5:
                stats_rows = []
                for sess in SESSIONS:
                    name = sess["name"]
                    if name in ("Overnight", "Lunch Lull", "Cash Close"):
                        continue
                    stats_rows.append({
                        "סשן": name,
                        "שעות ישראל": f"{sess['il_start']}–{sess['il_end']}",
                        "שעות ET": f"{sess['et_start']}–{sess['et_end']}",
                        "משתתפים": sess["participants"][:50] + "..." if len(sess["participants"]) > 50 else sess["participants"],
                    })
                st.dataframe(pd.DataFrame(stats_rows), use_container_width=True, hide_index=True)

    # ──────────────────────────────────────────────────────────────────────────
    # TAB 7 — Checklist
    # ──────────────────────────────────────────────────────────────────────────
    with tabs[6]:
        st.subheader("✅ צ'קליסט לפני כניסה לעסקה")

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
            ("good_session",  "הסשן הנוכחי מתאים למסחר (לא Lunch Lull / Overnight)"),
            ("near_level",    "המחיר קרוב לרמת מפתח (PDH/PDL/POC/FVG ±0.25%)"),
            ("no_divergence", "אין Delta Divergence ב-4 נרות אחרונים"),
            ("not_judas",     "לא בחלון Judas Swing (09:15–09:30 ET)"),
            ("has_stack",     "יש Stacked Imbalance בכיוון האפשרי"),
            ("has_fvg",       "יש Fair Value Gap פעיל"),
            ("vol_ok",        "Volume לפחות 80% מהממוצע"),
            ("near_ob",       "מחיר קרוב ל-Order Block פעיל (±0.3%)"),
            ("has_ms",        "יש אות BOS / MSS ברור בנתוני היום"),
            ("or_breakout",   "Opening Range פרץ בכיוון ברור (מחוץ ל-OR)"),
            ("atr_ok",        "ATR Range לא מוצה — נותרו יותר מ-20% מהrange היומי"),
        ]

        score = sum(1 for key, _ in checklist if conditions.get(key, False))
        total = len(checklist)
        score_pct = score / total * 100

        st.metric("ציון צ'קליסט", f"{score}/{total}")
        st.progress(score / total)

        if score_pct >= 70:
            st.success(f"✅ {score}/{total} תנאים מתקיימים ({score_pct:.0f}%) — **תנאים טובים לסחרות**")
        elif score_pct >= 45:
            st.warning(f"⚠️ {score}/{total} תנאים מתקיימים ({score_pct:.0f}%) — **אזהרה: לא כל התנאים קיימים**")
        else:
            st.error(f"❌ {score}/{total} תנאים מתקיימים ({score_pct:.0f}%) — **לא מומלץ לסחור עכשיו**")

        st.divider()
        st.markdown("### פירוט תנאים")
        for key, label in checklist:
            passed = conditions.get(key, False)
            icon = "✅" if passed else "❌"
            st.markdown(f"{icon} {label}")

        st.divider()
        st.markdown("### 📝 הערות ידניות")
        note = st.text_area("הכנס הערות לתכנון המסחר היום:", height=100,
                             placeholder="לדוגמה: אין נתונים כלכליים היום, VIX נמוך, trend שורי מאתמול...")

        st.markdown("#### 📋 ICT Pre-Trade Framework")
        st.markdown("""
1. **מה הbias היומי?** (Daily Chart — Higher High/Low?)
2. **מה הsession bias?** (Asian range: above/below?)
3. **מה ה-PD Array הכי קרובה?** (FVG? OB? Breaker?)
4. **מה הliquidity pool הכי קרובה?** (Stops מעל high / מתחת low?)
5. **מתי ה-killzone?** (מצפה לאות בתוך הchallzone בלבד)
6. **האם יש confirmation?** (MSS? BOS? Delta?)
        """)

"""Trade Journal — per-trade log, factor analysis, equity curve."""
import sys
from pathlib import Path

_MD   = Path(__file__).resolve().parents[1]
_ROOT = _MD.parent
for p in (str(_MD), str(_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from modules.trade_tracker import (
    load_trade_log, update_trade_manual, evaluate_open_trades,
    factor_win_rates, equity_curve, summary_stats, load_trade_settings,
)
from modules.nq_data import get_todays_nq_data


# ─── helpers ──────────────────────────────────────────────────────────────────

def _outcome_icon(o: str) -> str:
    return {"WIN": "✅", "LOSS": "❌", "PARTIAL": "🟡", "OPEN": "🔵"}.get(o, "—")


def _dir_icon(d: str) -> str:
    return {"LONG": "🟢 LONG", "SHORT": "🔴 SHORT", "NEUTRAL": "⚪"}.get(d, d)


# ─── Tab 1: Trade Log ─────────────────────────────────────────────────────────

def _tab_log():
    st.subheader("📋 Trade Log")

    df_all = load_trade_log()
    settings = load_trade_settings()

    if "trade_type" not in df_all.columns:
        df_all["trade_type"] = "real"

    real_df  = df_all[df_all["trade_type"] == "real"]
    paper_df = df_all[df_all["trade_type"] == "paper"]

    # ── Stats strip ─────────────────────────────────────────────
    col_eval, col_real, col_paper = st.columns([1, 2, 2])
    with col_eval:
        if st.button("🔄 Evaluate Open Trades", type="primary", use_container_width=True):
            df_price = get_todays_nq_data()
            n = evaluate_open_trades(df_price)
            if n:
                st.success(f"Updated {n} trade(s).")
                st.rerun()
            else:
                st.info("No open trades resolved.")

    with col_real:
        st.markdown("**🎯 Real Trades**")
        rs = summary_stats(real_df)
        rc1, rc2, rc3, rc4 = st.columns(4)
        rc1.metric("Total",   rs["total"])
        rc2.metric("Win %",   f"{rs['win_rate']}%")
        rc3.metric("Total R", f"{rs['total_r']:+.1f}R")
        rc4.metric("Open",    len(real_df[real_df["outcome"] == "OPEN"]))

    with col_paper:
        st.markdown("**📄 Paper Trades** *(ML training)*")
        ps = summary_stats(paper_df)
        pc1, pc2, pc3, pc4 = st.columns(4)
        pc1.metric("Total",   ps["total"])
        pc2.metric("Win %",   f"{ps['win_rate']}%")
        pc3.metric("Total R", f"{ps['total_r']:+.1f}R")
        pc4.metric("Open",    len(paper_df[paper_df["outcome"] == "OPEN"]))

    if df_all.empty:
        st.info("No trades yet — opens automatically when a directional signal fires.")
        return

    # ── Filters ──────────────────────────────────────────────────
    st.markdown("---")
    f1, f2, f3, f4 = st.columns(4)
    type_filter    = f1.multiselect("Type",    ["real", "paper"], default=["real", "paper"])
    outcome_filter = f2.multiselect("Outcome", ["OPEN", "WIN", "LOSS", "PARTIAL"],
                                    default=["OPEN", "WIN", "LOSS", "PARTIAL"])
    dir_filter     = f3.multiselect("Direction", ["LONG", "SHORT"], default=["LONG", "SHORT"])
    date_range     = f4.date_input("Date range", value=[], help="Leave empty for all dates")

    df = df_all.copy()
    mask = (
        df["trade_type"].isin(type_filter) &
        df["outcome"].isin(outcome_filter) &
        df["direction"].isin(dir_filter)
    )
    if len(date_range) == 2:
        df["_date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
        mask = mask & df["_date"].between(date_range[0], date_range[1])

    view = df[mask].sort_values("timestamp", ascending=False).copy()

    # Display columns
    display_cols = ["trade_type", "id", "date", "time_il", "direction", "entry", "stop",
                    "target", "rr", "confidence", "nearest_zone_score", "htf_bias",
                    "session_name", "outcome", "actual_rr", "notes"]
    present = [c for c in display_cols if c in view.columns]
    view_show = view[present].copy()
    view_show["trade_type"] = view_show["trade_type"].map(lambda t: "🎯 Real" if t == "real" else "📄 Paper")
    view_show["direction"]  = view_show["direction"].map(lambda d: _dir_icon(d))
    view_show["outcome"]    = view_show["outcome"].map(lambda o: f"{_outcome_icon(o)} {o}")
    st.dataframe(view_show, use_container_width=True, hide_index=True)

    # ── Manual outcome entry (real trades only) ───────────────
    st.markdown("### ✏️ Update Trade Outcome")
    open_trades = df_all[(df_all["outcome"] == "OPEN") & (df_all.get("trade_type", "real") == "real")]
    if open_trades.empty:
        st.info("No open trades to update.")
        return

    with st.form("manual_update"):
        trade_opts = {
            f"{row['id']} | {row.get('direction','?')} @ {row.get('entry','?')} [{row.get('time_il','?')}]": row["id"]
            for _, row in open_trades.iterrows()
        }
        selected_label = st.selectbox("Select Trade", list(trade_opts.keys()))
        selected_id    = trade_opts[selected_label]

        c1, c2, c3 = st.columns(3)
        outcome_choice = c1.selectbox("Outcome", ["WIN", "LOSS", "PARTIAL"])
        exit_price     = c2.number_input("Exit Price (optional)", value=0.0, step=1.0)
        notes          = c3.text_input("Notes", placeholder="reason, slippage, etc.")

        if st.form_submit_button("💾 Save"):
            ep = exit_price if exit_price != 0.0 else None
            if update_trade_manual(selected_id, outcome_choice, ep, notes):
                st.success(f"Trade {selected_id} → {outcome_choice}")
                st.rerun()
            else:
                st.error("Trade not found.")


# ─── Tab 2: Factor Analysis ───────────────────────────────────────────────────

def _tab_factors():
    st.subheader("📊 Factor Win Rate Analysis")
    df_all = load_trade_log()

    # Use all trade types for factor analysis — paper trades provide ML training data
    closed = df_all[df_all["outcome"].isin(["WIN", "LOSS", "PARTIAL"])]
    paper_closed = closed[closed.get("trade_type", pd.Series(dtype=str)) == "paper"] if "trade_type" in closed.columns else pd.DataFrame()
    real_closed  = closed[closed.get("trade_type", pd.Series(dtype=str)) == "real"]  if "trade_type" in closed.columns else closed

    st.caption(
        f"📊 {len(real_closed)} real completed trades · "
        f"📄 {len(paper_closed)} paper completed trades · "
        f"Total for analysis: **{len(closed)}**"
    )

    if len(closed) < 3:
        st.info(f"Need at least 3 completed trades for analysis. Currently: {len(closed)}.")
        st.caption("Paper trades accumulate automatically when a directional signal fires — they will appear here once evaluated.")
        return
    df = df_all

    wr = factor_win_rates(df)
    if wr.empty:
        st.info("Not enough data per factor yet.")
        return

    # Win rate bar chart
    fig = px.bar(
        wr, x="Win %", y="Factor", color="Value",
        orientation="h", text="Win %",
        color_discrete_sequence=px.colors.qualitative.Set2,
        hover_data=["Trades", "Wins", "Avg R:R"],
    )
    fig.update_layout(
        height=max(300, len(wr) * 28),
        template="plotly_dark",
        margin=dict(l=0, r=20, t=30, b=0),
        legend=dict(orientation="h", y=-0.15),
        xaxis=dict(range=[0, 100], title="Win %"),
        yaxis=dict(title=""),
    )
    fig.add_vline(x=50, line_dash="dash", line_color="#aaa", opacity=0.5)
    st.plotly_chart(fig, use_container_width=True)

    # Table
    st.dataframe(wr, use_container_width=True, hide_index=True)

    # Factor correlation heatmap (needs enough data)
    if len(closed) >= 10:
        st.markdown("### 🔥 Score Correlation with Outcome")
        num_cols = [c for c in ["f_htf", "f_zone", "f_delta", "f_session",
                                "f_stacked", "f_sentiment", "f_ml",
                                "confidence", "nearest_zone_score"] if c in closed.columns]
        if num_cols:
            corr_df = closed[num_cols + ["win"]].rename(
                columns={"win": "WIN?"}
            ).dropna()
            if len(corr_df) >= 5:
                corr = corr_df.corr()[["WIN?"]].drop("WIN?").sort_values("WIN?", ascending=False)
                fig2 = px.bar(
                    corr.reset_index(), x="WIN?", y="index",
                    orientation="h", color="WIN?",
                    color_continuous_scale=["#d50000", "#9e9e9e", "#00c853"],
                    range_color=[-1, 1],
                    labels={"WIN?": "Correlation with WIN", "index": "Factor"},
                )
                fig2.update_layout(
                    height=280, template="plotly_dark",
                    margin=dict(l=0, r=20, t=10, b=0),
                )
                st.plotly_chart(fig2, use_container_width=True)

    # Insight summary
    if not wr.empty:
        st.markdown("### 💡 Key Insights")
        best  = wr.iloc[0]
        worst = wr.iloc[-1]
        st.success(f"**Best predictor:** {best['Factor']} = {best['Value']} → {best['Win %']}% win rate ({best['Trades']} trades)")
        if worst["Win %"] < 45:
            st.warning(f"**Weak predictor:** {worst['Factor']} = {worst['Value']} → {worst['Win %']}% win rate — consider tightening entry conditions when this is present")


# ─── Tab 3: Equity Curve ──────────────────────────────────────────────────────

def _tab_equity():
    st.subheader("📈 Equity Curve (Cumulative R)")
    settings  = load_trade_settings()
    risk_per  = settings.get("risk_per_trade", 1.0)
    df        = load_trade_log()
    eq        = equity_curve(df, risk_per_trade=risk_per)

    if eq.empty:
        st.info("No completed trades yet. Equity curve will appear after first WIN or LOSS.")
        return

    stats = summary_stats(df)

    # KPI row
    c1, c2, c3, c4 = st.columns(4)
    total_r = float(eq["cumR"].iloc[-1])
    max_dd  = float((eq["cumR"] - eq["cumR"].cummax()).min())
    c1.metric("Net P&L",    f"{total_r:+.1f}R",
              delta=f"${total_r * risk_per:+.0f}" if risk_per != 1.0 else None)
    c2.metric("Win Rate",   f"{stats['win_rate']}%")
    c3.metric("Avg R:R",    f"{stats['avg_rr_wins']}" if stats['avg_rr_wins'] else "—")
    c4.metric("Max Drawdown", f"{max_dd:.1f}R", delta_color="inverse")

    # Equity curve
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=eq["date"] + " " + eq["time_il"].astype(str),
        y=eq["cumR"],
        mode="lines+markers",
        line=dict(color="#00c853" if total_r >= 0 else "#d50000", width=2),
        marker=dict(
            color=eq["r"].apply(lambda r: "#00c853" if r > 0 else ("#ffd600" if r > -0.5 else "#d50000")),
            size=8,
        ),
        name="Cumulative R",
        hovertemplate="%{x}<br>Cumulative R: <b>%{y:.2f}</b><extra></extra>",
    ))
    fig.add_hline(y=0, line_color="#aaa", line_dash="dash", opacity=0.4)
    fig.update_layout(
        height=350, template="plotly_dark",
        margin=dict(l=0, r=20, t=20, b=0),
        yaxis=dict(title="Cumulative R"),
        xaxis=dict(title=""),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Per-trade R waterfall
    st.markdown("### Per-Trade P&L")
    colors = eq["r"].apply(lambda r: "#00c853" if r > 0 else ("#ffd600" if r >= 0 else "#d50000"))
    fig2 = go.Figure(go.Bar(
        x=list(range(1, len(eq) + 1)),
        y=eq["r"],
        marker_color=colors,
        hovertemplate="Trade #%{x}<br>R: <b>%{y:.2f}</b><extra></extra>",
    ))
    fig2.update_layout(
        height=200, template="plotly_dark",
        margin=dict(l=0, r=0, t=10, b=0),
        xaxis_title="Trade #", yaxis_title="R",
    )
    st.plotly_chart(fig2, use_container_width=True)

    # Trade history table
    with st.expander("Trade History", expanded=False):
        disp = eq[["date", "time_il", "direction", "r", "cumR"]].copy()
        disp["direction"] = disp["direction"].map(lambda d: _dir_icon(d))
        disp["r"]    = disp["r"].map(lambda v: f"{v:+.2f}R")
        disp["cumR"] = disp["cumR"].map(lambda v: f"{v:+.2f}R")
        st.dataframe(disp, use_container_width=True, hide_index=True)


# ─── Main ─────────────────────────────────────────────────────────────────────

def app():
    st.title("📓 Trade Journal")

    t1, t2, t3 = st.tabs(["📋 Trade Log", "📊 Factor Analysis", "📈 Equity Curve"])
    with t1:
        _tab_log()
    with t2:
        _tab_factors()
    with t3:
        _tab_equity()

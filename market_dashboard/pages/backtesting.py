"""
Backtesting Page — UI for running historical signal replay and parameter sweeps.
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import pytz
import streamlit as st

_MD   = Path(__file__).resolve().parents[1]
_ROOT = _MD.parent
for p in (str(_MD), str(_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from modules.nq_data import load_nq_daily_cache
from modules.backtest_engine import (
    run_backtest, sweep_parameter, results_to_records, DEFAULT_PARAMS,
)
from modules.backtest_metrics import compare_params, equity_curve_df

IL_TZ = pytz.timezone("Asia/Jerusalem")


# ─── helpers ──────────────────────────────────────────────────────────────────

def _metric_color(val: float, higher_better: bool = True) -> str:
    if val > 0:
        return "normal" if higher_better else "inverse"
    return "inverse" if higher_better else "normal"


def _render_kpi_row(m: dict) -> None:
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Total Trades", m["total_trades"])
    c2.metric("Win Rate",     f"{m['win_rate']}%")
    c3.metric("Profit Factor",m["profit_factor"])
    c4.metric("Total R",      f"{m['total_r']:+.1f}R")
    c5.metric("Max DD",       f"{m['max_drawdown_r']:.1f}R")
    c6.metric("Sharpe",       m["sharpe"])


def _render_equity_curve(trades: list) -> None:
    eq = equity_curve_df(trades)
    if eq.empty:
        st.info("No trades to chart.")
        return
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=eq["date"], y=eq["cumulative_r"],
        mode="lines+markers",
        name="Equity (R)",
        line=dict(color="#00c851", width=2),
        fill="tozeroy",
        fillcolor="rgba(0,200,81,0.08)",
    ))
    fig.add_hline(y=0, line_dash="dash", line_color="#888")
    fig.update_layout(
        title="Equity Curve (Cumulative R)", height=320,
        xaxis_title="Date", yaxis_title="R",
        plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
        font_color="#ccc",
        margin=dict(l=10, r=10, t=40, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_distribution(trades: list) -> None:
    if not trades:
        return
    rvals = [t.actual_rr for t in trades]
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=rvals, nbinsx=30,
        marker_color=[
            "#00c851" if r > 0 else "#ff4444" for r in rvals
        ],
        name="R Distribution",
    ))
    fig.add_vline(x=0, line_dash="dash", line_color="#fff")
    fig.update_layout(
        title="R Distribution", height=260,
        xaxis_title="Actual R", yaxis_title="Count",
        plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
        font_color="#ccc", showlegend=False,
        margin=dict(l=10, r=10, t=40, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_dow_chart(dow_wr: dict) -> None:
    days = list(dow_wr.keys())
    vals = [dow_wr[d] or 0 for d in days]
    colors = ["#00c851" if v >= 55 else "#ff4444" if v < 45 else "#f0a500" for v in vals]
    fig = go.Figure(go.Bar(x=days, y=vals, marker_color=colors, text=[f"{v:.0f}%" for v in vals], textposition="outside"))
    fig.add_hline(y=50, line_dash="dot", line_color="#888")
    fig.update_layout(
        title="Win Rate by Day of Week", height=220, yaxis_range=[0, 100],
        plot_bgcolor="#0e1117", paper_bgcolor="#0e1117", font_color="#ccc",
        margin=dict(l=10, r=10, t=40, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)


# ─── tabs ──────────────────────────────────────────────────────────────────────

def _tab_run_backtest() -> None:
    st.subheader("Run Backtest")

    st.caption(
        "⚠️ **Limitations:** Daily OHLCV only (no tick/5m data), entry at next-day open, "
        "stop/target checked at daily High/Low. Useful for parameter sensitivity — not a tick-accurate P&L simulator."
    )

    df_daily = load_nq_daily_cache()
    if df_daily.empty or len(df_daily) < 60:
        st.error("Insufficient daily data — load NQ data first (NQ Analysis page).")
        return

    min_date = pd.Timestamp(df_daily.index.min()).date() if hasattr(df_daily.index, "min") else datetime(2023, 1, 1).date()
    max_date = pd.Timestamp(df_daily.index.max()).date() if hasattr(df_daily.index, "max") else datetime.now().date()

    # ── Parameter controls ───────────────────────────────────────────────────
    with st.expander("⚙️ Signal Parameters", expanded=True):
        col1, col2, col3 = st.columns(3)
        p_conf  = col1.slider("Min Confidence",  55, 85, 60, key="bt_conf")
        p_zone  = col2.slider("Min Zone Score",   4,  9,  5, key="bt_zone")
        p_rr    = col3.slider("Min R:R",        1.2, 3.0, 1.5, step=0.1, key="bt_rr")
        col4, col5 = st.columns(2)
        p_margin = col4.slider("Min Margin (pts)", 10, 25, 15, key="bt_margin")
        p_atr    = col5.slider("ATR Multiplier",  1.0, 2.5, 1.5, step=0.1, key="bt_atr")
        p_bars   = st.slider("Max Bars Held (days)", 1, 10, 3, key="bt_bars")

    col_a, col_b = st.columns(2)
    start_d = col_a.date_input("Start Date", value=max(min_date, (datetime.now() - timedelta(days=365)).date()), min_value=min_date, max_value=max_date)
    end_d   = col_b.date_input("End Date",   value=max_date, min_value=min_date, max_value=max_date)

    if st.button("▶ Run Backtest", type="primary", key="btn_run_bt"):
        params = {
            "signal.min_confidence": p_conf,
            "signal.min_zone_score": p_zone,
            "signal.min_rr":         p_rr,
            "signal.min_margin":     p_margin,
            "risk.atr_mult":         p_atr,
        }
        with st.spinner(f"Running backtest {start_d} → {end_d}…"):
            result = run_backtest(
                df_daily,
                params=params,
                start_date=str(start_d),
                end_date=str(end_d),
                max_bars_held=p_bars,
            )

        st.session_state["bt_result"] = result
        st.session_state["bt_params"] = params

    if "bt_result" not in st.session_state:
        return

    result = st.session_state["bt_result"]
    m = result.metrics

    if m.get("error"):
        st.error(f"Backtest error: {m['error']}")
        return

    st.divider()

    # ── KPIs ──────────────────────────────────────────────────────────────────
    _render_kpi_row(m)

    col1, col2 = st.columns(2)
    col1.metric("Wins / Losses / Partials",
                f"{m['wins']} / {m['losses']} / {m['partials']}")
    col2.metric("Avg R per Trade", f"{m['avg_r']:+.2f}R")

    st.divider()

    # ── Charts ────────────────────────────────────────────────────────────────
    lc, rc = st.columns([2, 1])
    with lc:
        _render_equity_curve(result.trades)
    with rc:
        _render_distribution(result.trades)

    _render_dow_chart(m.get("dow_win_rates", {}))

    # ── Trade table ──────────────────────────────────────────────────────────
    with st.expander("📋 All Trades"):
        rows = results_to_records(result)
        if rows:
            df = pd.DataFrame(rows)
            def color_outcome(val):
                c = {"WIN": "#00c851", "LOSS": "#ff4444", "PARTIAL": "#f0a500"}.get(val, "")
                return f"color: {c}; font-weight: bold" if c else ""
            st.dataframe(
                df.style.map(color_outcome, subset=["outcome"]),
                use_container_width=True, hide_index=True,
            )
        else:
            st.info("No trades fired with current parameters.")

    # ── Score win-rate breakdown ─────────────────────────────────────────────
    swr = m.get("score_win_rates", {})
    if any(v is not None for v in swr.values()):
        st.markdown("**Win Rate by Signal Score**")
        sc1, sc2 = st.columns(2)
        sc1.metric("Score < 70", f"{swr.get('score<70','?')}%")
        sc2.metric("Score ≥ 70", f"{swr.get('score>=70','?')}%")


def _tab_parameter_sweep() -> None:
    st.subheader("Parameter Sweep")
    st.caption("Test multiple values of a single parameter and compare results.")

    df_daily = load_nq_daily_cache()
    if df_daily.empty or len(df_daily) < 60:
        st.error("Insufficient daily data.")
        return

    col1, col2 = st.columns(2)
    sweep_param = col1.selectbox(
        "Parameter to sweep",
        options=[
            "signal.min_confidence", "signal.min_zone_score",
            "signal.min_rr", "signal.min_margin", "risk.atr_mult",
        ],
    )
    lookback = col2.slider("Lookback (days)", 60, 730, 180, step=30, key="sweep_lookback")

    # Value range based on parameter
    range_map = {
        "signal.min_confidence": (55, 85, 5),
        "signal.min_zone_score": (4, 9, 1),
        "signal.min_rr":         (1.2, 3.0, 0.3),
        "signal.min_margin":     (10, 25, 5),
        "risk.atr_mult":         (1.0, 2.5, 0.25),
    }
    lo, hi, step = range_map.get(sweep_param, (1, 10, 1))
    test_vals = list(round(lo + i * step, 2) for i in range(int((hi - lo) / step) + 1))

    base_params = DEFAULT_PARAMS.copy()

    if st.button("▶ Run Sweep", type="primary", key="btn_sweep"):
        end_date   = df_daily.index.max() if hasattr(df_daily.index, "max") else None
        start_date = pd.Timestamp(end_date) - pd.Timedelta(days=lookback) if end_date is not None else None

        df_window = df_daily.copy()
        if start_date is not None:
            df_window = df_window[df_window.index >= start_date]

        with st.spinner(f"Testing {len(test_vals)} values of {sweep_param}…"):
            sweep_results = sweep_parameter(df_window, sweep_param, test_vals, base_params)

        st.session_state["sweep_results"] = sweep_results
        st.session_state["sweep_param"]   = sweep_param

    if "sweep_results" not in st.session_state:
        return

    rows = st.session_state["sweep_results"]
    param_label = st.session_state["sweep_param"]

    if not rows:
        st.info("No results.")
        return

    df = pd.DataFrame(rows)
    display_cols = ["value", "total_trades", "win_rate", "profit_factor",
                    "total_r", "max_drawdown_r", "sharpe", "expectancy"]
    df_display = df[[c for c in display_cols if c in df.columns]].copy()
    df_display.columns = [c.replace("_", " ").title() for c in df_display.columns]

    st.markdown(f"**Results for `{param_label}` — sorted by Profit Factor**")

    def highlight_best(s):
        styles = [""] * len(s)
        best_idx = s.idxmax() if s.dtype in (float, int) else None
        if best_idx is not None:
            styles[best_idx] = "background-color: #1a3a1a; font-weight: bold"
        return styles

    numeric_cols = df_display.select_dtypes(include="number").columns.tolist()
    st.dataframe(
        df_display.style.apply(highlight_best, subset=["Profit Factor"] if "Profit Factor" in df_display.columns else []),
        use_container_width=True, hide_index=True,
    )

    # Best value chart
    if "Value" in df_display.columns and "Profit Factor" in df_display.columns:
        fig = go.Figure(go.Bar(
            x=df_display["Value"].tolist(),
            y=df_display["Profit Factor"].tolist(),
            marker_color=[
                "#00c851" if v >= 1.5 else "#f0a500" if v >= 1.0 else "#ff4444"
                for v in df_display["Profit Factor"].tolist()
            ],
        ))
        fig.add_hline(y=1.0, line_dash="dot", line_color="#888", annotation_text="Breakeven")
        fig.update_layout(
            title=f"Profit Factor by {param_label}",
            xaxis_title=param_label, yaxis_title="Profit Factor",
            height=280, plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
            font_color="#ccc", margin=dict(l=10, r=10, t=40, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)

    best = rows[0]
    st.success(
        f"Best value: **{best['value']}** — "
        f"Win Rate {best.get('win_rate','?')}% | "
        f"Profit Factor {best.get('profit_factor','?')} | "
        f"Total R {best.get('total_r','?'):+.1f}R"
    )


def _tab_compare() -> None:
    st.subheader("Compare Two Parameter Sets")
    st.caption("Run baseline vs. proposed parameters side-by-side.")

    df_daily = load_nq_daily_cache()
    if df_daily.empty:
        st.error("No daily data.")
        return

    lookback = st.slider("Lookback (days)", 60, 730, 180, step=30, key="cmp_lookback")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Baseline (Current)**")
        a_conf  = st.slider("Min Confidence",  55, 85, 60, key="a_conf")
        a_zone  = st.slider("Min Zone Score",   4,  9,  5, key="a_zone")
        a_rr    = st.slider("Min R:R",        1.2, 3.0, 1.5, step=0.1, key="a_rr")
        a_atr   = st.slider("ATR Mult",       1.0, 2.5, 1.5, step=0.1, key="a_atr")

    with col2:
        st.markdown("**Proposed (Test)**")
        b_conf  = st.slider("Min Confidence",  55, 85, 65, key="b_conf")
        b_zone  = st.slider("Min Zone Score",   4,  9,  6, key="b_zone")
        b_rr    = st.slider("Min R:R",        1.2, 3.0, 1.8, step=0.1, key="b_rr")
        b_atr   = st.slider("ATR Mult",       1.0, 2.5, 1.5, step=0.1, key="b_atr")

    if st.button("▶ Compare", type="primary", key="btn_compare"):
        params_a = {"signal.min_confidence": a_conf, "signal.min_zone_score": a_zone,
                    "signal.min_rr": a_rr, "risk.atr_mult": a_atr, "signal.min_margin": 15}
        params_b = {"signal.min_confidence": b_conf, "signal.min_zone_score": b_zone,
                    "signal.min_rr": b_rr, "risk.atr_mult": b_atr, "signal.min_margin": 15}

        end_date   = df_daily.index.max()
        start_date = pd.Timestamp(end_date) - pd.Timedelta(days=lookback)
        df_window  = df_daily[df_daily.index >= start_date].copy()

        with st.spinner("Running both backtests…"):
            result_a = run_backtest(df_window, params=params_a)
            result_b = run_backtest(df_window, params=params_b)

        st.divider()

        # Side-by-side KPIs
        st.markdown("**📊 Results**")
        kc1, kc2 = st.columns(2)
        with kc1:
            st.markdown("**Baseline**")
            _render_kpi_row(result_a.metrics)
        with kc2:
            st.markdown("**Proposed**")
            _render_kpi_row(result_b.metrics)

        st.divider()

        # Comparison table
        cmp_df = compare_params(result_a.metrics, result_b.metrics)
        st.dataframe(cmp_df, use_container_width=True, hide_index=True)

        # Equity curves overlaid
        eq_a = equity_curve_df(result_a.trades)
        eq_b = equity_curve_df(result_b.trades)
        if not eq_a.empty or not eq_b.empty:
            fig = go.Figure()
            if not eq_a.empty:
                fig.add_trace(go.Scatter(x=eq_a["date"], y=eq_a["cumulative_r"],
                              name="Baseline", line=dict(color="#4d9de0", width=2)))
            if not eq_b.empty:
                fig.add_trace(go.Scatter(x=eq_b["date"], y=eq_b["cumulative_r"],
                              name="Proposed", line=dict(color="#00c851", width=2)))
            fig.add_hline(y=0, line_dash="dash", line_color="#888")
            fig.update_layout(
                title="Equity Curves — Baseline vs Proposed", height=320,
                plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
                font_color="#ccc", margin=dict(l=10, r=10, t=40, b=10),
            )
            st.plotly_chart(fig, use_container_width=True)

        # Verdict
        pf_a = result_a.metrics.get("profit_factor", 0)
        pf_b = result_b.metrics.get("profit_factor", 0)
        if pf_b > pf_a * 1.05:
            st.success(f"✅ Proposed is better (PF {pf_b:.2f} vs {pf_a:.2f}). Consider applying.")
        elif pf_b < pf_a * 0.95:
            st.error(f"❌ Proposed is worse (PF {pf_b:.2f} vs {pf_a:.2f}). Keep current params.")
        else:
            st.warning(f"⚠️ Results are similar (PF {pf_b:.2f} vs {pf_a:.2f}). Not enough evidence to change.")


# ─── main ─────────────────────────────────────────────────────────────────────

def app() -> None:
    st.title("📉 Backtesting")

    tab1, tab2, tab3 = st.tabs([
        "▶ Run Backtest",
        "🔍 Parameter Sweep",
        "⚖️ Compare Params",
    ])

    with tab1:
        _tab_run_backtest()
    with tab2:
        _tab_parameter_sweep()
    with tab3:
        _tab_compare()

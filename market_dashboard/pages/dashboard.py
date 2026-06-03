import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
from pathlib import Path
import subprocess
import joblib
from datetime import datetime, date

matplotlib.use('Agg')

_ROOT = Path(__file__).resolve().parents[2]
_SCORE_PATH  = _ROOT / "scores_news" / "config" / "score_log.csv"
_MERGED_PATH = _ROOT / "scores_news" / "ml_model" / "merged_scores_mes.csv"
_MODEL_PATH  = _ROOT / "scores_news" / "ml_model" / "model.pkl"
_PERF_PATH   = _ROOT / "scores_news" / "ml_model" / "ml_performance_log.csv"
_FINAL_SCORE = _ROOT / "scores_news" / "cat_scores" / "final_score.py"
_MES_UPDATE  = _ROOT / "scores_news" / "data_sources" / "mes_prices_history_table.py"
_RSS_FETCH   = _ROOT / "scores_news" / "data_sources" / "fetch_news_rss.py"

_SCORE_COLS = [
    "sentiment_score", "macro_score", "bonds_score",
    "futures_vix_score", "mes_score", "sectors_score",
]
_SCORE_LABELS = {
    "sentiment_score":   "Sentiment",
    "macro_score":       "Macro",
    "bonds_score":       "Bonds",
    "futures_vix_score": "Futures/VIX",
    "mes_score":         "MES",
    "sectors_score":     "Sectors",
}


@st.cache_data(ttl=300)
def _load_score_log() -> pd.DataFrame:
    df = pd.read_csv(_SCORE_PATH)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)


@st.cache_data(ttl=300)
def _load_ml_prediction() -> str | None:
    try:
        df = pd.read_csv(_MERGED_PATH)
        df["date"] = pd.to_datetime(df["date"])
        latest = df.sort_values("date").iloc[-1]
        model = joblib.load(_MODEL_PATH)
        X = pd.DataFrame(
            [[latest[c] for c in ["sentiment_score", "macro_score", "bonds_score",
                                  "futures_vix_score", "sectors_score", "mes_score"]]],
            columns=["sentiment", "macro", "bonds", "futures_vix", "sectors", "mes"],
        )
        pred = model.predict(X)[0]
        return {1: "LONG", -1: "SHORT", 0: "NEUTRAL"}.get(pred, "NEUTRAL")
    except Exception:
        return None


@st.cache_data(ttl=300)
def _load_model_accuracy() -> tuple[float | None, float | None, int]:
    try:
        df = pd.read_csv(_PERF_PATH)
        valid = df[df["correct"].notna() & (df["correct"] != "")]
        if valid.empty:
            return None, None, 0
        all_acc    = (valid["correct"] == "✅").mean() * 100
        recent_acc = (valid.tail(10)["correct"] == "✅").mean() * 100
        return round(all_acc, 1), round(recent_acc, 1), len(valid)
    except Exception:
        return None, None, 0


def _make_env() -> dict:
    import os
    env = os.environ.copy()
    env["PYTHONPATH"] = str(_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    env["PYTHONIOENCODING"] = "utf-8"
    return env


def _run_script(script: Path, env: dict) -> tuple[bool, str]:
    try:
        r = subprocess.run(
            ["python", str(script)],
            capture_output=True, text=True, timeout=120,
            cwd=str(_ROOT), env=env,
            encoding="utf-8", errors="replace",
        )
        out = (r.stdout or "") + (r.stderr or "")
        return r.returncode == 0, out.strip()[:800]
    except subprocess.TimeoutExpired:
        return False, "timeout (>2 min)"
    except Exception as e:
        return False, str(e)


def _run_daily_pipeline() -> list[dict]:
    env = _make_env()
    steps = [
        {"name": "📈 Update MES_data.csv",   "script": _MES_UPDATE,  "required": True},
        {"name": "📰 Fetch RSS sentiment",    "script": _RSS_FETCH,   "required": False},
        {"name": "🧮 Calculate final_score",  "script": _FINAL_SCORE, "required": True},
    ]
    results = []
    for step in steps:
        ok, out = _run_script(step["script"], env)
        results.append({"name": step["name"], "ok": ok,
                        "required": step["required"], "output": out})
        if not ok and step["required"]:
            break
    st.cache_data.clear()
    return results


def _show_run_button(label: str = "▶️ Run Daily Analysis", key: str = "btn_run_pipeline"):
    if st.button(label, type="primary", key=key):
        with st.spinner("Running pipeline… (up to 5 min)"):
            results = _run_daily_pipeline()
        all_ok = True
        for r in results:
            icon = "✅" if r["ok"] else ("⚠️" if not r["required"] else "❌")
            st.markdown(f"{icon} **{r['name']}**")
            if r["output"]:
                with st.expander("Output", expanded=not r["ok"]):
                    st.code(r["output"])
            if not r["ok"] and r["required"]:
                all_ok = False
        if all_ok:
            st.success("✅ Analysis complete — data updated")
            st.rerun()
        else:
            st.error("❌ Critical step failed — check output above")


def _bias_banner(bias: str, score: int, last_date: date) -> None:
    clean = bias.replace("✅", "").replace("❌", "").replace("🔒", "").strip()
    if "LONG" in clean:
        bg, border, color, icon = "#091f10", "#00c853", "#00c853", "✅"
        score_interp = "Bullish environment — conditions favor upside"
    elif "SHORT" in clean:
        bg, border, color, icon = "#1f0909", "#d50000", "#ef5350", "❌"
        score_interp = "Bearish environment — conditions favor downside"
    else:
        bg, border, color, icon = "#111827", "#546e7a", "#90a4ae", "🔒"
        score_interp = "Mixed signals — no clear directional edge"

    # Score interpretation
    if score >= 70:
        strength = "Strong"
    elif score >= 60:
        strength = "Moderate"
    elif score >= 40:
        strength = "Weak"
    else:
        strength = "Strong"  # strong short

    st.markdown(
        f'<div style="background:{bg};border:2px solid {border};border-radius:12px;'
        f'padding:20px 28px;margin-bottom:20px;">'
        f'<div style="display:flex;align-items:center;gap:20px;">'
        f'<div style="font-size:3em;line-height:1;">{icon}</div>'
        f'<div style="flex:1;">'
        f'<div style="color:{color};font-size:1.7em;font-weight:800;letter-spacing:.04em;">'
        f'{strength} {clean}</div>'
        f'<div style="color:#aaa;font-size:.85em;margin-top:4px;">{score_interp}</div>'
        f'</div>'
        f'<div style="text-align:right;">'
        f'<div style="color:#fff;font-size:2em;font-weight:700;">{score}'
        f'<span style="font-size:.45em;color:#666;">/100</span></div>'
        f'<div style="color:#666;font-size:.78em;">{last_date.strftime("%A %d %b %Y")}</div>'
        f'</div>'
        f'</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def _score_card_html(label: str, score: float, delta: float | None) -> str:
    if score >= 60:
        bar_color = "#00c853"
    elif score >= 40:
        bar_color = "#ff9800"
    else:
        bar_color = "#ef5350"

    pct = min(100, int(score))

    if delta is None:
        delta_html = ""
    elif delta > 0:
        delta_html = f'<div style="color:#00c853;font-size:.72em;margin-top:3px;">▲ +{delta:.0f}</div>'
    elif delta < 0:
        delta_html = f'<div style="color:#ef5350;font-size:.72em;margin-top:3px;">▼ {delta:.0f}</div>'
    else:
        delta_html = '<div style="color:#555;font-size:.72em;margin-top:3px;">→ no change</div>'

    return (
        f'<div style="background:#111827;border:1px solid #1f2937;border-radius:10px;'
        f'padding:14px 12px;text-align:center;height:100%;">'
        f'<div style="color:#9ca3af;font-size:.78em;font-weight:500;margin-bottom:8px;">'
        f'{label.upper()}</div>'
        f'<div style="color:#f9fafb;font-size:1.6em;font-weight:700;line-height:1;">'
        f'{score:.0f}</div>'
        f'<div style="color:#4b5563;font-size:.65em;">/ 100</div>'
        f'{delta_html}'
        f'<div style="background:#1f2937;border-radius:4px;height:4px;margin-top:10px;overflow:hidden;">'
        f'<div style="background:{bar_color};width:{pct}%;height:100%;border-radius:4px;opacity:.85;"></div>'
        f'</div>'
        f'</div>'
    )


def _divergence_alert(bias: str, ml_pred: str | None) -> None:
    if not ml_pred:
        return
    bias_clean = bias.replace("✅", "").replace("❌", "").replace("🔒", "").strip()
    # Only alert when they disagree and neither is NEUTRAL
    if bias_clean == ml_pred:
        return
    if "NEUTRAL" in bias_clean or ml_pred == "NEUTRAL":
        return
    st.warning(
        f"⚠️ **Signal divergence** — Score model says **{bias_clean}** "
        f"but ML model says **{ml_pred}**. "
        "Treat both signals with caution when they conflict."
    )


def app():
    col_title, col_refresh = st.columns([7, 1])
    with col_title:
        st.title("📊 Market Sentiment Dashboard")
    with col_refresh:
        st.write("")
        if st.button("🔄", use_container_width=True, help="Refresh cache"):
            st.cache_data.clear()
            st.rerun()

    # ── Load score log ─────────────────────────────────────
    try:
        df = _load_score_log()
    except FileNotFoundError:
        st.error("❌ score_log.csv not found — run initial analysis first.")
        _show_run_button("▶️ Run Initial Analysis", key="btn_run_initial")
        return
    except Exception as e:
        st.error(f"❌ Data load error: {e}")
        return

    age = (date.today() - df["date"].max().date()).days
    last_day  = df.iloc[-1]
    prev_day  = df.iloc[-2] if len(df) >= 2 else None
    score_cols = [c for c in _SCORE_COLS if c in df.columns]

    # ── Stale data warning ─────────────────────────────────
    if age >= 1:
        st.warning(
            f"⚠️ Data is **{age} day(s) old** — last update: **{df['date'].max().date()}**"
        )
        _show_run_button(key="btn_run_stale")
    else:
        st.caption(f"Last updated: {df['date'].max().date()} · refreshed {datetime.now().strftime('%H:%M:%S')}")

    # ── Decision Banner ────────────────────────────────────
    final_score = int(last_day.get("final_score", 50) or 50)
    bias = str(last_day.get("bias", "NEUTRAL") or "NEUTRAL")
    _bias_banner(bias, final_score, df["date"].max().date())

    # ── ML prediction divergence alert ────────────────────
    ml_pred = _load_ml_prediction()
    _divergence_alert(bias, ml_pred)

    # ── ML card + MES price card ───────────────────────────
    col_ml, col_mes = st.columns(2)

    with col_ml:
        all_acc, recent_acc, n_days = _load_model_accuracy()
        if ml_pred:
            pred_color = "#00c853" if ml_pred == "LONG" else "#ef5350" if ml_pred == "SHORT" else "#90a4ae"
            if all_acc is not None:
                acc_html = (
                    f'<div style="color:#9ca3af;font-size:.8em;margin-top:8px;">'
                    f'Accuracy · last 10d: <b style="color:#f9fafb;">{recent_acc:.0f}%</b>'
                    f' &nbsp;·&nbsp; all-time: <b style="color:#f9fafb;">{all_acc:.0f}%</b>'
                    f' ({n_days}d)</div>'
                )
            else:
                acc_html = '<div style="color:#4b5563;font-size:.8em;margin-top:8px;">No accuracy data yet</div>'
            st.markdown(
                f'<div style="background:#111827;border:1px solid #1f2937;border-radius:10px;padding:16px;">'
                f'<div style="color:#9ca3af;font-size:.78em;font-weight:500;margin-bottom:6px;">🧠 ML MODEL</div>'
                f'<div style="color:{pred_color};font-size:1.4em;font-weight:700;">{ml_pred}</div>'
                f'{acc_html}'
                f'</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div style="background:#111827;border:1px solid #1f2937;border-radius:10px;padding:16px;">'
                '<div style="color:#9ca3af;font-size:.78em;margin-bottom:6px;">🧠 ML MODEL</div>'
                '<div style="color:#4b5563;">Not available</div>'
                '</div>',
                unsafe_allow_html=True,
            )

    with col_mes:
        mes_open  = last_day.get("open")
        mes_close = last_day.get("close")
        daily_chg = last_day.get("daily_change_pct")
        if mes_open and mes_close:
            chg_color = "#00c853" if (daily_chg or 0) >= 0 else "#ef5350"
            sign      = "+" if (daily_chg or 0) >= 0 else ""
            chg_val   = f"{sign}{float(daily_chg):.2f}%" if daily_chg is not None else "—"
            st.markdown(
                f'<div style="background:#111827;border:1px solid #1f2937;border-radius:10px;padding:16px;">'
                f'<div style="color:#9ca3af;font-size:.78em;font-weight:500;margin-bottom:6px;">📈 MES LAST CLOSE</div>'
                f'<div style="display:flex;align-items:baseline;gap:12px;">'
                f'<span style="color:#f9fafb;font-size:1.4em;font-weight:700;">{float(mes_close):.2f}</span>'
                f'<span style="color:{chg_color};font-size:.95em;font-weight:600;">{chg_val}</span>'
                f'</div>'
                f'<div style="color:#4b5563;font-size:.8em;margin-top:6px;">Open: {float(mes_open):.2f}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div style="background:#111827;border:1px solid #1f2937;border-radius:10px;padding:16px;">'
                '<div style="color:#9ca3af;font-size:.78em;margin-bottom:6px;">📈 MES LAST CLOSE</div>'
                '<div style="color:#4b5563;">Price data unavailable</div>'
                '</div>',
                unsafe_allow_html=True,
            )

    # ── Category Score Cards ───────────────────────────────
    st.markdown("<div style='margin-top:20px;'></div>", unsafe_allow_html=True)
    st.markdown("**📊 Category Scores**")
    if score_cols:
        cols = st.columns(len(score_cols))
        for i, col_name in enumerate(score_cols):
            score = float(last_day.get(col_name) or 50)
            delta = None
            if prev_day is not None:
                prev = float(prev_day.get(col_name) or 50)
                delta = score - prev
            label = _SCORE_LABELS.get(col_name, col_name.replace("_score", "").capitalize())
            with cols[i]:
                st.markdown(_score_card_html(label, score, delta), unsafe_allow_html=True)

    # ── Score Legend ───────────────────────────────────────
    st.markdown(
        '<div style="color:#4b5563;font-size:.75em;margin-top:8px;">'
        '🟢 60–100: bullish &nbsp;·&nbsp; 🟡 40–59: neutral &nbsp;·&nbsp; 🔴 0–39: bearish'
        '</div>',
        unsafe_allow_html=True,
    )

    # ── Trend Charts ───────────────────────────────────────
    if len(df) >= 2:
        st.markdown("---")
        st.markdown("**📈 Score Trends**")
        chart_df = df.tail(30).set_index("date")
        if score_cols:
            st.line_chart(chart_df[score_cols], height=220)

        plot_cols = [c for c in ["final_score", "daily_change_pct"] if c in chart_df.columns]
        if len(plot_cols) == 2:
            st.markdown("**🔁 Final Score vs MES Daily Change %**")
            st.line_chart(chart_df[plot_cols], height=200)

    # ── Heatmap ────────────────────────────────────────────
    if len(df) >= 3 and score_cols:
        st.markdown("---")
        st.markdown("**🌡 Score Heatmap — Last 15 Sessions**")
        heat_df = df.tail(15).set_index("date")[score_cols].copy()
        heat_df.index = pd.to_datetime(heat_df.index).strftime("%d %b")
        # Rename columns for display
        heat_df.columns = [_SCORE_LABELS.get(c, c) for c in heat_df.columns]

        n_rows = len(heat_df)
        fig, ax = plt.subplots(figsize=(10, max(3, n_rows * 0.45)))
        sns.heatmap(
            heat_df,
            cmap="RdYlGn",
            annot=True,
            fmt=".0f",
            linewidths=0.4,
            vmin=0, vmax=100,
            ax=ax,
            cbar_kws={"shrink": 0.7},
        )
        fig.patch.set_facecolor("#0e1117")
        ax.set_facecolor("#0e1117")
        ax.tick_params(colors="#9ca3af", labelsize=8)
        ax.set_ylabel("")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    # ── Pipeline & Model Management ────────────────────────
    with st.expander("⚙️ Pipeline & Model Management", expanded=False):
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**Run daily analysis manually**")
            _show_run_button(key="btn_run_manual")
        with col_b:
            st.markdown("**Check yesterday's prediction** (after 20:00 IL)")
            if st.button("📊 Compare prediction"):
                if datetime.now().hour >= 20:
                    try:
                        r = subprocess.run(
                            ["python",
                             str(_ROOT / "scores_news" / "ml_model" / "performance_tracker.py")],
                            capture_output=True, text=True,
                            encoding="utf-8", errors="replace",
                        )
                        st.success("✅ Done")
                        st.code((r.stdout or "") + (r.stderr or ""))
                    except Exception as e:
                        st.error(f"Error: {e}")
                else:
                    st.warning("⏳ Available after 20:00 IL time")

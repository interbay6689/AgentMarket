"""System page: Settings | Alerts | System Info — merged into tabs."""
import sys
from pathlib import Path

_MD = Path(__file__).resolve().parents[1]
_ROOT = _MD.parent
for p in (str(_MD), str(_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

import streamlit as st
import pandas as pd
from modules.config import load_config, save_config
from modules.alerts import load_alerts
from modules.trade_tracker import load_trade_settings, save_trade_settings


def _tab_settings():
    st.subheader("⚙️ Settings")
    cfg = load_config()
    th = cfg.get("thresholds", {})

    with st.form("settings_form"):
        st.markdown("**Thresholds**")
        c1, c2 = st.columns(2)
        inv_thresh = c1.number_input("Yield Inversion Thresh (%)", 0.0, 100.0,
                                      value=float(th.get("yield_inversion", 2.0)))
        vix_thresh = c2.number_input("VIX Spike Thresh (%)", 0.0, 100.0,
                                      value=float(th.get("vix_spike", 12.0)))

        st.markdown("**Category Weights**")
        w = cfg.get("weights", {})
        c1, c2, c3 = st.columns(3)
        w_sentiment = c1.number_input("Sentiment", 0.0, 1.0, value=float(w.get("sentiment", 0.20)), step=0.05)
        w_macro = c2.number_input("Macro", 0.0, 1.0, value=float(w.get("macro", 0.20)), step=0.05)
        w_bonds = c3.number_input("Bonds", 0.0, 1.0, value=float(w.get("bonds", 0.15)), step=0.05)
        c1, c2, c3 = st.columns(3)
        w_vix = c1.number_input("Futures/VIX", 0.0, 1.0, value=float(w.get("futures_vix", 0.15)), step=0.05)
        w_sectors = c2.number_input("Sectors", 0.0, 1.0, value=float(w.get("sectors", 0.15)), step=0.05)
        w_mes = c3.number_input("MES", 0.0, 1.0, value=float(w.get("mes", 0.15)), step=0.05)

        st.markdown("**RSS Sources**")
        rss = st.text_area("RSS Sources (one URL per line)", "\n".join(cfg.get("rss_list", [])))

        submitted = st.form_submit_button("💾 Save Settings")
        if submitted:
            total_w = w_sentiment + w_macro + w_bonds + w_vix + w_sectors + w_mes
            if abs(total_w - 1.0) > 0.01:
                st.warning(f"Weights sum to {total_w:.2f} — should equal 1.0")
            new_cfg = {
                "thresholds": {"yield_inversion": inv_thresh, "vix_spike": vix_thresh},
                "weights": {
                    "bonds": w_bonds, "macro": w_macro, "sentiment": w_sentiment,
                    "futures_vix": w_vix, "sectors": w_sectors, "mes": w_mes,
                },
                "rss_list": [u.strip() for u in rss.splitlines() if u.strip()],
            }
            save_config(new_cfg)
            st.success("Settings saved!")


def _tab_alerts():
    st.subheader("⚠️ Active Alerts")
    alerts_df = load_alerts()
    if alerts_df.empty:
        st.success("✅ No active alerts at this time.")
    else:
        st.dataframe(alerts_df, use_container_width=True)

    st.markdown("---")
    st.subheader("📋 Alerts Log")
    alerts_path = _ROOT / "market_dashboard" / "logs" / "alerts_log.txt"
    if alerts_path.exists():
        lines = alerts_path.read_text(encoding="utf-8").splitlines()
        st.code("\n".join(lines[-100:]), language=None)
    else:
        st.info("Log is empty.")


def _tab_system():
    st.subheader("🖥️ System Status")

    # Pipeline file status
    files = {
        "score_log.csv": _ROOT / "scores_news" / "config" / "score_log.csv",
        "merged_scores_mes.csv": _ROOT / "scores_news" / "ml_model" / "merged_scores_mes.csv",
        "model.pkl": _ROOT / "scores_news" / "ml_model" / "model.pkl",
        "ml_performance_log.csv": _ROOT / "scores_news" / "ml_model" / "ml_performance_log.csv",
        "weights_history.csv": _ROOT / "scores_news" / "ml_model" / "weights_history.csv",
        "NQ_data.csv": _ROOT / "scores_news" / "config" / "NQ_data.csv",
        "MES_data.csv": _ROOT / "scores_news" / "config" / "MES_data.csv",
    }
    rows = []
    for name, path in files.items():
        exists = path.exists()
        if exists:
            import os, datetime
            mtime = datetime.datetime.fromtimestamp(os.path.getmtime(path)).strftime("%Y-%m-%d %H:%M")
            size = f"{os.path.getsize(path) / 1024:.1f} KB"
        else:
            mtime = "—"
            size = "—"
        rows.append({"File": name, "Status": "✅" if exists else "❌", "Modified": mtime, "Size": size})
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # Error log
    st.markdown("**Error Log (last 50 lines)**")
    err_path = _ROOT / "market_dashboard" / "logs" / "error_log.txt"
    if err_path.exists():
        lines = err_path.read_text(encoding="utf-8").splitlines()
        st.code("\n".join(lines[-50:]), language=None)
    else:
        st.info("Error log is empty.")


def _tab_sentiment_feed():
    st.subheader("📰 Sentiment Feed")
    logs_dir = _ROOT / "scores_news" / "logs"
    if not logs_dir.exists():
        st.info("Logs directory not found.")
        return

    csv_files = sorted(logs_dir.glob("sentiment_raw_*.csv"), reverse=True)
    if not csv_files:
        csv_files = sorted(logs_dir.glob("*.csv"), reverse=True)

    if not csv_files:
        st.info("No sentiment_raw files yet.")
        return

    chosen = st.selectbox("Select file:", [f.name for f in csv_files])
    path = logs_dir / chosen
    try:
        df = pd.read_csv(path)
        df.columns = df.columns.str.strip()

        if "sentiment_score" in df.columns:
            min_score = st.slider("Filter by minimum score:", 0, 100, 0)
            df = df[df["sentiment_score"] >= min_score]

        search = st.text_input("Search titles:", "")
        if search and "title" in df.columns:
            df = df[df["title"].str.contains(search, case=False, na=False)]

        st.dataframe(df.head(200), use_container_width=True)
    except Exception as e:
        st.error(f"Error loading file: {e}")


def _tab_trade_tracking():
    st.subheader("📓 Trade Tracking Settings")
    ts = load_trade_settings()

    with st.form("trade_tracking_form"):
        st.markdown("**Logging**")
        c1, c2 = st.columns(2)
        auto_log      = c1.toggle("Auto-log alerts",    value=ts.get("auto_log", True),
                                   help="Automatically save each trade alert to trade_log.csv")
        auto_evaluate = c2.toggle("Auto-evaluate open trades", value=ts.get("auto_evaluate", True),
                                   help="Automatically resolve open trades against price when Journal page opens")

        st.markdown("**Entry Filters** — trade is logged only when ALL conditions met")
        c1, c2, c3 = st.columns(3)
        min_conf  = c1.number_input("Min Confidence (%)", 0, 100,
                                    value=int(ts.get("min_confidence", 60)), step=5)
        min_zone  = c2.number_input("Min Zone Score (0–10)", 0, 10,
                                    value=int(ts.get("min_zone_score", 5)), step=1)
        min_rr    = c3.number_input("Min R:R",  0.5, 5.0,
                                    value=float(ts.get("min_rr", 1.5)), step=0.1)

        st.markdown("**Deduplication & Evaluation**")
        c1, c2 = st.columns(2)
        dedup   = c1.number_input("Dedup window (min)", 5, 120,
                                   value=int(ts.get("dedup_minutes", 30)), step=5,
                                   help="Same direction won't be logged again within this window")
        eval_h  = c2.number_input("Evaluation interval (hours)", 1, 24,
                                   value=int(ts.get("eval_interval_h", 2)), step=1,
                                   help="How long to wait before checking if stop/target was hit")

        st.markdown("**Risk & Equity Curve**")
        risk_per = st.number_input("Risk per trade ($)", 10.0, 10_000.0,
                                    value=float(ts.get("risk_per_trade", 100.0)), step=10.0,
                                    help="Used only for dollar P&L display in Equity Curve tab")

        if st.form_submit_button("💾 Save Trade Tracking Settings"):
            new_settings = {
                "auto_log":        auto_log,
                "auto_evaluate":   auto_evaluate,
                "min_confidence":  min_conf,
                "min_zone_score":  min_zone,
                "min_rr":          min_rr,
                "dedup_minutes":   dedup,
                "eval_interval_h": eval_h,
                "risk_per_trade":  risk_per,
            }
            save_trade_settings(new_settings)
            st.success("Trade tracking settings saved!")

    # Clear log option
    st.markdown("---")
    st.markdown("**Danger Zone**")
    from modules.trade_tracker import TRADE_LOG_PATH
    if TRADE_LOG_PATH.exists():
        size = TRADE_LOG_PATH.stat().st_size / 1024
        st.caption(f"trade_log.csv — {size:.1f} KB")
    if st.button("🗑️ Clear Trade Log", type="secondary"):
        if st.session_state.get("_confirm_clear_log"):
            import pandas as pd
            from modules.trade_tracker import COLUMNS
            pd.DataFrame(columns=COLUMNS).to_csv(TRADE_LOG_PATH, index=False)
            st.session_state.pop("_confirm_clear_log", None)
            st.success("Trade log cleared.")
            st.rerun()
        else:
            st.session_state["_confirm_clear_log"] = True
            st.warning("Click again to confirm — this will delete all trade history.")


def app():
    st.title("⚙️ System")
    t1, t2, t3, t4, t5 = st.tabs(["⚙️ Settings", "⚠️ Alerts", "🖥️ System Info", "📰 Sentiment Feed", "📓 Trade Tracking"])
    with t1:
        _tab_settings()
    with t2:
        _tab_alerts()
    with t3:
        _tab_system()
    with t4:
        _tab_sentiment_feed()
    with t5:
        _tab_trade_tracking()

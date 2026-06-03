import streamlit as st
import pandas as pd
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]


def app():
    st.title("🧠 ML Model Learning")

    perf_log     = _ROOT / "scores_news" / "ml_model" / "ml_performance_log.csv"
    weights_hist = _ROOT / "scores_news" / "ml_model" / "weights_history.csv"

    # Daily prediction success
    st.subheader("✅ Daily Prediction Results")
    try:
        df_perf = pd.read_csv(perf_log)
        df_perf["date"]    = pd.to_datetime(df_perf["date"])
        df_perf["success"] = df_perf["correct"].apply(lambda x: 1 if x == "✅" else 0)
        st.bar_chart(df_perf.set_index("date")["success"])
    except FileNotFoundError:
        st.info("ml_performance_log.csv not yet created — generated automatically after running final_score.py for at least one day.")
    except Exception as e:
        st.error(f"Error loading ml_performance_log.csv: {e}")

    # Cumulative accuracy
    st.subheader("📈 Cumulative Accuracy")
    try:
        df_perf["cumulative_accuracy"] = df_perf["success"].expanding().mean() * 100
        st.line_chart(df_perf.set_index("date")["cumulative_accuracy"])
    except Exception:
        st.info("Not enough data to compute cumulative accuracy.")

    # Weight evolution
    st.subheader("⚖️ Category Weight Evolution")
    try:
        df_weights = pd.read_csv(weights_hist)
        df_weights["date"] = pd.to_datetime(df_weights["date"])
        df_weights = df_weights.set_index("date").sort_index()
        st.line_chart(df_weights)

        st.markdown("### 🔎 Latest Weights")
        last_row = df_weights.tail(1)
        st.dataframe(last_row.T.rename(columns={last_row.index[0]: "Weight"}))
    except FileNotFoundError:
        st.info("weights_history.csv not yet created — generated automatically after running final_score.py for at least one day.")
    except Exception as e:
        st.error(f"Error loading weights_history.csv: {e}")

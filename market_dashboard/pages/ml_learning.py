import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

st.set_page_config(page_title="🧠 תהליך הלמידה - ML", layout="wide")
st.title("📚 תהליך הלמידה של המודל")

_ROOT = Path(__file__).resolve().parents[2]
perf_log = _ROOT / "scores_news" / "ml_model" / "ml_performance_log.csv"
weights_hist = _ROOT / "scores_news" / "ml_model" / "weights_history.csv"

# === 1. גרף הצלחות יומיות ===
st.subheader("✅ הצלחות יומיות")
try:
    df_perf = pd.read_csv(perf_log)
    df_perf['date'] = pd.to_datetime(df_perf['date'])
    df_perf['success'] = df_perf['correct'].apply(lambda x: 1 if x == '✅' else 0)
    st.bar_chart(df_perf.set_index('date')['success'])
except Exception as e:
    st.error(f"שגיאה בטעינת ml_performance_log.csv: {e}")

# === 2. גרף דיוק מצטבר ===
st.subheader("📈 דיוק מצטבר")
try:
    df_perf['cumulative_accuracy'] = df_perf['success'].expanding().mean() * 100
    st.line_chart(df_perf.set_index('date')['cumulative_accuracy'])
except:
    st.warning("אין מספיק נתונים לחישוב דיוק מצטבר.")

# === 3. שינוי משקלים לאורך זמן ===
st.subheader("🧠 שינוי משקלים - weights")
try:
    df_weights = pd.read_csv(weights_hist)
    df_weights['date'] = pd.to_datetime(df_weights['date'])
    df_weights = df_weights.set_index('date').sort_index()
    st.line_chart(df_weights)

    st.markdown("### 🔎 טבלת משקלים אחרונה")
    st.dataframe(df_weights.tail(1).T.rename(columns={df_weights.tail(1).index[0]: "Weight"}))
except Exception as e:
    st.error(f"שגיאה בטעינת weights_history.csv: {e}")
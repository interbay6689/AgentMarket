import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import subprocess
import joblib
import matplotlib
from datetime import datetime

matplotlib.use('Agg')

_ROOT = Path(__file__).resolve().parents[2]
_SCORE_PATH  = _ROOT / "scores_news" / "config" / "score_log.csv"
_MERGED_PATH = _ROOT / "scores_news" / "ml_model" / "merged_scores_mes.csv"
_MODEL_PATH  = _ROOT / "scores_news" / "ml_model" / "model.pkl"
_PERF_PATH   = _ROOT / "scores_news" / "ml_model" / "ml_performance_log.csv"


@st.cache_data(ttl=300)
def _load_score_log():
    df = pd.read_csv(_SCORE_PATH)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df.dropna(subset=["date"]).sort_values("date")


@st.cache_data(ttl=300)
def _load_ml_prediction():
    try:
        df = pd.read_csv(_MERGED_PATH)
        df['date'] = pd.to_datetime(df['date'])
        latest = df.sort_values("date").iloc[-1]
        model = joblib.load(_MODEL_PATH)
        X = pd.DataFrame(
            [[latest[c] for c in ['sentiment_score', 'macro_score', 'bonds_score',
                                  'futures_vix_score', 'sectors_score', 'mes_score']]],
            columns=['sentiment', 'macro', 'bonds', 'futures_vix', 'sectors', 'mes']
        )
        pred = model.predict(X)[0]
        return {1: "✅ LONG", -1: "❌ SHORT", 0: "🔒 NEUTRAL"}.get(pred, "🔒")
    except Exception as e:
        return f"לא זמין ({e})"


@st.cache_data(ttl=300)
def _load_model_accuracy():
    try:
        df = pd.read_csv(_PERF_PATH)
        acc = (df['correct'] == '✅').mean() * 100
        return f"{acc:.1f}% ({len(df)} ימים)"
    except Exception:
        return "לא זמין"


def app():
    # === כותרת + כפתור רענן ===
    col_title, col_refresh = st.columns([6, 1])
    with col_title:
        st.title("📊 Market Sentiment Dashboard – החלטת מסחר יומית רשמית")
    with col_refresh:
        st.write("")
        if st.button("🔄 רענן", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

    st.caption(f"עודכן לאחרונה: {datetime.now().strftime('%H:%M:%S')} | מתעדכן אוטומטית כל 5 דקות")

    # === תחזית ML ===
    st.subheader("🧠 תחזית בינה מלאכותית")
    col1, col2 = st.columns(2)
    col1.metric("📍 המלצת AI יומית", _load_ml_prediction())
    col2.metric("🎯 דיוק מצטבר", _load_model_accuracy())

    with st.expander("🧠 ניהול חיזוי יומי"):
        if st.button("🚀 הרץ תחזית יומית"):
            st.cache_data.clear()
            st.success(f"✅ תחזית: {_load_ml_prediction()}")

        if st.button("📊 בדוק הצלחת חיזוי"):
            if datetime.now().hour >= 20:
                try:
                    result = subprocess.run(
                        ["python", str(_ROOT / "scores_news" / "ml_model" / "performance_tracker.py")],
                        capture_output=True, text=True
                    )
                    st.success("📈 התחזית הושוותה בהצלחה!")
                    st.code(result.stdout)
                except Exception as e:
                    st.error(f"שגיאה: {e}")
            else:
                st.warning("⏳ ניתן לבדוק רק לאחר השעה 20:00 (שעון ישראל)")

    # === הפעל ניתוח יומי ===
    with st.expander("🚀 הפעל ניתוח יומי"):
        if st.button("הרץ final_score.py"):
            try:
                result = subprocess.run(
                    ["python", str(_ROOT / "scores_news" / "cat_scores" / "final_score.py")],
                    capture_output=True, text=True
                )
                st.success("✅ הרצה הסתיימה בהצלחה")
                st.code(result.stdout)
                st.cache_data.clear()
                st.rerun()
            except Exception as e:
                st.error(f"שגיאה: {e}")

    # === טעינת נתוני ציונים ===
    try:
        df = _load_score_log()
        last_day = df.iloc[-1]
        score_columns = [col for col in df.columns if col.endswith("_score")]

        st.subheader(f"📆 {last_day['date'].date()} — החלטה יומית")
        col1, col2 = st.columns(2)
        col1.metric("🎯 ציון משוקלל", last_day.get("final_score", "—"))
        col2.metric("📌 המלצה", last_day.get("bias", "—"))

        # כרטיסי קטגוריות
        st.markdown("### 📊 ציוני קטגוריות")
        cols = st.columns(len(score_columns))
        for i, col_name in enumerate(score_columns):
            score = last_day[col_name]
            color = "🟢" if score >= 60 else "🟡" if score >= 40 else "🔴"
            with cols[i]:
                st.metric(label=col_name.replace("_score", "").capitalize(),
                          value=f"{score:.0f}", delta=color)

        # גרף מגמות
        st.markdown("### 📈 מגמות לפי קטגוריות")
        st.line_chart(df.set_index("date")[score_columns], height=250)

        # השוואת Final Score ל-MES
        st.markdown("### 🔁 השוואת Final Score ל־MES")
        plot_cols = [c for c in ["final_score", "daily_change_pct"] if c in df.columns]
        if plot_cols:
            st.line_chart(df.set_index("date")[plot_cols], height=250)
        else:
            st.info("📁 daily_change_pct יוצג לאחר הרצת final_score.py")

        # Heatmap
        st.markdown("### 🌡 Heatmap ציונים אחרונים")
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.heatmap(
            df.set_index("date")[score_columns].tail(15),
            cmap="RdYlGn", annot=True, fmt=".0f", linewidths=0.5, ax=ax
        )
        st.pyplot(fig)
        plt.close(fig)

        # בר-צ'ארט יום אחרון
        st.markdown("### 📊 קטגוריות – היום האחרון")
        latest_scores = {c.replace("_score", ""): last_day[c] for c in score_columns}
        st.bar_chart(pd.DataFrame.from_dict(latest_scores, orient="index", columns=["score"]))

    except Exception as e:
        st.error(f"❌ שגיאה בטעינת הנתונים: {e}")

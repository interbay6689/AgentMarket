import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import subprocess
from sklearn.ensemble import RandomForestClassifier
import joblib
import matplotlib
from datetime import datetime
import subprocess

matplotlib.use('Agg')

def app():
    st.title("📊 Market Sentiment Dashboard – החלטת מסחר יומית רשמית")

    # st.page_link("pages/ml_learning.py", label="🧠 תהליך הלמידה", icon="📊") # כפתור בראש הדף מתחת לכותרת
    with st.sidebar:
        if st.button("🧠 תהליך הלמידה"):
            switch_page("pages/ml_learning")

    # === המלצה יומית ע"י מודל ML ===
    def load_ml_prediction():
        try:
            df = pd.read_csv(
                r"/AgentMarket/scores_news/ml_model/merged_scores_mes.csv")
            df['date'] = pd.to_datetime(df['date'])
            latest = df.sort_values("date").iloc[-1]

            model = joblib.load(r"C:\Users\inter\PycharmProjects\FuturesMarketAI\scores_news\ml_model\model.pkl")
            X = pd.DataFrame([[latest[col] for col in ['sentiment_score', 'macro_score', 'bonds_score',
                                                       'futures_vix_score', 'sectors_score', 'mes_score']]],
                             columns=['sentiment', 'macro', 'bonds', 'futures_vix', 'sectors', 'mes'])
            pred = model.predict(X)[0]
            label = {1: "✅ LONG", -1: "❌ SHORT", 0: "🔒 NEUTRAL"}.get(pred, "🔒")
            return label

        except Exception as e:
            return f"שגיאה: {e}"

    def load_model_accuracy():
        try:
            df = pd.read_csv(
                r"/AgentMarket/scores_news/ml_model/ml_performance_log.csv")
            acc = (df['correct'] == '✅').mean() * 100
            return f"{acc:.1f}% ({len(df)} ימים)"
        except:
            return "לא זמין"

    # 🧠 הצגת תחזית יומית של המודל
    st.subheader("🧠 תחזית בינה מלאכותית")
    col1, col2 = st.columns(2)
    col1.metric("📍 המלצת AI יומית", load_ml_prediction())
    col2.metric("🎯 דיוק מצטבר", load_model_accuracy())

    with st.expander("🧠 ניהול חיזוי יומי"):
        if st.button("🚀 הרץ תחזית יומית"):
            try:
                pred = load_ml_prediction()
                st.success(f"✅ תחזית חיה: {pred}")
            except Exception as e:
                st.error(f"שגיאה בהרצת תחזית: {e}")

        if st.button("📊 בדוק הצלחת חיזוי"):
            current_hour = datetime.now().hour
            if current_hour >= 20:
                try:
                    result = subprocess.run(
                        ["python",
                         r"C:\Users\inter\PycharmProjects\FuturesMarketAI\scores_news\ml_model\performance_tracker.py"],
                        capture_output=True, text=True
                    )
                    st.success("📈 התחזית הושוותה בהצלחה מול התוצאה בפועל!")
                    st.code(result.stdout)
                except Exception as e:
                    st.error(f"שגיאה בהרצת performance_tracker: {e}")
            else:
                st.warning("⏳ ניתן לבדוק הצלחת תחזית רק לאחר השעה 20:00 (שעון ישראל)")

    score_path = Path(r"C:/Users/inter/PycharmProjects/FuturesMarketAI/scores_news/config/score_log.csv")
    mes_path = Path(r"C:/Users/inter/PycharmProjects/FuturesMarketAI/scores_news/config/MES_data.csv")

    # כפתור הרצה
    with st.expander("🚀 הפעל ניתוח יומי"):
        if st.button("הרץ final_score.py"):
            try:
                result = subprocess.run(
                    ["python", r"C:\Users\inter\PycharmProjects\FuturesMarketAI\scores_news\cat_scores\final_score.py"],
                    capture_output=True, text=True
                )
                st.success("✅ הרצה הסתיימה בהצלחה")
                st.code(result.stdout)
            except Exception as e:
                st.error(f"שגיאה בהרצה: {e}")

    try:
        df = pd.read_csv(score_path)
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"]).sort_values("date")
        last_day = df.iloc[-1]

        # ציונים אחרונים
        score_columns = [col for col in df.columns if col.endswith("_score")]
        final_score = last_day["final_score"]
        bias = last_day["bias"]

        # הצגה עליונה
        st.subheader(f"📆 {last_day['date'].date()} — החלטה יומית")
        col1, col2 = st.columns(2)
        col1.metric("🎯 ציון משוקלל", final_score)
        col2.metric("📌 המלצה", bias)

        # כרטיסים לפי קטגוריה
        st.markdown("### 📊 ציוני קטגוריות")
        cols = st.columns(len(score_columns))
        for i, col_name in enumerate(score_columns):
            cat = col_name.replace("_score", "").capitalize()
            score = last_day[col_name]
            with cols[i]:
                color = "🟢" if score >= 60 else "🟡" if score >= 40 else "🔴"
                st.metric(label=cat, value=f"{score:.0f}", delta=color)

        # גרף קווי של קטגוריות
        st.markdown("### 📈 מגמות לפי קטגוריות")
        st.line_chart(df.set_index("date")[score_columns], height=250)

        # השוואה ל־MES
        if mes_path.exists():
            df_mes = pd.read_csv(mes_path)
            df_mes["Date"] = pd.to_datetime(df_mes["Date"], errors="coerce")
            df_mes = df_mes.dropna(subset=["Date"])
            df_mes = df_mes.rename(columns={"Date": "date"})
            df_merged = pd.merge(df, df_mes, on="date", how="left")

            st.markdown("### 🔁 השוואת Final Score ל־MES")
            df_plot = df_merged[["date", "final_score", "daily_change_pct"]].set_index("date")
            st.line_chart(df_plot, height=250)
        else:
            st.warning("📁 MES_data.csv לא נמצא")

        # Heatmap
        st.markdown("### 🌡 Heatmap ציונים אחרונים")
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.heatmap(
            df.set_index("date")[score_columns].tail(15),
            cmap="RdYlGn", annot=True, fmt=".0f", linewidths=0.5, ax=ax
        )
        st.pyplot(fig)

        # עמודות לפי היום האחרון
        st.markdown("### 📊 קטגוריות – היום האחרון")
        latest_scores = {col.replace("_score", ""): last_day[col] for col in score_columns}
        df_bar = pd.DataFrame.from_dict(latest_scores, orient="index", columns=["score"])
        st.bar_chart(df_bar)

    except Exception as e:
        st.error(f"❌ שגיאה בטעינת הנתונים: {e}")

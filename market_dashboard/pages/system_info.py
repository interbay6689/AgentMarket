import streamlit as st
import pandas as pd
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

def app():
    st.title("🖥️ מידע מערכת")

    _ROOT = Path(__file__).resolve().parents[2]

    tab1, tab2, tab3 = st.tabs(["🔴 סנטימנט שלילי (NSSM)", "⚠️ התרעות מערכת", "📈 ציונים יומיים"])

    # === טאב 1: סנטימנט שלילי מ־NSSM ===
    with tab1:
        st.subheader("🧠 ניתוח סנטימנט שלילי מ־NSSM")
        file_path = _ROOT / "scores_news" / "logs" / "sentiment_new.csv"

        if not file_path.exists():
            st.error(f"⚠️ הקובץ sentiment_new.csv לא נמצא בנתיב:\n{file_path}")
        else:
            try:
                df = pd.read_csv(file_path)
                df.columns = df.columns.str.strip()
                if "datetime" not in df.columns or "sentiment_score" not in df.columns:
                    raise ValueError("העמודות 'datetime' או 'sentiment_score' לא קיימות בקובץ.")

                df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
                df = df.dropna(subset=["datetime", "sentiment_score"])
                df["category"] = df["category"].fillna("Unknown")

                # 🎯 סינון לפי קטגוריה
                available_categories = sorted(df["category"].unique())
                selected_categories = st.multiselect("🎯 סנן לפי קטגוריה", options=available_categories,
                                                     default=available_categories)
                df = df[df["category"].isin(selected_categories)]

                # 🔎 חיפוש מילות מפתח
                search_term = st.text_input("🔎 חיפוש מילות מפתח בכותרת", "")
                if search_term:
                    df = df[df["title"].str.contains(search_term, case=False, na=False)]

                # 🔥 סינון סנטימנט נמוך
                df_filtered = df[df["sentiment_score"] < 40]
                df_filtered = df_filtered.sort_values("datetime", ascending=False)

                if df_filtered.empty:
                    st.success("✅ לא נמצאו כתבות עם סנטימנט שלילי (score < 40).")
                else:
                    for _, row in df_filtered.iterrows():
                        color = "red" if row["sentiment_score"] < 30 else "orange"
                        icon = "🟥" if row["sentiment_score"] < 20 else "🟧" if row["sentiment_score"] < 30 else "🟨"
                        link_html = f"<a href='{row['link']}' target='_blank'>🔗 קרא עוד</a>" if "link" in row and pd.notna(
                            row["link"]) else ""

                        st.markdown(f"""
                        <div style='border: 1px solid #ccc; border-left: 5px solid {color}; padding: 10px; margin-bottom: 10px;'>
                            <strong>🕒 {row["datetime"]}</strong><br>
                            <strong>📰 {row["title"]}</strong><br>
                            <span style='color:{color};'><strong>Sentiment Score:</strong> {row["sentiment_score"]} {icon}</span><br>
                            <strong>Source:</strong> {row.get("source", "N/A")}<br>
                            <em>{row["summary"]}</em><br>
                            {link_html}
                        </div>
                        """, unsafe_allow_html=True)
            except Exception as e:
                st.error(f"❌ שגיאה בטעינת הנתונים: {e}")

    # === טאב 2: לוג התרעות ===
    with tab2:
        st.subheader("📢 לוג התרעות")
        alerts_path = _ROOT / "market_dashboard" / "logs" / "alerts_log.txt"
        if alerts_path.exists():
            lines = alerts_path.read_text(encoding="utf-8").splitlines()
            for line in lines[-500:]:
                st.markdown(f"- {line}")
        else:
            st.warning("⚠️ הקובץ alerts_log.txt לא נמצא.")

    # === טאב 3: ציוני קטגוריות יומיים ===
    with tab3:
        st.info("הטאב הועבר לדשבורד.")


import streamlit as st
import pandas as pd
from pathlib import Path

_ROOT       = Path(__file__).resolve().parents[2]
_SCORE_PATH = _ROOT / "scores_news" / "config" / "score_log.csv"


@st.cache_data(ttl=300)
def _load_score_history() -> pd.DataFrame:
    if not _SCORE_PATH.exists():
        return pd.DataFrame()
    df = pd.read_csv(_SCORE_PATH)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df.dropna(subset=["date"]).sort_values("date")


def app():
    st.title("🔍 פירוט קטגוריה")

    df = _load_score_history()
    if df.empty:
        st.info("אין נתונים היסטוריים עדיין. הרץ ניתוח יומי תחילה.")
        return

    score_cols = [c for c in df.columns if c.endswith("_score") and c != "final_score"]
    if not score_cols:
        st.warning("לא נמצאו עמודות ציון ב-score_log.csv.")
        return

    cat = st.selectbox("בחר קטגוריה:", score_cols,
                       format_func=lambda c: c.replace("_score", "").capitalize())

    st.subheader(f"📈 היסטוריה: {cat.replace('_score', '').capitalize()}")
    st.line_chart(df.set_index("date")[cat], height=280)

    col1, col2, col3 = st.columns(3)
    col1.metric("ממוצע", f"{df[cat].mean():.1f}")
    col2.metric("מינימום", f"{df[cat].min():.0f}")
    col3.metric("מקסימום", f"{df[cat].max():.0f}")

    st.subheader("📋 טבלת נתונים (30 ימים אחרונים)")
    display_cols = ["date", cat, "final_score", "bias"]
    existing = [c for c in display_cols if c in df.columns]
    st.dataframe(df[existing].tail(30).sort_values("date", ascending=False),
                 use_container_width=True)

import streamlit as st
from modules.data import load_time_series, load_significant_changes


def app():
    st.title("🔍 פירוט קטגוריה")
    categories = load_time_series().columns.tolist()
    cat = st.selectbox("בחר קטגוריה:", categories)

    ts = load_time_series()[cat]  # סדרת זמן של ציונים
    st.line_chart(ts)

    st.subheader("מקורות שגרמו לשינוי משמעותי")
    changes = load_significant_changes(cat)
    # DataFrame עם ['timestamp','source','old_score','new_score']
    st.table(changes)

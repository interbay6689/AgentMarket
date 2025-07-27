import streamlit as st
from modules.alerts import load_alerts

def app():
    st.title("⚠️ מערכת התראות")
    alerts_df = load_alerts()
    # ['time','type','description','value']
    st.dataframe(alerts_df, use_container_width=True)

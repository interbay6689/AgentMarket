# pages/settings.py
import streamlit as st
from modules.config import load_config, save_config

def app():
    st.title("⚙️ הגדרות")
    cfg = load_config()
    th = cfg.get("thresholds", {})
    inv_thresh = st.slider(
        "Yield Inversion Thresh (%)", 0.0, 100.0,
        value=th.get("yield_inversion", 2.0)
    )
    vix_thresh = st.slider(
        "VIX Spike Thresh (%)", 0.0, 100.0,
        value=th.get("vix_spike", 12.0)
    )

    with st.form("settings_form"):
        st.subheader("Thresholds")
        st.subheader("Weights")
        w_bonds = st.number_input("Bonds Weight", min_value=0.0, max_value=1.0, value=cfg['weights']['bonds'])
        w_macro = st.number_input("Macro Weight", min_value=0.0, max_value=1.0, value=cfg['weights']['macro'])
        w_sentiment = st.number_input("Sentiment Weight", min_value=0.0, max_value=1.0, value=cfg['weights']['sentiment'])
        w_vix = st.number_input("VIX Weight", min_value=0.0, max_value=1.0, value=cfg['weights'].get('vix', 0.20))
        w_sectors = st.number_input("Sectors Weight", min_value=0.0, max_value=1.0, value=cfg['weights']['sectors'])
        w_mes = st.number_input("MES Weight", min_value=0.0, max_value=1.0, value=cfg['weights']['mes'])

        st.subheader("RSS Sources")
        rss = st.text_area("כתובות RSS, שורה חדשה לכל מקור", "\n".join(cfg['rss_list']))

        submitted = st.form_submit_button("שמור הגדרות")
        if submitted:
            new_cfg = {
                'thresholds': {
                    'yield_inversion': inv_thresh,
                    'vix_spike': vix_thresh
                },
                'weights': {
                    'bonds': w_bonds,
                    'macro': w_macro,
                    'sentiment': w_sentiment,
                    'vix': w_vix,
                    'sectors': w_sectors,
                    'mes': w_mes
                },
                'rss_list': [u.strip() for u in rss.splitlines() if u.strip()]
            }
            save_config(new_cfg)
            st.success("ההגדרות נשמרו!")

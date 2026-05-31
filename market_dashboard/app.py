# app.py
import sys
from pathlib import Path

# מוסיף את תיקיית market_dashboard לשביל החיפוש של Python
sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st
from pages import settings, dashboard, system_info, alerts_page, category_detail, nq_order_flow
from pathlib import Path
import os
import csv

def init_logs():
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)

    files = {
        "error_log.txt": "=== Error Log Initialized ===\n",
        "alerts_log.txt": "=== Alerts Log Initialized ===\n",
        "score_log.csv": None
    }

    for file_name, content in files.items():
        path = Path(log_dir) / file_name
        if not path.exists():
            with open(path, "w", encoding="utf-8") as f:
                if content is not None:
                    f.write(content)
                else:
                    writer = csv.writer(f)
                    writer.writerow(["date", "macro", "sentiment", "bonds", "sectors", "vix", "total_score"])

init_logs()

st.set_page_config(page_title="Market Impact Dashboard", layout="wide")


PAGES = {
    "🏠 Dashboard": dashboard,
    "📊 NQ Order Flow": nq_order_flow,
    "🔍 Category Detail": category_detail,
    "⚠️ Alerts": alerts_page,
    "⚙️ Settings": settings,
    "🖥️ System Info": system_info,
}

st.sidebar.title("תפריט")
choice = st.sidebar.radio("", list(PAGES.keys()))
page = PAGES[choice]
page.app()  # מפעיל את הפונקציה app() בכל מודול

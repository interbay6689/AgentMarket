# app.py
import sys
import importlib
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).parent))  # market_dashboard/
sys.path.insert(0, str(_ROOT))                  # AgentMarket root — for scores_news

import streamlit as st
import os
import csv


def init_logs():
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    files = {
        "error_log.txt": "=== Error Log Initialized ===\n",
        "alerts_log.txt": "=== Alerts Log Initialized ===\n",
        "score_log.csv": None,
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

# שמות מודולים — נטענים רק כשהמשתמש מנווט אליהם
PAGE_MODULES = {
    "🏠 Dashboard":        "dashboard",
    "📊 NQ Order Flow":    "nq_order_flow",
    "🧠 ML Learning":      "ml_learning",
    "🔍 Category Detail":  "category_detail",
    "⚠️ Alerts":           "alerts_page",
    "⚙️ Settings":         "settings",
    "🖥️ System Info":      "system_info",
}

st.sidebar.title("תפריט")
choice = st.sidebar.radio("", list(PAGE_MODULES.keys()))

# טעינה עצלה — Python מחזיק את המודול ב-sys.modules אחרי הטעינה הראשונה
page = importlib.import_module(f"pages.{PAGE_MODULES[choice]}")
page.app()

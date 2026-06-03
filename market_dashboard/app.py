# app.py
import sys
import importlib
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).parent))  # market_dashboard/
sys.path.insert(0, str(_ROOT))                  # AgentMarket root — for scores_news

# Load .env from the project root (AgentMarket/.env) if python-dotenv is installed.
# This lets developers set API keys in a local .env file without touching env vars.
try:
    from dotenv import load_dotenv
    _env_file = _ROOT / ".env"
    if _env_file.exists():
        load_dotenv(_env_file, override=False)  # override=False: system env vars win
except ImportError:
    pass  # python-dotenv not installed — env vars must be set manually

import os as _os
_PLACEHOLDERS = {"PASTE_YOUR_NEW_KEY_HERE", "sk-ant-...", "", "your_key_here"}
for _k in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY"):
    if _os.environ.get(_k, "") in _PLACEHOLDERS:
        _os.environ.pop(_k, None)

import streamlit as st
import os
import csv


def init_logs():
    log_dir = _ROOT / "market_dashboard" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "error_log.txt").touch()
    (log_dir / "alerts_log.txt").touch()


init_logs()

st.set_page_config(
    page_title="AgentMarket — NQ Trading",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Minimal sidebar CSS
st.markdown("""
<style>
[data-testid="stSidebar"] { min-width: 180px; max-width: 200px; }
[data-testid="stSidebar"] .stRadio label { font-size: 0.95rem; }
</style>
""", unsafe_allow_html=True)

PAGE_MODULES = {
    "🎯 Signals":      "trade_signals",
    "📊 Dashboard":    "dashboard",
    "📈 NQ Analysis":  "nq_order_flow",
    "📉 Backtest":     "backtesting",
    "🧪 A/B Tracking": "ab_tracking",
    "🧠 Learning":     "ml_learning",
    "📓 Journal":      "trade_journal",
    "🤖 Agents":       "agents_dashboard",
    "⚙️ System":       "system_page",
}

st.sidebar.title("AgentMarket")
choice = st.sidebar.radio("", list(PAGE_MODULES.keys()))

# Lazy-load — Python caches the module in sys.modules after first import
page = importlib.import_module(f"pages.{PAGE_MODULES[choice]}")
page.app()

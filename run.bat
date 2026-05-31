@echo off
cd /d %~dp0
call venv\Scripts\activate
streamlit run market_dashboard/app.py
pause

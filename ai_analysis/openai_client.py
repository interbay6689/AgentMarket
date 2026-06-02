# Legacy shim — all functionality moved to claude_client.py
# This file exists only for backward compatibility. Do not add new code here.
from ai_analysis.claude_client import analyze_text, analyze_json, analyze_article  # noqa: F401

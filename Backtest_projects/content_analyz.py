"""
content_analyz.py — Economic category analysis of financial news articles.

Fetches article URLs via link_news() (newspaper3k + BeautifulSoup extraction),
then uses Claude to identify economic categories and their directional impact
on futures markets (-3 to +3 per category).

Run: python -m Backtest_projects.content_analyz  (from AgentMarket root)
"""
import sys
from pathlib import Path

# Ensure project root is importable when run directly
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import os
import requests
from bs4 import BeautifulSoup

from ai_analysis.claude_client import analyze_text
from Backtest_projects.fetch_news_rss import link_news

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0 Safari/537.36"
    )
}


def extract_text_from_url(url: str, timeout: int = 10) -> str:
    """Extract article body text via BeautifulSoup."""
    try:
        response = requests.get(url, headers=_HEADERS, timeout=timeout)
        soup = BeautifulSoup(response.text, "html.parser")
        paragraphs = soup.find_all("p")
        return "\n".join(
            p.get_text() for p in paragraphs if len(p.get_text()) > 40
        )[:3000]
    except Exception as e:
        return f"ERROR extracting content: {e}"


def analyze_article_categories(link: str) -> str:
    """
    Extract article text and run Claude economic category analysis.

    Returns a formatted string with category, impact (-3..+3),
    relevance %, and supporting quote.
    """
    article_text = extract_text_from_url(link)
    if article_text.startswith("ERROR"):
        return article_text

    prompt = f"""הטקסט הבא הוא כתבת חדשות כלכלית. נתח את התוכן והצג את הקטגוריות הכלכליות \
המרכזיות שהכתבה עוסקת בהן.

עבור כל קטגוריה רלוונטית (אינפלציה, ריבית, שוק ההון, פוליטיקה, מט"ח, אנרגיה, גיאופוליטי, \
אבטלה, נדל"ן, סנטימנט כלכלי):
- שם הקטגוריה
- השפעה: מספר בין -3 ל+3 (שלילי = דובי לNQ, חיובי = שורי)
- אחוז רלוונטיות (0–100%)
- משפט תמיכה קצר מהכתבה

כתבה:
\"\"\"{article_text}\"\"\"
"""
    system = (
        "אתה מומחה כלכלה המזהה השפעות כתבות חדשות על חוזים עתידיים (NQ, ES). "
        "ענה בעברית. היה תמציתי ומדויק."
    )
    result = analyze_text(prompt, deep=True, system_override=system, max_tokens=1800)
    return result if result else "ERROR: Claude did not return a response"


if __name__ == "__main__":
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("❌ ANTHROPIC_API_KEY לא מוגדר.")
        sys.exit(1)

    urls = link_news()
    if not urls:
        print("⚠️ לא נמצאו כתבות.")
        sys.exit(0)

    for i, url in enumerate(urls, 1):
        print(f"\n🔗 ({i}/{len(urls)}) מנתח כתבה:\n{url}")
        result = analyze_article_categories(url)
        print(result)
        print("-" * 60)

"""
main.py — AgentMarket entry point for daily news analysis.

Fetches articles from RSS feeds, runs Claude analysis per article,
scores market impact (1-10), and saves results to analysis_results/.

Run: python main.py  (from the AgentMarket root directory)
"""
import datetime
import json
import os
import sys
from pathlib import Path

# Ensure project root is on the path when run directly
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from ai_analysis.claude_client import analyze_text, analyze_article
from rss_feeds.feeds_list import RSS_FEEDS
from utils.helpers import parse_rss


def calculate_impact_score(analysis_text: str) -> int:
    prompt = (
        "על בסיס הניתוח הבא, תן ניקוד בין 1 (השפעה נמוכה) ל-10 (השפעה גבוהה מאוד) "
        "על שוק המניות NASDAQ ו-S&P 500.\n\n"
        f"ניתוח:\n{analysis_text}\n\n"
        "ספק אך ורק מספר שלם בין 1 ל-10 ללא שום טקסט נוסף."
    )
    raw = analyze_text(prompt)
    try:
        return max(1, min(10, int(raw.strip())))
    except ValueError:
        return 0


def get_recommendation(analysis_text: str) -> str:
    prompt = (
        "בהתבסס על הניתוח הבא, המלץ על פעולת השקעה אחת בלבד: לונג, שורט או ללא פעולה.\n\n"
        f"ניתוח:\n{analysis_text}\n\n"
        "ענה רק עם אחת מהאפשרויות: לונג, שורט, ללא פעולה."
    )
    return analyze_text(prompt).strip()


def save_results(daily_scores: list) -> None:
    results_dir = _ROOT / "analysis_results"
    results_dir.mkdir(exist_ok=True)
    today = datetime.date.today().isoformat()
    filepath = results_dir / f"daily_scores_{today}.json"
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(daily_scores, f, ensure_ascii=False, indent=4)
    print(f"✅ התוצאות נשמרו: {filepath}")


def main():
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("❌ ANTHROPIC_API_KEY לא מוגדר. הגדר את משתנה הסביבה לפני הרצה.")
        sys.exit(1)

    daily_scores = []
    today = datetime.date.today().isoformat()

    for feed_name, url in RSS_FEEDS.items():
        print(f"\n📡 טוען נתונים מ-{feed_name}...")
        try:
            articles = parse_rss(url)
        except Exception as e:
            print(f"  ⚠️ שגיאה בטעינת פיד: {e}")
            continue

        for article in articles:
            title = article.get("title", "")
            summary = article.get("summary", "")
            link = article.get("link", "")

            print(f"  📄 מנתח: {title[:70]}...")

            # Deep analysis via Claude
            analysis = analyze_article(title=title, body=summary, language="he")
            impact_score = calculate_impact_score(analysis)
            recommendation = get_recommendation(analysis)

            print(f"     ניקוד השפעה: {impact_score}/10  |  המלצה: {recommendation}")

            entry = {
                "date": today,
                "feed_name": feed_name,
                "title": title,
                "analysis": analysis,
                "impact_score": impact_score,
                "recommendation": recommendation,
                "article_link": link,
            }
            daily_scores.append(entry)

            # WhatsApp alert for high-impact articles (Twilio env vars required)
            if impact_score >= 7:
                try:
                    from utils.whatsapp_alerts import send_whatsapp_message
                    msg = (
                        f"📢 התראת שוק חשובה!\n"
                        f"כותרת: {title}\n"
                        f"ניקוד: {impact_score}/10  |  המלצה: {recommendation}\n\n"
                        f"{analysis[:300]}...\n\n{link}"
                    )
                    send_whatsapp_message(msg, os.environ.get("WHATSAPP_TO", ""))
                except Exception:
                    pass  # Twilio not configured — skip silently

    save_results(daily_scores)
    print(f"\n✅ סיכום: {len(daily_scores)} כתבות נותחו")


if __name__ == "__main__":
    main()

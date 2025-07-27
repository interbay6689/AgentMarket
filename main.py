import datetime
import json
import os

from AgentMarket.ai_analysis.openai_client import analyze_text
from AgentMarket.rss_feeds.feeds_list import RSS_FEEDS
from AgentMarket.utils.helpers import parse_rss


def calculate_impact_score(analysis_text):
    prompt = f"""
    על בסיס הניתוח הבא, תן ניקוד בין 1 (השפעה נמוכה) ל-10 (השפעה גבוהה מאוד) על שוק המניות NASDAQ ו-S&P 500.

    ניתוח:
    {analysis_text}

    ספק אך ורק מספר שלם בין 1 ל-10 ללא שום טקסט נוסף או הסברים.
    """
    score = analyze_text(prompt)
    try:
        return int(score)
    except ValueError:
        print(f"שגיאה בפענוח הניקוד: '{score}', מוחזר ערך ברירת מחדל 0.")
        return 0

def save_results(daily_scores):
    results_dir = "analysis_results"
    os.makedirs(results_dir, exist_ok=True)
    today = datetime.date.today().isoformat()
    filepath = os.path.join(results_dir, f"daily_scores_{today}.json")

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(daily_scores, f, ensure_ascii=False, indent=4)
    print(f"התוצאות נשמרו בהצלחה בקובץ {filepath}")

def main():
    daily_scores = []
    today = datetime.date.today().isoformat()

    for feed_name, url in RSS_FEEDS.items():
        print(f"טוען נתונים מ-{feed_name}...")
        articles = parse_rss(url)

        for article in articles:
            prompt = f"""
            נתח את סיכום המאמר הבא וספק תובנות מקצועיות בעברית על ההשפעה שלו על שוק החוזים העתידיים (NASDAQ ו-S&P 500).
            כלול המלצה לפעולת השקעה (לונג/שורט/ללא פעולה) בהתחשב בניתוח.

            כותרת: {article['title']}
            סיכום: {article['summary']}
            """
            analysis = analyze_text(prompt)
            impact_score = calculate_impact_score(analysis)

            recommendation_prompt = f"""
            בהתבסס על הניתוח הבא, המלץ על פעולת השקעה אחת בלבד:
            לונג, שורט או ללא פעולה.

            ניתוח:
            {analysis}

            ענה רק עם אחת מהאפשרויות: לונג, שורט, ללא פעולה.
            """
            recommendation = analyze_text(recommendation_prompt).strip()

            print(f"כותרת המאמר: {article['title']}")
            print(f"ניתוח השפעות:\n{analysis}")
            print(f"ניקוד השפעה (Impact Score): {impact_score}")
            print(f"המלצת השקעה: {recommendation}\n{'-' * 80}\n")

            daily_scores.append({
                "date": today,
                "feed_name": feed_name,
                "title": article['title'],
                "analysis": analysis,
                "impact_score": impact_score,
                "recommendation": recommendation,
                "article_link": article['link']
            })

            # שליחת התראה לוואטסאפ רק על השפעה משמעותית (למשל, ניקוד 7 ומעלה)
            if impact_score >= 7:
                message_body = f"""
        📢 התראה חשובה משוק החוזים העתידיים:
        כותרת: {article['title']}
        ניקוד השפעה: {impact_score}/10
        המלצת השקעה: {recommendation}

        ניתוח קצר:
        {analysis[:300]}...

        לינק למאמר:
        {article['link']}
                """
        #        send_whatsapp_message(message_body, "+972522621552")  # Whatsapp notifications

    save_results(daily_scores)

if __name__ == "__main__":
    main()

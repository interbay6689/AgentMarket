# import sys
# import os
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import os
import pandas as pd
import datetime
from pathlib import Path
from scores_news.cat_scores.nlp_utils import analyze_articles

# ---------------------------------------------------------------
# נתיב בסיס - מאפשר גישה יחסית לקבצי לוג במקום שימוש בנתיב קבוע
# הקובץ נמצא תחת scores_news/cat_scores ולכן parents[2] מחזיר את שורש הפרויקט.
BASE_DIR = Path(__file__).resolve().parents[2]
LOG_DIR = BASE_DIR / "scores_news" / "logs"


def load_sentiment_data(date: str | None = None) -> pd.DataFrame:
    """טוען את קובץ הכתבות שנשמר בשלב הקודם מתוך תיקיית logs.

    הפרמטר `date` בפורמט YYYY-MM-DD. אם לא סופק, יטען את היום הנוכחי.
    """
    if not date:
        date = datetime.datetime.now().strftime("%Y-%m-%d")
    path = LOG_DIR / f"sentiment_raw_{date}.csv"
    if not path.exists():
        raise FileNotFoundError(f"⚠️ קובץ sentiment לא נמצא: {path}")
    return pd.read_csv(path)

def calculate_sentiment_score(df: pd.DataFrame) -> tuple[int, str]:
    articles = df.to_dict(orient="records")
    analyzed = analyze_articles(articles)
    labeled_df = pd.DataFrame(analyzed)

    total = len(labeled_df)
    pos = (labeled_df["sentiment_label"] == "positive").sum()
    neg = (labeled_df["sentiment_label"] == "negative").sum()
    neu = (labeled_df["sentiment_label"] == "neutral").sum()

    if total == 0:
        return 50, "🔒 אין מספיק נתונים לחישוב סנטימנט"

    # שקילה: positive = +1, neutral = 0.5, negative = 0
    score = round(((pos * 1.0) + (neu * 0.5)) / total * 100)

    explanation = (
        f"✅ חיוביות: {pos}, ❌ שליליות: {neg}, ⚪ נייטרליות: {neu}, סה״כ: {total}"
    )
    return score, explanation


if __name__ == "__main__":
    df = load_sentiment_data()
    score, _ = calculate_sentiment_score(df)
    print(score)
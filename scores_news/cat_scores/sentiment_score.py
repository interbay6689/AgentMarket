# import sys
# import os
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import os
import pandas as pd
import datetime
from scores_news.cat_scores.nlp_utils import analyze_articles


def load_sentiment_data(date=None):
    """טוען את קובץ הכתבות שנשמר מהשלב הקודם"""
    if not date:
        date = datetime.datetime.now().strftime("%Y-%m-%d")
    path = f"/AgentMarket/scores_news\\logs\\sentiment_raw_{date}.csv"
    if not os.path.exists(path):
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
# test_sentiment_score.py

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from fetch_news_rss_async import fetch_all_sentiment_articles_async



from data_sources.fetch_news_rss import fetch_all_sentiment_articles

def test_sentiment_articles_structure():
    print("🔍 מתחיל בדיקת sentiment...")
    df = fetch_all_sentiment_articles_async()
    print(f"📊 נמצאו {len(df)} כתבות")
    assert not df.empty
    assert "title" in df.columns
    assert df["title"].str.len().max() > 10, "❌ הכותרות ריקות או קצרות מדי"
    assert df["published"].notnull().all(), "❌ קיימות כתבות ללא תאריך פרסום"
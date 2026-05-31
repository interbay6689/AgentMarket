# fetch_news_rss.py
import logging
import os
from datetime import datetime
from pathlib import Path

import feedparser
import pandas as pd
import yaml

_ROOT = Path(__file__).resolve().parents[2]

# from utils.cache import RedisCache, fetch_url_with_cache
from scores_news.utils.cache import RedisCache, fetch_url_with_cache

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)

def load_sentiment_feeds(config_path=None):
    """טוען קישורי RSS לפי קטגוריית sentiment"""
    if config_path is None:
        config_path = _ROOT / "scores_news" / "config" / "sources.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config["rss_feeds"]["sentiment"]

def fetch_feed_articles(feed_url, cache, timeout=30, retries=1):
    """משיכת כתבות מ־RSS כולל שימוש ב־Redis Cache"""

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        )
    }

    try:
        raw_feed = fetch_url_with_cache(feed_url, cache, headers=headers, timeout=timeout, retries=retries)
        feed = feedparser.parse(raw_feed)
    except Exception as e:
        logger.error(f"Error fetching feed {feed_url}: {e}")
        return []

    articles = []
    for entry in feed.entries:
        articles.append({
            "title": entry.get("title", ""),
            "summary": entry.get("summary", ""),
            "link": entry.get("link", ""),
            "published": entry.get("published", ""),
            "source": feed_url
        })

    return articles

def fetch_all_sentiment_articles():
    """משיכת כל הכתבות מכל פידי הסנטימנט עם Cache"""
    feeds = load_sentiment_feeds()
    all_articles = []

    cache = RedisCache(ttl_seconds=600)  # Cache ל־10 דקות

    for url in feeds:
        logger.info(f"📡 טוען RSS: {url}")
        try:
            articles = fetch_feed_articles(url, cache)
            logger.info(f"✅ {len(articles)} כתבות מ־{url}")
            all_articles.extend(articles)
        except Exception as e:
            logger.warning(f"⚠️ שגיאה בהבאת נתונים מ: {url} – {e}")

    return pd.DataFrame(all_articles)

def fetch_and_save_sentiment_data(date=None):
    """פונקציה ציבורית לשימוש גם מבחוץ"""
    if not date:
        date = datetime.now().strftime("%Y-%m-%d")

    df = fetch_all_sentiment_articles()

    if df.empty:
        logger.warning("⚠️ לא נמצאו כתבות. קובץ לא יישמר.")
        return

    output_dir = str(_ROOT / "scores_news" / "logs")
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, f"sentiment_raw_{date}.csv")
    df.to_csv(output_path, index=False)
    logger.info(f"✅ נשמרו {len(df)} כתבות מהקטגוריה 'sentiment' אל הקובץ: {output_path}")

# קריאה ישירה
if __name__ == "__main__":
    try:
        fetch_and_save_sentiment_data()
    except Exception as e:
        logger.error(f"❌ שגיאת ריצה: {e}")

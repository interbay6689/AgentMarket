# fetch_news_rss.py
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
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

# Set FETCH_FULL_TEXT=1 to extract full article body (slower but richer for Claude scoring).
_FETCH_FULL_TEXT = os.environ.get("FETCH_FULL_TEXT", "0").strip() == "1"
_FULL_TEXT_WORKERS = 6
_FULL_TEXT_MAX_CHARS = 3000


def _extract_full_text(url: str, timeout: int = 8) -> str:
    """Extract article body text. Tries newspaper3k then BeautifulSoup."""
    # Primary: newspaper3k
    try:
        from newspaper import Article
        art = Article(url)
        art.config.browser_user_agent = "Mozilla/5.0"
        art.config.request_timeout = timeout
        art.download()
        art.parse()
        if len(art.text) > 200:
            return art.text[:_FULL_TEXT_MAX_CHARS]
    except Exception:
        pass

    # Fallback: BeautifulSoup
    try:
        import requests
        from bs4 import BeautifulSoup
        resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=timeout)
        soup = BeautifulSoup(resp.content, "html.parser")
        paras = [p.get_text() for p in soup.find_all("p") if len(p.get_text()) > 40]
        return "\n".join(paras)[:_FULL_TEXT_MAX_CHARS]
    except Exception:
        pass

    return ""


def _enrich_full_text(articles: list) -> list:
    """Parallel full-text extraction for a list of article dicts (in-place update)."""
    def _fetch(item):
        idx, art = item
        text = _extract_full_text(art.get("link", ""))
        return idx, text

    with ThreadPoolExecutor(max_workers=_FULL_TEXT_WORKERS) as pool:
        futures = {pool.submit(_fetch, (i, a)): i for i, a in enumerate(articles)}
        for fut in as_completed(futures):
            try:
                idx, text = fut.result()
                articles[idx]["full_text"] = text
            except Exception:
                pass
    return articles


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
            "source": feed_url,
            "full_text": "",
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

    if _FETCH_FULL_TEXT and all_articles:
        logger.info(f"📄 מוציא טקסט מלא ל-{len(all_articles)} כתבות (FETCH_FULL_TEXT=1)…")
        all_articles = _enrich_full_text(all_articles)
        filled = sum(1 for a in all_articles if a.get("full_text"))
        logger.info(f"✅ טקסט מלא נמשך עבור {filled}/{len(all_articles)} כתבות")

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

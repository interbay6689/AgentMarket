# fetch_news_rss_async.py
import aiohttp
import asyncio
import feedparser
import pandas as pd
import yaml
from nlp_utils import analyze_articles

async def fetch_single_feed(session, url):
    try:
        async with session.get(url, timeout=10) as response:
            text = await response.text()
            parsed = feedparser.parse(text)
            articles = []
            for entry in parsed.entries:
                articles.append({
                    "title": entry.get("title", ""),
                    "summary": entry.get("summary", ""),
                    "link": entry.get("link", ""),
                    "published": entry.get("published", ""),
                    "source": url
                })
            return articles
    except Exception as e:
        print(f"⚠️ שגיאה בהבאת {url}: {e}")
        return []

async def fetch_all_feeds(urls):
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_single_feed(session, url) for url in urls]
        results = await asyncio.gather(*tasks)
        all_articles = [item for sublist in results for item in sublist]
        return all_articles

def load_sentiment_feeds(config_path="../config/sources.yaml"):
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config["rss_feeds"]["sentiment"]

def fetch_all_sentiment_articles_async():
    """פונקציה ראשית – שליפה אסינכרונית"""
    urls = load_sentiment_feeds()
    articles = asyncio.run(fetch_all_feeds(urls))
    articles = analyze_articles(articles)
    return pd.DataFrame(articles)

# בדיקה עצמאית:
if __name__ == "__main__":
    df = fetch_all_sentiment_articles_async()
    print(f"✅ נמצאו {len(df)} כתבות")
    print(df.head(3))

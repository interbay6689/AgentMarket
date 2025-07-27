import os
import logging
import time
from datetime import datetime
import feedparser
from scores_news.cat_scores.nlp_utils import analyze_articles
from scores_news.utils.cache import RedisCache
import pandas as pd
import json

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))


# הגדרות נתיב לסקריפט הראשי
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# הגדרות לוגים
LOG_PATH = os.path.join(SCRIPT_DIR, "..", "scores_news", "logs", "rss_listener.log")
os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)

logging.basicConfig(
    filename=LOG_PATH,
    filemode="a",
    format="%(asctime)s - %(levelname)s: %(message)s",
    level=logging.INFO
)
logging.getLogger().addHandler(logging.StreamHandler())  # לוגים גם למסך

# הדפסת מיקום קובץ הלוג
logging.info(f"📍 Writing logs to: {os.path.abspath(LOG_PATH)}")

# קובץ CSV של כתבות
CSV_PATH = r"/AgentMarket/scores_news/logs/sentiment_new.csv"

# אתחול Redis
cache = RedisCache()

# קריאת פידים מתוך קובץ yaml
import yaml

CONFIG_PATH = os.path.join(SCRIPT_DIR, "..", "config", "sources.yaml")
with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    feed_config = yaml.safe_load(f)


def fetch_feed(url):
    for attempt in range(2):
        try:
            logging.info(f"Fetching RSS feed from internet: {url}")
            return feedparser.parse(url)
        except Exception as e:
            logging.warning(f"Attempt {attempt + 1} failed for {url}: {e}")
    logging.error(f"Failed to fetch {url} after 2 attempts")
    return None

def handle_new_entries(articles, category):
    logging.info(f"🧪 Handling {len(articles)} articles in category {category}")
    new_entries = []
    for entry in articles:
        unique_id = entry.get("link", "")
        cache_key = f"entry:{unique_id}"
        if not cache.get(cache_key):
            title = entry.get("title", "").strip()
            summary = entry.get("summary", "").strip()
            link = entry.get("link", "").strip()
            published = entry.get("published", "")
            source = entry.get("source", "")

            logging.info(f"🆕 New entry found in {category}: {title}")
            cache.set(cache_key, "seen", ttl=600)

            # ניתוח סנטימנט
            try:
                analyzed = analyze_articles([entry])[0]
                sentiment_score = analyzed.get("sentiment_score", None)
                logging.info(f"🧠 Sentiment score: {sentiment_score}")
            except Exception as e:
                logging.error(f"❌ Failed to analyze article: {e}")
                sentiment_score = None

            new_entries.append({
                "datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "title": title,
                "summary": summary,
                "link": link,
                "published": published,
                "source": source,
                "category": category,
                "sentiment_score": sentiment_score
            })

    if new_entries:
        df = pd.DataFrame(new_entries)
        df.to_csv(CSV_PATH, mode="a", index=False, header=not os.path.exists(CSV_PATH))
        logging.info(f"📁 Saved {len(new_entries)} new entries to {CSV_PATH}")


def main():
    logging.info("=== Starting Continuous RSS Listener ===")
    feeds = feed_config["rss_feeds"]  # וידוא גישה נכונה
    while True:
        for category, urls in feeds.items():
            logging.info(f"Checking RSS category: {category}")
            for url in urls:
                logging.info(f"🔗 Fetching from URL: {url} (Category: {category})")

                cache_key = f"rss_cache:{url}"
                raw_data = cache.get(cache_key)

                if raw_data:
                    try:
                        parsed_entries = json.loads(raw_data)
                        feed = type("Feed", (object,), {"entries": parsed_entries})
                    except Exception as e:
                        logging.error(f"❌ Failed to parse cached JSON for {cache_key}: {e}")
                        feed = None
                else:
                    feed = fetch_feed(url)
                    if feed:
                        # נרמל את הנתונים לשמירה תקינה ב־Redis
                        entries = []
                        for entry in feed.entries:
                            entries.append({
                                "title": entry.get("title", ""),
                                "summary": entry.get("summary", ""),
                                "link": entry.get("link", ""),
                                "published": entry.get("published", ""),
                                "source": entry.get("source", "")
                            })
                        cache.set(cache_key, json.dumps(entries), ttl=600)
                        feed = type("Feed", (object,), {"entries": entries})

                if feed:
                    logging.info(f"✅ Got {len(feed.entries)} entries from: {url}")
                    handle_new_entries(feed.entries, category)

        logging.info("Sleeping for 300 seconds...\n")
        time.sleep(300)


if __name__ == "__main__":
    main()

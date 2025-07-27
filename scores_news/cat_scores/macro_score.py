import yaml
import os
import feedparser
import pandas as pd
from datetime import datetime
from pathlib import Path
from scores_news.cat_scores.nlp_utils import analyze_articles

# ---------------------------------------------------------------
# נתיב בסיס - משמש למציאת sources.yaml יחסית לפרויקט
BASE_DIR = Path(__file__).resolve().parents[2]
CONFIG_PATH = BASE_DIR / "scores_news" / "config" / "sources.yaml"


def load_macro_feeds(config_path: Path | None = None) -> list:
    """טוען קישורי RSS של macro מתוך sources.yaml.

    אם לא סופק נתיב, ישתמש בנתיב יחסית תחת CONFIG_PATH.
    """
    path = Path(config_path) if config_path else CONFIG_PATH
    if not path.exists():
        raise FileNotFoundError(f"sources.yaml לא נמצא: {path}")
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config.get("rss_feeds", {}).get("macro", [])


def fetch_macro_news():
    """משוך את כל הודעות המאקרו לפי RSS"""
    urls = load_macro_feeds()
    all_articles = []

    for url in urls:
        try:
            print(f"📡 טוען RSS מאקרו: {url}")
            feed = feedparser.parse(url)
            for entry in feed.entries:
                all_articles.append({
                    "title": entry.get("title", ""),
                    "summary": entry.get("summary", ""),
                    "link": entry.get("link", ""),
                    "published": entry.get("published", ""),
                    "source": url
                })
        except Exception as e:
            print(f"⚠️ שגיאה ב־{url}: {e}")

    return pd.DataFrame(all_articles)


def calculate_macro_score(df: pd.DataFrame) -> tuple[int, str]:
    articles = df.to_dict(orient="records")
    enriched = analyze_articles(articles)
    df_enriched = pd.DataFrame(enriched)

    pos = (df_enriched["sentiment_label"] == "positive").sum()
    neg = (df_enriched["sentiment_label"] == "negative").sum()
    total = pos + neg

    if total == 0:
        return 50, "🔒 אין מספיק חדשות מאקרו רלוונטיות היום"

    score = round(pos / total * 100)
    explanation = f"✅ חיוביות: {pos}, ❌ שליליות: {neg}, סה״כ: {total}"
    return score, explanation


if __name__ == "__main__":
    df = fetch_macro_news()
    score, explanation = calculate_macro_score(df)
    print(f"📊 Macro Score: {score} | {explanation}")

import os
import pandas as pd
from datetime import datetime
from pathlib import Path

try:
    import yaml
    _yaml_ok = True
except ImportError:
    _yaml_ok = False

try:
    import feedparser
    _feedparser_ok = True
except ImportError:
    _feedparser_ok = False

try:
    from scores_news.cat_scores.nlp_utils import analyze_articles
except ImportError:
    from nlp_utils import analyze_articles

# ---------------------------------------------------------------
# נתיב בסיס - משמש למציאת sources.yaml יחסית לפרויקט
BASE_DIR = Path(__file__).resolve().parents[2]
CONFIG_PATH = BASE_DIR / "scores_news" / "config" / "sources.yaml"


def load_macro_feeds(config_path: Path | None = None) -> list:
    """טוען קישורי RSS של macro מתוך sources.yaml."""
    if not _yaml_ok:
        return []
    path = Path(config_path) if config_path else CONFIG_PATH
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config.get("rss_feeds", {}).get("macro", [])


def fetch_macro_news():
    """משוך את כל הודעות המאקרו לפי RSS"""
    if not _feedparser_ok:
        return pd.DataFrame()
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
    if df.empty:
        return 50, "No macro data available"
    articles = df.to_dict(orient="records")
    enriched = analyze_articles(articles)
    df_enriched = pd.DataFrame(enriched)

    if df_enriched.empty or "sentiment_label" not in df_enriched.columns:
        return 50, "No macro articles analyzed"

    pos = (df_enriched["sentiment_label"] == "positive").sum()
    neg = (df_enriched["sentiment_label"] == "negative").sum()
    total = pos + neg

    if total == 0:
        return 50, "No relevant macro news today"

    score = round(pos / total * 100)
    explanation = f"✅ חיוביות: {pos}, ❌ שליליות: {neg}, סה״כ: {total}"
    return score, explanation


if __name__ == "__main__":
    df = fetch_macro_news()
    score, explanation = calculate_macro_score(df)
    print(f"📊 Macro Score: {score} | {explanation}")

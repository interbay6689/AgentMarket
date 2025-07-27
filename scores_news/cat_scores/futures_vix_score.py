# futures_vix_score.py


import feedparser
import yaml
import pandas as pd
from pathlib import Path
from scores_news.cat_scores.nlp_utils import analyze_articles
import yfinance as yf

# ---------------------------------------------------------------
# נתיב בסיס למציאת sources.yaml באופן יחסי
BASE_DIR = Path(__file__).resolve().parents[2]
CONFIG_PATH = BASE_DIR / "scores_news" / "config" / "sources.yaml"

def load_futures_feeds(config_path: Path | None = None) -> list:
    """טוען קישורי RSS של futures/vix מתוך sources.yaml.

    אם לא סופק נתיב, ישתמש בנתיב ברירת מחדל.
    """
    path = Path(config_path) if config_path else CONFIG_PATH
    if not path.exists():
        raise FileNotFoundError(f"sources.yaml לא נמצא: {path}")
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config.get("rss_feeds", {}).get("futures_vix", [])


def fetch_futures_news():
    urls = load_futures_feeds()
    articles = []
    for url in urls:
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries:
                articles.append({
                    "title": entry.get("title", ""),
                    "summary": entry.get("summary", ""),
                    "link": entry.get("link", ""),
                    "published": entry.get("published", ""),
                    "source": url
                })
        except Exception as e:
            print(f"⚠️ שגיאה ב־{url}: {e}")
    return pd.DataFrame(articles)


def calculate_futures_score(df: pd.DataFrame) -> tuple[int, str]:
    articles = df.to_dict(orient="records")
    enriched = analyze_articles(articles)
    df_enriched = pd.DataFrame(enriched)

    pos = (df_enriched["sentiment_label"] == "positive").sum()
    neg = (df_enriched["sentiment_label"] == "negative").sum()
    total = pos + neg

    if total == 0:
        return 50, "🔒 אין מספיק מידע רלוונטי"

    score = round(pos / total * 100)
    explanation = f"✅ חוזים חיוביים: {pos}, ❌ שליליים: {neg}, סה״כ: {total}"

    return score, explanation

def detect_vix_spike() -> float:
    """
    מושך את מדד ה־VIX דרך yfinance,
    ומחשב את השינוי באחוזים בין הסגירה של היום לאתמול.
    """
    try:
        vix = yf.Ticker('^VIX')
        hist = vix.history(period='2d')['Close'].dropna()
        if len(hist) < 2:
            return 0.0
        prev, last = hist.iloc[-2], hist.iloc[-1]
        return (last - prev) / prev * 100
    except Exception as e:
        print(f"⚠️ שגיאה ב־detect_vix_spike: {e}")
        return 0.0


if __name__ == "__main__":
    df = fetch_futures_news()
    score, explanation = calculate_futures_score(df)
    print(f"📈 Futures/VIX Score: {score} | {explanation}")

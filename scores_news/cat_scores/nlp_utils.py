# utils/nlp_utils.py

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import threading

analyzer = SentimentIntensityAnalyzer()
_sentiment_cache = {}
_lock = threading.Lock()

def fast_sentiment_score(text: str) -> float:
    """שימוש במנוע VADER — מחזיר סנטימנט בין -1 ל־1"""
    return analyzer.polarity_scores(text)["compound"]

def classify_sentiment(score: float) -> str:
    """תווית סנטימנט על פי סף"""
    if score > 0.3:
        return "positive"
    elif score < -0.3:
        return "negative"
    else:
        return "neutral"

def cached_sentiment(text: str) -> tuple:
    """בודק אם הטקסט כבר חושב — אם לא, שומר בזיכרון"""
    with _lock:
        if text in _sentiment_cache:
            return _sentiment_cache[text]

    score = fast_sentiment_score(text)
    label = classify_sentiment(score)

    with _lock:
        _sentiment_cache[text] = (score, label)

    return score, label

def analyze_articles(articles: list) -> list:
    """
    מקבל רשימת כתבות ומחזיר אותן עם שדות sentiment_score + sentiment_label
    """
    results = []

    def process(article):
        title = str(article.get('title') or '')
        summary = str(article.get('summary') or '')
        text = f"{title} {summary}"
        score, label = cached_sentiment(text)
        article['sentiment_score'] = int((score + 1) * 50)  # מיפוי ל־0–100
        article['sentiment_label'] = label
        results.append(article)

    threads = []
    for article in articles:
        t = threading.Thread(target=process, args=(article,))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    return results

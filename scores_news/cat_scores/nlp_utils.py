# utils/nlp_utils.py
import threading

_analyzer = None
_sentiment_cache = {}
_lock = threading.Lock()


def _get_analyzer():
    global _analyzer
    if _analyzer is None:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        _analyzer = SentimentIntensityAnalyzer()
    return _analyzer


def fast_sentiment_score(text: str) -> float:
    return _get_analyzer().polarity_scores(text)["compound"]


def classify_sentiment(score: float) -> str:
    if score > 0.3:
        return "positive"
    elif score < -0.3:
        return "negative"
    return "neutral"


def cached_sentiment(text: str) -> tuple:
    with _lock:
        if text in _sentiment_cache:
            return _sentiment_cache[text]
    score = fast_sentiment_score(text)
    label = classify_sentiment(score)
    with _lock:
        _sentiment_cache[text] = (score, label)
    return score, label


def analyze_articles(articles: list) -> list:
    results = []

    def process(article):
        title = str(article.get('title') or '')
        summary = str(article.get('summary') or '')
        score, label = cached_sentiment(f"{title} {summary}")
        article['sentiment_score'] = int((score + 1) * 50)
        article['sentiment_label'] = label
        results.append(article)

    threads = [threading.Thread(target=process, args=(a,)) for a in articles]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return results

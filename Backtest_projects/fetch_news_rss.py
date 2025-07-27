import feedparser
import requests
from bs4 import BeautifulSoup
from newspaper import Article
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer

def extract_text_manual(url: str) -> str:
    """Fallback: מוריד בעזרת requests ואז BeautifulSoup"""
    resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
    soup = BeautifulSoup(resp.content, "html.parser")
    paragraphs = soup.find_all("p")
    text = "\n".join(p.get_text() for p in paragraphs)
    return text

def clean_html(html: str) -> str:
    """מסיר תגי HTML ומחזיר טקסט"""
    return BeautifulSoup(html, "html.parser").get_text(separator=" ")

def fetch_and_summarize(rss_url: str, max_articles: int = 5, summary_sentences: int = 3):
    feed = feedparser.parse(rss_url)
    summaries = []

    for entry in feed.entries[:max_articles]:
        title     = entry.title
        published = entry.get("published", "Unknown date")
        url       = entry.link

        # 1) נסיון ראשון: Newspaper עם UA
        article = Article(url)
        article.config.browser_user_agent = "Mozilla/5.0"
        text = ""
        try:
            article.download()
            article.parse()
            text = article.text
        except Exception:
            # 2) fallback ראשון: extract ידני
            text = extract_text_manual(url)

        # 3) אם הטקסט קצר מדי או מכיל הודעת JS/ad-blocker
        if len(text) < 200 or "Please enable JS" in text:
            # השתמש בתקציר מה–RSS
            text = clean_html(entry.get("description", ""))

        # סיכום עם Sumy
        parser     = PlaintextParser.from_string(text, Tokenizer("english"))
        summarizer = LexRankSummarizer()
        sentences  = summarizer(parser.document, summary_sentences)
        summary    = " ".join(str(s) for s in sentences)

        summaries.append({
            "title":     title,
            "published": published,
            "description":   summary,
            "url":       url
        })

    return summaries

def link_news():
    RSS_FEED_URL = "https://www.investing.com/rss/news.rss"
    articles = fetch_and_summarize(RSS_FEED_URL, max_articles=5, summary_sentences=4)

    links = []

    for art in articles:
        # print(f"Title:   {art['title']}")
        # print(f"Date:    {art['published']}")
        # print(f"Description: {art['description']}")
        links.append(f"{art['url']}")
        # print(f"URL:     {art['url']}")
        # print("-" * 60)
    return links

if __name__ == "__main__":
    link_news()

    # for i in link_news():
    #     print(i)

import feedparser

def parse_rss(url):
    feed = feedparser.parse(url)
    return [{"title": entry.title, "link": entry.link, "summary": entry.summary} for entry in feed.entries]

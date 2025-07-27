import feedparser

test_url = "https://www.moneyworks4me.com/company/news/latest-stock-news-rss/commodity-news"
feed = feedparser.parse(test_url)
print(f"Items received: {len(feed.entries)}")

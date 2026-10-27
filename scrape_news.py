"""
scrape_news.py

Pulls recent financial headlines from free RSS feeds (no API key needed),
counts how often each of your chosen keywords appears in each headline,
and stores everything in the `news` table of market_news.db.

Install dependency first:
    pip install feedparser

Usage:
    python scrape_news.py

Note: RSS feeds only give you recent headlines (usually the last few
days), not deep history. For backfilling further back, you would use
a paid news archive API instead. For a class project, running this
daily/every-few-days and letting the database accumulate is the
realistic approach - same idea as the price scraper.
"""

import json
import sqlite3
from datetime import datetime, timezone

import feedparser

DB_PATH = "market_news.db"

# Free financial news RSS feeds. Add more as you find useful ones.
RSS_FEEDS = {
    "CNBC": "https://www.cnbc.com/id/100003114/device/rss/rss.html",
    "MarketWatch": "https://www.marketwatch.com/rss/topstories",
    "Investing.com": "https://www.investing.com/rss/news.rss",
}

# Keywords you want to track frequency of. Edit this list to match
# what you're hypothesizing might relate to market movement.
KEYWORDS = [
    "inflation", "interest rate", "fed", "rate cut", "rate hike",
    "recession", "gdp", "unemployment", "oil", "gold", "tariff",
    "war", "opec",
]


def count_keywords(text: str) -> dict:
    text_lower = text.lower()
    counts = {kw: text_lower.count(kw) for kw in KEYWORDS}
    # Only keep keywords that actually appeared, to keep stored JSON small
    return {kw: c for kw, c in counts.items() if c > 0}


def fetch_and_store(db_path: str = DB_PATH) -> None:
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    for source_name, feed_url in RSS_FEEDS.items():
        print(f"Fetching {source_name}...")
        feed = feedparser.parse(feed_url)

        if feed.bozo:
            # bozo = 1 means feedparser had trouble parsing this feed
            print(f"  -> warning: could not cleanly parse {source_name}, "
                  f"skipping ({feed.bozo_exception})")
            continue

        rows_written = 0
        for entry in feed.entries:
            headline = entry.get("title", "").strip()
            if not headline:
                continue

            # Use the entry's own publish date if available, else today
            if "published_parsed" in entry and entry.published_parsed:
                pub_date = datetime(*entry.published_parsed[:6]).strftime("%Y-%m-%d")
            else:
                pub_date = today_str

            kw_counts = count_keywords(headline)

            cur.execute(
                """
                INSERT INTO news (date, source, headline, keyword_counts)
                VALUES (?, ?, ?, ?)
                """,
                (pub_date, source_name, headline, json.dumps(kw_counts)),
            )
            rows_written += 1

        print(f"  -> wrote {rows_written} headlines from {source_name}")

    conn.commit()
    conn.close()


if __name__ == "__main__":
    fetch_and_store()

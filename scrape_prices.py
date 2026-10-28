"""
scrape_prices.py

Downloads daily closing prices for a list of tickers using yfinance,
computes the daily percentage return, and stores everything in the
`prices` table of market_news.db.

Install dependency first:
    pip install yfinance

Usage:
    python scrape_prices.py
"""

import sqlite3
import pandas as pd
import yfinance as yf

DB_PATH = "market_news.db"

# Add/remove tickers here.
# Stocks use normal symbols; commodities and currencies use Yahoo's
# futures/FX notation.
TICKERS = {
    "AAPL": "AAPL",       # Apple (example stock)
    "GOLD": "GC=F",       # Gold futures
    "OIL": "CL=F",        # Crude oil futures
    "USDINR": "USDINR=X",  # USD/INR exchange rate
    "RELIANCE.NS": "RELIANCE.NS",  # Reliance Industries (NSE) - model.py TARGET
}

# How much history to pull on each run. "1mo", "3mo", "1y", "5y", "max" etc.
PERIOD = "6mo"


def fetch_and_store(period: str = PERIOD, db_path: str = DB_PATH) -> None:
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    for label, symbol in TICKERS.items():
        print(f"Fetching {label} ({symbol})...")

        # download() returns a pandas DataFrame indexed by date
        data = yf.download(symbol, period=period, progress=False)

        if data.empty:
            print(f"  -> no data returned for {symbol}, skipping")
            continue

        # Newer yfinance versions sometimes return columns as a
        # MultiIndex (e.g. ("Close", "AAPL")) even for a single ticker.
        # Flatten it to plain column names so data["Close"] is always
        # a single Series, not a nested DataFrame.
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.droplevel(1)

        # Daily percentage return: (today - yesterday) / yesterday
        data["Return"] = data["Close"].pct_change()

        rows_written = 0
        for date, row in data.iterrows():
            date_str = date.strftime("%Y-%m-%d")
            close_val = float(row["Close"])
            return_val = row["Return"]
            return_val = None if pd.isna(return_val) else float(return_val)

            # INSERT OR REPLACE = safe to re-run; updates existing rows
            # instead of throwing a duplicate-key error.
            cur.execute(
                """
                INSERT OR REPLACE INTO prices (date, ticker, close, return)
                VALUES (?, ?, ?, ?)
                """,
                (date_str, label, close_val, return_val),
            )
            rows_written += 1

        print(f"  -> wrote {rows_written} rows for {label}")

    conn.commit()
    conn.close()


if __name__ == "__main__":
    fetch_and_store()

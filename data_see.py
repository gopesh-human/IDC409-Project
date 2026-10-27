import sqlite3
import pandas as pd

conn = sqlite3.connect("market_news.db")
prices_df = pd.read_sql("SELECT * FROM prices", conn)
news_df = pd.read_sql("SELECT * FROM news", conn)
conn.close()

print(prices_df.head())
print(prices_df.shape)   # (rows, columns) - quick sanity check on row count
print(news_df.head())

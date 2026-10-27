"""
db_setup.py

Run this ONCE to create the SQLite database file and its two tables:
    - prices : daily closing price + return, per ticker
    - news   : daily headlines + keyword counts

Usage:
    python db_setup.py
"""

import sqlite3

DB_PATH = "market_news.db"


def create_tables(db_path: str = DB_PATH) -> None:
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS prices (
            date    TEXT NOT NULL,
            ticker  TEXT NOT NULL,
            close   REAL NOT NULL,
            return  REAL,
            PRIMARY KEY (date, ticker)
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS news (
            id             INTEGER PRIMARY KEY AUTOINCREMENT,
            date           TEXT NOT NULL,
            source         TEXT NOT NULL,
            headline       TEXT NOT NULL,
            keyword_counts TEXT
        )
    """)

    conn.commit()
    conn.close()
    print(f"Database ready at: {db_path}")


if __name__ == "__main__":
    create_tables()

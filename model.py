"""
model.py

Reads market_news.db, builds a daily feature table, and trains:
  - Regression : predict next-day return
  - Classification : predict next-day direction (Up/Down)
with a time-based train/test split and naive baselines.

Usage:
    pip install pandas numpy scikit-learn
    python model.py

Also works cell-by-cell in a notebook (copy sections under the # ---- headers).
"""

import json
import sqlite3

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (confusion_matrix, f1_score, mean_absolute_error,
                             mean_squared_error, r2_score)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

DB_PATH = "market_news.db"
TARGET = "RELIANCE.NS"      # the ticker you want to predict; must exist in `prices`
TEST_FRACTION = 0.2         # last 20% of dates = test set
SEED = 42


# ---- 1. Load ---------------------------------------------------------------
conn = sqlite3.connect(DB_PATH)
prices = pd.read_sql("SELECT date, ticker, close FROM prices", conn)
news = pd.read_sql("SELECT * FROM news", conn)
conn.close()

# Re-running the news scraper can insert the same headline more than once.
news = news.drop_duplicates(subset=["date", "source", "headline"]).reset_index(drop=True)

prices["date"] = pd.to_datetime(prices["date"])
news["date"] = pd.to_datetime(news["date"])


# ---- 2. Price features -----------------------------------------------------
# Wide table: one column per ticker. Different markets close on different days
# (holidays), so forward-fill closes before computing returns.
close = prices.pivot(index="date", columns="ticker", values="close").sort_index()

if TARGET not in close.columns:
    raise SystemExit(f"{TARGET} not found in prices. Available: {list(close.columns)}")

# Keep only days the target actually traded, then forward-fill the OTHER tickers.
# (Otherwise holidays become fake 0% returns, which get labelled "Down".)
close = close[close[TARGET].notna()].ffill()
ret = close.pct_change()

feat = pd.DataFrame(index=close.index)
for t in close.columns:
    feat[f"{t}_ret_lag0"] = ret[t]                       # today's return
    feat[f"{t}_ret_lag1"] = ret[t].shift(1)
    feat[f"{t}_ret_mean5"] = ret[t].rolling(5).mean()
    feat[f"{t}_vol5"] = ret[t].rolling(5).std()
    feat[f"{t}_ma5_over_ma10"] = (
        close[t].rolling(5).mean() / close[t].rolling(10).mean() - 1
    )


# ---- 3. News features (one row per day) -----------------------------------
def parse_counts(s):
    try:
        return json.loads(s) if s else {}
    except (TypeError, ValueError):
        return {}

kw = pd.DataFrame(news["keyword_counts"].map(parse_counts).tolist()).fillna(0)
kw.index = news.index
kw.columns = [f"kw_{c.replace(' ', '_')}" for c in kw.columns]

news_f = pd.concat([news[["date"]], kw], axis=1)
news_f["n_headlines"] = 1
if "sentiment" in news.columns:
    news_f["sentiment"] = news["sentiment"]

agg = {c: "sum" for c in news_f.columns if c.startswith("kw_")}
agg["n_headlines"] = "sum"
if "sentiment" in news_f.columns:
    agg["sentiment"] = "mean"
daily_news = news_f.groupby("date").agg(agg)

# News dated on a non-trading day (weekend/holiday) is attached to the last trading day before it.
pos = close.index.searchsorted(daily_news.index, side="right") - 1
daily_news = daily_news[pos >= 0]
daily_news.index = close.index[pos[pos >= 0]]
daily_news = daily_news.groupby(level=0).agg(agg)

df = feat.join(daily_news, how="left")
news_cols = list(daily_news.columns)
df[news_cols] = df[news_cols].fillna(0)
df["has_news"] = (df["n_headlines"] > 0).astype(int)


# ---- 4. Targets ------------------------------------------------------------
df["y_ret"] = ret[TARGET].shift(-1)            # next-day return
df["y_dir"] = (df["y_ret"] > 0).astype(int)    # 1 = Up, 0 = Down
df["today_ret"] = ret[TARGET]                  # used for the naive baseline

df = df.dropna(subset=["y_ret"] + [c for c in df.columns if c.startswith(TARGET)])
df = df.replace([np.inf, -np.inf], np.nan).dropna()

df.to_csv("clean_dataset.csv")                 # cleaned CSV deliverable
print(f"Dataset: {df.shape[0]} rows, {df.shape[1]} cols "
      f"({df['has_news'].sum()} days have news)")
if df["has_news"].mean() < 0.5:
    print("WARNING: most days have no news, so the 'price+news' comparison is not "
          "meaningful yet. Keep running scrape_news.py daily to build up history.")


# ---- 5. Time-based split ---------------------------------------------------
split = int(len(df) * (1 - TEST_FRACTION))
train, test = df.iloc[:split], df.iloc[split:]
print(f"Train: {train.index[0].date()} -> {train.index[-1].date()} ({len(train)})")
print(f"Test : {test.index[0].date()} -> {test.index[-1].date()} ({len(test)})")

price_cols = [c for c in feat.columns]
FEATURE_SETS = {
    "price_only": price_cols,
    "price+news": price_cols + news_cols,
}


# ---- 6. Regression: next-day return ---------------------------------------
def reg_scores(y_true, y_pred):
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "R2": r2_score(y_true, y_pred),
    }

reg_rows = []
y_tr, y_te = train["y_ret"], test["y_ret"]

reg_rows.append({"model": "baseline: tomorrow=today", "features": "-",
                 **reg_scores(y_te, test["today_ret"])})
reg_rows.append({"model": "baseline: predict train mean", "features": "-",
                 **reg_scores(y_te, np.full(len(y_te), y_tr.mean()))})

regressors = {
    "LinearRegression": make_pipeline(StandardScaler(), LinearRegression()),
    "RandomForest": RandomForestRegressor(
        n_estimators=300, max_depth=5, min_samples_leaf=5,
        random_state=SEED, n_jobs=-1),
}
for fs_name, cols in FEATURE_SETS.items():
    for m_name, model in regressors.items():
        model.fit(train[cols], y_tr)
        reg_rows.append({"model": m_name, "features": fs_name,
                         **reg_scores(y_te, model.predict(test[cols]))})

reg_results = pd.DataFrame(reg_rows)
print("\n=== Regression (next-day return) ===")
print(reg_results.round(5).to_string(index=False))


# ---- 7. Classification: next-day direction --------------------------------
yc_tr, yc_te = train["y_dir"], test["y_dir"]
clf_rows = []
conf_mats = {}

majority = int(yc_tr.mode()[0])
pred = np.full(len(yc_te), majority)
clf_rows.append({"model": "baseline: majority class", "features": "-",
                 "accuracy": (pred == yc_te).mean(), "F1": f1_score(yc_te, pred, zero_division=0)})
pred = (test["today_ret"] > 0).astype(int)
clf_rows.append({"model": "baseline: tomorrow=today", "features": "-",
                 "accuracy": (pred == yc_te).mean(), "F1": f1_score(yc_te, pred, zero_division=0)})

classifiers = {
    "LogisticRegression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
    "DecisionTree": DecisionTreeClassifier(max_depth=4, min_samples_leaf=10, random_state=SEED),
    "RandomForest": RandomForestClassifier(
        n_estimators=300, max_depth=5, min_samples_leaf=5,
        random_state=SEED, n_jobs=-1),
    "SVM": make_pipeline(StandardScaler(), SVC(kernel="rbf", C=1.0)),
}
for fs_name, cols in FEATURE_SETS.items():
    for m_name, model in classifiers.items():
        model.fit(train[cols], yc_tr)
        pred = model.predict(test[cols])
        clf_rows.append({"model": m_name, "features": fs_name,
                         "accuracy": (pred == yc_te).mean(),
                         "F1": f1_score(yc_te, pred, zero_division=0)})
        conf_mats[(m_name, fs_name)] = confusion_matrix(yc_te, pred, labels=[0, 1])

clf_results = pd.DataFrame(clf_rows)
print("\n=== Classification (Up=1 / Down=0) ===")
print(clf_results.round(4).to_string(index=False))

print("\nConfusion matrices (rows = actual Down/Up, cols = predicted Down/Up):")
for (m_name, fs_name), cm in conf_mats.items():
    print(f"\n{m_name} [{fs_name}]\n{cm}")


# ---- 8. Feature importance (does news matter?) ----------------------------
rf = classifiers["RandomForest"]
rf.fit(train[FEATURE_SETS["price+news"]], yc_tr)
imp = pd.Series(rf.feature_importances_, index=FEATURE_SETS["price+news"])
print("\nTop 10 RandomForest features (price+news):")
print(imp.sort_values(ascending=False).head(10).round(4))

reg_results.to_csv("results_regression.csv", index=False)
clf_results.to_csv("results_classification.csv", index=False)
print("\nSaved: clean_dataset.csv, results_regression.csv, results_classification.csv")

"""Daily sentiment aggregation + technical features. Row t only uses information known at close of day t."""
import numpy as np
import pandas as pd


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    d = close.diff()
    up, dn = d.clip(lower=0).rolling(n).mean(), (-d.clip(upper=0)).rolling(n).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def build_features(prices: pd.DataFrame, news: pd.DataFrame) -> pd.DataFrame:
    df = pd.DataFrame(index=prices.index)
    df["close"] = prices["Close"]
    df["ret"] = df["close"].pct_change()
    df["vol5"] = df["ret"].rolling(5).std()
    df["rsi"] = rsi(df["close"])
    df["ma_gap"] = df["close"] / df["close"].rolling(10).mean() - 1
    df["vol_chg"] = prices["Volume"].pct_change()
    g = news.groupby("day")["score"]
    s = pd.DataFrame({"sent_mean": g.mean(), "sent_std": g.std(), "n_articles": g.size()})
    df = df.join(s)
    df["n_articles"] = df["n_articles"].fillna(0)
    df["sent_mean"] = df["sent_mean"].fillna(0)
    df["sent_std"] = df["sent_std"].fillna(0)
    df["sent_3d"] = df["sent_mean"].rolling(3).mean()
    df["sent_5d"] = df["sent_mean"].rolling(5).mean()
    # Targets: next-day return/direction (shift(-1) -> features at t predict t+1)
    df["next_ret"] = df["ret"].shift(-1)
    df["next_up"] = (df["next_ret"] > 0).astype(int)
    return df

"""Average abnormal return around extreme-sentiment days."""
import pandas as pd


def event_study(df: pd.DataFrame, q: float = 0.1, window=range(-2, 4)) -> pd.DataFrame:
    d = df[df["n_articles"] > 0]
    ab = df["ret"] - df["ret"].mean()
    hi, lo = d["sent_mean"].quantile(1 - q), d["sent_mean"].quantile(q)
    out = {}
    for name, days in {"positive events": d.index[d["sent_mean"] >= hi], "negative events": d.index[d["sent_mean"] <= lo]}.items():
        pos = [df.index.get_loc(x) for x in days]
        out[name] = {k: pd.Series([ab.iloc[p + k] for p in pos if 0 <= p + k < len(df)]).mean() for k in window}
    return pd.DataFrame(out).rename_axis("day offset")

"""Long/flat backtest from walk-forward predictions, with transaction costs."""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .model import PRICE, SENT


def _stats(r: pd.Series) -> dict:
    eq = (1 + r).cumprod()
    return dict(total_return=eq.iloc[-1] - 1, sharpe=np.sqrt(252) * r.mean() / r.std() if r.std() > 0 else np.nan,
                max_drawdown=(eq / eq.cummax() - 1).min())


def backtest(df: pd.DataFrame, use_sentiment=True, cost=0.001, splits=5) -> pd.DataFrame:
    f = PRICE + (SENT if use_sentiment else [])
    d = df[f + ["next_up", "next_ret"]].replace([np.inf, -np.inf], np.nan).dropna()
    pos = pd.Series(0.0, index=d.index)
    for tr, te in TimeSeriesSplit(n_splits=splits).split(d):
        m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=500, class_weight="balanced"))
        m.fit(d.iloc[tr][f], d.iloc[tr]["next_up"])
        pos.iloc[te] = (m.predict_proba(d.iloc[te][f])[:, 1] > 0.5).astype(float)
    test = d.index[d.index >= d.index[TimeSeriesSplit(n_splits=splits).split(d).__next__()[1][0]]]
    pos, nr = pos.loc[test], d.loc[test, "next_ret"]
    strat = pos * nr - cost * pos.diff().abs().fillna(pos.abs())
    return pd.DataFrame({"Strategy": _stats(strat), "Buy & hold": _stats(nr)}).T

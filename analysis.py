"""Lead-lag correlations and Granger causality."""
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from statsmodels.tsa.stattools import grangercausalitytests


def lag_correlations(df: pd.DataFrame, max_lag: int = 5) -> pd.DataFrame:
    """corr(sentiment_t, return_{t+k}). k>0: sentiment leads; k<0: returns lead."""
    rows = []
    for k in range(-max_lag, max_lag + 1):
        d = pd.concat([df["sent_mean"], df["ret"].shift(-k)], axis=1).dropna()
        if len(d) < 10:
            continue
        pr, pp = pearsonr(d.iloc[:, 0], d.iloc[:, 1])
        sr, sp = spearmanr(d.iloc[:, 0], d.iloc[:, 1])
        rows.append(dict(lag=k, pearson=pr, p_pearson=pp, spearman=sr, p_spearman=sp, n=len(d)))
    return pd.DataFrame(rows)


def granger(df: pd.DataFrame, max_lag: int = 3) -> dict:
    """Min p-value across lags, both directions."""
    d = df[["ret", "sent_mean"]].dropna()
    out = {}
    for name, cols in {"sentiment -> returns": ["ret", "sent_mean"],
                       "returns -> sentiment": ["sent_mean", "ret"]}.items():
        try:
            res = grangercausalitytests(d[cols], max_lag, verbose=False)
            out[name] = min(r[0]["ssr_ftest"][1] for r in res.values())
        except Exception:
            out[name] = float("nan")
    return out

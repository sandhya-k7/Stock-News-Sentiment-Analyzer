"""Walk-forward next-day direction models: price-only vs price+sentiment vs naive baselines."""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

PRICE = ["ret", "vol5", "rsi", "ma_gap", "vol_chg"]
SENT = ["sent_mean", "sent_std", "n_articles", "sent_3d", "sent_5d"]


def _cv(df, feats, mk, splits=5):
    d = df[feats + ["next_up"]].replace([np.inf, -np.inf], np.nan).dropna()
    X, y = d[feats].values, d["next_up"].values
    pred, prob, truth = [], [], []
    for tr, te in TimeSeriesSplit(n_splits=splits).split(X):  # chronological, no shuffling
        m = mk().fit(X[tr], y[tr])
        pred += list(m.predict(X[te]))
        prob += list(m.predict_proba(X[te])[:, 1])
        truth += list(y[te])
    t = np.array(truth)
    auc = roc_auc_score(t, prob) if len(set(t)) > 1 else np.nan
    return dict(accuracy=accuracy_score(t, pred), balanced_acc=balanced_accuracy_score(t, pred), auc=auc, n_test=len(t))


def compare_models(df: pd.DataFrame) -> pd.DataFrame:
    lr = lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=500, class_weight="balanced"))
    rf = lambda: RandomForestClassifier(200, max_depth=4, min_samples_leaf=5, random_state=42, class_weight="balanced")
    rows = {f"{n} | {fs}": _cv(df, f, mk)
            for n, mk in [("LogReg", lr), ("RandForest", rf)]
            for fs, f in [("price only", PRICE), ("price + sentiment", PRICE + SENT)]}
    d = df.dropna(subset=["next_up"])
    rows["Baseline | majority class"] = dict(accuracy=max(d["next_up"].mean(), 1 - d["next_up"].mean()),
                                             balanced_acc=0.5, auc=0.5, n_test=len(d))
    return pd.DataFrame(rows).T


def shap_importance(df: pd.DataFrame) -> pd.Series:
    """Mean |SHAP| per feature for a random forest fit on the chronologically first 80% of rows."""
    import shap
    feats = PRICE + SENT
    d = df[feats + ["next_up"]].replace([np.inf, -np.inf], np.nan).dropna()
    tr = d.iloc[: int(len(d) * 0.8)]
    rf = RandomForestClassifier(200, max_depth=4, min_samples_leaf=5, random_state=42).fit(tr[feats], tr["next_up"])
    sv = shap.TreeExplainer(rf).shap_values(tr[feats])
    sv = sv[1] if isinstance(sv, list) else (sv[:, :, 1] if np.ndim(sv) == 3 else sv)
    return pd.Series(np.abs(sv).mean(0), index=feats).sort_values()

"""Compare sentiment engines on labeled financial text."""
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from .sentiment import label

GOLD = {0: "negative", 1: "positive", 2: "neutral"}
LABELS = ["negative", "neutral", "positive"]


def evaluate(engines: dict, n: int = 1000):
    from datasets import load_dataset
    v = load_dataset("zeroshot/twitter-financial-news-sentiment", split="validation").shuffle(seed=42).select(range(n))
    y = [GOLD[x] for x in v["label"]]
    rows, cms = {}, {}
    for name, eng in engines.items():
        pred = [label(s) for s in eng.score(v["text"])]
        rows[name] = dict(accuracy=accuracy_score(y, pred), macro_f1=f1_score(y, pred, average="macro"))
        cms[name] = pd.DataFrame(confusion_matrix(y, pred, labels=LABELS), index=LABELS, columns=LABELS)
    return pd.DataFrame(rows).T, cms

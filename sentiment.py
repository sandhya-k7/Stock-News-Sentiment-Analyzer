"""Sentiment engines behind one interface: score(texts) -> list of floats in [-1, 1]."""
import pandas as pd


class Vader:
    name = "VADER"

    def __init__(self):
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        self.a = SentimentIntensityAnalyzer()

    def score(self, texts):
        return [self.a.polarity_scores(t)["compound"] for t in texts]


class FinBert:
    """ProsusAI/finbert (or a path to your fine-tuned checkpoint). Score = P(pos) - P(neg)."""
    name = "FinBERT"

    def __init__(self, model="ProsusAI/finbert"):
        import torch
        from transformers import pipeline
        self.p = pipeline("text-classification", model=model, top_k=None, truncation=True,
                          max_length=512, device=0 if torch.cuda.is_available() else -1)

    def score(self, texts):
        out = self.p(list(texts), batch_size=16)
        return [{x["label"].lower(): x["score"] for x in o}.get("positive", 0)
                - {x["label"].lower(): x["score"] for x in o}.get("negative", 0) for o in out]


def label(s: float, thr: float = 0.15) -> str:
    return "positive" if s > thr else "negative" if s < -thr else "neutral"


def score_articles(news: pd.DataFrame, engine) -> pd.DataFrame:
    news = news.copy()
    news["score"] = engine.score(news["text"].tolist())
    news["label"] = news["score"].map(label)
    return news

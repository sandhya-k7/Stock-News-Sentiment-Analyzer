"""Fine-tune FinBERT. Run: python -m src.sentiment_finetune   (use a GPU / Colab)
Data: zeroshot/twitter-financial-news-sentiment (0 bearish, 1 bullish, 2 neutral)."""
import numpy as np
import torch
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
from transformers import (AutoModelForSequenceClassification, AutoTokenizer, EarlyStoppingCallback,
                          Trainer, TrainingArguments, set_seed)

BASE, OUT = "ProsusAI/finbert", "models/finbert-ft"
REMAP = {0: 1, 1: 0, 2: 2}  # dataset -> FinBERT ids (0 positive, 1 negative, 2 neutral)


class WeightedTrainer(Trainer):
    def __init__(self, *a, class_weights=None, **k):
        super().__init__(*a, **k)
        self.cw = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kw):
        labels = inputs.pop("labels")
        out = model(**inputs)
        w = self.cw.to(out.logits.device)
        loss = torch.nn.functional.cross_entropy(out.logits, labels, weight=w)
        return (loss, out) if return_outputs else loss


def metrics(p):
    pred = p.predictions.argmax(-1)
    return {"accuracy": accuracy_score(p.label_ids, pred), "macro_f1": f1_score(p.label_ids, pred, average="macro")}


def main():
    set_seed(42)
    tok = AutoTokenizer.from_pretrained(BASE)
    model = AutoModelForSequenceClassification.from_pretrained(BASE)
    ds = load_dataset("zeroshot/twitter-financial-news-sentiment")
    ds = ds.map(lambda b: {**tok(b["text"], truncation=True, max_length=128),
                           "labels": [REMAP[x] for x in b["label"]]}, batched=True)
    ds = ds.remove_columns(["text", "label"])
    y = np.array(ds["train"]["labels"])
    cw = torch.tensor(compute_class_weight("balanced", classes=np.arange(3), y=y), dtype=torch.float)
    args = TrainingArguments(OUT, num_train_epochs=5, per_device_train_batch_size=16, learning_rate=2e-5,
                             eval_strategy="epoch", save_strategy="epoch", load_best_model_at_end=True,
                             metric_for_best_model="macro_f1", save_total_limit=1, seed=42, report_to="none")
    t = WeightedTrainer(model=model, args=args, train_dataset=ds["train"], eval_dataset=ds["validation"],
                        tokenizer=tok, compute_metrics=metrics, class_weights=cw,
                        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)])
    t.train()
    t.save_model(OUT)
    tok.save_pretrained(OUT)
    print(t.evaluate())


if __name__ == "__main__":
    main()

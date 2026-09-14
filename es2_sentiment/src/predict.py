"""Inferenza: classifica nuove frasi come 'positive' o 'negative'.

Esempio:
    python src/predict.py "I really loved this movie" "What a waste of time"
    echo "great product" | python src/predict.py
"""
from __future__ import annotations

import sys

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from config import Config

DEFAULT_MODEL_DIR = "outputs/best_model"


def load(model_dir: str, device: str):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device)
    model.eval()
    return tokenizer, model


@torch.no_grad()
def predict(sentences, tokenizer, model, device, max_length=128):
    enc = tokenizer(
        sentences,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=max_length,
    ).to(device)
    logits = model(**enc).logits
    probs = torch.softmax(logits, dim=-1)
    conf, pred = probs.max(dim=-1)
    id2label = model.config.id2label
    return [(s, id2label[int(p)], float(c)) for s, p, c in zip(sentences, pred, conf)]


def main():
    cfg = Config()
    sentences = sys.argv[1:]
    if not sentences:
        sentences = [line.strip() for line in sys.stdin if line.strip()]
    if not sentences:
        print("Nessuna frase fornita.")
        return

    tokenizer, model = load(DEFAULT_MODEL_DIR, cfg.device)
    for text, label, conf in predict(sentences, tokenizer, model, cfg.device, cfg.max_length):
        print(f"[{label:>8}  {conf:5.1%}]  {text}")


if __name__ == "__main__":
    main()

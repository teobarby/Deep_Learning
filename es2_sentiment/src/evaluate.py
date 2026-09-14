"""Valutazione di un modello gia' addestrato sul set di validazione.

Stampa il classification report e salva la matrice di confusione come immagine.

Esempio:
    python src/evaluate.py
    python src/evaluate.py --model_dir outputs/best_model
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
)

from config import Config
from data import build_datasets


@torch.no_grad()
def collect_predictions(model, loader, device):
    y_true, y_pred = [], []
    for batch in loader:
        labels = batch.pop("labels")
        batch = {k: v.to(device) for k, v in batch.items()}
        logits = model(**batch).logits
        preds = logits.argmax(dim=-1).cpu().numpy()
        y_pred.extend(preds.tolist())
        y_true.extend(labels.numpy().tolist())
    return np.array(y_true), np.array(y_pred)


def main():
    cfg = Config()
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir", default="outputs/best_model")
    p.add_argument("--dataset", default=cfg.dataset, choices=["sst2", "csv"])
    p.add_argument("--csv_path", default=cfg.csv_path)
    args = p.parse_args()
    cfg.dataset = args.dataset
    cfg.csv_path = args.csv_path
    cfg.model_name = args.model_dir  # tokenizza con lo stesso modello/tokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir).to(cfg.device)
    model.eval()

    tokenized, _ = build_datasets(cfg)
    collator = DataCollatorWithPadding(tokenizer=tokenizer)
    loader = DataLoader(tokenized["validation"], batch_size=cfg.batch_size, collate_fn=collator)

    y_true, y_pred = collect_predictions(model, loader, cfg.device)
    target_names = [cfg.id2label[0], cfg.id2label[1]]
    print(classification_report(y_true, y_pred, target_names=target_names, digits=4))

    out = Path(cfg.output_dir) / "confusion_matrix.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=target_names)
    disp.plot(cmap="Blues", colorbar=False)
    disp.figure_.savefig(out, dpi=150, bbox_inches="tight")
    print(f"[salvato] matrice di confusione in {out}")


if __name__ == "__main__":
    main()

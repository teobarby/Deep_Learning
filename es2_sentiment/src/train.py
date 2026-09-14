"""Fine-tuning del transformer per sentiment analysis.

Esempio d'uso:
    python src/train.py                       # SST-2, default
    python src/train.py --epochs 3 --batch_size 32
    python src/train.py --dataset csv --csv_path data/kaggle.csv \
        --text_column review --label_column sentiment
"""
from __future__ import annotations

import argparse

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import (
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)

from config import Config
from data import build_datasets
from model import build_model


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds),
        "precision": precision_score(labels, preds),
        "recall": recall_score(labels, preds),
    }


def parse_args() -> Config:
    cfg = Config()
    p = argparse.ArgumentParser(description="Fine-tuning sentiment analysis (transformer)")
    p.add_argument("--model_name", default=cfg.model_name)
    p.add_argument("--dataset", default=cfg.dataset, choices=["sst2", "csv"])
    p.add_argument("--csv_path", default=cfg.csv_path)
    p.add_argument("--text_column", default=cfg.text_column)
    p.add_argument("--label_column", default=cfg.label_column)
    p.add_argument("--epochs", type=int, default=cfg.epochs)
    p.add_argument("--batch_size", type=int, default=cfg.batch_size)
    p.add_argument("--learning_rate", type=float, default=cfg.learning_rate)
    p.add_argument("--max_length", type=int, default=cfg.max_length)
    p.add_argument("--max_train_samples", type=int, default=cfg.max_train_samples)
    p.add_argument("--max_eval_samples", type=int, default=cfg.max_eval_samples)
    p.add_argument("--resume", action="store_true",
                   help="riprende dall'ultimo checkpoint in output_dir se presente")
    args = p.parse_args()

    cfg.resume = args.resume
    for k, v in vars(args).items():
        if k != "resume":
            setattr(cfg, k, v)
    return cfg


def main():
    cfg = parse_args()
    set_seed(cfg.seed)
    print(f"[device] {cfg.device}  |  modello: {cfg.model_name}  |  dataset: {cfg.dataset}")

    tokenized, tokenizer = build_datasets(cfg)
    print(f"[dati] train={len(tokenized['train'])}  val={len(tokenized['validation'])}")

    model = build_model(cfg)
    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    use_fp16 = cfg.device == "cuda"
    training_args = TrainingArguments(
        output_dir=cfg.output_dir,
        eval_strategy="steps",
        save_strategy="steps",
        eval_steps=250,
        save_steps=250,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        learning_rate=cfg.learning_rate,
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size,
        num_train_epochs=cfg.epochs,
        weight_decay=cfg.weight_decay,
        warmup_ratio=cfg.warmup_ratio,
        logging_steps=50,
        seed=cfg.seed,
        fp16=use_fp16,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        processing_class=tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics,
    )

    resume = getattr(cfg, "resume", False)
    trainer.train(resume_from_checkpoint=resume)
    metrics = trainer.evaluate()
    print("[valutazione finale]", {k: round(v, 4) for k, v in metrics.items() if isinstance(v, float)})

    final_dir = f"{cfg.output_dir}/best_model"
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"[salvato] modello in {final_dir}")


if __name__ == "__main__":
    main()

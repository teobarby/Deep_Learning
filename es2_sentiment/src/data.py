"""Caricamento e tokenizzazione del dataset di sentiment analysis.

Supporta due sorgenti:
  - "sst2": dataset GLUE/SST-2 scaricato da HuggingFace (frasi pos/neg).
  - "csv" : file locale (es. dataset Kaggle) con una colonna di testo e una
            colonna di label binaria (0/1 oppure 'negative'/'positive').

Tre split con ruoli distinti:
  - train:      addestramento;
  - validation: scelta del checkpoint migliore durante il training;
  - test:       valutazione finale, usata una sola volta. Se lo stesso set servisse
                sia a scegliere il modello sia a misurarlo, il risultato sarebbe
                ottimistico.
Per SST-2 il test ufficiale non ha label pubbliche: il validation ufficiale
(872 frasi) diventa il test, e il validation si ricava da una parte del train.
"""
from __future__ import annotations

import pandas as pd
from datasets import Dataset, DatasetDict, load_dataset
from transformers import AutoTokenizer

from config import Config


def _subsample(ds: Dataset, n: int | None, seed: int) -> Dataset:
    if n is not None and n < len(ds):
        ds = ds.shuffle(seed=seed).select(range(n))
    return ds


def _normalize_labels(ds: Dataset, label_column: str) -> Dataset:
    """Porta le label a interi 0/1 anche se nel CSV sono stringhe."""
    sample = ds[label_column][0]
    if isinstance(sample, str):
        mapping = {"negative": 0, "neg": 0, "0": 0,
                   "positive": 1, "pos": 1, "1": 1}

        def _map(ex):
            ex[label_column] = mapping[str(ex[label_column]).strip().lower()]
            return ex

        ds = ds.map(_map)
    return ds


def load_raw(cfg: Config) -> DatasetDict:
    """Restituisce un DatasetDict con split 'train', 'validation' e 'test'."""
    if cfg.dataset == "sst2":
        # Repo moderno con split gia' pronti (idx, sentence, label).
        raw = load_dataset("stanfordnlp/sst2")
        # Il test ufficiale ha label -1 (non pubbliche): il validation ufficiale
        # fa da test, e un nuovo validation si ricava dal train.
        # Nota: il train di SST-2 contiene anche sotto-frasi delle stesse frasi,
        # quindi il validation ricavato qui somiglia al train e le sue metriche
        # sono ottimistiche. Va bene per scegliere il checkpoint; il risultato da
        # riportare e' quello sul test (frasi complete, mai viste in training).
        split = raw["train"].train_test_split(test_size=cfg.val_size, seed=cfg.seed)
        return DatasetDict(train=split["train"], validation=split["test"],
                           test=raw["validation"])

    if cfg.dataset == "csv":
        if not cfg.csv_path:
            raise ValueError("cfg.csv_path deve essere impostato per dataset='csv'.")
        df = pd.read_csv(cfg.csv_path)
        ds = Dataset.from_pandas(df, preserve_index=False)
        ds = _normalize_labels(ds, cfg.label_column)
        # 80% train, 10% validation, 10% test
        first = ds.train_test_split(test_size=0.2, seed=cfg.seed)
        second = first["test"].train_test_split(test_size=0.5, seed=cfg.seed)
        return DatasetDict(train=first["train"], validation=second["train"],
                           test=second["test"])

    raise ValueError(f"Dataset non riconosciuto: {cfg.dataset}")


def build_datasets(cfg: Config):
    """Carica, sottocampiona e tokenizza il dataset.

    Ritorna: (tokenized_datasets, tokenizer)
    """
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    raw = load_raw(cfg)

    raw["train"] = _subsample(raw["train"], cfg.max_train_samples, cfg.seed)
    raw["validation"] = _subsample(raw["validation"], cfg.max_eval_samples, cfg.seed)
    # il test non viene sottocampionato: e' il riferimento del risultato finale
    raw["test"] = _subsample(raw["test"], cfg.max_test_samples, cfg.seed)

    text_col = cfg.text_column
    label_col = cfg.label_column

    def tokenize(batch):
        enc = tokenizer(
            batch[text_col],
            truncation=True,
            max_length=cfg.max_length,
        )
        enc["labels"] = batch[label_col]
        return enc

    keep = raw["train"].column_names
    tokenized = raw.map(tokenize, batched=True, remove_columns=keep)
    return tokenized, tokenizer

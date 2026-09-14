"""Costruzione del modello: transformer encoder-only per classificazione di sequenze.

Si parte da un modello pre-addestrato della famiglia BERT (default: DistilBERT)
e si aggiunge una testa di classificazione (fine-tuning). Questo corrisponde
all'uso di un encoder-only transformer descritto nelle slide (BERT).
"""
from transformers import AutoModelForSequenceClassification

from config import Config


def build_model(cfg: Config):
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model_name,
        num_labels=cfg.num_labels,
        id2label=cfg.id2label,
        label2id=cfg.label2id,
    )
    return model

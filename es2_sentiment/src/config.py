"""Configurazione centrale per l'esperimento di sentiment analysis.

Architettura: transformer encoder-only (famiglia BERT), allineato alle slide
14.Transformers (BERT, encoder bidirezionale). Il task e' una classificazione
binaria di sequenze testuali in 'positive' / 'negative'.
"""
from dataclasses import dataclass, field
from pathlib import Path

import torch

# Directory radice del progetto (es2_sentiment/)
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"


def get_device() -> str:
    """Sceglie il miglior device disponibile: CUDA > Apple MPS > CPU."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@dataclass
class Config:
    # --- Modello (encoder-only, famiglia BERT) ---
    model_name: str = "distilbert-base-uncased"
    num_labels: int = 2
    max_length: int = 128  # lunghezza massima della sequenza (token)

    # --- Dataset ---
    # "sst2": frasi brevi etichettate pos/neg (GLUE, scaricato da HuggingFace).
    # "csv":  dataset locale (es. da Kaggle) con colonne testo/label.
    dataset: str = "sst2"
    text_column: str = "sentence"
    label_column: str = "label"
    # Subsampling: numero massimo di esempi per split (None = tutti).
    max_train_samples: int | None = 20000
    max_eval_samples: int | None = 2000
    # Solo per dataset == "csv"
    csv_path: str | None = None

    # --- Training ---
    epochs: int = 2
    batch_size: int = 16
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    seed: int = 42

    # --- Runtime ---
    device: str = field(default_factory=get_device)
    output_dir: str = str(OUTPUT_DIR)

    # Mappa indice -> etichetta leggibile
    id2label: dict = field(default_factory=lambda: {0: "negative", 1: "positive"})
    label2id: dict = field(default_factory=lambda: {"negative": 0, "positive": 1})

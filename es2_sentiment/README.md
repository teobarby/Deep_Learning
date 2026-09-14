# Esercizio 2 — Sentiment Analysis con Transformer

Sistema di **text classification** che classifica frasi (sequenze testuali) come
**positive** o **negative**, basato su un'architettura **transformer encoder-only**
(famiglia BERT), allineato alle slide `14.Transformers` del corso.

## Approccio

- **Modello**: `distilbert-base-uncased` pre-addestrato (encoder-only, BERT-family),
  con una testa di classificazione binaria addestrata in fine-tuning.
- **Dataset di default**: **SST-2** (Stanford Sentiment Treebank, `stanfordnlp/sst2`),
  frasi brevi etichettate pos/neg — corrisponde al task "frasi positive/negative".
- **Dataset alternativo (Kaggle)**: qualunque CSV con una colonna di testo e una di
  label binaria, via `--dataset csv`.

## Struttura

```
es2_sentiment/
├── src/
│   ├── config.py      # iperparametri, scelta device (CUDA/MPS/CPU)
│   ├── data.py        # caricamento + tokenizzazione (SST-2 o CSV Kaggle)
│   ├── model.py       # modello transformer per classificazione
│   ├── train.py       # fine-tuning (HuggingFace Trainer)
│   ├── evaluate.py    # classification report + matrice di confusione
│   └── predict.py     # inferenza su nuove frasi
├── notebook/          # notebook di dimostrazione
├── requirements.txt
└── outputs/           # modello addestrato + grafici (generato)
```

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Uso

```bash
# Addestramento (SST-2, default)
python src/train.py

# Con dataset Kaggle in CSV
python src/train.py --dataset csv --csv_path data/kaggle.csv \
    --text_column review --label_column sentiment

# Valutazione del modello salvato (report + matrice di confusione)
python src/evaluate.py

# Inferenza su nuove frasi
python src/predict.py "I really loved this movie" "What a waste of time"
```

Tutti gli script vanno lanciati dalla cartella `es2_sentiment/` (gli import di
`src/` sono relativi a questa directory).

## Note

- Il device viene scelto automaticamente: CUDA → Apple **MPS** → CPU.
- Il subsampling è controllato da `--max_train_samples` / `--max_eval_samples`
  (default 20000 / 2000) per ridurre i tempi di training.

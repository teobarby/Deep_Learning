# Esercizio 2 — Sentiment Analysis con Transformer

Sistema di **text classification** che classifica frasi (sequenze testuali) come
**positive** o **negative**, basato su un'architettura **transformer encoder-only**
(famiglia BERT), allineato alle slide `14.Transformers` del corso.

## Approccio

- **Modello**: `distilbert-base-uncased` pre-addestrato (encoder-only, BERT-family),
  con una testa di classificazione binaria addestrata in fine-tuning.
- **Dataset di default**: **SST-2** (Stanford Sentiment Treebank, `stanfordnlp/sst2`),
  frasi brevi etichettate pos/neg — corrisponde al task "frasi positive/negative".
- **Split**: train per l'addestramento, validation (2000 frasi ricavate dal train)
  per scegliere il checkpoint, **test** (le 872 frasi del validation ufficiale, dato
  che il test ufficiale non ha label) per la valutazione finale, usato una sola volta.
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
└── outputs/           # generato: best_model/, checkpoint, confusion_matrix_<split>.png
```

## Risultati

DistilBERT fine-tunato per 2 epoche su 20.000 frasi di SST-2 (~1,5 min su RTX 4070 Ti).
Metriche sul **test set** (872 frasi), F1/precision/recall riferiti alla classe *positive*:

| Accuracy | F1 | Precision | Recall |
|---|---|---|---|
| 89,8% | 0,902 | 0,885 | 0,919 |

Errori: 53 frasi negative classificate positive, 36 positive classificate negative.
Sul validation (usato per scegliere il checkpoint) l'accuracy è 93,3%: più alta
perché quelle frasi provengono dal train di SST-2, che contiene sotto-frasi delle
stesse recensioni.

## Setup

macOS / Linux:
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Windows (PowerShell), con GPU NVIDIA: installare prima torch con CUDA, altrimenti
pip installa la versione solo CPU.
```powershell
py -m venv venv
venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
```

Serve `transformers` 5.x: il warmup è espresso come frazione dei passi
(`warmup_steps=0.1`), che le versioni 4.x non accettano.

## Uso

```bash
# Addestramento (SST-2, default)
python src/train.py

# Con dataset Kaggle in CSV
python src/train.py --dataset csv --csv_path data/kaggle.csv \
    --text_column review --label_column sentiment

# Valutazione del modello salvato (report + matrice di confusione)
python src/evaluate.py                # validation
python src/evaluate.py --split test   # valutazione finale (numeri da riportare)

# Inferenza su nuove frasi
python src/predict.py "I really loved this movie" "What a waste of time"
```

Tutti gli script vanno lanciati dalla cartella `es2_sentiment/` (gli import di
`src/` sono relativi a questa directory).

## Note

- Il device viene scelto automaticamente: CUDA → Apple **MPS** → CPU.
- Il subsampling è controllato da `--max_train_samples` / `--max_eval_samples`
  (default 20000 / 2000) per ridurre i tempi di training. Il test non viene mai
  sottocampionato.

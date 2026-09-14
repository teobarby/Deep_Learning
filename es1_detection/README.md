# Esercizio 1 — Object Detection (YOLO from scratch) su Object365

Riconoscitore di oggetti con **bounding box** basato su un detector in stile
**YOLO implementato da zero** in PyTorch (nessuna libreria di detection).
Segue le tecniche viste a lezione (slide `8.ObjectDetection`): griglia di celle,
**anchor box**, **IoU**, **non-max suppression**.

## Dataset

**Object365** (365 classi, 600k immagini, ~10M bbox). Il dataset completo è
enorme (~365 GB): si effettua **subsampling** su due assi.

- **Immagini**: scaricati 5 shard parquet da `surenreddy/object365` (HuggingFace),
  ~5000 immagini con immagini JPEG + annotazioni bbox incluse (~620 MB).
- **Classi**: ridotte a **8 categorie comuni** — Person, Car, Chair, Bottle, Cup,
  Lamp, Hat, Book.

`prepare_data.py` filtra le box a queste classi, salva immagini + label in
formato YOLO (`class cx cy w h` normalizzati) e calcola gli **anchor box** con
k-means (distanza 1−IoU) sulle dimensioni delle box.

## Architettura

- **Backbone** stile Darknet-tiny: blocchi Conv-BN-LeakyReLU + 5 max-pool che
  portano l'immagine 416×416 a una griglia **13×13** (downsampling 32×).
- **Testa** 1×1 che produce, per ognuna delle **5 anchor** per cella:
  `(tx, ty, tw, th, objectness, 8 logit di classe)`.
- **Loss** (stile YOLOv2, in `loss.py`): assegnazione di ogni oggetto alla cella
  del suo centro e all'anchor con IoU di forma massima; loss di coordinate,
  objectness (con *ignore mask* per anchor a IoU alta) e classificazione.
- **Inferenza** (`detect.py`): decodifica, soglia di confidenza e **non-max
  suppression** per classe.

```
es1_detection/
├── src/
│   ├── config.py        # classi, anchor, geometria griglia, iperparametri
│   ├── prepare_data.py  # subset Object365 -> immagini + label YOLO + anchor
│   ├── dataset.py       # Dataset PyTorch
│   ├── utils.py         # IoU, conversioni bbox, NMS (from scratch)
│   ├── model.py         # backbone CNN + testa di detection
│   ├── loss.py          # loss YOLO + costruzione target
│   ├── train.py         # training loop
│   ├── detect.py        # inferenza + disegno bounding box
│   └── evaluate.py      # mAP@0.5 sul validation set
├── notebook/            # demo
└── requirements.txt
```

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Uso

```bash
# 1. Preparazione dati (scarica gli shard se non in cache, poi filtra)
python src/prepare_data.py

# 2. Training (device automatico: CUDA/MPS/CPU)
python src/train.py --epochs 40

# 3. Valutazione (mAP@0.5)
python src/evaluate.py

# 4. Detection su una foto (es. scattata con la fotocamera)
python src/detect.py path/foto.jpg --out outputs/detection.jpg
```

## Note

- Detector **single-scale**: rileva bene oggetti di dimensioni medie; oggetti
  molto piccoli o molto grandi sono più difficili (limite noto di YOLO a scala
  singola, migliorabile con predizioni multi-scala tipo YOLOv3).
- Le classi sono fortemente sbilanciate (Person ≫ altre), coerente con Object365.
- Training from-scratch senza pesi pre-addestrati: le prestazioni assolute sono
  didattiche, non allo stato dell'arte.

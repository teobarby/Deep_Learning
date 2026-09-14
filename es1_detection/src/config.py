"""Configurazione dell'esercizio 1: object detection YOLO 'from scratch'.

Detector single-scale in stile YOLO: griglia SxS di celle, A anchor box per
cella, IoU per assegnare gli oggetti agli anchor, non-max suppression in
inferenza. Tutto implementato a mano (nessuna libreria di detection).
"""
from __future__ import annotations

from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"

# --- Sottoinsieme di Object365: 8 classi comuni ---
# La mappatura category_id (Object365, 1..365) -> indice di classe (0..7).
OBJ365_ID_TO_CLASS = {
    1: 0,   # Person
    6: 1,   # Car
    3: 2,   # Chair
    9: 3,   # Bottle
    11: 4,  # Cup
    7: 5,   # Lamp
    5: 6,   # Hat
    19: 7,  # Book
}
CLASS_NAMES = ["Person", "Car", "Chair", "Bottle", "Cup", "Lamp", "Hat", "Book"]
NUM_CLASSES = len(CLASS_NAMES)

# --- Geometria del detector ---
# Risoluzione e finezza della griglia.
#
# Il dataset contiene moltissimi oggetti piccoli: il lato minore mediano e' di
# circa 25 px (a 416). Un detector a scala singola non puo' rilevare oggetti
# piu' piccoli di una cella della griglia, quindi lo STRIDE determina il limite
# di risoluzione:
#
#   stride 32 -> griglia 13x13, cella 32px -> 58,8% degli oggetti sotto il limite
#   stride 16 -> griglia 26x26, cella 16px -> 34,8%
#   stride  8 -> griglia 52x52, cella  8px -> 13,8%   <-- scelto
#
# Con stride 8 il backbone si ferma prima (layer2 di ResNet18): si guadagna
# risoluzione spaziale al prezzo di feature un po' meno profonde.
IMG_SIZE = 416          # immagini quadrate 416x416
STRIDE = 8              # fattore di downsampling del backbone
GRID = IMG_SIZE // STRIDE  # S = 52 celle per lato
NUM_ANCHORS = 5         # A anchor box per cella

# Anchor box scelti A MANO (come indicato nelle slide del corso: "the number n
# and shape of anchor boxes is usually determined by hand"), in unita'
# NORMALIZZATE [0,1]. Le taglie sono state fissate guardando la distribuzione
# reale delle dimensioni degli oggetti del dataset (percentili 25/50/75/90 piu'
# una taglia grande): la mediana degli oggetti e' 0,068 x 0,116, quindi servono
# anchor molto piccoli. Con questi valori solo il 12% delle box e' mal coperto
# (era il 33% con anchor scelti "a occhio").
ANCHORS = [
    (0.03, 0.05),   # molto piccolo   (~percentile 25)
    (0.07, 0.12),   # piccolo         (~mediana)
    (0.14, 0.26),   # medio           (~percentile 75)
    (0.26, 0.46),   # grande          (~percentile 90)
    (0.60, 0.70),   # molto grande
]

# --- Backbone ---
# True : backbone ResNet18 PRE-ADDESTRATO su ImageNet e poi fine-tunato
#        (transfer learning, argomento del corso - deck 7.TransferLearning).
# False: backbone convoluzionale scritto interamente da zero.
# In entrambi i casi la testa di detection, gli anchor, la loss YOLO, l'IoU e il
# non-max suppression restano implementati a mano.
PRETRAINED_BACKBONE = True

# Normalizzazione richiesta dai pesi pre-addestrati su ImageNet.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# --- Iperparametri di training ---
BATCH_SIZE = 16
EPOCHS = 20
# Nel fine-tuning si usa un learning rate piu' basso, per non distruggere le
# feature gia' apprese dal backbone pre-addestrato.
LEARNING_RATE = 1e-4 if PRETRAINED_BACKBONE else 1e-3
WEIGHT_DECAY = 5e-4
LAMBDA_COORD = 5.0      # peso della loss di coordinate

# Peso della loss di 'no object'. In loss.py il termine e' normalizzato come
# MEDIA sulle celle vuote (non come somma), quindi lambda esprime direttamente
# il rapporto fra il peso complessivo di "qui non c'e' nulla" e quello di "qui
# c'e' un oggetto".
#
# Riferimento: nello YOLO originale (griglia 7x7x2 = 98 predizioni, ~4 oggetti
# per immagine, somma con lambda 0,5) il rapporto aggregato e' circa 12:1.
# Con la normalizzazione a media serve quindi un lambda dell'ordine di 10.
#
# Un valore troppo basso (es. 0,5) rende la penalita' per i falsi positivi ~24
# volte piu' debole del riferimento: la rete predice oggetti quasi ovunque e la
# precision crolla, perche' l'NMS non puo' sopprimere box che non si sovrappongono.
LAMBDA_NOOBJ = 10.0
SEED = 42

# --- Inferenza ---
CONF_THRESH = 0.25      # soglia di confidenza per tenere una predizione
NMS_IOU_THRESH = 0.45   # soglia IoU per il non-max suppression


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_anchors() -> torch.Tensor:
    """Restituisce gli anchor scelti a mano in unita' di CELLA della griglia.

    ANCHORS e' normalizzato [0,1]; qui viene moltiplicato per GRID (A, 2).
    """
    return torch.tensor(ANCHORS, dtype=torch.float32) * GRID


DEVICE = get_device()

"""Configurazione dell'esercizio 1: object detection YOLO 'from scratch'.

Detector single-scale in stile YOLO: griglia SxS di celle, A anchor box per
cella, IoU per assegnare gli oggetti agli anchor, non-max suppression in
inferenza. Tutto implementato a mano (nessuna libreria di detection).
"""
from __future__ import annotations

import json
import os
from pathlib import Path

# torch viene importato solo dove serve (get_device, load_anchors): cosi'
# prepare_data.py e analyze_sizes.py, che lavorano solo sui file, girano anche
# senza torch.

# --- Override per gli esperimenti ---
# Le varianti si lanciano senza modificare questo file, passando un JSON nella
# variabile d'ambiente ES1_CONFIG (PowerShell):
#     $env:ES1_CONFIG = '{"EXPERIMENT": "stride16", "STRIDE": 16}'
#     python src/train.py
# EXPERIMENT sceglie la sottocartella di outputs/ in cui finiscono checkpoint e
# storico, cosi' gli esperimenti non si sovrascrivono. La configurazione effettiva
# viene salvata nel checkpoint e verificata al caricamento. EVAL_MIN_OBJ_SIZE non
# e' sovrascrivibile: definisce il compito su cui si confrontano gli esperimenti.
_OVERRIDABLE = {
    "EXPERIMENT", "IMG_SIZE", "STRIDE", "ANCHORS", "MIN_OBJ_SIZE",
    "BACKBONE", "PRETRAINED_BACKBONE", "BATCH_SIZE", "EPOCHS", "LEARNING_RATE",
    "LR_SCHEDULE", "FREEZE_BACKBONE_EPOCHS", "WEIGHT_DECAY", "SCALE_JITTER",
    "LAMBDA_COORD", "LAMBDA_NOOBJ", "CONF_THRESH", "NMS_IOU_THRESH",
}
CONFIG_OVERRIDES = json.loads(os.environ.get("ES1_CONFIG") or "{}")
_unknown = set(CONFIG_OVERRIDES) - _OVERRIDABLE
if _unknown:
    raise ValueError(f"ES1_CONFIG contiene chiavi non valide: {sorted(_unknown)}")


def _cfg(name: str, default):
    """Valore di configurazione: override da ES1_CONFIG se presente, altrimenti default."""
    return CONFIG_OVERRIDES.get(name, default)


EXPERIMENT = _cfg("EXPERIMENT", "")
# ES1_ROOT permette di eseguire una copia congelata di src/ (es. salvata nella
# cartella di un esperimento) continuando a usare data/ e outputs/ del progetto.
ROOT = Path(os.environ.get("ES1_ROOT") or Path(__file__).resolve().parent.parent)
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs" / EXPERIMENT if EXPERIMENT else ROOT / "outputs"

# --- Sottoinsieme di Object365: 8 classi di oggetti medio-grandi ---
# Scelte sui dati (src/analyze_sizes.py e statistiche per categoria sugli shard):
# categorie frequenti il cui lato minore resta sopra il limite di risoluzione del
# detector. Bottle, Cup e Hat (lato minore mediano ~0,03 del lato immagine)
# sarebbero quasi interamente sotto soglia; Book e' per 2/3 box 'crowd' (intere
# librerie).
#
# Categorie "sorelle" quasi indistinguibili sono unite in una classe (es. Car,
# SUV, Van, Pickup): tenendone una sola, le altre resterebbero nell'immagine come
# sfondo e la rete riceverebbe segnali contraddittori su oggetti identici.
#
# La mappatura category_id (Object365, 1..365) -> indice di classe (0..7).
# Gli ID sono stati verificati visivamente ritagliando esempi dagli shard; Dining
# Table (99) e Coffee Table (168) sono esclusi perche' non verificabili (esempi
# assenti o errati).
OBJ365_ID_TO_CLASS = {
    1: 0,                               # Person
    3: 1, 48: 1,                        # Chair, Stool
    10: 2, 169: 2,                      # Desk, Side Table
    13: 3,                              # Cabinet/shelf
    6: 4, 35: 4, 50: 4, 88: 4,          # Car, SUV, Van, Pickup Truck
    7: 5,                               # Lamp
    17: 6,                              # Picture/Frame
    38: 7,                              # Monitor/TV
}
CLASS_NAMES = ["Person", "Chair", "Table", "Cabinet", "Car", "Lamp", "Picture", "Monitor"]
NUM_CLASSES = len(CLASS_NAMES)

# --- Geometria del detector ---
# Risoluzione e finezza della griglia. Un detector a scala singola non puo'
# rilevare oggetti piu' piccoli di una cella, quindi lo STRIDE determina il
# limite di risoluzione; ma fermando il backbone prima (stride piccolo) si perde
# profondita' e campo recettivo, e gli oggetti grandi non vengono visti per intero.
# Scelto sperimentalmente (mAP@0.5 sul validation, 20 epoche):
#   stride  8 (griglia 52x52, campo recettivo ~131px) -> 0,221
#   stride 16 (griglia 26x26, campo recettivo ~275px) -> 0,342   <-- scelto
#   stride 32 (griglia 13x13, campo recettivo ~563px) -> 0,294
IMG_SIZE = _cfg("IMG_SIZE", 416)    # immagini quadrate IMG_SIZE x IMG_SIZE
STRIDE = _cfg("STRIDE", 16)         # fattore di downsampling del backbone
GRID = IMG_SIZE // STRIDE           # S celle per lato (52 con i default)

# Anchor box scelti A MANO (come indicato nelle slide del corso: "the number n
# and shape of anchor boxes is usually determined by hand"), in unita'
# NORMALIZZATE [0,1]. Le taglie sono state fissate guardando la distribuzione
# reale delle dimensioni degli oggetti del dataset (percentili 25/50/75/90 piu'
# una taglia grande): la mediana degli oggetti e' 0,068 x 0,116, quindi servono
# anchor molto piccoli. Con questi valori solo il 12% delle box e' mal coperto
# (era il 33% con anchor scelti "a occhio").
ANCHORS = [tuple(a) for a in _cfg("ANCHORS", [
    (0.03, 0.05),   # molto piccolo   (~percentile 25)
    (0.07, 0.12),   # piccolo         (~mediana)
    (0.14, 0.26),   # medio           (~percentile 75)
    (0.26, 0.46),   # grande          (~percentile 90)
    (0.60, 0.70),   # molto grande
])]
NUM_ANCHORS = len(ANCHORS)          # A anchor box per cella

# --- Regioni ignorate: oggetti piccoli e box 'crowd' ---
# Due tipi di box non sono oggetti che il detector deve imparare a localizzare:
#   - oggetti piu' piccoli di una cella (sotto il limite di risoluzione);
#   - box 'crowd' di Object365 (iscrowd=1): una sola box attorno a un gruppo di
#     oggetti, es. una folla o uno scaffale di libri.
# Cancellarle sarebbe dannoso: gli oggetti restano nell'immagine e la loss
# insegnerebbe che li' "non c'e' nulla". Sono quindi REGIONI IGNORATE (come in
# COCO): in training le celle interessate non ricevono la loss no-object, in
# valutazione non contano ne' come oggetti da trovare ne' come falsi positivi.
#
# MIN_OBJ_SIZE: soglia di TRAINING sul lato minore (unita' normalizzate [0,1]).
# Di default una cella: esclusione e limite di risoluzione coincidono.
# 0 disattiva il meccanismo.
MIN_OBJ_SIZE = _cfg("MIN_OBJ_SIZE", 1.0 / GRID)

# EVAL_MIN_OBJ_SIZE: soglia di VALUTAZIONE, cioe' la definizione del compito
# ("oggetti con lato minore >= 1/20 dell'immagine"). E' volutamente fissa e
# indipendente dalla griglia: configurazioni diverse vanno confrontate sullo
# stesso insieme di oggetti, altrimenti una soglia piu' alta toglierebbe gli
# oggetti difficili e gonfierebbe le metriche.
EVAL_MIN_OBJ_SIZE = 1.0 / 20

# --- Backbone ---
# True : backbone ResNet18 PRE-ADDESTRATO su ImageNet e poi fine-tunato
#        (transfer learning, argomento del corso - deck 7.TransferLearning).
# False: backbone convoluzionale scritto interamente da zero.
# In entrambi i casi la testa di detection, gli anchor, la loss YOLO, l'IoU e il
# non-max suppression restano implementati a mano.
PRETRAINED_BACKBONE = _cfg("PRETRAINED_BACKBONE", True)
# Rete pre-addestrata da usare: "resnet18" o "resnet34" (stessi canali per
# stadio, ResNet34 ha piu' blocchi e quindi feature piu' profonde).
# ResNet34 scelto sperimentalmente: 0,437 contro 0,385 di ResNet18 (stride 16 +
# augmentation, 20 epoche).
BACKBONE = _cfg("BACKBONE", "resnet34")

# Normalizzazione richiesta dai pesi pre-addestrati su ImageNet.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# --- Iperparametri di training ---
BATCH_SIZE = _cfg("BATCH_SIZE", 16)
# 30 epoche: con augmentation e ResNet34 il mAP saliva ancora all'epoca 20.
EPOCHS = _cfg("EPOCHS", 30)
# Nel fine-tuning si usa un learning rate piu' basso, per non distruggere le
# feature gia' apprese dal backbone pre-addestrato.
LEARNING_RATE = _cfg("LEARNING_RATE", 1e-4 if PRETRAINED_BACKBONE else 1e-3)
# "none": learning rate costante; "cosine": decresce da LEARNING_RATE a ~0 lungo
# le EPOCHS con andamento a coseno (passi piu' piccoli verso la fine del training).
# "cosine": 0,393 contro 0,385 e curva finale piu' stabile (ResNet18, stride 16).
LR_SCHEDULE = _cfg("LR_SCHEDULE", "cosine")
# Transfer learning in due fasi: per le prime N epoch il backbone pre-addestrato
# resta congelato e si addestrano solo collo e testa (feature extraction), poi si
# sblocca tutto (fine-tuning). 0 = fine-tuning da subito.
FREEZE_BACKBONE_EPOCHS = _cfg("FREEZE_BACKBONE_EPOCHS", 0)
WEIGHT_DECAY = _cfg("WEIGHT_DECAY", 5e-4)
# Ampiezza dello zoom casuale in data augmentation: fattore in [1-s, 1+s].
# 0 = disattivato (solo flip e color jitter). Con 0,3: 0,385 contro 0,342 senza,
# e scompare l'overfitting (la val loss non risale piu' nelle ultime epoche).
SCALE_JITTER = _cfg("SCALE_JITTER", 0.3)
LAMBDA_COORD = _cfg("LAMBDA_COORD", 5.0)      # peso della loss di coordinate

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
LAMBDA_NOOBJ = _cfg("LAMBDA_NOOBJ", 10.0)
SEED = 42

# --- Inferenza ---
CONF_THRESH = _cfg("CONF_THRESH", 0.25)          # soglia di confidenza
NMS_IOU_THRESH = _cfg("NMS_IOU_THRESH", 0.45)    # soglia IoU per il NMS


def get_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_anchors() -> "torch.Tensor":
    """Restituisce gli anchor scelti a mano in unita' di CELLA della griglia.

    ANCHORS e' normalizzato [0,1]; qui viene moltiplicato per GRID (A, 2).
    """
    import torch

    return torch.tensor(ANCHORS, dtype=torch.float32) * GRID


def is_small(w: float, h: float, threshold: float | None = None) -> bool:
    """True se una box (w,h normalizzati) ha lato minore sotto soglia.

    Senza threshold usa MIN_OBJ_SIZE (soglia di training).
    """
    return min(w, h) < (MIN_OBJ_SIZE if threshold is None else threshold)


# Parametri che danno significato ai pesi: se differiscono tra training e
# inferenza il modello da' risultati sbagliati in silenzio. Vengono salvati nel
# checkpoint e verificati al caricamento.
_WEIGHT_KEYS = ("IMG_SIZE", "STRIDE", "NUM_ANCHORS", "ANCHORS",
                "BACKBONE", "PRETRAINED_BACKBONE", "CLASS_NAMES")
# Valore implicito delle chiavi introdotte dopo i primi checkpoint.
_LEGACY_DEFAULTS = {"BACKBONE": "resnet18"}


def checkpoint_config() -> dict:
    """Configurazione da salvare nel checkpoint insieme ai pesi."""
    return {
        "IMG_SIZE": IMG_SIZE,
        "STRIDE": STRIDE,
        "NUM_ANCHORS": NUM_ANCHORS,
        "ANCHORS": [list(a) for a in ANCHORS],
        "BACKBONE": BACKBONE,
        "PRETRAINED_BACKBONE": PRETRAINED_BACKBONE,
        "CLASS_NAMES": list(CLASS_NAMES),
        "MIN_OBJ_SIZE": MIN_OBJ_SIZE,
        # solo a scopo di riproducibilita' (non verificati al caricamento)
        "EXPERIMENT": EXPERIMENT,
        "OVERRIDES": dict(CONFIG_OVERRIDES),
        "BATCH_SIZE": BATCH_SIZE,
        "LEARNING_RATE": LEARNING_RATE,
        "LR_SCHEDULE": LR_SCHEDULE,
        "FREEZE_BACKBONE_EPOCHS": FREEZE_BACKBONE_EPOCHS,
        "WEIGHT_DECAY": WEIGHT_DECAY,
        "SCALE_JITTER": SCALE_JITTER,
        "LAMBDA_COORD": LAMBDA_COORD,
        "LAMBDA_NOOBJ": LAMBDA_NOOBJ,
        "EVAL_MIN_OBJ_SIZE": EVAL_MIN_OBJ_SIZE,
    }


def check_checkpoint_config(ck: dict) -> None:
    """Verifica che un checkpoint sia compatibile con questo config.py.

    Errore se differisce un parametro dei pesi; solo un avviso se differisce
    MIN_OBJ_SIZE (valutare con una soglia diversa da quella di training e' lecito).
    """
    saved = ck.get("config")
    if saved is None:
        print("[attenzione] checkpoint senza configurazione salvata: "
              "compatibilita' con config.py non verificabile")
        return
    current = checkpoint_config()
    saved = {**_LEGACY_DEFAULTS, **saved}
    diff = [k for k in _WEIGHT_KEYS if saved.get(k) != current[k]]
    if diff:
        lines = "\n".join(f"  {k}: checkpoint={saved.get(k)!r}  config.py={current[k]!r}"
                          for k in diff)
        raise ValueError("Checkpoint addestrato con una configurazione diversa "
                         "da config.py:\n" + lines)
    if saved.get("MIN_OBJ_SIZE") != current["MIN_OBJ_SIZE"]:
        print(f"[attenzione] MIN_OBJ_SIZE diverso: training={saved.get('MIN_OBJ_SIZE')}"
              f"  attuale={current['MIN_OBJ_SIZE']}")


_device: str | None = None


def __getattr__(name: str):
    """DEVICE e' calcolato al primo accesso ('from config import DEVICE'), non
    all'import del modulo, per non richiedere torch a chi non lo usa."""
    global _device
    if name == "DEVICE":
        if _device is None:
            _device = get_device()
        return _device
    raise AttributeError(f"module 'config' has no attribute {name!r}")

# Esercizio 1 — Object Detection (YOLO da zero) su Object365

Riconoscitore di oggetti con **bounding box** in stile **YOLO**, implementato a mano
in PyTorch (nessuna libreria di detection): griglia di celle, **anchor box**, target,
**loss**, **IoU** e **non-max suppression** sono codice nostro. L'unico componente
importato è il backbone convoluzionale pre-addestrato (*transfer learning*).

**Risultato sul test set: mAP@0.5 = 0,444** (precision 0,66, recall 0,53, F1 0,59),
contro 0,221 della configurazione di partenza.

Guida di studio completa: [`docs/Studio_Esercizio1_Object_Detection.pdf`](../docs/Studio_Esercizio1_Object_Detection.pdf).

## Dataset

**Object365** (365 classi, 600k immagini). Il dataset completo è ingestibile su un PC,
quindi si fa **subsampling**: 15 shard parquet da HuggingFace
(`surenreddy/object365`, ~1,9 GB) → **14.361 immagini**, divise in
train 11.467 / validation 1.395 / test 1.499.

**8 classi**, scelte misurando frequenza e dimensione delle box: categorie frequenti
e di dimensione medio-grande, con le categorie "sorelle" unite (un SUV e un'auto non
possono essere uno oggetto e l'altro sfondo).

| Classe | Categorie Object365 |
|---|---|
| Person | Person |
| Chair | Chair, Stool |
| Table | Desk, Side Table |
| Cabinet | Cabinet/shelf |
| Car | Car, SUV, Van, Pickup Truck |
| Lamp | Lamp |
| Picture | Picture/Frame |
| Monitor | Monitor/TV |

**Regioni ignorate** (come in COCO): le box `iscrowd` (una box attorno a un gruppo di
oggetti, il 15% delle annotazioni) e gli oggetti più piccoli di una cella della griglia
non generano obiettivi né penalità. Cancellarle insegnerebbe alla rete che lì "non c'è
nulla", penalizzando anche gli oggetti veri della stessa classe.

**Il compito valutato**: trovare gli oggetti con **lato minore ≥ 1/20** del lato
dell'immagine. La soglia è fissa e indipendente dalla griglia, così configurazioni
diverse si confrontano sugli stessi oggetti.

## Configurazione finale

Scelta con 8 esperimenti controllati, uno per modifica (vedi *Esperimenti*).

| | Valore |
|---|---|
| Risoluzione / stride | 416 px, stride 16 → griglia 26×26 |
| Backbone | ResNet34 pre-addestrata su ImageNet, troncata a `layer3` |
| Anchor | 5 forme scelte a mano |
| Augmentation | flip orizzontale, luminosità/contrasto, zoom e traslazione ±30% |
| Training | Adam, lr 10⁻⁴ con decadimento coseno, weight decay 5·10⁻⁴, batch 16, 30 epoche |
| Selezione | checkpoint con **mAP@0.5** migliore sul validation |
| Soglia d'uso | quella che massimizza l'F1 sul validation (0,9), salvata nel checkpoint |

## Struttura

```
es1_detection/
├── src/
│   ├── config.py          # classi, geometria, anchor, iperparametri, override
│   ├── prepare_data.py    # shard Object365 -> immagini + label YOLO (con flag crowd)
│   ├── analyze_sizes.py   # statistiche su dimensioni, griglia e anchor
│   ├── dataset.py         # Dataset PyTorch + augmentation
│   ├── utils.py           # IoU, conversioni bbox, NMS (scritti a mano)
│   ├── model.py           # backbone + collo + testa di detection
│   ├── loss.py            # target, regioni ignorate, loss YOLO, decodifica
│   ├── train.py           # training, valutazione per epoca, checkpoint
│   ├── evaluate.py        # precision/recall/F1, AP e mAP@0.5
│   └── detect.py          # detection su una foto
├── tools/                 # coda esperimenti e generatori della relazione
├── notebook/demo.ipynb
└── outputs/               # generato: checkpoint, storico, report, foto
```

## Setup

macOS / Linux:
```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

Windows (PowerShell) con GPU NVIDIA — installare prima torch con CUDA, altrimenti
pip installa la versione solo CPU:
```powershell
py -m venv venv
venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
```

## Uso

```bash
python src/prepare_data.py                      # scarica 15 shard e prepara i dati
python src/analyze_sizes.py                     # statistiche su dimensioni e griglia
python src/train.py                             # training (~64 min su RTX 4070 Ti)
python src/evaluate.py                          # metriche sul validation
python src/evaluate.py --split test             # valutazione finale (una sola volta)
python src/detect.py foto.jpg --out risultato.jpg
```

Tutti i comandi vanno lanciati dalla cartella `es1_detection/`.

### Esperimenti

Ogni variante si lancia **senza modificare il codice**, passando un JSON nella
variabile d'ambiente `ES1_CONFIG`. `EXPERIMENT` sceglie la sottocartella di `outputs/`,
così i run non si sovrascrivono; la configurazione viene salvata nel checkpoint e
verificata al caricamento.

```powershell
$env:ES1_CONFIG = '{"EXPERIMENT": "prova_stride8", "STRIDE": 8}'
python src\train.py
```

Per una coda di esperimenti in sequenza, ognuno con la propria copia congelata del
codice: `tools\run_queue.ps1 -QueueFile tools\coda_ablation.json`.

## Esperimenti e risultati

mAP@0.5 sul validation; ogni riga cambia **una sola** cosa rispetto alla sua base.

| Esperimento | Base | Modifica | mAP@0.5 |
|---|---|---|---|
| e0 | — | configurazione di partenza (stride 8, ResNet18) | 0,221 |
| e1 | e0 | 8 anchor (4 scale × 2 rapporti) | 0,217 |
| e2 | e0 | **stride 16** | 0,342 |
| e3 | e0 | stride 32 | 0,294 |
| e4 | e2 | **zoom e traslazione ±30%** | 0,385 |
| e5 | e4 | **learning rate coseno** | 0,393 |
| e6 | e4 | **ResNet34** | 0,437 |
| e7 | e4 | backbone congelato per 3 epoche | 0,396 |
| finale | e4 | ResNet34 + coseno + 30 epoche | **0,465** |

**Test set** (1.499 immagini, 8.383 oggetti): mAP@0.5 **0,444**, precision 0,660,
recall 0,531, F1 0,588.

| Classe | AP@0.5 | | Classe | AP@0.5 |
|---|---|---|---|---|
| Person | 0,720 | | Lamp | 0,433 |
| Picture | 0,525 | | Chair | 0,432 |
| Car | 0,524 | | Cabinet | 0,277 |
| Monitor | 0,453 | | Table | 0,190 |

## Note e limiti

- **Detector a scala singola**: lo stride è un compromesso tra oggetti piccoli e
  grandi. Con stride 8 il campo recettivo (~131 px su 416) è troppo piccolo per gli
  oggetti grandi; con stride 32 le celle da 32 px non separano gli oggetti medi.
- **Errori principali**: box imprecise (42% dei falsi positivi e 31% degli oggetti
  mancati) e confidenza poco separata (il 41% degli oggetti mancati era predetto
  correttamente ma sotto soglia). Gli errori di classe sono solo il 2,5%.
- **Etichette incomplete**: parte dei falsi positivi sono oggetti reali non annotati,
  quindi il valore misurato sottostima leggermente il modello.
- **Classi sbilanciate**: Person è metà delle box; Table e Cabinet restano difficili.
- Il ridimensionamento a quadrato deforma le proporzioni: un letterbox è il
  miglioramento più immediato per foto molto panoramiche.

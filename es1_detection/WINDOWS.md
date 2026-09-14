# Addestramento su Windows (GPU)

Istruzioni autosufficienti per addestrare il detector su un PC Windows con GPU
NVIDIA, e riportare il modello sul Mac.

---

## 1. Preparazione

Scompatta `es1_windows.zip`, ad esempio in `C:\DL\`.
Ottieni la cartella `C:\DL\es1_detection\` con dentro `src/`, `data/`, `notebook/`.

Apri **PowerShell** e crea l'ambiente:

```powershell
cd C:\DL\es1_detection
py -m venv venv
venv\Scripts\activate

# PyTorch con CUDA (NON il pacchetto base: quello gira su CPU!)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

pip install numpy pillow matplotlib tqdm jupyter
```

## 2. Verifica che la GPU sia vista  ← PASSO IMPORTANTE

```powershell
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Deve stampare `True` e il nome della GPU.
Se stampa `False` hai la versione CPU: `pip uninstall torch torchvision` e
reinstalla con `--index-url` come sopra.

## 3. Addestramento

```powershell
python src\train.py --epochs 20 --batch_size 16
```

- **Batch 16, non di piu'**: la griglia e' 52x52, quindi ci sono 13.520
  predizioni per immagine (contro le 845 di una griglia 13x13) e la memoria GPU
  richiesta e' molto maggiore. Se va in *out of memory*, scendi a `8`.
  Se invece gira comodo, puoi provare `24`.
- Se si interrompe: `python src\train.py --resume`
- Assicurati che il PC non vada in sospensione durante il training.

La loss iniziale deve essere nell'ordine di **7-10** (non centinaia) e deve
scendere. Componenti tipiche all'inizio: coord ~4, obj ~0.7, noobj ~0.35, cls ~2.

Output: una riga per epoch. La `train` loss deve scendere.
I checkpoint finiscono in `outputs\best.pt` (migliore) e `outputs\last.pt`.

## 4. Valutazione

```powershell
python src\evaluate.py
```

Stampa precision e recall per classe a IoU >= 0.5. **Salva questo output**
(copia-incolla): serve per la relazione e la presentazione.

Detection su una foto:
```powershell
python src\detect.py C:\percorso\foto.jpg --out outputs\detection.jpg
```

Notebook dimostrativo:
```powershell
jupyter notebook notebook\demo.ipynb
```

## 5. Riporta il risultato sul Mac

Copia **un solo file** sul Mac:

```
C:\DL\es1_detection\outputs\best.pt
        |
        v
<progetto>/es1_detection/outputs/best.pt
```

Il checkpoint e' indipendente dalla piattaforma: il codice lo carica con
`map_location`, quindi un modello addestrato su CUDA funziona su Mac (MPS).
Sul Mac ci sono gia' codice, dati, relazione e presentazione: manca solo questo file.

---

## ATTENZIONE: non modificare config.py

Il checkpoint contiene **solo i pesi**, non la configurazione. Se addestri con
parametri diversi e poi carichi il modello sul Mac con la config attuale,
ottieni risultati sbagliati in silenzio (o un errore).

**Sicuri** (solo training, non cambiano il modello):
`--epochs`, `--batch_size`, `--lr`, `--resume`

**Da NON toccare** (o da replicare identici anche nel `config.py` del Mac):
`IMG_SIZE`, `GRID`, `ANCHORS`, `NUM_ANCHORS`, `PRETRAINED_BACKBONE`

Configurazione attuale (deve restare cosi'):

| Parametro | Valore |
|---|---|
| IMG_SIZE | 416 |
| STRIDE | **8** -> griglia **52x52** |
| NUM_ANCHORS | 5, scelti a mano |
| PRETRAINED_BACKBONE | True (ResNet18 troncata a layer2 + "collo") |
| LEARNING_RATE | 1e-4 (fine-tuning) |
| Augmentation | flip orizzontale + color jitter |

Lo stride 8 e' stato scelto perche' il dataset e' pieno di oggetti piccoli:
con stride 32 il 58,8% degli oggetti era piu' piccolo di una cella della
griglia (quindi non rilevabile); con stride 8 la quota scende al 13,8%.

---

## Se qualcosa non va

- **Errore del DataLoader all'avvio**: e' un problema noto di multiprocessing su
  Windows. Apri `src/train.py` e metti `num_workers=0` nei due `DataLoader`.
- **Out of memory (CUDA)**: abbassa il batch, es. `--batch_size 16`.
- **Training lentissimo**: quasi certamente stai girando su CPU, rifai il passo 2.

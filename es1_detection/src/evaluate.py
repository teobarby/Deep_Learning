"""Valutazione del detector: precision, recall, F1 e mAP a IoU>=0.5.

Per ogni classe si confrontano le predizioni con la ground truth usando l'IoU
(la misura vista a lezione per valutare le bounding box): una predizione e' un
vero positivo (TP) se ha IoU >= 0.5 con un oggetto reale della stessa classe non
ancora abbinato, altrimenti e' un falso positivo (FP). Da qui:

    precision = TP / (TP + FP)      recall = TP / (numero di oggetti reali)
    F1 = 2 * precision * recall / (precision + recall)

Precision e recall dipendono dalla soglia di confidenza. Per confrontare modelli
(calibrati in modo diverso) si usa l'AP@0.5, l'area sotto la curva
precision-recall ottenuta facendo variare la soglia, mediata sulle classi (mAP).
Si riporta anche l'F1 massimo e la soglia che lo ottiene, da usare in inferenza.

Il compito e' definito da EVAL_MIN_OBJ_SIZE, fissa e indipendente dalla griglia
del modello, cosi' configurazioni diverse si confrontano sugli stessi oggetti.
Regioni ignorate, con le stesse regole di COCO:
  - oggetti con lato minore sotto soglia e box 'crowd' non contano tra gli
    oggetti da trovare;
  - una predizione abbinata (IoU >= 0.5) a un oggetto sotto soglia, o contenuta
    per almeno meta' in una box crowd della sua classe, non conta;
  - una predizione non abbinata e a sua volta sotto soglia non conta: il compito
    non riguarda oggetti di quella dimensione.

train.py usa il mAP@0.5 del validation set per scegliere il checkpoint
migliore. Il test set va usato una sola volta, a
configurazione e soglie ormai fissate: se lo si usa per decidere, i suoi numeri
smettono di essere una stima onesta.

Uso:
    python src/evaluate.py                      # validation set
    python src/evaluate.py --split test         # valutazione finale
    python src/evaluate.py --weights outputs/best.pt --conf 0.25
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from config import (
    BATCH_SIZE,
    CLASS_NAMES,
    CONF_THRESH,
    DEVICE,
    EVAL_MIN_OBJ_SIZE,
    IMG_SIZE,
    NMS_IOU_THRESH,
    NUM_CLASSES,
    OUTPUT_DIR,
    check_checkpoint_config,
    is_small,
    load_anchors,
)
from dataset import Object365Detection, collate_fn
from detect import postprocess
from model import YOLO
from utils import iou_xyxy, xywh_to_xyxy

IOU_MATCH = 0.5  # soglia di IoU per considerare una predizione corretta
IOA_CROWD = 0.5  # frazione della predizione dentro una box crowd per ignorarla
EVAL_CONF_MIN = 0.01  # confidenza minima delle predizioni raccolte (per mAP e F1 max)


def _size_bin(side: float, min_size: float) -> str:
    """Fascia di dimensione di un oggetto valido, in multipli della soglia."""
    return "medi" if side < 3 * min_size else "grandi"


def _ioa(box: list[float], boxes: list[list[float]]) -> torch.Tensor:
    """Intersezione tra box e ciascuna di boxes, divisa per l'area di box."""
    a = torch.tensor(box)
    b = torch.tensor(boxes)
    w = (torch.min(a[2], b[:, 2]) - torch.max(a[0], b[:, 0])).clamp(min=0)
    h = (torch.min(a[3], b[:, 3]) - torch.max(a[1], b[:, 1])).clamp(min=0)
    area = ((a[2] - a[0]) * (a[3] - a[1])).clamp(min=1e-9)
    return w * h / area


def _prf(tp: int, fp: int, n_gt: int) -> tuple[float, float, float]:
    """Precision, recall e F1 dai conteggi."""
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / n_gt if n_gt else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f1


def _new_stats(min_size: float) -> dict:
    return {
        "min_size": min_size,
        "tp": [0] * NUM_CLASSES,       # veri positivi per classe
        "fp": [0] * NUM_CLASSES,       # falsi positivi per classe
        "n_gt": [0] * NUM_CLASSES,     # oggetti validi per classe
        "tp_sz": {"medi": 0, "grandi": 0},
        "n_gt_sz": {"medi": 0, "grandi": 0},
        "n_small_gt": 0,               # oggetti sotto soglia
        "n_crowd_gt": 0,               # box crowd
        # una voce per predizione: (classe, score, esito "tp"/"fp"/"ign", fascia)
        "preds": [],
    }


def _match_image(results, gt: torch.Tensor, st: dict) -> None:
    """Abbina le predizioni di un'immagine alla ground truth, aggiornando st.

    L'esito di una predizione dipende solo dalle predizioni con score piu' alto
    (abbinamento greedy per score decrescente), quindi e' lo stesso a qualunque
    soglia di confidenza: basta calcolarlo una volta a soglia bassa e poi filtrare.
    """
    min_size = st["min_size"]
    valid = {c: [] for c in range(NUM_CLASSES)}   # box xyxy degli oggetti da trovare
    sizes = {c: [] for c in range(NUM_CLASSES)}   # fascia di dimensione di ognuno
    small = {c: [] for c in range(NUM_CLASSES)}   # oggetti sotto soglia
    crowd = {c: [] for c in range(NUM_CLASSES)}   # box crowd
    for row in gt:
        c = int(row[0])
        box = xywh_to_xyxy(row[1:5]).tolist()
        w, h = float(row[3]), float(row[4])
        if len(row) > 5 and row[5] > 0:
            crowd[c].append(box)
            st["n_crowd_gt"] += 1
        elif is_small(w, h, min_size):
            small[c].append(box)
            st["n_small_gt"] += 1
        else:
            b = _size_bin(min(w, h), min_size)
            valid[c].append(box)
            sizes[c].append(b)
            st["n_gt"][c] += 1
            st["n_gt_sz"][b] += 1
    matched = {c: [False] * len(valid[c]) for c in range(NUM_CLASSES)}

    # predizioni ordinate per score decrescente: le piu' sicure si abbinano prima
    for cls, score, box in results:
        pbox = torch.tensor(box).unsqueeze(0)
        if valid[cls]:
            ious = iou_xyxy(pbox, torch.tensor(valid[cls]))
            ious[torch.tensor(matched[cls])] = -1.0   # ogni oggetto si abbina una volta
            best = int(ious.argmax())
            if float(ious[best]) >= IOU_MATCH:
                matched[cls][best] = True
                st["preds"].append((cls, score, "tp", sizes[cls][best]))
                continue
        x1, y1, x2, y2 = box
        if ((small[cls] and float(iou_xyxy(pbox, torch.tensor(small[cls])).max()) >= IOU_MATCH)
                or (crowd[cls] and float(_ioa(box, crowd[cls]).max()) >= IOA_CROWD)
                or is_small(x2 - x1, y2 - y1, min_size)):
            st["preds"].append((cls, score, "ign", None))
            continue
        st["preds"].append((cls, score, "fp", None))


def _average_precision(ranked_is_tp: list[bool], n_gt: int) -> float:
    """AP a IoU 0.5: area sotto la curva precision-recall.

    ranked_is_tp: esito delle predizioni di una classe, per score decrescente.
    Precision interpolata (massimo a recall maggiore o uguale), come in VOC/COCO.
    """
    if n_gt == 0 or not ranked_is_tp:
        return 0.0
    tp = fp = 0
    precision, recall = [], []
    for is_tp in ranked_is_tp:
        tp += is_tp
        fp += not is_tp
        precision.append(tp / (tp + fp))
        recall.append(tp / n_gt)
    for i in range(len(precision) - 2, -1, -1):
        precision[i] = max(precision[i], precision[i + 1])
    ap, prev_r = 0.0, 0.0
    for p, r in zip(precision, recall):
        ap += (r - prev_r) * p
        prev_r = r
    return ap


def _summarize(st: dict, conf_thresh: float) -> None:
    """Metriche a partire dagli esiti delle predizioni.

    - precision/recall/F1 alla soglia conf_thresh;
    - F1 massimo al variare della soglia (e soglia corrispondente);
    - AP@0.5 per classe e mAP@0.5, indipendenti dalla soglia: sono le metriche
      per confrontare esperimenti, perche' modelli diversi sono calibrati in modo
      diverso e una soglia fissa ne favorirebbe alcuni.
    """
    counted = sorted((p for p in st["preds"] if p[2] != "ign"), key=lambda p: -p[1])
    n_gt = sum(st["n_gt"])
    st["n_ignored_pred"] = sum(1 for p in st["preds"] if p[2] == "ign")

    st["tp"] = [0] * NUM_CLASSES
    st["fp"] = [0] * NUM_CLASSES
    st["tp_sz"] = {"medi": 0, "grandi": 0}
    for cls, score, status, size in counted:
        if score <= conf_thresh:
            break
        if status == "tp":
            st["tp"][cls] += 1
            st["tp_sz"][size] += 1
        else:
            st["fp"][cls] += 1
    st["precision"], st["recall"], st["f1"] = _prf(sum(st["tp"]), sum(st["fp"]), n_gt)

    st["f1_max"], st["best_conf"] = 0.0, conf_thresh
    for t in [round(0.05 * k, 2) for k in range(1, 20)]:
        tp = sum(1 for _, s, status, _ in counted if s > t and status == "tp")
        fp = sum(1 for _, s, status, _ in counted if s > t and status == "fp")
        f1 = _prf(tp, fp, n_gt)[2]
        # a parita' di F1 si preferisce la soglia piu' alta (meno falsi positivi)
        if f1 > 0 and f1 >= st["f1_max"]:
            st["f1_max"], st["best_conf"] = f1, t

    st["ap"] = [_average_precision([p[2] == "tp" for p in counted if p[0] == c], st["n_gt"][c])
                for c in range(NUM_CLASSES)]
    present = [c for c in range(NUM_CLASSES) if st["n_gt"][c] > 0]
    st["map50"] = sum(st["ap"][c] for c in present) / len(present) if present else 0.0


@torch.no_grad()
def evaluate_model(model, loader, anchors, device, conf_thresh, nms_thresh,
                   min_size: float = EVAL_MIN_OBJ_SIZE) -> dict:
    """Valuta il modello su tutte le immagini del loader (senza augmentation).

    Le predizioni si raccolgono da EVAL_CONF_MIN in su; precision/recall/F1 sono
    riportati a conf_thresh, mAP@0.5 e F1 massimo su tutte le soglie.
    """
    model.eval()
    st = _new_stats(min_size)
    for imgs, boxes in loader:
        batch_results = postprocess(model(imgs.to(device)), anchors,
                                    min(EVAL_CONF_MIN, conf_thresh), nms_thresh)
        for results, gt in zip(batch_results, boxes):
            _match_image(results, gt, st)
    _summarize(st, conf_thresh)
    return st


def print_report(st: dict, conf_thresh: float) -> None:
    min_size = st["min_size"]
    print(f"Precision/recall/F1 a confidenza > {conf_thresh}; AP@0.5 indipendente dalla soglia")
    print(f"{'classe':10s} {'precision':>10s} {'recall':>8s} {'F1':>7s} {'AP@0.5':>7s} {'#GT':>6s}")
    for c in range(NUM_CLASSES):
        if st["n_gt"][c] == 0:
            continue
        p, r, f1 = _prf(st["tp"][c], st["fp"][c], st["n_gt"][c])
        print(f"{CLASS_NAMES[c]:10s} {p:10.4f} {r:8.4f} {f1:7.4f} {st['ap'][c]:7.4f} "
              f"{st['n_gt'][c]:6d}")
    print("-" * 53)
    print(f"{'totale':10s} {st['precision']:10.4f} {st['recall']:8.4f} "
          f"{st['f1']:7.4f} {st['map50']:7.4f} {sum(st['n_gt']):6d}")
    print(f"\nmAP@0.5 = {st['map50']:.4f}   F1 massimo = {st['f1_max']:.4f} "
          f"(confidenza > {st['best_conf']})")

    # --- recall per fascia di dimensione (stile COCO) ---
    # La precision non e' scomponibile per fascia (un falso positivo non ha un
    # oggetto reale a cui riferirsi), quindi qui si riporta il recall.
    print(f"\nRecall per dimensione (lato minore; soglia = {min_size:.3f} del lato, "
          f"{min_size * IMG_SIZE:.0f}px a {IMG_SIZE}px)")
    print(f"{'fascia':10s} {'recall':>8s} {'#GT':>7s}  {'lato minore':>16s}")
    labels = {"medi": f"{min_size:.3f}-{3 * min_size:.3f}", "grandi": f">= {3 * min_size:.3f}"}
    for name in ("medi", "grandi"):
        n = st["n_gt_sz"][name]
        if n:
            print(f"{name:10s} {st['tp_sz'][name] / n:8.4f} {n:7d}  {labels[name]:>16s}")

    tot = sum(st["n_gt"]) + st["n_small_gt"] + st["n_crowd_gt"]
    print(f"\nRegioni ignorate: {st['n_small_gt']} oggetti sotto soglia e "
          f"{st['n_crowd_gt']} box crowd, su {tot} box totali; "
          f"{st['n_ignored_pred']} predizioni non conteggiate")


def evaluate(weights, conf_thresh, split: str = "val", nms_thresh: float = NMS_IOU_THRESH,
             min_size: float = EVAL_MIN_OBJ_SIZE):
    """Carica un checkpoint, lo valuta su uno split e stampa il report.

    Ritorna (precision, recall) complessivi.
    """
    model = YOLO().to(DEVICE)
    ck = torch.load(weights, map_location=DEVICE)
    check_checkpoint_config(ck)
    model.load_state_dict(ck["model"])

    ds = Object365Detection(split)  # niente augmentation
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False,
                        collate_fn=collate_fn, num_workers=2)
    print(f"[valutazione] split={split}  immagini={len(ds)}  conf={conf_thresh}")
    st = evaluate_model(model, loader, load_anchors(), DEVICE, conf_thresh, nms_thresh,
                        min_size)
    print_report(st, conf_thresh)
    return st["precision"], st["recall"]


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--weights", default=str(Path(OUTPUT_DIR) / "best.pt"))
    p.add_argument("--split", default="val", choices=["train", "val", "test"])
    p.add_argument("--conf", type=float, default=CONF_THRESH)
    p.add_argument("--nms", type=float, default=NMS_IOU_THRESH)
    p.add_argument("--min_size", type=float, default=EVAL_MIN_OBJ_SIZE,
                   help="lato minore minimo degli oggetti da valutare (frazione del lato)")
    args = p.parse_args()
    evaluate(args.weights, args.conf, args.split, args.nms, args.min_size)

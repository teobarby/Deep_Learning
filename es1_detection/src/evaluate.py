"""Valutazione del detector: precision e recall a IoU>=0.5 sul validation set.

Per ogni classe si confrontano le predizioni con la ground truth usando l'IoU
(la misura vista a lezione per valutare le bounding box): una predizione e' un
vero positivo (TP) se ha IoU >= 0.5 con un oggetto reale non ancora abbinato,
altrimenti e' un falso positivo (FP). Da qui:

    precision = TP / (TP + FP)      recall = TP / (numero di oggetti reali)

Uso:
    python src/evaluate.py
    python src/evaluate.py --weights outputs/best.pt --conf 0.25
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image

from config import (
    CLASS_NAMES,
    CONF_THRESH,
    DEVICE,
    IMG_SIZE,
    NMS_IOU_THRESH,
    NUM_CLASSES,
    OUTPUT_DIR,
    STRIDE,
    load_anchors,
)
from dataset import Object365Detection
from detect import detect
from model import YOLO
from utils import iou_xyxy, xywh_to_xyxy

IOU_MATCH = 0.5  # soglia di IoU per considerare una predizione corretta

# Fasce di dimensione (lato minore dell'oggetto, in pixel a IMG_SIZE).
# Il confine "piccolo" e' la dimensione di una cella della griglia (STRIDE): un
# oggetto piu' piccolo di una cella e' sotto il limite di risoluzione del
# detector a scala singola e non e' praticamente rilevabile.
SIZE_BINS = [
    ("piccoli",  0.0,        float(STRIDE)),       # < 1 cella
    ("medi",     float(STRIDE), 3.0 * STRIDE),     # 1-3 celle
    ("grandi",   3.0 * STRIDE, float("inf")),      # > 3 celle
]


def _size_bin(box_xywh) -> str:
    """Fascia di dimensione di una box (cx,cy,w,h normalizzati)."""
    side = min(float(box_xywh[2]), float(box_xywh[3])) * IMG_SIZE
    for name, lo, hi in SIZE_BINS:
        if lo <= side < hi:
            return name
    return SIZE_BINS[-1][0]


def evaluate(weights, conf_thresh):
    anchors = load_anchors()
    model = YOLO().to(DEVICE)
    ck = torch.load(weights, map_location=DEVICE)
    model.load_state_dict(ck["model"])
    model.eval()

    val = Object365Detection("val")
    tp = {c: 0 for c in range(NUM_CLASSES)}   # veri positivi per classe
    fp = {c: 0 for c in range(NUM_CLASSES)}   # falsi positivi per classe
    n_gt = {c: 0 for c in range(NUM_CLASSES)}  # oggetti reali per classe

    # stesse quantita' suddivise per fascia di dimensione dell'oggetto
    bins = [b[0] for b in SIZE_BINS]
    tp_sz = {b: 0 for b in bins}
    n_gt_sz = {b: 0 for b in bins}

    for idx in range(len(val)):
        name = val.ids[idx]
        img = Image.open(val.img_dir / f"{name}.jpg").convert("RGB")
        gt = val._load_boxes(val.lbl_dir / f"{name}.txt")  # (M,5) cls,cx,cy,w,h

        results = detect(model, img, anchors, DEVICE, conf_thresh, NMS_IOU_THRESH)

        # GT in xyxy normalizzate, per classe (con la fascia di dimensione)
        gt_by_cls = {c: [] for c in range(NUM_CLASSES)}
        sz_by_cls = {c: [] for c in range(NUM_CLASSES)}
        for row in gt:
            c = int(row[0])
            gt_by_cls[c].append(xywh_to_xyxy(row[1:5]).tolist())
            b = _size_bin(row[1:5])
            sz_by_cls[c].append(b)
            n_gt[c] += 1
            n_gt_sz[b] += 1
        matched = {c: [False] * len(gt_by_cls[c]) for c in range(NUM_CLASSES)}

        # predizioni ordinate per score decrescente (gia' cosi' da detect)
        for cls, score, box in results:
            gts = gt_by_cls[cls]
            if gts:
                ious = iou_xyxy(torch.tensor(box).unsqueeze(0), torch.tensor(gts))
                best = int(ious.argmax())
                if float(ious[best]) >= IOU_MATCH and not matched[cls][best]:
                    tp[cls] += 1
                    matched[cls][best] = True
                    tp_sz[sz_by_cls[cls][best]] += 1
                    continue
            fp[cls] += 1

    print(f"{'classe':10s} {'precision':>10s} {'recall':>8s} {'#GT':>6s}")
    tot_tp = tot_fp = tot_gt = 0
    for c in range(NUM_CLASSES):
        if n_gt[c] == 0:
            continue
        prec = tp[c] / (tp[c] + fp[c] + 1e-9)
        rec = tp[c] / (n_gt[c] + 1e-9)
        tot_tp += tp[c]; tot_fp += fp[c]; tot_gt += n_gt[c]
        print(f"{CLASS_NAMES[c]:10s} {prec:10.4f} {rec:8.4f} {n_gt[c]:6d}")

    prec = tot_tp / (tot_tp + tot_fp + 1e-9)
    rec = tot_tp / (tot_gt + 1e-9)
    print("-" * 38)
    print(f"{'totale':10s} {prec:10.4f} {rec:8.4f} {tot_gt:6d}")

    # --- recall per fascia di dimensione (stile COCO) ---
    # La precision non e' scomponibile per fascia (un falso positivo non ha un
    # oggetto reale a cui riferirsi), quindi qui si riporta il recall.
    cell = STRIDE
    print(f"\nRecall per dimensione dell'oggetto (lato minore, a {IMG_SIZE}px; "
          f"1 cella = {cell}px)")
    print(f"{'fascia':10s} {'recall':>8s} {'#GT':>7s}  {'soglia':>16s}")
    labels = {
        "piccoli": f"< {cell}px (1 cella)",
        "medi": f"{cell}-{3*cell}px",
        "grandi": f"> {3*cell}px",
    }
    for b in bins:
        r = tp_sz[b] / (n_gt_sz[b] + 1e-9)
        print(f"{b:10s} {r:8.4f} {n_gt_sz[b]:7d}  {labels[b]:>16s}")
    small_frac = n_gt_sz["piccoli"] / (sum(n_gt_sz.values()) + 1e-9)
    print(f"\nGli oggetti piu' piccoli di una cella sono il {small_frac*100:.1f}% del "
          f"totale: sotto il limite\ndi risoluzione di un detector a scala singola "
          f"con stride {STRIDE}.")
    return prec, rec


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--weights", default=str(Path(OUTPUT_DIR) / "best.pt"))
    p.add_argument("--conf", type=float, default=CONF_THRESH)
    args = p.parse_args()
    evaluate(args.weights, args.conf)

"""Inferenza: rileva oggetti in un'immagine e disegna le bounding box.

Applica il modello, decodifica le predizioni, filtra per confidenza e applica
il non-max suppression (from scratch). Salva un'immagine annotata.

Uso:
    python src/detect.py path/immagine.jpg
    python src/detect.py path/immagine.jpg --conf 0.3 --out result.jpg
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont, ImageOps

from dataset import image_transform

from config import (
    CLASS_NAMES,
    CONF_THRESH,
    DEVICE,
    NMS_IOU_THRESH,
    OUTPUT_DIR,
    check_checkpoint_config,
    load_anchors,
)
from loss import decode_predictions
from model import YOLO
from utils import nms, xywh_to_xyxy

_COLORS = [
    (255, 99, 71), (60, 179, 113), (65, 105, 225), (255, 165, 0),
    (147, 112, 219), (0, 206, 209), (255, 20, 147), (154, 205, 50),
]


# Massimo di box candidate per immagine prima dell'NMS. Con un modello ancora poco
# addestrato (valutazione durante il training) migliaia di box possono superare la
# soglia, e l'NMS, quadratico nel numero di box, diventerebbe lentissimo.
MAX_PRE_NMS = 300


@torch.no_grad()
def postprocess(pred, anchors, conf_thresh, nms_thresh):
    """Da output grezzo (B,A,S,S,5+C) a detection finali, una lista per immagine.

    Ogni detection e' (classe_idx, score, (x1,y1,x2,y2)) in coord [0,1]; le liste
    sono ordinate per score decrescente.
    """
    boxes_grid, obj_logit, cls_logit = decode_predictions(pred, anchors.to(pred.device))
    S = pred.shape[2]
    batch_results = []
    for b in range(pred.shape[0]):
        boxes = boxes_grid[b].reshape(-1, 4) / S               # (N,4) xywh in [0,1]
        obj = torch.sigmoid(obj_logit[b]).reshape(-1)          # (N,)
        cls_prob = torch.softmax(cls_logit[b].reshape(-1, len(CLASS_NAMES)), dim=-1)
        cls_score, cls_idx = cls_prob.max(dim=-1)              # (N,)
        scores = obj * cls_score

        keep = scores > conf_thresh
        boxes, scores, cls_idx = boxes[keep], scores[keep], cls_idx[keep]
        if scores.numel() > MAX_PRE_NMS:
            top = scores.topk(MAX_PRE_NMS).indices
            boxes, scores, cls_idx = boxes[top], scores[top], cls_idx[top]
        # le poche box rimaste si gestiscono su CPU: evita una sync GPU per box
        boxes, scores, cls_idx = boxes.cpu(), scores.cpu(), cls_idx.cpu()

        results = []
        if scores.numel() > 0:
            boxes_xyxy = xywh_to_xyxy(boxes).clamp(0, 1)
            for c in cls_idx.unique():                          # NMS per classe
                m = cls_idx == c
                idx = nms(boxes_xyxy[m], scores[m], nms_thresh)
                for box, s in zip(boxes_xyxy[m][idx], scores[m][idx]):
                    results.append((int(c), float(s), tuple(box.tolist())))
            results.sort(key=lambda r: r[1], reverse=True)
        batch_results.append(results)
    return batch_results


@torch.no_grad()
def detect(model, img_pil, anchors, device, conf_thresh, nms_thresh):
    """Ritorna una lista di (classe_idx, score, (x1,y1,x2,y2)) in coord [0,1]."""
    # stessa trasformazione usata in training (inclusa la normalizzazione)
    x = image_transform()(img_pil).unsqueeze(0).to(device)
    return postprocess(model(x), anchors, conf_thresh, nms_thresh)[0]


def _load_font(size: int = 16):
    """Font di sistema, cercato su macOS / Windows / Linux; fallback al default."""
    for path in (
        "/System/Library/Fonts/Supplemental/Arial.ttf",   # macOS
        "C:/Windows/Fonts/arial.ttf",                     # Windows
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",  # Linux
    ):
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            continue
    return ImageFont.load_default()


def draw(img_pil, results):
    img = img_pil.convert("RGB").copy()
    d = ImageDraw.Draw(img)
    W, H = img.size
    font = _load_font(16)
    for cls, score, (x1, y1, x2, y2) in results:
        color = _COLORS[cls % len(_COLORS)]
        px = [x1 * W, y1 * H, x2 * W, y2 * H]
        d.rectangle(px, outline=color, width=3)
        label = f"{CLASS_NAMES[cls]} {score:.2f}"
        tb = d.textbbox((0, 0), label, font=font)
        tw, th = tb[2] - tb[0], tb[3] - tb[1]
        d.rectangle([px[0], px[1] - th - 4, px[0] + tw + 4, px[1]], fill=color)
        d.text((px[0] + 2, px[1] - th - 3), label, fill=(255, 255, 255), font=font)
    return img


def main():
    p = argparse.ArgumentParser()
    p.add_argument("image", type=str)
    p.add_argument("--weights", default=str(Path(OUTPUT_DIR) / "best.pt"))
    p.add_argument("--conf", type=float, default=None,
                   help="soglia di confidenza (default: quella salvata nel checkpoint)")
    p.add_argument("--nms", type=float, default=NMS_IOU_THRESH)
    p.add_argument("--out", default=str(Path(OUTPUT_DIR) / "detection.jpg"))
    args = p.parse_args()

    anchors = load_anchors()
    model = YOLO().to(DEVICE)
    ck = torch.load(args.weights, map_location=DEVICE)
    check_checkpoint_config(ck)
    model.load_state_dict(ck["model"])
    model.eval()

    # soglia che massimizza l'F1 sul validation set, se registrata nel checkpoint
    conf = args.conf if args.conf is not None else ck.get("best_conf", CONF_THRESH)
    # le foto del telefono salvano spesso la rotazione nei metadati EXIF invece
    # di ruotare i pixel: senza applicarla la rete vedrebbe l'immagine coricata
    img = ImageOps.exif_transpose(Image.open(args.image)).convert("RGB")
    results = detect(model, img, anchors, DEVICE, conf, args.nms)
    print(f"[detect] {len(results)} oggetti rilevati:")
    for cls, score, box in results:
        print(f"    {CLASS_NAMES[cls]:8s} {score:.2f}  bbox(norm)={tuple(round(b,3) for b in box)}")
    out = draw(img, results)
    out.save(args.out)
    print(f"[salvato] {args.out}")


if __name__ == "__main__":
    main()

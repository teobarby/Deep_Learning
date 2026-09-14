"""Funzioni geometriche implementate a mano: IoU, conversioni bbox, NMS.

Convenzioni:
  - "xywh": (cx, cy, w, h) con centro e dimensioni.
  - "xyxy": (x1, y1, x2, y2) angoli.
Le coordinate possono essere normalizzate [0,1] o in pixel a seconda del contesto;
le funzioni sono agnostiche rispetto all'unita' di misura.
"""
from __future__ import annotations

import torch


def xywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    """(..., 4) da (cx,cy,w,h) a (x1,y1,x2,y2)."""
    cx, cy, w, h = boxes.unbind(-1)
    return torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=-1)


def xyxy_to_xywh(boxes: torch.Tensor) -> torch.Tensor:
    x1, y1, x2, y2 = boxes.unbind(-1)
    return torch.stack([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1], dim=-1)


def iou_xyxy(boxes_a: torch.Tensor, boxes_b: torch.Tensor) -> torch.Tensor:
    """IoU elemento-per-elemento tra due insiemi di box in formato xyxy.

    boxes_a, boxes_b: (..., 4) broadcastabili. Ritorna (...) IoU.
    Corrisponde alla 'Intersection over Union' delle slide (valutazione bbox).
    """
    x1 = torch.max(boxes_a[..., 0], boxes_b[..., 0])
    y1 = torch.max(boxes_a[..., 1], boxes_b[..., 1])
    x2 = torch.min(boxes_a[..., 2], boxes_b[..., 2])
    y2 = torch.min(boxes_a[..., 3], boxes_b[..., 3])

    inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
    area_a = (boxes_a[..., 2] - boxes_a[..., 0]).clamp(min=0) * \
             (boxes_a[..., 3] - boxes_a[..., 1]).clamp(min=0)
    area_b = (boxes_b[..., 2] - boxes_b[..., 0]).clamp(min=0) * \
             (boxes_b[..., 3] - boxes_b[..., 1]).clamp(min=0)
    union = area_a + area_b - inter + 1e-9
    return inter / union


def wh_iou(wh_a: torch.Tensor, wh_b: torch.Tensor) -> torch.Tensor:
    """IoU tra box considerando solo (w,h), centrati nell'origine.

    Serve per assegnare un oggetto all'anchor box piu' simile per forma.
    wh_a: (N,2), wh_b: (M,2) -> ritorna (N,M).
    """
    wh_a = wh_a.unsqueeze(1)  # (N,1,2)
    wh_b = wh_b.unsqueeze(0)  # (1,M,2)
    inter = torch.min(wh_a, wh_b).prod(dim=-1)      # (N,M)
    area_a = wh_a.prod(dim=-1)
    area_b = wh_b.prod(dim=-1)
    return inter / (area_a + area_b - inter + 1e-9)


def nms(boxes_xyxy: torch.Tensor, scores: torch.Tensor, iou_thresh: float) -> torch.Tensor:
    """Non-Max Suppression 'from scratch' (algoritmo delle slide).

    Mentre restano box: prendi quello con score massimo, scarta tutti gli altri
    con IoU > soglia rispetto a esso.

    Ritorna gli indici dei box mantenuti, ordinati per score decrescente.
    """
    if boxes_xyxy.numel() == 0:
        return torch.empty((0,), dtype=torch.long, device=boxes_xyxy.device)

    order = scores.argsort(descending=True)
    keep = []
    while order.numel() > 0:
        i = order[0]
        keep.append(i.item())
        if order.numel() == 1:
            break
        rest = order[1:]
        ious = iou_xyxy(boxes_xyxy[i].unsqueeze(0), boxes_xyxy[rest])
        order = rest[ious <= iou_thresh]
    return torch.tensor(keep, dtype=torch.long, device=boxes_xyxy.device)

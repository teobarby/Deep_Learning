"""Loss YOLO implementata da zero.

Dato l'output grezzo del modello (B, A, S, S, 5+C) e le box di ground truth,
si costruisce il target di griglia:
  - ogni oggetto viene assegnato alla cella che contiene il suo centro e
    all'anchor box con IoU (di sola forma w,h) massima;
  - gli anchor 'responsabili' hanno objectness target 1 e ricevono la loss di
    coordinate e di classe;
  - gli oggetti sotto la soglia MIN_OBJ_SIZE non hanno anchor responsabili: la
    cella che ne contiene il centro e' ignorata (nessuna loss, per ogni anchor);
  - le box 'crowd' non hanno anchor responsabili: sono ignorate tutte le celle il
    cui centro cade dentro la box (almeno quella del centro della box);
  - tutti gli altri anchor hanno objectness target 0.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from config import (
    LAMBDA_COORD,
    LAMBDA_NOOBJ,
    is_small,
)
from utils import wh_iou


def _cells_inside(lo: float, hi: float, S: int) -> range:
    """Indici delle celle (su un asse) il cui centro (k+0.5)/S cade in [lo, hi]."""
    first = max(math.ceil(lo * S - 0.5), 0)
    last = min(math.floor(hi * S - 0.5), S - 1)
    return range(first, last + 1)


def decode_predictions(pred: torch.Tensor, anchors: torch.Tensor):
    """Decodifica i logit in box in unita' di CELLA della griglia.

    pred: (B, A, S, S, 5+C). anchors: (A, 2) in unita' di cella.
    Ritorna (boxes_xywh_grid, obj_logit, class_logit).
    """
    B, A, S, _, _ = pred.shape
    device = pred.device
    # griglia degli offset di cella
    gy, gx = torch.meshgrid(torch.arange(S, device=device),
                            torch.arange(S, device=device), indexing="ij")
    gx = gx.view(1, 1, S, S).float()
    gy = gy.view(1, 1, S, S).float()
    aw = anchors[:, 0].view(1, A, 1, 1).to(device)
    ah = anchors[:, 1].view(1, A, 1, 1).to(device)

    tx, ty, tw, th = pred[..., 0], pred[..., 1], pred[..., 2], pred[..., 3]
    bx = torch.sigmoid(tx) + gx
    by = torch.sigmoid(ty) + gy
    bw = aw * torch.exp(tw)
    bh = ah * torch.exp(th)
    boxes = torch.stack([bx, by, bw, bh], dim=-1)  # (B,A,S,S,4) in celle
    return boxes, pred[..., 4], pred[..., 5:]


class YOLOLoss(nn.Module):
    def __init__(self, anchors: torch.Tensor):
        super().__init__()
        self.register_buffer("anchors", anchors)  # (A,2) in celle
        self.bce = nn.BCEWithLogitsLoss(reduction="sum")
        self.mse = nn.MSELoss(reduction="sum")
        self.ce = nn.CrossEntropyLoss(reduction="sum")

    def forward(self, pred: torch.Tensor, targets: list[torch.Tensor]):
        B, A, S, _, _ = pred.shape
        device = pred.device
        anchors = self.anchors.to(device)

        pred_obj = pred[..., 4]
        pred_cls = pred[..., 5:]

        # --- costruzione dei target di responsabilita' su CPU ---
        # (evita migliaia di micro-operazioni/sync sulla GPU: gli assegnamenti
        #  scalari si fanno su tensori CPU, poi si sposta tutto sul device.)
        anchors_cpu = anchors.detach().cpu()
        obj_mask = torch.zeros((B, A, S, S), dtype=torch.bool)
        ignore_mask = torch.zeros((B, A, S, S), dtype=torch.bool)
        tx = torch.zeros((B, A, S, S))
        ty = torch.zeros((B, A, S, S))
        tw = torch.zeros((B, A, S, S))
        th = torch.zeros((B, A, S, S))
        tcls = torch.zeros((B, A, S, S), dtype=torch.long)

        n_obj = 0
        for b in range(B):
            gt = targets[b]
            if gt.numel() == 0:
                continue
            cls = gt[:, 0].long()
            gxy = gt[:, 1:3] * S             # centro in celle
            gwh = (gt[:, 3:5] * S).clamp(min=1e-6)  # w,h in celle
            gij = gxy.long().clamp(0, S - 1)
            gi, gj = gij[:, 0], gij[:, 1]

            best_a = wh_iou(gwh, anchors_cpu).argmax(dim=1)  # (N,) anchor migliore
            for n in range(gt.shape[0]):
                a, j, i = int(best_a[n]), int(gj[n]), int(gi[n])
                if gt.shape[1] > 5 and gt[n, 5] > 0:
                    # box crowd: ignora le celle coperte dalla box
                    cx, cy, w, h = gt[n, 1:5].tolist()
                    cols = _cells_inside(cx - w / 2, cx + w / 2, S) or range(i, i + 1)
                    rows = _cells_inside(cy - h / 2, cy + h / 2, S) or range(j, j + 1)
                    ignore_mask[b, :, rows.start:rows.stop, cols.start:cols.stop] = True
                    continue
                if is_small(float(gt[n, 3]), float(gt[n, 4])):
                    # sotto soglia: nessun target, ma neppure la penalita'
                    # no-object sulla sua cella
                    ignore_mask[b, :, j, i] = True
                    continue
                obj_mask[b, a, j, i] = True
                tx[b, a, j, i] = gxy[n, 0] - i
                ty[b, a, j, i] = gxy[n, 1] - j
                tw[b, a, j, i] = torch.log(gwh[n, 0] / anchors_cpu[a, 0] + 1e-16)
                th[b, a, j, i] = torch.log(gwh[n, 1] / anchors_cpu[a, 1] + 1e-16)
                tcls[b, a, j, i] = cls[n]
                n_obj += 1

        obj_mask = obj_mask.to(device)
        ignore_mask = ignore_mask.to(device)
        tx, ty = tx.to(device), ty.to(device)
        tw, th = tw.to(device), th.to(device)
        tcls = tcls.to(device)

        # --- maschera noobj: anchor non responsabili e non ignorati ---
        noobj_mask = ~obj_mask & ~ignore_mask

        n_obj = max(n_obj, 1)

        # --- componenti della loss (normalizzate per numero di oggetti) ---
        # coordinate: sigmoid(tx,ty) vs offset; (tw,th) vs log-target
        ptx = torch.sigmoid(pred[..., 0])
        pty = torch.sigmoid(pred[..., 1])
        loss_xy = self.mse(ptx[obj_mask], tx[obj_mask]) + \
                  self.mse(pty[obj_mask], ty[obj_mask])
        loss_wh = self.mse(pred[..., 2][obj_mask], tw[obj_mask]) + \
                  self.mse(pred[..., 3][obj_mask], th[obj_mask])
        loss_coord = LAMBDA_COORD * (loss_xy + loss_wh) / n_obj

        # objectness: media sugli anchor responsabili
        loss_obj = self.bce(pred_obj[obj_mask],
                            torch.ones_like(pred_obj[obj_mask])) / n_obj
        # no-object: media sulle celle VUOTE, non sul numero di oggetti.
        # Normalizzare per n_obj renderebbe il termine proporzionale al numero di
        # celle della griglia: passando da 13x13 a 52x52 (16 volte piu' celle) il
        # termine esploderebbe e la rete convergerebbe alla soluzione degenere
        # "non c'e' mai nulla". Con la media il peso e' indipendente dalla griglia.
        n_noobj = max(int(noobj_mask.sum()), 1)
        loss_noobj = LAMBDA_NOOBJ * self.bce(
            pred_obj[noobj_mask], torch.zeros_like(pred_obj[noobj_mask])) / n_noobj

        # classe
        if obj_mask.any():
            loss_cls = self.ce(pred_cls[obj_mask], tcls[obj_mask]) / n_obj
        else:
            loss_cls = torch.zeros((), device=device)

        total = loss_coord + loss_obj + loss_noobj + loss_cls
        parts = {
            "coord": loss_coord.item(),
            "obj": loss_obj.item(),
            "noobj": loss_noobj.item(),
            "cls": loss_cls.item(),
        }
        return total, parts

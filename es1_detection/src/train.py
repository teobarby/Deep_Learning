"""Training del detector YOLO 'from scratch'.

Il checkpoint migliore (best.pt) e' scelto sul mAP@0.5 del validation set, non
sulla val loss: la loss e' una somma pesata di termini (coordinate, objectness,
classe) e puo' scendere senza che le detection effettive migliorino. Il mAP non
dipende dalla soglia di confidenza, a differenza di precision/recall/F1. Ogni checkpoint contiene anche la configurazione con cui
e' stato addestrato, verificata al caricamento.

Uso:
    python src/train.py
    python src/train.py --epochs 40 --batch_size 16
    python src/train.py --eval_every 2      # valuta l'F1 ogni 2 epoch
    python src/train.py --resume            # riprende dall'ultimo checkpoint
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from config import (
    BATCH_SIZE,
    CONF_THRESH,
    CONFIG_OVERRIDES,
    DEVICE,
    EPOCHS,
    FREEZE_BACKBONE_EPOCHS,
    LEARNING_RATE,
    LR_SCHEDULE,
    NMS_IOU_THRESH,
    OUTPUT_DIR,
    SEED,
    WEIGHT_DECAY,
    check_checkpoint_config,
    checkpoint_config,
    load_anchors,
)
from dataset import Object365Detection, collate_fn
from evaluate import evaluate_model
from loss import YOLOLoss
from model import YOLO


def set_backbone_frozen(model, frozen: bool) -> None:
    """Congela/sblocca i pesi del backbone (feature extraction vs fine-tuning)."""
    for p in model.features.parameters():
        p.requires_grad = not frozen


def run_epoch(model, loader, crit, optimizer, device, train: bool):
    model.train(train)
    if train and not next(model.features.parameters()).requires_grad:
        # backbone congelato: anche le statistiche di batch norm restano quelle
        # apprese su ImageNet
        model.features.eval()
    agg = {"total": 0.0, "coord": 0.0, "obj": 0.0, "noobj": 0.0, "cls": 0.0}
    n = 0
    for imgs, boxes in loader:
        imgs = imgs.to(device)
        with torch.set_grad_enabled(train):
            pred = model(imgs)
            loss, parts = crit(pred, boxes)
        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        agg["total"] += loss.item()
        for k in parts:
            agg[k] += parts[k]
        n += 1
    return {k: v / max(n, 1) for k, v in agg.items()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    p.add_argument("--lr", type=float, default=LEARNING_RATE)
    p.add_argument("--eval_every", type=int, default=1,
                   help="ogni quante epoch calcolare le metriche sul validation set")
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()

    torch.manual_seed(SEED)
    device = DEVICE
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    print(f"[device] {device}")
    print(f"[output] {OUTPUT_DIR}  override={CONFIG_OVERRIDES or 'nessuno'}")

    train_ds = Object365Detection("train", augment=True)
    val_ds = Object365Detection("val")  # niente augmentation in validazione
    print(f"[dati] train={len(train_ds)}  val={len(val_ds)}")
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate_fn, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            collate_fn=collate_fn, num_workers=2)

    anchors = load_anchors()
    model = YOLO().to(device)
    crit = YOLOLoss(anchors).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 weight_decay=WEIGHT_DECAY)
    if LR_SCHEDULE == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    elif LR_SCHEDULE == "none":
        scheduler = None
    else:
        raise ValueError(f"LR_SCHEDULE non valido: {LR_SCHEDULE!r}")

    ckpt_path = Path(OUTPUT_DIR) / "last.pt"
    best_path = Path(OUTPUT_DIR) / "best.pt"
    history_path = Path(OUTPUT_DIR) / "history.json"  # metriche per epoch
    start_epoch = 0
    best_map = -1.0
    history = []
    if args.resume and ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location=device)
        check_checkpoint_config(ck)
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        if scheduler is not None and ck.get("scheduler"):
            scheduler.load_state_dict(ck["scheduler"])
        start_epoch = ck["epoch"] + 1
        best_map = ck.get("best_map50", best_map)
        if history_path.exists():
            history = json.loads(history_path.read_text())[:start_epoch]
        print(f"[resume] riparto dall'epoch {start_epoch}")

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        lr = optimizer.param_groups[0]["lr"]
        frozen = model.pretrained and epoch < FREEZE_BACKBONE_EPOCHS
        set_backbone_frozen(model, frozen)
        tr = run_epoch(model, train_loader, crit, optimizer, device, train=True)
        if scheduler is not None:
            scheduler.step()
        va = run_epoch(model, val_loader, crit, optimizer, device, train=False)
        line = (f"[epoch {epoch+1}/{args.epochs}] "
                f"train={tr['total']:.3f} (coord={tr['coord']:.2f} obj={tr['obj']:.2f} "
                f"noobj={tr['noobj']:.2f} cls={tr['cls']:.2f})  "
                f"val={va['total']:.3f}  lr={lr:.1e}{' [backbone congelato]' if frozen else ''}")

        record = {"epoch": epoch + 1, "lr": lr, "frozen": frozen, "train": tr, "val": va}
        map50 = None
        if (epoch + 1) % args.eval_every == 0 or epoch + 1 == args.epochs:
            m = evaluate_model(model, val_loader, anchors, device,
                               CONF_THRESH, NMS_IOU_THRESH)
            map50 = m["map50"]
            line += (f"  mAP50={map50:.3f}  F1max={m['f1_max']:.3f}@{m['best_conf']}"
                     f"  (P={m['precision']:.3f} R={m['recall']:.3f} a {CONF_THRESH})")
            record.update(map50=map50, f1_max=m["f1_max"], best_conf=m["best_conf"],
                          ap=m["ap"], precision=m["precision"], recall=m["recall"],
                          f1=m["f1"])
        record["seconds"] = round(time.time() - t0, 1)
        print(line + f"  ({record['seconds']:.0f}s)")
        history.append(record)
        history_path.write_text(json.dumps(history, indent=1))

        is_best = map50 is not None and map50 > best_map
        if is_best:
            best_map = map50
        ck = {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
              "scheduler": scheduler.state_dict() if scheduler is not None else None,
              "epoch": epoch, "best_map50": best_map, "config": checkpoint_config(),
              "train_args": vars(args)}
        if map50 is not None:
            ck["best_conf"] = m["best_conf"]
        torch.save(ck, ckpt_path)
        if is_best:
            torch.save(ck, best_path)
            print(f"    -> nuovo best (mAP50={best_map:.3f}) salvato in {best_path}")

    print("[fine] training completato.")


if __name__ == "__main__":
    main()

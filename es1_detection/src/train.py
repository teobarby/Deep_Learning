"""Training del detector YOLO 'from scratch'.

Uso:
    python src/train.py
    python src/train.py --epochs 40 --batch_size 16
    python src/train.py --resume            # riprende dall'ultimo checkpoint
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from config import (
    BATCH_SIZE,
    DEVICE,
    EPOCHS,
    LEARNING_RATE,
    OUTPUT_DIR,
    SEED,
    WEIGHT_DECAY,
    load_anchors,
)
from dataset import Object365Detection, collate_fn
from loss import YOLOLoss
from model import YOLO


def run_epoch(model, loader, crit, optimizer, device, train: bool):
    model.train(train)
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
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()

    torch.manual_seed(SEED)
    device = DEVICE
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    print(f"[device] {device}")

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

    ckpt_path = Path(OUTPUT_DIR) / "last.pt"
    best_path = Path(OUTPUT_DIR) / "best.pt"
    start_epoch = 0
    best_val = float("inf")
    if args.resume and ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        start_epoch = ck["epoch"] + 1
        best_val = ck.get("best_val", best_val)
        print(f"[resume] riparto dall'epoch {start_epoch}")

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        tr = run_epoch(model, train_loader, crit, optimizer, device, train=True)
        va = run_epoch(model, val_loader, crit, optimizer, device, train=False)
        dt = time.time() - t0
        print(f"[epoch {epoch+1}/{args.epochs}] "
              f"train={tr['total']:.3f} (coord={tr['coord']:.2f} obj={tr['obj']:.2f} "
              f"noobj={tr['noobj']:.2f} cls={tr['cls']:.2f})  "
              f"val={va['total']:.3f}  ({dt:.0f}s)")

        ck = {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
              "epoch": epoch, "best_val": best_val}
        torch.save(ck, ckpt_path)
        if va["total"] < best_val:
            best_val = va["total"]
            ck["best_val"] = best_val
            torch.save(ck, best_path)
            print(f"    -> nuovo best (val={best_val:.3f}) salvato in {best_path}")

    print("[fine] training completato.")


if __name__ == "__main__":
    main()

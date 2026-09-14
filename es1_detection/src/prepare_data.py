"""Preparazione del subset di Object365 per il detector YOLO.

Passi:
  1. Legge gli shard parquet gia' scaricati da HuggingFace (surenreddy/object365).
  2. Tiene solo le bounding box delle 8 classi selezionate.
  3. Scarta le immagini senza alcun oggetto delle classi scelte.
  4. Salva le immagini in data/images/{train,val}/ e le label in formato YOLO
     (una riga per box: "class cx cy w h", normalizzate in [0,1]) in data/labels/.

Gli anchor box sono scelti a mano in config.ANCHORS (non ricavati dai dati).

Uso:
    python src/prepare_data.py                 # usa tutti gli shard in cache
    python src/prepare_data.py --max_images 3000
"""
from __future__ import annotations

import argparse
import glob
import io
import json
import os
import random

import pyarrow.parquet as pq
from PIL import Image

from config import (
    DATA_DIR,
    OBJ365_ID_TO_CLASS,
    SEED,
)

HF_CACHE_GLOB = os.path.expanduser(
    "~/.cache/huggingface/hub/datasets--surenreddy--object365/snapshots/*/data/train-patch00-*.parquet"
)


def prepare(max_images: int | None, val_ratio: float = 0.1):
    random.seed(SEED)
    shards = sorted(glob.glob(HF_CACHE_GLOB))
    if not shards:
        raise FileNotFoundError(
            "Nessuno shard trovato in cache. Scarica prima i parquet di "
            "surenreddy/object365.")
    print(f"[prepare] {len(shards)} shard trovati")

    for split in ("train", "val"):
        (DATA_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (DATA_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

    kept = 0
    per_class = {c: 0 for c in range(len(OBJ365_ID_TO_CLASS))}
    uid = 0

    for shard in shards:
        tbl = pq.read_table(shard, columns=["image", "width", "height", "annotations"])
        images = tbl["image"].to_pylist()
        widths = tbl["width"].to_pylist()
        heights = tbl["height"].to_pylist()
        anns = tbl["annotations"].to_pylist()

        for img_struct, W, H, ann_json in zip(images, widths, heights, anns):
            if max_images is not None and kept >= max_images:
                break
            objs = json.loads(ann_json)
            lines = []
            for o in objs:
                cid = o["category_id"]
                if cid not in OBJ365_ID_TO_CLASS:
                    continue
                cls = OBJ365_ID_TO_CLASS[cid]
                x, y, w, h = o["bbox"]  # formato xywh in pixel (angolo alto-sx)
                # centro normalizzato
                cx = (x + w / 2) / W
                cy = (y + h / 2) / H
                nw = w / W
                nh = h / H
                # scarta box degeneri o fuori immagine
                if nw <= 0 or nh <= 0 or cx <= 0 or cy <= 0 or cx >= 1 or cy >= 1:
                    continue
                cx, cy = min(cx, 1.0), min(cy, 1.0)
                nw, nh = min(nw, 1.0), min(nh, 1.0)
                lines.append((cls, cx, cy, nw, nh))
                per_class[cls] += 1

            if not lines:
                continue

            split = "val" if random.random() < val_ratio else "train"
            name = f"{uid:06d}"
            uid += 1
            img = Image.open(io.BytesIO(img_struct["bytes"])).convert("RGB")
            img.save(DATA_DIR / "images" / split / f"{name}.jpg", quality=90)
            with open(DATA_DIR / "labels" / split / f"{name}.txt", "w") as f:
                for cls, cx, cy, nw, nh in lines:
                    f.write(f"{cls} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n")
            kept += 1
            if kept % 500 == 0:
                print(f"[prepare] {kept} immagini salvate...")
        if max_images is not None and kept >= max_images:
            break

    print(f"[prepare] immagini tenute: {kept}  |  box totali: {sum(per_class.values())}")
    from config import CLASS_NAMES
    for c, n in per_class.items():
        print(f"    {CLASS_NAMES[c]:8s}: {n}")
    # Gli anchor box sono scelti a mano in config.ANCHORS (non ricavati dai dati).


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--max_images", type=int, default=None)
    p.add_argument("--val_ratio", type=float, default=0.1)
    args = p.parse_args()
    prepare(args.max_images, args.val_ratio)

"""Preparazione del subset di Object365 per il detector YOLO.

Passi:
  1. Scarica i primi N shard parquet da HuggingFace (surenreddy/object365); gli
     shard gia' presenti nella cache di HuggingFace non vengono riscaricati.
  2. Tiene solo le bounding box delle categorie mappate in OBJ365_ID_TO_CLASS.
  3. Scarta le immagini senza alcun oggetto (non crowd) delle classi scelte.
  4. Divide le immagini in train / val / test e salva le immagini in
     data/images/<split>/ e le label in data/labels/<split>/, una riga per box:
     "class cx cy w h crowd" (formato YOLO normalizzato in [0,1], piu' il flag
     crowd = 1 per le box iscrowd, che saranno regioni ignorate).

Il validation set serve a scegliere checkpoint e iperparametri; il test set resta
da parte per la valutazione finale, cosi' i numeri riportati non sono gonfiati
dalle scelte fatte guardando il validation.

Gli anchor box sono scelti a mano in config.ANCHORS (non ricavati dai dati).

Uso:
    python src/prepare_data.py                 # 15 shard (~1,9 GB)
    python src/prepare_data.py --num_shards 2 --max_images 1000
"""
from __future__ import annotations

import argparse
import io
import json
import random

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from PIL import Image

from config import (
    CLASS_NAMES,
    DATA_DIR,
    OBJ365_ID_TO_CLASS,
    SEED,
)

HF_REPO = "surenreddy/object365"
# Il dataset e' diviso in shard parquet da ~120 MB l'uno:
# data/train-patch00-000.parquet ... train-patch00-034.parquet, poi patch01, ...
SHARD_NAME = "data/train-patch00-{:03d}.parquet"
MAX_SHARDS = 35
SPLITS = ("train", "val", "test")


def download_shards(num_shards: int) -> list[str]:
    """Scarica (o recupera dalla cache) i primi num_shards shard; ritorna i path."""
    if not 1 <= num_shards <= MAX_SHARDS:
        raise ValueError(f"num_shards deve essere tra 1 e {MAX_SHARDS}")
    paths = []
    for k in range(num_shards):
        print(f"[prepare] shard {k + 1}/{num_shards}")
        paths.append(hf_hub_download(HF_REPO, SHARD_NAME.format(k), repo_type="dataset"))
    return paths


def pick_split(val_ratio: float, test_ratio: float) -> str:
    r = random.random()
    if r < val_ratio:
        return "val"
    if r < val_ratio + test_ratio:
        return "test"
    return "train"


def prepare(max_images: int | None, num_shards: int,
            val_ratio: float = 0.1, test_ratio: float = 0.1):
    random.seed(SEED)
    # Dati preesistenti prodotti con altri parametri resterebbero nelle cartelle,
    # e la stessa immagine potrebbe comparire in due split (es. train e test).
    if any((DATA_DIR / "images").glob("*/*.jpg")):
        raise FileExistsError(
            f"{DATA_DIR / 'images'} contiene gia' immagini: elimina {DATA_DIR / 'images'} "
            f"e {DATA_DIR / 'labels'} prima di rigenerare il dataset.")
    shards = download_shards(num_shards)

    for split in SPLITS:
        (DATA_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (DATA_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

    kept = 0
    per_class = {c: 0 for c in range(len(CLASS_NAMES))}
    crowd_per_class = {c: 0 for c in range(len(CLASS_NAMES))}
    per_split = {s: 0 for s in SPLITS}
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
                crowd = int(bool(o.get("iscrowd")))
                lines.append((cls, cx, cy, nw, nh, crowd))

            # serve almeno un oggetto vero: un'immagine con sole box crowd non
            # contiene nulla da localizzare
            if not any(crowd == 0 for *_, crowd in lines):
                continue
            for cls, *_, crowd in lines:
                (crowd_per_class if crowd else per_class)[cls] += 1

            split = pick_split(val_ratio, test_ratio)
            per_split[split] += 1
            name = f"{uid:06d}"
            uid += 1
            img = Image.open(io.BytesIO(img_struct["bytes"])).convert("RGB")
            img.save(DATA_DIR / "images" / split / f"{name}.jpg", quality=90)
            with open(DATA_DIR / "labels" / split / f"{name}.txt", "w") as f:
                for cls, cx, cy, nw, nh, crowd in lines:
                    f.write(f"{cls} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f} {crowd}\n")
            kept += 1
            if kept % 500 == 0:
                print(f"[prepare] {kept} immagini salvate...")
        if max_images is not None and kept >= max_images:
            break

    print(f"[prepare] immagini tenute: {kept}  |  box: {sum(per_class.values())}  "
          f"(+ {sum(crowd_per_class.values())} crowd, ignorate)")
    print("    " + "  ".join(f"{s}={n}" for s, n in per_split.items()))
    for c, n in per_class.items():
        print(f"    {CLASS_NAMES[c]:8s}: {n:6d}  (+ {crowd_per_class[c]} crowd)")
    # Gli anchor box sono scelti a mano in config.ANCHORS (non ricavati dai dati).


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--num_shards", type=int, default=15)
    p.add_argument("--max_images", type=int, default=None)
    p.add_argument("--val_ratio", type=float, default=0.1)
    p.add_argument("--test_ratio", type=float, default=0.1)
    args = p.parse_args()
    prepare(args.max_images, args.num_shards, args.val_ratio, args.test_ratio)

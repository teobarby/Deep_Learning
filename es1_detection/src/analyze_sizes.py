"""Analisi delle dimensioni degli oggetti: griglia, soglie e anchor.

Un detector a scala singola con griglia G x G non localizza oggetti con lato
minore piu' piccolo di una cella, cioe' 1/G del lato dell'immagine. Per diverse
griglie lo script riporta quanti oggetti finirebbero sotto soglia (e quindi
ignorati), per classe, e quante immagini restano con almeno un oggetto valido.
Riporta anche i percentili di larghezza e altezza degli oggetti del compito
(lato minore >= EVAL_MIN_OBJ_SIZE), da cui scegliere a mano gli anchor box.

La griglia dipende da risoluzione e stride, G = IMG_SIZE / STRIDE, ad esempio:
    416/8 = 52    416/16 = 26    320/16 = 20    256/16 = 16    416/32 = 13

Le box crowd sono escluse (sono regioni ignorate). Legge solo le label prodotte
da prepare_data.py: non serve la GPU.

Uso:
    python src/analyze_sizes.py
    python src/analyze_sizes.py --grids 52 26 20 16 13 10
"""
from __future__ import annotations

import argparse

from config import (
    ANCHORS,
    CLASS_NAMES,
    DATA_DIR,
    EVAL_MIN_OBJ_SIZE,
    GRID,
    IMG_SIZE,
    MIN_OBJ_SIZE,
    NUM_CLASSES,
    STRIDE,
)

PERCENTILES = (10, 25, 50, 75, 90)


def load_objects() -> tuple[list[tuple[str, int, float, float]], int, int]:
    """Ritorna (oggetti, n_immagini, n_crowd); oggetto = (immagine, classe, w, h)."""
    objects = []
    n_images = n_crowd = 0
    for split in ("train", "val", "test"):
        for path in sorted((DATA_DIR / "labels" / split).glob("*.txt")):
            n_images += 1
            for line in path.read_text().splitlines():
                values = line.split()
                if not values:
                    continue
                if len(values) > 5 and values[5] == "1":
                    n_crowd += 1
                    continue
                objects.append((f"{split}/{path.stem}", int(values[0]),
                                float(values[3]), float(values[4])))
    return objects, n_images, n_crowd


def percentile(sorted_vals: list[float], q: float) -> float:
    """Percentile q (0-100) di una lista gia' ordinata (nearest-rank)."""
    if not sorted_vals:
        return float("nan")
    return sorted_vals[round(q / 100 * (len(sorted_vals) - 1))]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--grids", type=int, nargs="+", default=[52, 26, 20, 16, 13])
    args = p.parse_args()

    objects, n_images, n_crowd = load_objects()
    if not objects:
        raise SystemExit("Nessuna label trovata: esegui prima src/prepare_data.py")
    print(f"[dati] immagini={n_images}  oggetti={len(objects)}  (+ {n_crowd} box crowd escluse)")
    print(f"[config] {IMG_SIZE}px / stride {STRIDE} -> griglia {GRID}x{GRID}, "
          f"MIN_OBJ_SIZE={MIN_OBJ_SIZE:.4f}, EVAL_MIN_OBJ_SIZE={EVAL_MIN_OBJ_SIZE:.4f}\n")

    print("Lato minore (frazione del lato immagine): percentili per classe")
    print(f"{'classe':10s} {'#obj':>6s} " + " ".join(f"{'p' + str(q):>7s}" for q in PERCENTILES))
    for c in range(NUM_CLASSES):
        s = sorted(min(w, h) for _, cls, w, h in objects if cls == c)
        print(f"{CLASS_NAMES[c]:10s} {len(s):6d} "
              + " ".join(f"{percentile(s, q):7.3f}" for q in PERCENTILES))

    print("\nOggetti VALIDI per classe (lato minore >= 1 cella), per griglia")
    print(f"{'griglia':>8s} {'soglia':>7s} {'ignorati':>9s} {'img utili':>10s} "
          + " ".join(f"{n[:7]:>7s}" for n in CLASS_NAMES))
    for g in args.grids:
        thr = 1.0 / g
        kept = [0] * NUM_CLASSES
        useful_images = set()  # immagini con almeno un oggetto valido
        for img, cls, w, h in objects:
            if min(w, h) >= thr:
                kept[cls] += 1
                useful_images.add(img)
        ignored = 100 * (len(objects) - sum(kept)) / len(objects)
        print(f"{f'{g}x{g}':>8s} {thr:7.4f} {ignored:8.1f}% {len(useful_images):10d} "
              + " ".join(f"{k:7d}" for k in kept))

    task = [(w, h) for _, _, w, h in objects if min(w, h) >= EVAL_MIN_OBJ_SIZE]
    ws = sorted(w for w, _ in task)
    hs = sorted(h for _, h in task)
    ratios = sorted(h / w for w, h in task)
    print(f"\nOggetti del compito (lato minore >= {EVAL_MIN_OBJ_SIZE:.3f}): {len(task)}")
    print(f"{'':10s} " + " ".join(f"{'p' + str(q):>7s}" for q in PERCENTILES))
    for name, vals in (("larghezza", ws), ("altezza", hs), ("h / w", ratios)):
        print(f"{name:10s} " + " ".join(f"{percentile(vals, q):7.3f}" for q in PERCENTILES))
    print("\nAnchor attuali (w, h): " + "  ".join(f"({w:.2f}, {h:.2f})" for w, h in ANCHORS))


if __name__ == "__main__":
    main()

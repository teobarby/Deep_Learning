"""Dati e figure reali dell'esercizio 1 per la relazione di studio (solo CPU)."""
import json
import os
import random
import sys
from collections import Counter
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""   # la GPU e' occupata dagli esperimenti
os.environ.pop("ES1_CONFIG", None)
ROOT = Path(r"C:\Users\matte\IdeaProjects\Deep_Learning\es1_detection")
OUT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

import config

plt.rcParams["font.family"] = "Arial"
facts = {}

# ------------------------------------------------------------------ modello
def count(module):
    return sum(p.numel() for p in module.parameters())


def receptive_field(stride):
    """Campo recettivo teorico (px) dell'uscita del collo, per ResNet18 troncata."""
    rf, jump = 1, 1
    layers = [(7, 2), (3, 2)] + [(3, 1)] * 4                       # conv1, maxpool, layer1
    layers += [(3, 2)] + [(3, 1)] * 3                              # layer2 -> stride 8
    if stride >= 16:
        layers += [(3, 2)] + [(3, 1)] * 3                          # layer3
    if stride >= 32:
        layers += [(3, 2)] + [(3, 1)] * 3                          # layer4
    layers += [(3, 1)] * 2                                         # collo
    for k, s in layers:
        rf += (k - 1) * jump
        jump *= s
    return rf


model_info = {}
import importlib
for stride in (8, 16, 32):
    os.environ["ES1_CONFIG"] = json.dumps({"STRIDE": stride})
    for mod in ("config", "model"):
        if mod in sys.modules:
            importlib.reload(sys.modules[mod])
    import model as model_mod
    m = model_mod.YOLO()
    grid = config.IMG_SIZE // stride
    model_info[stride] = {
        "grid": grid, "predictions": config.NUM_ANCHORS * grid * grid,
        "backbone": count(m.features), "neck": count(m.neck), "head": count(m.head),
        "total": count(m), "rf": receptive_field(stride),
        "shape": list(m(torch.zeros(1, 3, config.IMG_SIZE, config.IMG_SIZE)).shape),
    }
os.environ.pop("ES1_CONFIG")
importlib.reload(sys.modules["config"])
facts["model"] = model_info

# ------------------------------------------------------------------ dataset
splits = {}
per_class = Counter()
crowd_class = Counter()
objects = []
for split in ("train", "val", "test"):
    files = sorted((config.DATA_DIR / "labels" / split).glob("*.txt"))
    splits[split] = len(files)
    for f in files:
        for line in f.read_text().splitlines():
            v = line.split()
            if not v:
                continue
            c, w, h, crowd = int(v[0]), float(v[3]), float(v[4]), int(v[5])
            if crowd:
                crowd_class[c] += 1
            else:
                per_class[c] += 1
                objects.append((c, w, h))
facts["splits"] = splits
facts["per_class"] = {config.CLASS_NAMES[c]: per_class[c] for c in range(config.NUM_CLASSES)}
facts["crowd_class"] = {config.CLASS_NAMES[c]: crowd_class[c] for c in range(config.NUM_CLASSES)}
facts["n_objects"] = len(objects)
facts["n_crowd"] = sum(crowd_class.values())
facts["task_objects"] = sum(1 for _, w, h in objects if min(w, h) >= config.EVAL_MIN_OBJ_SIZE)

# oggetti per classe: validi al compito e sotto soglia
fig, ax = plt.subplots(figsize=(7.2, 3.0))
names = config.CLASS_NAMES
valid = [sum(1 for c, w, h in objects if c == k and min(w, h) >= config.EVAL_MIN_OBJ_SIZE) for k in range(8)]
small = [per_class[k] - valid[k] for k in range(8)]
crowd = [crowd_class[k] for k in range(8)]
x = np.arange(8)
ax.bar(x, valid, color="#3b6ea5", label="oggetti del compito (lato min. >= 1/20)")
ax.bar(x, small, bottom=valid, color="#9dbbd9", label="oggetti piccoli (ignorati in valutazione)")
ax.bar(x, crowd, bottom=np.array(valid) + np.array(small), color="#e3a65b", label="box crowd (sempre ignorate)")
ax.set_xticks(x, names, fontsize=8.5); ax.set_ylabel("box", fontsize=9); ax.legend(fontsize=7.5)
ax.grid(axis="y", alpha=0.3); ax.tick_params(axis="y", labelsize=8)
fig.tight_layout(); fig.savefig(OUT / "classi.png", dpi=220); plt.close(fig)

# frazione di oggetti sotto una cella per griglia
grids = [52, 26, 20, 16, 13]
frac = [100 * sum(1 for _, w, h in objects if min(w, h) < 1 / g) / len(objects) for g in grids]
facts["ignored_by_grid"] = dict(zip(grids, frac))

# ------------------------------------------------------------------ anchor
random.seed(0)
sample = random.sample([(w, h) for _, w, h in objects if min(w, h) >= 1 / 52], 6000)
anchors_default = [tuple(a) for a in config.ANCHORS]
anchors_8 = [(0.054, 0.038), (0.029, 0.07), (0.108, 0.075), (0.058, 0.139),
             (0.215, 0.151), (0.116, 0.279), (0.43, 0.301), (0.232, 0.558)]


def shape_iou(a, b):
    inter = min(a[0], b[0]) * min(a[1], b[1])
    return inter / (a[0] * a[1] + b[0] * b[1] - inter)


def coverage(anchors):
    best = [max(shape_iou(b, a) for a in anchors) for b in sample]
    return 100 * sum(v < 0.5 for v in best) / len(best), float(np.mean(best))


facts["anchor_coverage"] = {"default": coverage(anchors_default), "8": coverage(anchors_8)}
fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.6), sharex=True, sharey=True)
for ax, anchors, title in ((axes[0], anchors_default, "5 anchor attuali"),
                          (axes[1], anchors_8, "8 anchor: 4 scale x 2 rapporti")):
    ax.scatter([w for w, _ in sample], [h for _, h in sample], s=2, alpha=0.18, color="#3b6ea5")
    ax.scatter([w for w, _ in anchors], [h for _, h in anchors], s=60, marker="X", color="#c0392b", zorder=3)
    ax.plot([0, 0.7], [0, 0.7], ls="--", lw=0.7, color="gray")
    bad, _ = coverage(anchors)
    ax.set_title(f"{title}: {bad:.1f}% box mal coperte", fontsize=9)
    ax.set_xlim(0, 0.7); ax.set_ylim(0, 0.9); ax.grid(alpha=0.3)
    ax.set_xlabel("larghezza (frazione immagine)", fontsize=8.5); ax.tick_params(labelsize=8)
axes[0].set_ylabel("altezza (frazione immagine)", fontsize=8.5)
fig.tight_layout(); fig.savefig(OUT / "anchor.png", dpi=220); plt.close(fig)

# ------------------------------------------------------------------ esempio reale: box valide, piccole, crowd
best = None
for f in sorted((config.DATA_DIR / "labels" / "val").glob("*.txt")):
    rows = [l.split() for l in f.read_text().splitlines() if l.strip()]
    n_crowd = sum(r[5] == "1" for r in rows)
    n_small = sum(r[5] == "0" and min(float(r[3]), float(r[4])) < config.EVAL_MIN_OBJ_SIZE for r in rows)
    n_valid = len(rows) - n_crowd - n_small
    if n_crowd >= 1 and n_small >= 2 and 3 <= n_valid <= 8:
        best = (f.stem, rows)
        break
facts["example_image"] = best[0] if best else None
if best:
    img = Image.open(config.DATA_DIR / "images" / "val" / f"{best[0]}.jpg").convert("RGB")
    W, H = img.size
    fig, ax = plt.subplots(figsize=(7, 7 * H / W))
    ax.imshow(img); ax.axis("off")
    for r in best[1]:
        c, cx, cy, w, h, cr = int(r[0]), *map(float, r[1:5]), r[5] == "1"
        is_small = not cr and min(w, h) < config.EVAL_MIN_OBJ_SIZE
        color, ls = ("#e67e22", "--") if cr else (("#f1c40f", ":") if is_small else ("#2ecc71", "-"))
        ax.add_patch(mpatches.Rectangle(((cx - w / 2) * W, (cy - h / 2) * H), w * W, h * H,
                                        fill=False, ec=color, lw=2, ls=ls))
        if not is_small:
            ax.text((cx - w / 2) * W, (cy - h / 2) * H - 3, config.CLASS_NAMES[c] + (" (crowd)" if cr else ""),
                    color="white", fontsize=7, bbox=dict(fc=color, ec="none", pad=1))
    handles = [mpatches.Patch(ec="#2ecc71", fc="none", lw=2, label="oggetto da trovare"),
               mpatches.Patch(ec="#f1c40f", fc="none", lw=2, ls=":", label="oggetto piccolo (ignorato)"),
               mpatches.Patch(ec="#e67e22", fc="none", lw=2, ls="--", label="box crowd (ignorata)")]
    ax.legend(handles=handles, loc="lower right", fontsize=7, framealpha=0.9)
    fig.tight_layout(); fig.savefig(OUT / "esempio_ignore.png", dpi=200); plt.close(fig)

(OUT / "facts.json").write_text(json.dumps(facts, indent=1), encoding="utf-8")
print(json.dumps(facts, indent=1))

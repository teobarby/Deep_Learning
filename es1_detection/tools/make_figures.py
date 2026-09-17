"""Figure didattiche per la relazione dell'esercizio 1."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mp
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from PIL import Image

OUT = Path(__file__).parent
DATA = Path(r"C:\Users\matte\IdeaProjects\Deep_Learning\es1_detection\data")
facts = json.loads((OUT / "facts.json").read_text())
plt.rcParams["font.family"] = "Arial"
ANCHORS = [(0.03, 0.05), (0.07, 0.12), (0.14, 0.26), (0.26, 0.46), (0.60, 0.70)]
extra = {}


def shape_iou(a, b):
    inter = min(a[0], b[0]) * min(a[1], b[1])
    return inter / (a[0] * a[1] + b[0] * b[1] - inter)


def box_iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    area = lambda r: (r[2] - r[0]) * (r[3] - r[1])
    return inter / (area(a) + area(b) - inter)


# ------------------------------------------------------------------ esempio: immagine 416x416 come la vede la rete
name = facts["example_image"]
img = Image.open(DATA / "images" / "val" / f"{name}.jpg").convert("RGB").resize((416, 416))
rows = [l.split() for l in (DATA / "labels" / "val" / f"{name}.txt").read_text().splitlines() if l.strip()]
persons = [tuple(map(float, r[1:5])) for r in rows if r[0] == "0" and r[5] == "0"]
gt = max(persons, key=lambda b: b[2] * b[3])        # la persona piu' grande
cx, cy, w, h = gt

# --- 1. griglia, cella responsabile, anchor
S = 13
fig, ax = plt.subplots(figsize=(5.6, 5.6))
ax.imshow(img); ax.set_xlim(0, 416); ax.set_ylim(416, 0); ax.axis("off")
for k in range(S + 1):
    ax.plot([k * 32, k * 32], [0, 416], color="white", lw=0.5, alpha=0.7)
    ax.plot([0, 416], [k * 32, k * 32], color="white", lw=0.5, alpha=0.7)
i, j = int(cx * S), int(cy * S)
ax.add_patch(mp.Rectangle((i * 32, j * 32), 32, 32, fc="#f1c40f", alpha=0.55, ec="#f1c40f", lw=2))
ax.add_patch(mp.Rectangle(((cx - w / 2) * 416, (cy - h / 2) * 416), w * 416, h * 416, fill=False, ec="#2ecc71", lw=2.2))
ax.plot(cx * 416, cy * 416, "o", color="#2ecc71", ms=6, mec="white")
ious = [shape_iou((w, h), a) for a in ANCHORS]
best = int(np.argmax(ious))
ccx, ccy = (i + 0.5) * 32, (j + 0.5) * 32
for k, (aw, ah) in enumerate(ANCHORS):
    ax.add_patch(mp.Rectangle((ccx - aw * 208, ccy - ah * 208), aw * 416, ah * 416, fill=False,
                              ec="#e74c3c" if k == best else "#ecf0f1", lw=2.2 if k == best else 1,
                              ls="-" if k == best else "--"))
ax.set_title(f"Griglia 13x13 (nel modello 52x52): la cella gialla contiene il centro della persona", fontsize=8.5)
fig.tight_layout(); fig.savefig(OUT / "griglia_anchor.png", dpi=200); plt.close(fig)
extra["grid_example"] = {"cell": [i, j], "gt_wh": [round(w, 3), round(h, 3)],
                         "anchor_ious": [round(v, 3) for v in ious], "best_anchor": best}

# --- 2. campo recettivo per stride
fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.5))
for ax, stride in zip(axes, (8, 16, 32)):
    rf = facts["model"][str(stride)]["rf"]
    g = 416 // stride
    ax.imshow(img); ax.set_xlim(0, 416); ax.set_ylim(416, 0); ax.axis("off")
    for k in range(g + 1):
        ax.plot([k * stride] * 2, [0, 416], color="white", lw=0.25 if stride == 8 else 0.5, alpha=0.5)
        ax.plot([0, 416], [k * stride] * 2, color="white", lw=0.25 if stride == 8 else 0.5, alpha=0.5)
    pcx, pcy = cx * 416, cy * 416
    cxs, cys = (int(pcx // stride) + 0.5) * stride, (int(pcy // stride) + 0.5) * stride
    ax.add_patch(mp.Rectangle((cxs - rf / 2, cys - rf / 2), rf, rf, fill=True, fc="#3498db", alpha=0.25, ec="#2980b9", lw=2))
    ax.add_patch(mp.Rectangle((cxs - stride / 2, cys - stride / 2), stride, stride, fc="#f1c40f", ec="#f1c40f"))
    ax.add_patch(mp.Rectangle(((cx - w / 2) * 416, (cy - h / 2) * 416), w * 416, h * 416, fill=False, ec="#2ecc71", lw=1.8))
    ax.set_title(f"stride {stride}: griglia {g}x{g}\ncampo recettivo ~{rf}px", fontsize=9)
fig.tight_layout(pad=1.2); fig.savefig(OUT / "campo_recettivo.png", dpi=200, bbox_inches="tight"); plt.close(fig)

# --- 3. decodifica delle coordinate (coordinate immagine: y verso il basso)
U = 2.0                                           # lato di una cella nel disegno
fig, ax = plt.subplots(figsize=(7.4, 5.0))
ax.set_xlim(-3.2, 11.2); ax.set_ylim(7.4, -1.6); ax.set_aspect("equal"); ax.axis("off")
for k in range(5):
    ax.plot([k * U, k * U], [0, 3 * U], color="#cccccc", lw=0.8)
for k in range(4):
    ax.plot([0, 4 * U], [k * U, k * U], color="#cccccc", lw=0.8)
cxc, cyc = 1 * U, 1 * U                           # cella (c_x, c_y) = (1, 1)
ax.add_patch(mp.Rectangle((cxc, cyc), U, U, fc="#fdf1c7", ec="#e6b800", lw=1.6, zorder=1))
sx, sy = 0.62 * U, 0.55 * U
bx, by = cxc + sx, cyc + sy
aw, ah = 2.0, 3.0                                 # anchor
pw, ph = aw * np.exp(0.35), ah * np.exp(0.12)     # box predetta
ax.add_patch(mp.Rectangle((bx - aw / 2, by - ah / 2), aw, ah, fill=False, ec="#7f8c8d", ls="--", lw=1.4, zorder=2))
ax.add_patch(mp.Rectangle((bx - pw / 2, by - ph / 2), pw, ph, fill=False, ec="#c0392b", lw=2.2, zorder=3))
ax.plot(bx, by, "o", color="#c0392b", ms=7, zorder=5)
# offset orizzontale: dal bordo sinistro della cella al centro
ax.annotate("", xy=(bx, cyc + 0.3), xytext=(cxc, cyc + 0.3), arrowprops=dict(arrowstyle="<->", color="#1f4e79", lw=1.2))
ax.text(cxc + sx / 2, cyc + 0.2, r"$\sigma(t_x)$", color="#1f4e79", fontsize=10, ha="center", va="bottom", zorder=6)
# offset verticale: dal bordo superiore della cella al centro
ax.annotate("", xy=(bx + 0.25, by), xytext=(bx + 0.25, cyc), arrowprops=dict(arrowstyle="<->", color="#1f4e79", lw=1.2))
ax.text(bx + 0.35, cyc + sy / 2 + 0.25, r"$\sigma(t_y)$", color="#1f4e79", fontsize=10, zorder=6)
# etichette esterne con frecce
ax.annotate("cella (c$_x$, c$_y$)\ncoordinate dell'angolo\nin alto a sinistra", xy=(cxc, cyc + U), xytext=(-3.1, 6.3),
            fontsize=8.5, color="#8a6d00", arrowprops=dict(arrowstyle="->", color="#8a6d00", lw=0.8))
ax.annotate("anchor (a$_w$, a$_h$): forma di partenza", xy=(bx + aw / 2, by + 1.1), xytext=(8.3, 6.3),
            fontsize=8.5, color="#5d6d6e", ha="left", arrowprops=dict(arrowstyle="->", color="#5d6d6e", lw=0.8))
ax.annotate("box predetta\n" + r"$b_w = a_w\,e^{t_w},\ b_h = a_h\,e^{t_h}$", xy=(bx + pw / 2, by - ph / 2 + 0.6),
            xytext=(8.3, -0.9), fontsize=8.5, color="#c0392b", arrowprops=dict(arrowstyle="->", color="#c0392b", lw=0.8))
ax.annotate("centro della box\n" + r"$(c_x+\sigma(t_x),\ c_y+\sigma(t_y))$", xy=(bx - 0.1, by + 0.05), xytext=(-3.1, -0.9),
            fontsize=8.5, color="#c0392b", arrowprops=dict(arrowstyle="->", color="#c0392b", lw=0.8,
                                                           connectionstyle="angle,angleA=0,angleB=90"))
fig.tight_layout(); fig.savefig(OUT / "decodifica.png", dpi=220); plt.close(fig)

# --- 4. IoU
A = (1.0, 1.0, 4.0, 3.5)
B = (2.2, 1.8, 5.2, 4.6)
iou_ab = box_iou(A, B)
fig, ax = plt.subplots(figsize=(4.4, 3.2))
ax.add_patch(mp.Rectangle(A[:2], A[2] - A[0], A[3] - A[1], fill=False, ec="#2ecc71", lw=2.2, label="ground truth"))
ax.add_patch(mp.Rectangle(B[:2], B[2] - B[0], B[3] - B[1], fill=False, ec="#c0392b", lw=2.2, label="predizione"))
ax.add_patch(mp.Rectangle((B[0], B[1]), A[2] - B[0], A[3] - B[1], fc="#f39c12", alpha=0.45, label="intersezione"))
ax.set_xlim(0.5, 5.8); ax.set_ylim(5.0, 0.5); ax.set_aspect("equal"); ax.axis("off")
ax.legend(fontsize=7.5, loc="lower left")
ax.set_title(f"IoU = intersezione / unione = {iou_ab:.2f}", fontsize=9)
fig.tight_layout(); fig.savefig(OUT / "iou.png", dpi=220); plt.close(fig)
extra["iou_example"] = {"A": A, "B": B, "iou": round(iou_ab, 3),
                        "inter": round((A[2] - B[0]) * (A[3] - B[1]), 3),
                        "areaA": (A[2] - A[0]) * (A[3] - A[1]), "areaB": round((B[2] - B[0]) * (B[3] - B[1]), 3)}

# --- 5. NMS
crop_box = ((cx - w / 2) * 416 - 30, (cy - h / 2) * 416 - 30, (cx + w / 2) * 416 + 30, (cy + h / 2) * 416 + 30)
x0, y0, x1, y1 = [int(round(v)) for v in crop_box]
x0, y0 = max(x0, 0), max(y0, 0)
x1, y1 = min(x1, 416), min(y1, 416)
crop = img.crop((x0, y0, x1, y1))
gx1, gy1 = (cx - w / 2) * 416 - x0, (cy - h / 2) * 416 - y0
gx2, gy2 = (cx + w / 2) * 416 - x0, (cy + h / 2) * 416 - y0
rng = np.random.default_rng(3)
cands = [((gx1, gy1, gx2, gy2), 0.91)]
for s in (0.84, 0.72, 0.55, 0.38):
    d = rng.normal(0, 12, 4)
    cands.append(((gx1 + d[0], gy1 + d[1], gx2 + d[2], gy2 + d[3]), s))
order = sorted(range(len(cands)), key=lambda k: -cands[k][1])
keep = []
remaining = order[:]
while remaining:
    k = remaining.pop(0)
    keep.append(k)
    remaining = [r for r in remaining if box_iou(cands[k][0], cands[r][0]) <= 0.45]
fig, axes = plt.subplots(1, 2, figsize=(6.6, 4.2))
for ax, title, idx in ((axes[0], "prima: 5 box sopra la soglia", range(len(cands))),
                       (axes[1], f"dopo l'NMS (IoU > 0,45 scartate): {len(keep)} box", keep)):
    ax.imshow(crop); ax.axis("off"); ax.set_title(title, fontsize=9)
    for rank, k in enumerate(sorted(idx, key=lambda k: -cands[k][1])):
        (a, b, c, d), s = cands[k]
        col = "#c0392b" if k == keep[0] else "#3498db"
        ax.add_patch(mp.Rectangle((a, b), c - a, d - b, fill=False, ec=col, lw=1.8))
        # etichette in colonna sotto l'immagine, ognuna collegata alla propria box
        ly = crop.size[1] - 12 - 16 * rank
        ax.annotate(f"{s:.2f}", xy=(c, d - 4), xytext=(crop.size[0] - 4, ly), ha="right", fontsize=7, color="white",
                    bbox=dict(fc=col, ec="none", pad=1), arrowprops=dict(arrowstyle="-", color=col, lw=0.8))
fig.tight_layout(); fig.savefig(OUT / "nms.png", dpi=200); plt.close(fig)
extra["nms_ious_with_best"] = [round(box_iou(cands[0][0], c[0]), 2) for c in cands[1:]]

# --- 6. curva precision-recall e AP
ranked = [True, True, False, True, True, False, True, False, False, True, False, False]
n_gt = 8
tp = fp = 0
P, R = [], []
for t in ranked:
    tp += t; fp += not t
    P.append(tp / (tp + fp)); R.append(tp / n_gt)
Pi = P[:]
for k in range(len(Pi) - 2, -1, -1):
    Pi[k] = max(Pi[k], Pi[k + 1])
ap = 0.0; prev = 0.0
for p, r in zip(Pi, R):
    ap += (r - prev) * p; prev = r
fig, ax = plt.subplots(figsize=(5.2, 3.4))
ax.plot(R, P, "o", color="#3b6ea5", ms=4, label="(recall, precision) scendendo con la soglia")
xs, ys = [0], [Pi[0]]
prev = 0
for p, r in zip(Pi, R):
    xs += [prev, r]; ys += [p, p]; prev = r
ax.fill_between(xs[1:], ys[1:], step=None, color="#f39c12", alpha=0.3, label=f"area = AP = {ap:.3f}")
ax.plot(xs[1:], ys[1:], color="#e67e22", lw=1.5, label="precision interpolata")
ax.set_xlim(0, 1.02); ax.set_ylim(0, 1.05); ax.set_xlabel("recall", fontsize=9); ax.set_ylabel("precision", fontsize=9)
ax.grid(alpha=0.3); ax.legend(fontsize=7.5, loc="lower left"); ax.tick_params(labelsize=8)
fig.tight_layout(); fig.savefig(OUT / "pr_curve.png", dpi=220); plt.close(fig)
extra["pr_example"] = {"ranked": ranked, "n_gt": n_gt, "ap": round(ap, 4), "final_recall": R[-1]}

# --- 7. architettura
fig, ax = plt.subplots(figsize=(6.4, 7.6))
ax.set_xlim(0, 10); ax.set_ylim(0, 16.4); ax.axis("off")
m8 = facts["model"]["8"]
boxes = [
    (15.5, "Immagine RGB ridimensionata a 416 x 416\nnormalizzazione ImageNet", "#f2f2f2", None),
    (13.5, "ResNet18: conv1 7x7 /2  ->  maxpool /2  ->  layer1\nlayer2 /2  (stop: stride 8)\n128 x 52 x 52", "#dfe9f5",
     "backbone\npre-addestrato\n" + f"{m8['backbone'] / 1e6:.2f}".replace(".", ",") + " M par."),
    (10.9, "Collo: 2 x (conv 3x3 -> batch norm -> ReLU)\n256 x 52 x 52", "#c9dcf0",
     f"{m8['neck'] / 1e6:.2f}".replace(".", ",") + " M par."),
    (8.8, "Testa: conv 1x1  ->  A x (5 + C) = 5 x 13 = 65 canali\n65 x 52 x 52", "#fbe0b0",
     f"{m8['head'] / 1e3:.1f}".replace(".", ",") + " k par."),
    (6.7, "Riorganizzazione: (A, S, S, 5 + C) = (5, 52, 52, 13)\nper ogni cella e anchor: t_x, t_y, t_w, t_h, objectness, 8 classi", "#fdf0d5", None),
]


def bh(t):
    return {0: 0.8, 1: 1.25}.get(t.count("\n"), 1.7)


for y, t, c, side in boxes:
    ax.add_patch(FancyBboxPatch((0.3, y - bh(t) / 2), 7.2, bh(t), boxstyle="round,pad=0.05,rounding_size=0.15",
                                fc=c, ec="#555555", lw=0.8))
    ax.text(3.9, y, t, ha="center", va="center", fontsize=8.2)
    if side:
        ax.text(8.8, y, side, ha="center", va="center", fontsize=8, style="italic", color="#333333")
for (ya, ta, *_), (yb, tb, *_) in zip(boxes[:-1], boxes[1:]):
    ax.add_patch(FancyArrowPatch((3.9, ya - bh(ta) / 2 - 0.05), (3.9, yb + bh(tb) / 2 + 0.05),
                                 arrowstyle="-|>", mutation_scale=10, color="#555555", lw=0.8))
# uscite affiancate: training e inferenza
bottom = [(0.3, "Training\nloss YOLO: coordinate,\nobjectness, no-object, classe\n(con regioni ignorate)"),
          (4.0, "Inferenza\ndecodifica -> score = obj x P(classe)\nsoglia di confidenza\n-> NMS per classe")]
for x0, t in bottom:
    ax.add_patch(FancyBboxPatch((x0, 2.6), 3.5, 2.2, boxstyle="round,pad=0.05,rounding_size=0.15",
                                fc="#e3f1dd", ec="#555555", lw=0.8))
    ax.text(x0 + 1.75, 3.7, t, ha="center", va="center", fontsize=8)
    ax.add_patch(FancyArrowPatch((x0 + 1.75, 6.7 - 0.68), (x0 + 1.75, 4.85), arrowstyle="-|>",
                                 mutation_scale=10, color="#555555", lw=0.8))
ax.set_ylim(2.2, 16.4)
fig.tight_layout(); fig.savefig(OUT / "architettura.png", dpi=220); plt.close(fig)

# --- 8. formule
formulas = {
    "f_decode": r"$b_x=\sigma(t_x)+c_x,\quad b_y=\sigma(t_y)+c_y,\quad b_w=a_w\,e^{t_w},\quad b_h=a_h\,e^{t_h}$",
    "f_iou": r"$\mathrm{IoU}(A,B)=\dfrac{|A\cap B|}{|A\cup B|}=\dfrac{|A\cap B|}{|A|+|B|-|A\cap B|}$",
    "f_score": r"$\mathrm{score}=\sigma(t_o)\cdot\max_c\,\mathrm{softmax}(t_{cls})_c$",
    "f_loss": r"$\mathcal{L}=\dfrac{\lambda_{coord}\,(\mathcal{L}_{xy}+\mathcal{L}_{wh})+\mathcal{L}_{obj}+\mathcal{L}_{cls}}{N_{obj}}\;+\;\lambda_{noobj}\,\dfrac{\mathcal{L}_{noobj}}{N_{vuote}}$",
    "f_targets": r"$\hat{t}_x = g_x - c_x,\quad \hat{t}_w = \log\dfrac{g_w}{a_w}$",
    "f_ap": r"$\mathrm{AP}=\sum_k (R_k-R_{k-1})\,P^{interp}_k,\qquad \mathrm{mAP}=\dfrac{1}{C}\sum_{c=1}^{C}\mathrm{AP}_c$",
    "f_metrics": r"$P=\dfrac{TP}{TP+FP}\qquad R=\dfrac{TP}{N_{oggetti}}\qquad F_1=\dfrac{2PR}{P+R}$",
}
for n, tex in formulas.items():
    fig = plt.figure(figsize=(0.01, 0.01))
    fig.text(0, 0, tex, fontsize=15)
    fig.savefig(OUT / f"{n}.png", dpi=300, bbox_inches="tight", pad_inches=0.04, transparent=True)
    plt.close(fig)

(OUT / "extra.json").write_text(json.dumps(extra, indent=1))
print(json.dumps(extra, indent=1))

"""Figure dei risultati: curve del run finale, confronto esperimenti, AP per classe, errori."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).parent
OUTPUTS = Path(r"C:\Users\matte\IdeaProjects\Deep_Learning\es1_detection\outputs")
FINAL = "final_s16_jitter_r34_cosine_e30"
CLASSES = ["Person", "Chair", "Table", "Cabinet", "Car", "Lamp", "Picture", "Monitor"]
plt.rcParams["font.family"] = "Arial"


def hist(name):
    return json.loads((OUTPUTS / name / "history.json").read_text())


# --- 1. curve del run finale ---
h = hist(FINAL)
ep = [r["epoch"] for r in h]
fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
ax[0].plot(ep, [r["train"]["total"] for r in h], label="train", color="#3b6ea5")
ax[0].plot(ep, [r["val"]["total"] for r in h], label="validation", color="#c0504d")
ax[0].set_xlabel("epoca"); ax[0].set_ylabel("loss"); ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3)
ax[1].plot(ep, [r["map50"] for r in h], "o-", ms=3, color="#4f8a3a", label="mAP@0.5")
ax[1].plot(ep, [r["f1_max"] for r in h], "s-", ms=3, color="#e3a65b", label="F1 massimo")
best = max(h, key=lambda r: r["map50"])
ax[1].axvline(best["epoch"], ls="--", lw=0.8, color="gray")
ax[1].annotate(f"checkpoint scelto\nmAP {best['map50']:.3f}".replace(".", ","), xy=(best["epoch"], best["map50"]),
               xytext=(best["epoch"] - 13, best["map50"] - 0.13), fontsize=8,
               arrowprops=dict(arrowstyle="->", lw=0.8, color="gray"))
ax[1].set_xlabel("epoca"); ax[1].set_ylabel("metrica sul validation"); ax[1].legend(fontsize=8, loc="lower right")
ax[1].grid(alpha=0.3); ax[1].set_ylim(0, 0.7)
for a in ax:
    a.tick_params(labelsize=8)
fig.tight_layout(); fig.savefig(OUT / "curve_finale.png", dpi=200); plt.close(fig)

# --- 2. confronto degli esperimenti ---
runs = [
    ("baseline (stride 8)", "e0_baseline"),
    ("+ 8 anchor", "e1_anchor8"),
    ("stride 16", "e2_stride16"),
    ("stride 32", "e3_stride32"),
    ("stride 16 + augment.", "e4_jitter_s16"),
    ("+ LR coseno", "e5_cosine_s16j"),
    ("+ ResNet34", "e6_resnet34_s16j"),
    ("+ backbone congelato", "e7_freeze3_s16j"),
    ("finale (30 epoche)", FINAL),
]
vals = [(lab, max(r["map50"] for r in hist(n))) for lab, n in runs]
fig, ax = plt.subplots(figsize=(8.6, 3.4))
colors = ["#9dbbd9"] * len(vals)
colors[2] = colors[4] = colors[6] = "#3b6ea5"
colors[-1] = "#4f8a3a"
bars = ax.bar(range(len(vals)), [v for _, v in vals], color=colors)
ax.bar_label(bars, labels=[f"{v:.3f}".replace(".", ",") for _, v in vals], fontsize=8, padding=2)
ax.set_xticks(range(len(vals)), [lab for lab, _ in vals], rotation=25, ha="right", fontsize=8)
ax.set_ylabel("mAP@0.5 (validation)", fontsize=9); ax.grid(axis="y", alpha=0.3); ax.set_ylim(0, 0.55)
ax.tick_params(axis="y", labelsize=8)
fig.tight_layout(); fig.savefig(OUT / "confronto_esperimenti.png", dpi=200); plt.close(fig)

# --- 3. AP per classe: baseline vs finale (validation) e test ---
ap_base = max(hist("e0_baseline"), key=lambda r: r["map50"])["ap"]
ap_final = best["ap"]
test_ap = [0.7197, 0.4316, 0.1903, 0.2772, 0.5238, 0.4328, 0.5249, 0.4532]   # da test_report.txt
x = np.arange(len(CLASSES))
fig, ax = plt.subplots(figsize=(8.6, 3.2))
ax.bar(x - 0.28, ap_base, 0.28, label="baseline (validation)", color="#c9c9c9")
ax.bar(x, ap_final, 0.28, label="modello finale (validation)", color="#3b6ea5")
ax.bar(x + 0.28, test_ap, 0.28, label="modello finale (test)", color="#4f8a3a")
ax.set_xticks(x, CLASSES, fontsize=8.5); ax.set_ylabel("AP@0.5", fontsize=9)
ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3); ax.tick_params(axis="y", labelsize=8)
fig.tight_layout(); fig.savefig(OUT / "ap_per_classe.png", dpi=200); plt.close(fig)

# --- 4. scomposizione degli errori ---
err = json.loads((OUT / "errori_finale.json").read_text(encoding="utf-8"))
fig, ax = plt.subplots(1, 2, figsize=(10, 3.2))
for a, (titolo, d) in zip(ax, (("Falsi positivi", err["fp"]), ("Oggetti mancati", err["fn"]))):
    items = sorted(d.items(), key=lambda kv: -kv[1])
    tot = sum(v for _, v in items)
    labels = [k.replace(" (IoU 0,1-0,5)", "").replace(" (nessun oggetto vicino)", "") for k, _ in items]
    a.barh(range(len(items)), [100 * v / tot for _, v in items], color="#3b6ea5")
    a.set_yticks(range(len(items)), labels, fontsize=8)
    a.invert_yaxis(); a.set_xlabel("% del totale", fontsize=8.5); a.set_title(f"{titolo} ({tot})", fontsize=9.5)
    a.grid(axis="x", alpha=0.3); a.tick_params(axis="x", labelsize=8)
    for i, (_, v) in enumerate(items):
        a.text(100 * v / tot + 1, i, str(v), va="center", fontsize=7.5)
fig.tight_layout(); fig.savefig(OUT / "errori.png", dpi=200); plt.close(fig)

summary = {"final_best": {k: best[k] for k in ("epoch", "map50", "f1_max", "best_conf", "ap")},
           "runs": {lab: v for lab, v in vals}, "test_ap": test_ap,
           "final_last_val_loss": h[-1]["val"]["total"], "min_val_loss": min(r["val"]["total"] for r in h),
           "seconds_per_epoch": round(sum(r["seconds"] for r in h) / len(h))}
(OUT / "risultati.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
print(json.dumps(summary, indent=1))

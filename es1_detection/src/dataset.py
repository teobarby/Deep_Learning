"""Dataset PyTorch per il detector YOLO.

Legge immagini + label prodotte da prepare_data.py (class cx cy w h crowd, con
coordinate normalizzate). Ogni immagine viene ridimensionata a IMG_SIZE x IMG_SIZE.
Le coordinate delle box sono normalizzate, quindi restano valide dopo il resize.
Le box sono tensori (N,6): class, cx, cy, w, h, crowd (1 = regione ignorata).

Data augmentation (solo in training, se augment=True):
  - zoom e traslazione casuali (se SCALE_JITTER > 0): l'immagine viene
    ingrandita o rimpicciolita di un fattore in [1-s, 1+s] e spostata; le box
    vengono trasformate e tagliate al bordo. Una box rimasta visibile per meno
    del 40% diventa regione ignorata: l'oggetto e' ancora parzialmente nella
    foto, e cancellarlo insegnerebbe che li' "non c'e' nulla";
  - flip orizzontale con probabilita' 0.5 (ribalta anche le bounding box);
  - leggero color jitter di luminosita'/contrasto (non tocca le box).

La costruzione del target di griglia (assegnazione agli anchor) e' fatta nella
loss a partire dalle box grezze: qui restituiamo semplicemente le box.
"""
from __future__ import annotations

import random
from pathlib import Path

import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset
from torchvision import transforms

from config import (
    DATA_DIR,
    IMAGENET_MEAN,
    IMAGENET_STD,
    IMG_SIZE,
    PRETRAINED_BACKBONE,
    SCALE_JITTER,
)

MIN_VISIBLE = 0.4   # sotto questa frazione visibile una box diventa regione ignorata
# colore di riempimento delle zone fuori immagine: la media di ImageNet, che dopo
# la normalizzazione diventa ~0
FILL_COLOR = tuple(round(255 * m) for m in IMAGENET_MEAN)


def image_transform():
    """Trasformazione immagine, condivisa tra training e inferenza.

    Con il backbone pre-addestrato serve la normalizzazione ImageNet; con il
    backbone da zero si resta ai valori in [0,1].
    """
    steps = [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),  # -> [0,1], (3,H,W)
    ]
    if PRETRAINED_BACKBONE:
        steps.append(transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD))
    return transforms.Compose(steps)


class Object365Detection(Dataset):
    def __init__(self, split: str, augment: bool = False):
        self.img_dir = DATA_DIR / "images" / split
        self.lbl_dir = DATA_DIR / "labels" / split
        self.ids = sorted(p.stem for p in self.img_dir.glob("*.jpg"))
        self.augment = augment
        self.jitter = transforms.ColorJitter(brightness=0.2, contrast=0.2)
        self.tf = image_transform()

    def __len__(self):
        return len(self.ids)

    @staticmethod
    def _zoom_translate(img, boxes):
        """Zoom e traslazione casuali; ritorna (immagine, box trasformate)."""
        zoom = random.uniform(1 - SCALE_JITTER, 1 + SCALE_JITTER)
        view = 1.0 / zoom                   # lato della finestra, in unita' dell'immagine
        lo, hi = sorted((0.0, 1.0 - view))  # zoom-in: finestra dentro; zoom-out: sporge
        x0, y0 = random.uniform(lo, hi), random.uniform(lo, hi)
        W, H = img.size
        img = img.transform((W, H), Image.Transform.EXTENT,
                            (x0 * W, y0 * H, (x0 + view) * W, (y0 + view) * H),
                            resample=Image.Resampling.BILINEAR, fillcolor=FILL_COLOR)
        if boxes.numel() == 0:
            return img, boxes

        cls, cx, cy, w, h, crowd = boxes.unbind(1)
        # angoli nelle coordinate della finestra
        x1, x2 = (cx - w / 2 - x0) / view, (cx + w / 2 - x0) / view
        y1, y2 = (cy - h / 2 - y0) / view, (cy + h / 2 - y0) / view
        full_area = (x2 - x1) * (y2 - y1)
        x1, x2 = x1.clamp(0, 1), x2.clamp(0, 1)
        y1, y2 = y1.clamp(0, 1), y2.clamp(0, 1)
        vis_area = (x2 - x1) * (y2 - y1)
        keep = (x2 > x1) & (y2 > y1)
        truncated = vis_area < MIN_VISIBLE * full_area
        crowd = torch.where(truncated, torch.ones_like(crowd), crowd)
        out = torch.stack([cls, (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1, crowd], dim=1)
        return img, out[keep]

    def _augment(self, img, boxes):
        """Zoom/traslazione, flip orizzontale (con box) e color jitter."""
        if SCALE_JITTER > 0:
            img, boxes = self._zoom_translate(img, boxes)
        if random.random() < 0.5:
            img = ImageOps.mirror(img)
            if boxes.numel() > 0:
                boxes[:, 1] = 1.0 - boxes[:, 1]  # cx -> 1 - cx (cy, w, h invariati)
        img = self.jitter(img)
        return img, boxes

    def _load_boxes(self, path: Path) -> torch.Tensor:
        boxes = []
        if path.exists():
            for line in path.read_text().splitlines():
                if not line.strip():
                    continue
                values = [float(v) for v in line.split()]
                if len(values) == 5:  # label senza flag crowd
                    values.append(0.0)
                boxes.append(values)
        if not boxes:
            return torch.zeros((0, 6), dtype=torch.float32)
        return torch.tensor(boxes, dtype=torch.float32)

    def __getitem__(self, idx):
        name = self.ids[idx]
        img = Image.open(self.img_dir / f"{name}.jpg").convert("RGB")
        boxes = self._load_boxes(self.lbl_dir / f"{name}.txt")
        if self.augment:
            img, boxes = self._augment(img, boxes)
        img = self.tf(img)
        return img, boxes


def collate_fn(batch):
    """Impila le immagini; le box restano una lista (lunghezza variabile)."""
    imgs = torch.stack([b[0] for b in batch], dim=0)
    boxes = [b[1] for b in batch]
    return imgs, boxes

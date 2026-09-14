"""Dataset PyTorch per il detector YOLO.

Legge immagini + label in formato YOLO (class cx cy w h normalizzati) prodotte
da prepare_data.py. Ogni immagine viene ridimensionata a IMG_SIZE x IMG_SIZE.
Le coordinate delle box sono normalizzate, quindi restano valide dopo il resize.

Data augmentation (solo in training, se augment=True), semplice:
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
)


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

    def _augment(self, img, boxes):
        """Flip orizzontale (con box) + color jitter. Modifica boxes in place."""
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
                cls, cx, cy, w, h = line.split()
                boxes.append([float(cls), float(cx), float(cy), float(w), float(h)])
        if not boxes:
            return torch.zeros((0, 5), dtype=torch.float32)
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

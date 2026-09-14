"""Backbone convoluzionale + testa di detection, scritti da zero.

Backbone convoluzionale: blocchi Conv-BN-ReLU con max-pooling che riducono
l'immagine di un fattore 32 (IMG_SIZE -> GRID, cioe' 224x224 -> 7x7 con la
configurazione attuale). La testa 1x1 produce A*(5+C) canali: per ogni anchor
-> (tx, ty, tw, th, objectness, C logit classe).
La rete e' completamente convoluzionale: funziona a qualsiasi IMG_SIZE multiplo
di 32, senza modifiche ai pesi.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from config import (
    GRID,
    IMG_SIZE,
    NUM_ANCHORS,
    NUM_CLASSES,
    PRETRAINED_BACKBONE,
    STRIDE,
)


def conv_block(in_c: int, out_c: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True),
    )


def _scratch_backbone() -> tuple[nn.Sequential, int]:
    """Backbone convoluzionale scritto da zero, con downsampling pari a STRIDE.

    Ogni max-pool dimezza la risoluzione: servono log2(STRIDE) pooling.
    """
    n_pool = int(math.log2(STRIDE))
    channels = [16, 32, 64, 128, 256]
    layers = []
    in_c = 3
    for i in range(n_pool):
        out_c = channels[i]
        layers += [conv_block(in_c, out_c), nn.MaxPool2d(2)]
        in_c = out_c
    # blocchi finali senza pooling (aumentano la profondita' semantica)
    layers += [conv_block(in_c, in_c * 2), conv_block(in_c * 2, in_c * 2)]
    return nn.Sequential(*layers), in_c * 2


def _pretrained_backbone() -> tuple[nn.Sequential, int]:
    """ResNet18 pre-addestrata su ImageNet, troncata allo STRIDE richiesto.

    In ResNet18 la risoluzione si dimezza a tappe: conv1 (/2), maxpool (/4),
    layer1 (/4), layer2 (/8), layer3 (/16), layer4 (/32). Fermandosi prima si
    ottiene una griglia piu' fine (utile per gli oggetti piccoli) al prezzo di
    feature meno profonde.
    """
    from torchvision.models import ResNet18_Weights, resnet18

    net = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    # children(): conv1, bn1, relu, maxpool, layer1, layer2, layer3, layer4, avgpool, fc
    layers = list(net.children())
    cut, out_ch = {
        8: (6, 128),    # fino a layer2
        16: (7, 256),   # fino a layer3
        32: (8, 512),   # fino a layer4
    }[STRIDE]
    return nn.Sequential(*layers[:cut]), out_ch


class YOLO(nn.Module):
    def __init__(self, num_classes: int = NUM_CLASSES, num_anchors: int = NUM_ANCHORS,
                 pretrained: bool = PRETRAINED_BACKBONE):
        super().__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.attrs = 5 + num_classes  # tx,ty,tw,th,obj + classi
        self.pretrained = pretrained

        self.features, out_ch = (_pretrained_backbone() if pretrained
                                 else _scratch_backbone())

        # "Collo": blocchi convoluzionali SENZA pooling. Troncando il backbone
        # per ottenere una griglia fine (stride 8) si perdono i layer profondi,
        # quindi le feature sono poco semantiche: questi blocchi recuperano
        # profondita' e campo recettivo mantenendo invariata la risoluzione.
        neck_ch = max(out_ch, 256)
        self.neck = nn.Sequential(
            conv_block(out_ch, neck_ch),
            conv_block(neck_ch, neck_ch),
        )
        self.head = nn.Conv2d(neck_ch, num_anchors * self.attrs, kernel_size=1)

    def forward(self, x):
        """Ritorna (B, A, S, S, 5+C) con i logit grezzi (non decodificati)."""
        b = x.size(0)
        feat = self.features(x)
        feat = self.neck(feat)
        out = self.head(feat)  # (B, A*attrs, S, S)
        s = out.size(-1)
        out = out.view(b, self.num_anchors, self.attrs, s, s)
        out = out.permute(0, 1, 3, 4, 2).contiguous()  # (B, A, S, S, attrs)
        return out


if __name__ == "__main__":
    m = YOLO()
    x = torch.randn(2, 3, IMG_SIZE, IMG_SIZE)
    y = m(x)
    n_params = sum(p.numel() for p in m.parameters())
    print("output:", tuple(y.shape), "| atteso:", (2, NUM_ANCHORS, GRID, GRID, 5 + NUM_CLASSES))
    print("parametri:", f"{n_params/1e6:.2f}M")

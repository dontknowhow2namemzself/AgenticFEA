#!/usr/bin/env python3
"""UNet for the stress-field surrogate (plate plan M12).

A plain encoder-decoder with skip connections: 2 input channels (mask, sdf/b)
-> 3 output channels (sxx, syy, sxy). Nothing exotic -- the point of this
project is the data pipeline and the honesty of the evaluation, so the model is
the standard baseline a reviewer would expect, not a tuned exotic architecture.

Why a UNet rather than a plain CNN stack: stress at a point is set both by the
LOCAL geometry (how close is the hole edge) and by GLOBAL context (where the
hole sits in the plate, how much ligament is left). The encoder's downsampling
gives the deep layers a receptive field that spans the plate; the skip
connections carry the fine detail back so the hole edge does not come out
blurred. That blurring is exactly the failure mode we expect to have to
measure -- the peak lives in ~0.04% of the pixels.

The canvas is 384x192, divisible by 16, so four 2x downsamples are exact and
no padding bookkeeping is needed.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def _block(cin, cout):
    """Two 3x3 convolutions. GroupNorm, not BatchNorm: batches here are small
    (a 384x192x3 field per sample), and BatchNorm's statistics get noisy and
    train/eval-inconsistent at small batch sizes."""
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1), nn.GroupNorm(8, cout), nn.GELU(),
        nn.Conv2d(cout, cout, 3, padding=1), nn.GroupNorm(8, cout), nn.GELU(),
    )


class UNet(nn.Module):
    def __init__(self, cin=2, cout=3, width=32, depth=4):
        super().__init__()
        self.depth = depth
        chans = [width * 2 ** i for i in range(depth + 1)]
        self.down = nn.ModuleList()
        c_prev = cin
        for c in chans[:-1]:
            self.down.append(_block(c_prev, c))
            c_prev = c
        self.bottom = _block(c_prev, chans[-1])
        self.pool = nn.MaxPool2d(2)
        self.up = nn.ModuleList()
        self.upconv = nn.ModuleList()
        for i in range(depth - 1, -1, -1):
            self.upconv.append(nn.ConvTranspose2d(chans[i + 1], chans[i], 2, stride=2))
            self.up.append(_block(chans[i] * 2, chans[i]))
        self.head = nn.Conv2d(chans[0], cout, 1)

    def forward(self, x):
        skips = []
        for blk in self.down:
            x = blk(x)
            skips.append(x)
            x = self.pool(x)
        x = self.bottom(x)
        for k, (upc, blk) in enumerate(zip(self.upconv, self.up)):
            x = upc(x)
            x = blk(torch.cat([x, skips[-(k + 1)]], dim=1))
        return self.head(x)


def masked_mse_loss(pred, true, mask):
    """MSE over material pixels only.

    The denominator is the number of material pixels, not the number of pixels
    in the canvas. Averaging over the whole canvas would weight each case by
    how much of the window its plate happens to fill -- see dataset.py.
    """
    m = mask.expand_as(pred)
    se = ((pred - true) ** 2) * m
    return se.sum() / m.sum().clamp(min=1.0)


def count_params(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    net = UNet()
    x = torch.randn(2, 2, 192, 384)
    y = net(x)
    print(f"UNet  params {count_params(net):,}")
    print(f"  in  {tuple(x.shape)}  ->  out {tuple(y.shape)}")
    m = (torch.rand(2, 1, 192, 384) > 0.1).float()
    print(f"  masked_mse_loss on random data: {masked_mse_loss(y, torch.randn_like(y), m):.4f}")

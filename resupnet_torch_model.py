"""PyTorch ResUpNet v2 for native Windows CUDA training."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNReLU(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, dilation: int = 1):
        super().__init__()
        padding = dilation if kernel_size == 3 else 0
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0):
        super().__init__()
        self.conv1 = ConvBNReLU(in_channels, out_channels)
        self.dropout = nn.Dropout2d(dropout) if dropout else nn.Identity()
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        out = self.conv1(x)
        out = self.dropout(out)
        out = self.conv2(out)
        return F.relu(out + self.shortcut(x), inplace=True)


class AttentionGate(nn.Module):
    def __init__(self, skip_channels: int, gating_channels: int, inter_channels: int):
        super().__init__()
        self.theta = nn.Conv2d(skip_channels, inter_channels, 1, bias=False)
        self.phi = nn.Conv2d(gating_channels, inter_channels, 1, bias=False)
        self.psi = nn.Conv2d(inter_channels, 1, 1)

    def forward(self, skip, gating):
        attn = F.relu(self.theta(skip) + self.phi(gating), inplace=True)
        attn = torch.sigmoid(self.psi(attn))
        return skip * attn


class ASPP(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.branches = nn.ModuleList(
            [
                ConvBNReLU(in_channels, out_channels, kernel_size=1, dilation=1),
                ConvBNReLU(in_channels, out_channels, kernel_size=3, dilation=2),
                ConvBNReLU(in_channels, out_channels, kernel_size=3, dilation=4),
                ConvBNReLU(in_channels, out_channels, kernel_size=3, dilation=8),
            ]
        )
        self.pool_proj = ConvBNReLU(in_channels, out_channels, kernel_size=1)
        self.out = ConvBNReLU(out_channels * 5, out_channels, kernel_size=1)

    def forward(self, x):
        size = x.shape[-2:]
        branches = [branch(x) for branch in self.branches]
        pooled = F.adaptive_avg_pool2d(x, 1)
        pooled = self.pool_proj(pooled)
        pooled = F.interpolate(pooled, size=size, mode="bilinear", align_corners=False)
        return self.out(torch.cat([*branches, pooled], dim=1))


class ResUpNetTorch(nn.Module):
    def __init__(self, in_channels: int = 4, base_filters: int = 32, dropout: float = 0.20):
        super().__init__()
        b = base_filters
        self.e1 = ResidualBlock(in_channels, b)
        self.e2 = ResidualBlock(b, b * 2, dropout * 0.5)
        self.e3 = ResidualBlock(b * 2, b * 4, dropout * 0.75)
        self.e4 = ResidualBlock(b * 4, b * 8, dropout)
        self.pool = nn.MaxPool2d(2)

        self.bottleneck = nn.Sequential(
            ResidualBlock(b * 8, b * 16, dropout),
            ASPP(b * 16, b * 16),
            nn.Dropout2d(dropout),
        )

        self.a4 = AttentionGate(b * 8, b * 16, b * 4)
        self.d4 = ResidualBlock(b * 16 + b * 8, b * 8, dropout)
        self.a3 = AttentionGate(b * 4, b * 8, b * 2)
        self.d3 = ResidualBlock(b * 8 + b * 4, b * 4, dropout * 0.75)
        self.a2 = AttentionGate(b * 2, b * 4, b)
        self.d2 = ResidualBlock(b * 4 + b * 2, b * 2, dropout * 0.5)
        self.a1 = AttentionGate(b, b * 2, max(b // 2, 1))
        self.d1 = ResidualBlock(b * 2 + b, b)
        self.out = nn.Conv2d(b, 1, 1)

    def _up(self, x, target):
        return F.interpolate(x, size=target.shape[-2:], mode="bilinear", align_corners=False)

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(self.pool(e1))
        e3 = self.e3(self.pool(e2))
        e4 = self.e4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))

        u4 = self._up(b, e4)
        d4 = self.d4(torch.cat([u4, self.a4(e4, u4)], dim=1))
        u3 = self._up(d4, e3)
        d3 = self.d3(torch.cat([u3, self.a3(e3, u3)], dim=1))
        u2 = self._up(d3, e2)
        d2 = self.d2(torch.cat([u2, self.a2(e2, u2)], dim=1))
        u1 = self._up(d2, e1)
        d1 = self.d1(torch.cat([u1, self.a1(e1, u1)], dim=1))
        return self.out(d1)


def dice_score_from_logits(logits, targets, smooth: float = 1e-6):
    probs = torch.sigmoid(logits)
    probs = probs.reshape(probs.shape[0], -1)
    targets = targets.reshape(targets.shape[0], -1)
    intersection = (probs * targets).sum(dim=1)
    denom = probs.sum(dim=1) + targets.sum(dim=1)
    return ((2 * intersection + smooth) / (denom + smooth)).mean()


def focal_tversky_loss(logits, targets, alpha=0.3, beta=0.7, gamma=0.75, smooth=1e-6):
    probs = torch.sigmoid(logits).reshape(-1)
    targets = targets.reshape(-1)
    tp = (probs * targets).sum()
    fp = ((1 - targets) * probs).sum()
    fn = (targets * (1 - probs)).sum()
    tversky = (tp + smooth) / (tp + alpha * fp + beta * fn + smooth)
    return (1 - tversky).pow(gamma)


def boundary_loss(logits, targets):
    probs = torch.sigmoid(logits)
    sobel_x = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=probs.dtype, device=probs.device)
    sobel_y = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=probs.dtype, device=probs.device)
    sobel_x = sobel_x.view(1, 1, 3, 3)
    sobel_y = sobel_y.view(1, 1, 3, 3)
    pred_x = F.conv2d(probs, sobel_x, padding=1)
    pred_y = F.conv2d(probs, sobel_y, padding=1)
    true_x = F.conv2d(targets, sobel_x, padding=1)
    true_y = F.conv2d(targets, sobel_y, padding=1)
    pred_mag = torch.sqrt(pred_x.square() + pred_y.square() + 1e-6)
    true_mag = torch.sqrt(true_x.square() + true_y.square() + 1e-6)
    return torch.mean(torch.abs(pred_mag - true_mag))


def combined_loss(logits, targets):
    bce = F.binary_cross_entropy_with_logits(logits, targets)
    dice_loss = 1.0 - dice_score_from_logits(logits, targets)
    return 0.40 * dice_loss + 0.35 * focal_tversky_loss(logits, targets) + 0.15 * boundary_loss(logits, targets) + 0.10 * bce

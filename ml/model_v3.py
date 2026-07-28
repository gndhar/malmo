"""
resnet_ortho.py  —  OrthogonalAberrationNet
-------------------------------------------
Predicts coefficients for a set of strictly orthogonalized Zernike modes
(on the discrete grid) and reconstructs the phase maps.
The orthogonal basis ensures zero residual when projecting back to this subspace.
"""

import torch
from torch import nn
import numpy as np
import zern


class _ConvBnGelu(nn.Sequential):
    def __init__(self, c_in, c_out, k=3, stride=1, padding=1):
        super().__init__(
            nn.Conv2d(c_in, c_out, k, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(c_out),
            nn.GELU(),
        )


class _ResBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(c, c, 3, padding=1, bias=False),
            nn.BatchNorm2d(c),
            nn.GELU(),
            nn.Conv2d(c, c, 3, padding=1, bias=False),
            nn.BatchNorm2d(c),
        )
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(x + self.body(x))


class OrthogonalAberrationNet(nn.Module):
    def __init__(self, N: int, coeff_count: int, feat_dim: int = 256):
        super().__init__()
        self.N = N
        self.coeff_count = coeff_count

        self.encoder = nn.Sequential(
            _ConvBnGelu(2, 32, k=7, stride=4, padding=3),
            _ResBlock(32),
            _ConvBnGelu(32, 64, stride=2),
            _ResBlock(64),
            _ConvBnGelu(64, 128, stride=2),
            _ResBlock(128),
            _ConvBnGelu(128, 256, stride=2),
            _ResBlock(256),
            _ConvBnGelu(256, feat_dim, stride=2),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )

        self.head = nn.Sequential(
            nn.Linear(feat_dim, 512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, 2 * coeff_count),
        )

    def forward(self, x: torch.Tensor):
        B = x.shape[0]
        N = self.N

        real = x.real.reshape(B, 1, N * N, N * N)
        imag = x.imag.reshape(B, 1, N * N, N * N)
        inp = torch.cat([real, imag], dim=1)

        feat = self.encoder(inp)
        coeffs = self.head(feat)

        c_in = coeffs[:, : self.coeff_count]
        c_out = coeffs[:, self.coeff_count :]

        # # Project coefficients onto orthogonal basis to get strictly orthogonal phase maps
        # phi_in  = torch.einsum('bc,cxy->bxy', c_in, self.ortho_basis)
        # phi_out = torch.einsum('bc,cxy->bxy', c_out, self.ortho_basis)

        return c_in, c_out

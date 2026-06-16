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
from config import config

def get_zernike_basis(N: int, coeff_count: int) -> torch.Tensor:
    """
    Evaluates Zernike modes and crops to center NxN to form a fixed basis.
    """
    nk = zern.cart.nk
    modes = []
    
    # Force zernike init if not done
    if N != config.N:
        _ = zern.generate_abberations([0.0]*nk)
        
    for i in range(coeff_count):
        c_np = np.zeros(nk)
        c_np[i] = 1.0
        # Evaluate Zernike phase surface (before exponentiation)
        phi = zern.cart.eval_grid(c_np, matrix=True)
        phi = np.nan_to_num(phi, nan=0.0)
        
        N2 = N // 2
        phi_crop = phi[N2:N2+N, N2:N2+N]
        modes.append(phi_crop.flatten())
        
    Z = np.stack(modes, axis=-1)  # (N*N, coeff_count)
    
    basis = Z.T.reshape(coeff_count, N, N).astype(np.float32)
    return torch.tensor(basis)


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


class AberrationNet(nn.Module):
    def __init__(self, N: int, coeff_count: int, feat_dim: int = 256):
        super().__init__()
        self.N           = N
        self.coeff_count = coeff_count

        self.encoder = nn.Sequential(
            _ConvBnGelu(2,   32, k=7, stride=4, padding=3),  
            _ResBlock(32),
            _ConvBnGelu(32,  64, stride=2),                  
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

        basis = get_zernike_basis(N, coeff_count)
        self.register_buffer("zernike_basis", basis)

    def forward(self, x: torch.Tensor):
        B = x.shape[0]
        N = self.N

        real = x.real.reshape(B, 1, N * N, N * N)
        imag = x.imag.reshape(B, 1, N * N, N * N)
        inp  = torch.cat([real, imag], dim=1)

        feat = self.encoder(inp)
        coeffs = self.head(feat)

        c_in  = coeffs[:, :self.coeff_count]
        c_out = coeffs[:, self.coeff_count:]

        # Project coefficients onto Zernike basis to get phase maps
        phi_in  = torch.einsum('bc,cxy->bxy', c_in, self.zernike_basis)
        phi_out = torch.einsum('bc,cxy->bxy', c_out, self.zernike_basis)

        return coeffs, phi_in, phi_out

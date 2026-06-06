import torch
import torch.nn as nn
import numpy as np
from zernike import RZern
from config import config


class ZernikeAberration(nn.Module):
    def __init__(self, N: int = config.N, zern_n: int = config.zern_n):
        super().__init__()
        self.N = N
        self.zern_n = zern_n

        # NumPy setup (runs once on CPU during initialization)
        x = y = np.linspace(-2.0, 2.0, 2 * self.N)
        xv, yv = np.meshgrid(x, y)

        cart = RZern(self.zern_n)
        cart.make_cart_grid(xv, yv)

        basis_list = []
        for i in range(cart.nk):
            c = np.zeros(cart.nk)
            c[i] = 1.0
            basis_list.append(cart.eval_grid(c, matrix=True))

        basis_tensor = torch.tensor(np.array(basis_list), dtype=torch.float32)
        # basis_tensor = torch.nan_to_num(basis_tensor, nan=0.0) <-

        # Lightning looks for this. It handles all .to(device) calls automatically.
        self.register_buffer("zernike_basis", basis_tensor)

    @property
    def num_coefficients(self) -> int:
        return self.zernike_basis.shape[0]

    def forward(self, coeffs: torch.Tensor) -> torch.Tensor:
        nk = self.num_coefficients

        # Slicing is inherently device-agnostic
        if coeffs.shape[-1] > nk:
            coeffs = coeffs[..., :nk]
        # Using .new_zeros() guarantees the padding matches the device/dtype of coeffs
        elif coeffs.shape[-1] < nk:
            missing_count = nk - coeffs.shape[-1]
            padding_shape = list(coeffs.shape[:-1]) + [missing_count]
            padding = coeffs.new_zeros(padding_shape)
            coeffs = torch.cat([coeffs, padding], dim=-1)

        phi = torch.einsum("...k,khw->...hw", coeffs, self.zernike_basis)
        return torch.nan_to_num(torch.exp(1j * phi), nan=0.0)  # <-

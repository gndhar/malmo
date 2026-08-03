import torch
import torch.nn as nn
import numpy as np
from zernike import RZern


class ZernikeAberration(nn.Module):
    def __init__(self, N: int, zern_n: int, dtype=torch.float32, npdtype=np.float32):
        super().__init__()
        self.N = N
        self.zern_n = zern_n
        self.dtype = dtype
        self.npdtype = npdtype

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

        raw_basis = np.array(basis_list)

        # 1. Create a binary mask where the Zernike polynomials are valid (Not NaN)
        # Since all basis functions share the same support, we can just check the first one.
        pupil_mask_np = (~np.isnan(raw_basis[0])).astype(self.npdtype)
        pupil_mask_tensor = torch.tensor(pupil_mask_np, dtype=self.dtype)

        # 2. Clean the basis tensor safely by replacing NaNs with 0.0
        basis_tensor = torch.tensor(raw_basis, dtype=self.dtype)
        basis_tensor = torch.nan_to_num(basis_tensor, nan=0.0)

        # Lightning handles all .to(device) calls automatically for these buffers
        self.register_buffer("zernike_basis", basis_tensor)
        self.register_buffer("pupil_mask", pupil_mask_tensor)

    @property
    def num_coefficients(self) -> int:
        return self.zernike_basis.shape[0]

    def forward(self, coeffs: torch.Tensor) -> torch.Tensor:
        phi = torch.einsum("...k,khw->...hw", coeffs, self.zernike_basis)

        # Compute complex wavefront. Outside the pupil, phi=0 -> exp(1j*0) = 1.0
        wavefront = torch.exp(1j * phi)

        # 3. Multiply by the binary mask to force the outside magnitude back to 0.0
        return wavefront * self.pupil_mask

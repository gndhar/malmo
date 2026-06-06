import torch
import torch.nn as nn
import torch.fft as fft


class Simulation(nn.Module):
    def __init__(self, N, dtype=torch.complex64):
        super().__init__()
        self.N = N

        # Create the grid once during initialization (on CPU by default)
        k_in = torch.zeros((N, N, 2 * N, 2 * N), dtype=dtype)
        x_idx = torch.arange(N)[:, None]
        y_idx = torch.arange(N)
        k_in[x_idx, y_idx, (N // 2) + x_idx, (N // 2) + y_idx] = 1.0 + 0.0j

        # Register it as a buffer.
        # Lightning will now manage its device and dtype automatically!
        self.register_buffer("k_in", k_in)

    def forward(self, ab_in, ab_out, obj):
        N = self.N

        k_inc = self.k_in * ab_in

        # Transform to Real Space
        r_inc = fft.ifft2(k_inc, dim=(-2, -1))

        # Interact with Object
        r_ref = r_inc * obj

        # Transform back to K-space
        k_ref = fft.fft2(r_ref, dim=(-2, -1))

        # Apply output aberrations
        k_out = k_ref * ab_out

        # Crop
        start = N // 2
        end = start + N

        k_ins = self.k_in[:, :, start:end, start:end]
        k_outs = k_out[:, :, start:end, start:end]

        return k_ins, k_outs

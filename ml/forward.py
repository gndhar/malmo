# verified
import torch
import torch.nn as nn


class Simulation(nn.Module):
    def __init__(self, N: int, dtype=torch.complex64):
        super().__init__()
        self.N = N

        # Create the initial k_in grid
        k_in = torch.zeros((N, N, 2 * N, 2 * N), dtype=dtype)
        x_idx = torch.arange(N)[:, None]
        y_idx = torch.arange(N)
        k_in[x_idx, y_idx, (N // 2) + x_idx, (N // 2) + y_idx] = 1.0 + 0.0j

        # Register the full grid as a buffer
        self.register_buffer("k_in", k_in)

        # Pre-compute and register the cropped version
        start = N // 2
        end = start + N
        k_in_cropped = k_in[:, :, start:end, start:end]

        # This acts as your property (accessible via self.k_in_cropped)
        # and automatically moves to the GPU when you call model.to(device)
        self.register_buffer("k_in_cropped", k_in_cropped)

    def forward(self, ab_in, ab_out, obj):
        k_inc = self.k_in * ab_in

        # Transform to Real Space
        r_inc = torch.fft.fftshift(
            torch.fft.ifft2(
                torch.fft.ifftshift(k_inc, dim=(-2, -1)), dim=(-2, -1), norm="ortho"
            ),
            dim=(-2, -1),
        )

        # Interact with Object
        r_ref = r_inc * obj

        # Transform back to K-space
        k_ref = torch.fft.fftshift(
            torch.fft.fft2(
                torch.fft.ifftshift(r_ref, dim=(-2, -1)), dim=(-2, -1), norm="ortho"
            ),
            dim=(-2, -1),
        )

        # Apply output aberrations
        k_out = k_ref * ab_out

        # Crop and return ONLY the output
        start = self.N // 2
        end = start + self.N
        k_outs = k_out[:, :, start:end, start:end]

        return k_outs
